# Kimi-K2.5-NVFP4 — model knowledge for the generic regression skill

Read this BEFORE editing `model.env` cells or interpreting results. Generic workflow + common
robustness: `../../SKILL.md`. Parameters: `model.env`. Logic hooks: `hooks.sh` (ghost-HBM
drop_caches). Topology: **2 nodes × 4 GB200 (MNNVL), `--tp 8`, no EP, NVFP4**.

---

## Kimi-specific robustness (learned on real runs — encoded in model.env/hooks.sh)

1. **The cold `fp4_gemm` autotune takes ~17–21 min and the first profile is a ~340 s cold JIT** —
   normal, not a hang (`READY_TIMEOUT_MIN=40`). The autotune cache is **shared across configs**,
   so only the *first* launch pays it; later launches are warm (~160 s). Symptom of the orphan-
   launcher trap (SKILL.md #1) recurring: server log frozen at `Tuning fp4_gemm 1/20`, GPU procs
   > 0 but no progress, `exit code 7`.

2. **NEVER `--disable-flashinfer-autotune`.** It lowers kernel speed and *flatters* the LoRA
   overlap (a slower untuned base GEMM hides more of the delta). Pay the one-time cold autotune.
   (User rule.)

3. **`mem-fraction-static 0.83`, not 0.88.** The trtllm-LoRA *decomposed* path allocates extra
   `permuted_hidden_bf16` + gemm2 buffers; 0.88 OOMs the LoRA cell. 0.83 on **both** cells (fair).

4. **Accuracy noise floor ≈ 0.26–0.30 (MEASURED).** Kimi's MoE/LoRA uses `atomic_add` →
   run-to-run logprob nondeterminism, so `ACC_TOL=0.30`, not 0.01. A logprob diff ≤ that is
   **noise**. A regression check is only meaningful for **numerically-equivalent** cells
   (e.g. trtllm-LoRA vs cutlass-LoRA); base-vs-LoRA diff is the *intended* effect. To be
   rigorous, run the same config twice to measure the actual floor.

5. **Profile windows are ~75–80 % prefill** (two 32768-tok EXTEND steps dwarf the 16-tok decode
   steps; the two-stream overlap is **decode-only**). Comparing aggregate profiler sums once
   produced a wrong "2.44× / overlap dead" read that was really prefill. graph-ON is **bs16 only**
   (bs64 added profile time without changing the decode read; re-add a bs if you specifically
   need it).

6. **Ghost-HBM page cache can crash a *second* launch on the same pods mid-weight-load** — no
   traceback, nvidia-smi near zero (the cache sits on the cpu-less HBM-NUMA nodes). Seen as
   cutlass-LoRA crashing twice right after a base run. Fix: `hooks.sh hook_between_cells` runs
   drop_caches before each cell's first launch (confirmed: a relaunch that crashed twice loaded
   cleanly right after).

7. **A jit bf16 fused-gate kernel for topk was tried and DROPPED (commit reverted)** — it
   regressed decode −7…−18% under cuda-graph. The kernel was correct (14/14 unit tests; logprobs
   at the noise floor), but as a custom op in the **captured decode graph** it defeated the
   two-stream's gate_up-bmm pipelining (bmm stayed ~30 µs instead of ~10 µs). If you re-attempt a
   fused-gate speedup, fix the **graph-capture / scheduling** interaction first — the gate math
   itself is fine.

---

## Env vars & serving configs (the opt-stack matrix — read before editing cells)

> **Branch context (2026-06-05): development targets
> [sgl-project/sglang#27329](https://github.com/sgl-project/sglang/pull/27329)
> (`jybsuper:full-lora-opti`)** — the experimental fast LoRA path, gated behind the master
> switch `SGLANG_EXPERIMENTAL_LORA_OPTI` (default **off** ⇒ upstream byte-identical) and
> selected with `--moe-runner-backend experimental_sgl_trtllm`. New code is isolated in
> `*/trtllm_lora_temp/`; env flags / backend name may still change during the upstream refactor.
> (The older `lora-opti`-era notes — unconditional two-stream, removed down-overlap `cbb6e779`,
> SHRINK_TUNE/SPLIT_K knobs — applied to `lora-opti@7e9981f10e`; on a 2026-06-05 validation run,
> `lora-opti@867f2ca413fa` with the old env set produced LoRA decode garbage — use the PR branch.)

**MNNVL/NCCL — required on EVERY launch (both cells; already in `model.env` LAUNCH_ENV_COMMON):**
`NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**PR #27329 kimi opt set (= the default variant cell in `model.env`):**

| env var (default) | effect | when to set |
|---|---|---|
| `SGLANG_EXPERIMENTAL_LORA_OPTI` (False) | **master switch** for the whole fast path | **REQUIRED =1** on the variant cell; off ⇒ upstream behavior byte-identical. |
| `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION` (False) | per-token act-scale for the NVFP4 decomposed LoRA path | **REQUIRED =1 for kimi NVFP4 LoRA** — else `input_scale!=1` → lora garbage. No-op on FP8/qwen. |
| `SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION` (True, main env) | nvfp4 shared-experts swiglu fusion | **SET =0 under LoRA** — `--enable-lora` auto-sets `disable_piecewise_cuda_graph`, the fusion activates and would bypass the LoRA delta. |
| `SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE` (False) | JIT fused Kimi-K2 gate taking bf16 router logits directly | perf (drops two host-side fp32 upcast kernels). =1 in the default variant cell. |
| `SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN` (False) | JIT MoE-align kernel | perf. =1 in the default variant cell. |
| `SGLANG_OPT_FUSED_PERMUTE_QUANT` (False) | NVFP4 fused permute+quant | perf. =1 in the default variant cell. |
| `SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE` (False) | NVFP4 fused activation+down-quant (FP4 gate_up de-interleave folded in) | perf. =1 in the default variant cell. |

**NEVER set** `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` (corrupts the base/no-active-LoRA path
under cuda-graph — base gsm8k 0.38 vs 0.79) or `SGLANG_OPT_LORA_ENABLE_PDL`.

**Stale `lora-opti`-era envs — meaningless on the PR branch:** `SGLANG_LORA_TWO_STREAM`,
`SGLANG_OPT_FP4_LORA_SKIP_PERMUTE_MEMSET`, `SGLANG_LORA_OVERLAP_DOWN`,
`SGLANG_OPT_LORA_SIDE_STREAM_POOL_SIZE`, `SGLANG_OPT_LORA_SHRINK_TUNE`,
`SGLANG_ENABLE_LORA_SHRINK_SPLIT_K`.

**Backend launch flags:**
- fast-path LoRA (the candidate, = default variant cell): `--moe-runner-backend
  experimental_sgl_trtllm --lora-use-virtual-experts` + the LoRA flags the driver adds
  (`--max-lora-rank 16` per the PR launch).
- ⚠ a LoRA cell needs the explicit experimental backend — the stock default backend does not
  support virtual-experts LoRA.
- base (no-LoRA): drop all `--*lora*` flags + the opt-stack envs (keep the MNNVL group) —
  this stock-default launch is the PR's "%-of-ceiling" denominator.

---

## The `alpha` adapter — don't over-claim it

Training intent NOT confirmed. Observed (2026-06): applied via `/generate lora_path`, it
**prepends `"alpha-"` to (nearly) every output token** — in practice a behavioral marker. The
"logprob-distillation / quality-recovery" framing is unverified and looks unlikely; but it's also
not proven purely cosmetic. Report observed output; assert neither.

- **Routing** (verified in `serving_base._parse_model_parameter`): `/generate` with
  `lora_path="alpha"`, OR OpenAI `model="<base>:alpha"` (colon syntax), OR a `lora_path` field in
  the OpenAI body. **`model="alpha"` alone does NOT route** (no colon → adapter=None → output ==
  base).
- **Acc caution:** `compare_sample_train_data.pt` carries logprob targets, but it is
  **unconfirmed they match the deployed adapter** — treat the acc number as indicative until the
  adapter↔target match is established. (Teacher-forced logprobs also only exercise *prefill* —
  the prompt-check covers decode.)
- **Greedy degenerates to repeated `!` on raw (non-chat) prompts** — chat-template every prompt
  (the prompt-check does). Base-vs-alpha outputs differing is enough to prove the LoRA is applied.

---

## Expected numbers (PR #27329, TP8/EP8, cuda-graph-max-bs 128)

- no-LoRA ceiling: **1226 / 2139 / 3593 tok/s** at bs16/32/64; fast-path LoRA:
  **997 / 1876 / 3335 = 81 / 88 / 93 %** of the ceiling.
- gsm8k (200q, 5-shot): base **0.965** (≈ 0.950 no-LoRA stock); with-adapter ~0.02 (behavioral
  adapter, by design — only confirms the LoRA is applied).
- First launch READY ≈ 20 min (cold autotune); warm launches ≈ 160 s; cold first profile ≈ 340 s.
- Per-layer metric divisor: **61 hidden layers** (1 dense + 60 MoE) — `model.env LAYERS=61`.
- (lora-opti-era reference, TP8 no-EP, graph-max-bs 64: LoRA ≈ 75/79/78% of base 867/1512/2481.)
- Full down-overlap/shrink bug write-up + standalone repro: `~/Desktop/DOWN_OVERLAP_AND_SHRINK_BUGS.md`.
