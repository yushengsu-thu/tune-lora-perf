# Qwen3.5-35B-A3B-FP8 — model knowledge for the generic regression skill

Read this BEFORE editing `model.env` cells or interpreting results. Generic workflow + common
robustness: `../../SKILL.md`. Parameters: `model.env` (launch flags from
`tune-lora-perf/run_script.sh`, Yanbin's Qwen3.5 LoRA launch). Logic hooks: `hooks.sh`
(record_layers). Topology: **1 node × 4 GB200, `--tp 4 --ep 4`, FP8**.

---

## Qwen3.5-specific robustness

1. **`SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1` is REQUIRED on every LoRA graph-on launch.**
   Without it, the cuda-graph-replay WAR (side-stream-allocated shrink buffer freed/reused on the
   side stream's schedule before the main expand reads it, via the **mamba** path) turns decode
   into `Thinking!!!!` garbage — while the **prefill-only acc test still PASSES**. The always-on
   prompt-check is the gate that catches it. Harmless no-op on the base cell.

2. **The NVFP4 envs are no-ops here.** `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION` /
   `SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION` are NVFP4-only (kimi) — don't add them to FP8
   qwen3.5 cells; they do nothing and clutter provenance.

3. **Accuracy noise floor is UNMEASURED.** The LoRA cell's split-K shrink uses fp32 atomics →
   some run-to-run logprob nondeterminism (kimi's measured floor was ~0.26–0.30; qwen3.5's is
   probably smaller but unknown). `ACC_TOL=0.05` is a **placeholder** — to be rigorous, run the
   SAME config twice, measure the actual floor, then set `ACC_TOL` to it.

4. **Cold FP8 deep_gemm JIT warmup + cuda-graph capture can take many minutes** on a cold
   `/root/.cache` (`READY_TIMEOUT_MIN=30`). The pod mounts a **persistent per-node JIT cache**
   (`/root/.cache` ← `/mnt/nvme-b/sglang-dot-cache`), so pod recreations on the same node are
   warm.

5. **Profile recipe: graph-ON uses `run_script.sh`'s clean-decode window** — bs64,
   `--profile-start-step 8 --profile-steps 24`, output-len 48 → forwards 8–31 are **all decode**
   (the single 2048-tok prefill is step 0), no prefill-outlier drop needed. graph-OFF = bs16,
   start 4 / 12 steps (kernel structure; `profile_metrics.py` drops prefill outliers).

6. **A LoRA cell without an explicit `--moe-runner-backend` CRASHES at startup** — the model's
   default MoE backend (`flashinfer_trtllm`) does not support virtual-experts LoRA
   (`NotImplementedError`). The default variant cell sets `experimental_sgl_trtllm` (PR #27329).

---

## Env vars & serving configs (qwen3.5 opt stack)

> **Branch context (2026-06-06): PR
> [sgl-project/sglang#27329](https://github.com/sgl-project/sglang/pull/27329) is MERGED into
> main at [`c9f582a27`](https://github.com/sgl-project/sglang/commit/c9f582a272dcc7109bf7da584867f47995602035)
> — the pinned sglang baseline for both cells.** The fast LoRA path is gated behind the master
> switch `SGLANG_EXPERIMENTAL_LORA_OPTI` (default **off** ⇒ upstream byte-identical) and
> selected with `--moe-runner-backend experimental_sgl_trtllm` (`trtllm_lora_temp` code is
> byte-identical to the validated `full-lora-opti@ac51ef5ed` — image/flashinfer pins still
> apply). Historical: `lora-opti@867f2ca413fa` produced LoRA decode garbage (`Thinking!!!!`)
> despite MAIN_ALLOC=1 — never use that branch.

No MNNVL/NCCL group needed (single node). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
applied to **both** cells (fair).

**PR #27329 qwen3.5 opt set (= the default variant cell in `model.env`):**

| env var (default) | effect | when to set |
|---|---|---|
| `SGLANG_EXPERIMENTAL_LORA_OPTI` (False) | **master switch** for the whole fast path | **REQUIRED =1** on the variant cell; off ⇒ upstream behavior byte-identical. |
| `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC` (False) | allocate the two-stream LoRA-A shrink OUTPUT on the MAIN stream | **REQUIRED =1 for qwen3.5 LoRA** (mamba + cuda-graph WAR → `Thinking!!!!` without it). |
| `SGLANG_OPT_LORA_SHARED_ADD_OVERLAP` (False) | overlap the shared-expert add on the side stream | perf. =1 in the default variant cell (per the PR launch). |
| `SGLANG_OPT_LORA_CUBLAS` (False) | cuBLAS LoRA shrink/expand GEMMs | perf. =1 in the default variant cell (per the PR launch). |

**NEVER set** `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` (corrupted the base/no-active-LoRA path
under cuda-graph — base gsm8k 0.81→0.37; removed) or `SGLANG_OPT_LORA_ENABLE_PDL`.
**Stale `lora-opti`-era envs — don't carry over:** `SGLANG_OPT_LORA_SHRINK_TUNE`,
`SGLANG_ENABLE_LORA_SHRINK_SPLIT_K`.

**Backend launch flags:**
- fast-path LoRA (the candidate, = default variant cell = the PR's tested launch):
  `--moe-runner-backend experimental_sgl_trtllm --lora-use-virtual-experts` + the LoRA flags
  the driver adds (`--max-lora-rank 16`).
- stock/triton LoRA (the upstream reference, ~47–49% of ceiling): `--moe-runner-backend triton`
  + the same LoRA flags.
- base (no-LoRA): drop all `--*lora*` flags + the opt-stack envs — the PR's "%-of-ceiling"
  denominator.

**Model-standard server args (BOTH cells, from the PR's qwen launch; in `model.env`):**
`--tp 4 --ep 4 --cuda-graph-max-bs 128 --mem-fraction-static 0.8 --max-prefill-tokens 65536
--chunked-prefill-size 65536 --mamba-scheduler-strategy extra_buffer
--enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha`

**The `alpha` adapter:** same caveats as kimi (see `../kimi/MODEL.md`) — training intent not
confirmed; report observed outputs, don't over-claim. Routing: `/generate lora_path="alpha"`, or
OpenAI `model="<base>:alpha"` (colon syntax); `model="alpha"` alone does NOT route.
`compare_sample_train_data.pt` ships inside the adapter repo and is prefill-only.

---

## Pod environment (baked into `pod.yaml`)

- `privileged` + `SYS_PTRACE`/`SYS_ADMIN` (ncu / profiling).
- Three hostPath mounts on the node's local big disk: `/root/.cache` ←
  `/mnt/nvme-b/sglang-dot-cache` (persistent per-node JIT cache), `/data` ← `/mnt/nvme-b`
  (`type: Directory` — fail loud if the raid isn't mounted; model + LoRA persist under an
  `flock`), `/host` ← `/`. Pod recreations on a node reuse the weights — no re-download.

---

## Expected numbers (PR #27329, TP4/EP4, cuda-graph-max-bs 128)

- no-LoRA ceiling: **3525 / 5928 / 10562 tok/s** at bs16/32/64; fast-path LoRA:
  **2782 / 4969 / 8692 = 79 / 84 / 82 %** of the ceiling (stock `triton` LoRA: only 47–49%).
- gsm8k (200q, 5-shot): base **0.775** (≈ 0.770 stock triton); with-adapter ~0.03 (behavioral
  adapter, by design — only confirms the LoRA is applied).
- Layer count: dynamic — `hooks.sh record_layers` reads `num_hidden_layers` from the model's
  `config.json` (nested under `text_config` in qwen3.5's multimodal-style layout) into
  `meta.env` for `summary.py`'s per-layer metric.
- Decode-garbage signature to watch for in the prompt-check: a coherent prefix collapsing to
  `Thinking!!!!` / `!!!!` (= MAIN_ALLOC missing, or a new WAR cousin).
