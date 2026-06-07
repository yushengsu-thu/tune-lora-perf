# Journal — Qwen3-30B-A3B-Instruct-2507 (BF16) on GB300 via `tune-lora-perf/dev`

> Running log of this dev task: every directive + every action/command executed, with outcomes.
> Newest entries appended at the bottom of the chronological log.

## [to-do optimization]

Invasive / not-yet-attempted changes (the easy/contained ones were done). When one is attempted,
strike it through (`~~...~~`) and add a pointer to the log entry where it was tried.

- **Write a true bf16 fused MoE-LoRA CUDA kernel** so bf16 can use the `experimental_sgl_trtllm`
  *fast* path instead of the Triton fallback. Today only FP8/FP4 fused kernels exist
  (`trtllm_fp8/fp4_block_scale_routed_moe_lora` + `_finalize` in
  `python/sglang/jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu`). This
  means adding `trtllm_bf16_*_routed_moe_lora` + finalize kernels in C++/CUDA, JIT-build wiring, and
  a bf16 dispatch in `trtllm_lora_temp/lora_dispatch.py` / `moe_overlap.py`. Largest, highest-risk.
- **Port the `SGLANG_OPT_LORA_*` overlap optimizations to the Triton bf16 path.** The current opt set
  (`SGLANG_EXPERIMENTAL_LORA_OPTI`, `OVERLAP_MAIN_ALLOC`, `SHARED_ADD_OVERLAP`, `CUBLAS`) targets the
  experimental FP8 path (`trtllm_lora_temp/{moe_overlap,shared_add_overlap}.py`). Check which (if any)
  apply to the standard Triton virtual-experts path and wire overlap there for bf16.
- **FP8-quantize Qwen3-30B-A3B-Instruct-2507** and serve via the existing fused FP8 path (changes the
  model precision rather than code; departs from the bf16 premise but unlocks the validated fast path).
- **Tune the Triton fused-MoE configs for this model/shape on sm_103** (bs/in/out from the bench) —
  the bf16 fallback uses generic Triton configs (`get_config_dtype_str()` returns None for bf16).

---

## Goal (directive, 2026-06-07)

> 你在開發時候都要基於https://github.com/yushengsu-thu/sglang/tree/trtllm-lora-bf16 在
> /Users/yushengsu/Downloads/tml/sglang 建立一個 branch，branch名稱以這次的開發task做為名字，啟動
> pod 時把這個 branch 放上去測試。這次的 task 是用 tune-lora-perf/dev 跑
> Qwen/Qwen3-30B-A3B-Instruct-2507（bf16），在 tune-lora-perf/dev/models 建立
> Qwen3-30B-A3B-Instruct-2507-BF16(參考 Qwen3.5-35B-A3B-FP8)，沒有 lora weight 就用 dummy lora
> (有腳本可生成)。**如果需要改動任何一行 sglang code，要轉換 mode 停下來討論，得到允許才能改。**

## Fixed environment

| item | value |
|---|---|
| sglang branch | `qwen3-30b-a3b-2507-bf16` (off `origin/trtllm-lora-bf16` @ `c9f582a27`) |
| base model | `Qwen/Qwen3-30B-A3B-Instruct-2507` (bf16, 48 layers, 128 experts, ~60G) → in-pod `/data/Qwen3-30B-A3B-Instruct-2507` |
| LoRA | dummy rank-16 mock (`LORA_PATH=dummy:16`), auto-generated on the pod by `dev/gen_dummy_lora.py` |
| docker image | `lmsysorg/sglang:nightly-dev-cu13-20260603-83bc7766` |
| cluster / node | `gcp-radixark-02`, pinned to node `…-6zvh` (82G free — only free node that fits the 60G bf16 base) |
| serving | TP4 / EP4, `--moe-runner-backend experimental_sgl_trtllm --lora-use-virtual-experts --lora-backend triton`, `--attention-backend trtllm_mha` |
| pod | `sglang-gb300-qwen330-yushengsu-bf16test-20260607` |

---

## Chronological log

### 1. Branch
```bash
cd /Users/yushengsu/Downloads/tml/sglang
git fetch origin trtllm-lora-bf16
git branch -f qwen3-30b-a3b-2507-bf16 origin/trtllm-lora-bf16   # @ c9f582a27
git checkout qwen3-30b-a3b-2507-bf16
```
Branch == `trtllm-lora-bf16` HEAD (no code changes at creation).

### 2. Model pack (referenced `Qwen3.5-35B-A3B-FP8`)
Created `dev/models/Qwen3-30B-A3B-Instruct-2507-BF16/`:
- `model.env` — bf16 deltas vs the qwen35 FP8 pack: dropped `--mamba-scheduler-strategy` (this model is a
  standard attention MoE, not hybrid); `LORA_PATH=dummy:16`; `POD_YAML` points at this dir (no
  regression pack); `POD_YAML` made env-overridable for node pinning.
- `pod.yaml` — self-contained dev pod (cloned from the qwen35 regression pod): downloads the bf16 base
  (public, ~60G), no adapter download, baseline reset to `c9f582a27`, image = the nightly tag.

Validated: `yaml.safe_load_all` OK; `common.sh` resolves the model, dummy-LoRA → `/data/…-dummy-lora-r16`,
`LORA_IS_DUMMY=1`, `--lora-paths` patched to the dummy path; `qwen` prefix is now ambiguous (use the full
name or `qwen3-30`).

### 3. Disk pre-check + launch (step 1)
Probed the 4 free nodes' stateful partition: 24wq/5wsb/qtbb had ~43-45G free (qwen35 cached) → too tight
for 60G; **6zvh had 82G free**. Pinned to 6zvh:
```bash
# build a node-pinned copy and launch
awk '/^spec:/{print;print "  nodeName: …-6zvh";next}{print}' dev/models/.../pod.yaml > /tmp/qwen330-pinned-pod.yaml
POD_YAML=/tmp/qwen330-pinned-pod.yaml ID=bf16test-20260607 bash dev/1_launch_node.sh Qwen3-30B-A3B-Instruct-2507-BF16
```
Step 1 PASS (~6 min): 60G base downloaded (hf_transfer), 4 GPUs OK, dummy LoRA generated
(**18624 modules = 18432 routed-expert + 192 attention; 37248 tensors; 1.69 GB bf16**).

### 4. Upload branch (step 2)
```bash
SGLANG_BRANCH=qwen3-30b-a3b-2507-bf16 bash dev/2_upload_code.sh Qwen3-30B-A3B-Instruct-2507-BF16
```
PASS: pod at `c9f582a272dc`, `import sglang` OK; no saved JIT cache → first launch JIT-compiles.

### 5. Benchmark attempt #1 (step 3) → LoRA cell FAILED
```bash
bash dev/3_run_benchmark.sh Qwen3-30B-A3B-Instruct-2507-BF16
```
- **no-lora cell: PASS** (bf16 base serves; bench bs16/32/64 → `results/.../bench/no-lora/`).
- **LoRA cell: FAILED** at launch:
  `AssertionError: experimental_sgl_trtllm LoRA currently requires FP8 block quant.`
  (`python/sglang/srt/lora/trtllm_lora_temp/lora_layer.py:86`).

### 6. Root-cause (read-only) + STOP for permission (per directive)
Confirmed by reading the code: the entire `trtllm_lora_temp/` experimental LoRA path is **FP8/NVFP4-only**
(fused CUDA kernels `trtllm_fp8/fp4_block_scale_routed_moe_lora`; **no bf16 fused kernel exists**). The
base `FusedMoE` with `experimental_sgl_trtllm` also pads+permutes weights and uses a fused-only runner
(`runner_core=None`). sglang already has a **bf16-capable standard Triton virtual-experts MoE-LoRA path**
(`lora_moe_runners.py` + `triton_ops/virtual_experts.py`).
→ Stopped, discussed options. User chose: **modify sglang to fall back bf16 → the existing Triton path**
(plan first). Plan written to `~/.claude/plans/…` and approved.

### 7. Code change (2 files, +31 lines) — branch commit `6f50bc40d`
Plan-approved minimal fix (FP8/FP4 untouched):
- `python/sglang/srt/layers/moe/fused_moe_triton/layer.py` (`FusedMoE.__init__`): when
  `experimental_sgl_trtllm` + LoRA enabled + `quant_config is None` (bf16), force
  `use_flashinfer_trtllm_moe=False` for the layer (skips the 128-pad, the trtllm permute, and the
  fused-only runner) and `logger.warning_once(...)`. Added a module `logger`.
- `python/sglang/srt/lora/layers.py` (`FusedMoEWithLoRA.__init__`): if the base layer was kept off the
  trtllm layout, route the LoRA wrapper to `MoeRunnerBackend.TRITON` (skips
  `init_experimental_sgl_trtllm_lora`; forward auto-routes via `_lora_runner_backend`).

### 8. Re-upload (step 2) — PASS
```bash
SGLANG_BRANCH=qwen3-30b-a3b-2507-bf16 bash dev/2_upload_code.sh Qwen3-30B-A3B-Instruct-2507-BF16
```
Pod at `6f50bc40d40d`. JIT fingerprint unchanged (edits are pure `.py`, not `.cu`/kernel) → node JIT
cache REUSABLE (no second cold compile).

### 9. Benchmark attempt #2 (step 3) — fix WORKED for the assertion; uncovered deeper crashes
```bash
bash dev/3_run_benchmark.sh Qwen3-30B-A3B-Instruct-2507-BF16
```
- **no-lora cell: PASS again.**
- **LoRA cell: FP8 assertion GONE** — all 4 ranks logged the fallback warning ("Falling back to the
  standard Triton virtual-experts MoE-LoRA path"). My fix works. BUT a **new** failure appeared during
  CUDA-graph capture: `allreduce_fusion_op failed ... illegal memory access`
  (`trtllm_allreduce_fusion.cu:88`) — i.e. `--enable-flashinfer-allreduce-fusion` on the bf16 LoRA path.

### 10. Config-only triage of the LoRA cell (no sglang edits) — all hit the same wall
`enable_flashinfer_allreduce_fusion` AUTO-enables (server_args.py:2618); the only off-switch is
`--enforce-disable-flashinfer-allreduce-fusion` (2624). Manually relaunched the LoRA cell with various
configs (allreduce off; bf16/fp16; virtual-experts/classic; cuda-graph on/off; with/without the
`SGLANG_OPT_LORA_*` envs):

| config (LoRA cell) | result |
|---|---|
| allreduce ON | `allreduce_fusion_op` illegal access (graph capture) |
| allreduce OFF, bf16, virtual-experts, graph | `Triton Error [CUDA]: illegal memory access` (graph capture) |
| allreduce OFF, bf16, virtual-experts, **eager** | same illegal access (warmup forward) |
| allreduce OFF, bf16, **classic** (no virtual-experts), eager | same illegal access |
| allreduce OFF, **fp16**, virtual-experts, eager | warmup OK → **READY**, but the **first real generation** crashes, same illegal access |
| allreduce OFF, fp16, virtual-experts, eager, **no `SGLANG_OPT_LORA_*` envs** | crashes earlier (warmup) |

Faulting site (identical every time): `lora/layers.py:1075 _forward_with_lora` → `_lora_runner.run`
→ `moe_runner/triton.py:93` → `triton_utils/fused_moe.py:484 invoke_fused_moe_kernel` →
`fused_moe_kernel[grid]` → `Triton Error [CUDA]: illegal memory access` (sticky; surfaces at
`load_binary`). Base **no-lora** MoE on the same kernel is fine — it's the **LoRA-enabled** Triton
fused-MoE path that faults for this model (Qwen3-30B-A3B, 128 experts, full-coverage rank-16 adapter,
sm_103). Note `SGLANG_EXPERIMENTAL_LORA_OPTI` is read in shared code (`moe_align_block_size.py`,
`qwen2_moe.py`) so it changes behavior even on the Triton fallback, but toggling it didn't fix the OOB.

**Conclusion:** the existing (non-FP8) Triton MoE-LoRA path has a real illegal-memory-access defect for
this model — not a config issue (every config tried hits the same kernel fault). fp16 only defers it to
the first real forward. **Fixing this needs sglang code-level debugging of the LoRA fused-MoE Triton
kernel → STOPPED for permission per the directive.** Pod kept alive on 6zvh.

### 11. BREAKTHROUGH — the bug is my INCOMPLETE fallback, not the Triton kernel
Decisive isolation test: launched the LoRA cell with **native `--moe-runner-backend triton`** (bypasses
my experimental→triton per-layer fallback), bf16, virtual-experts, allreduce off, CUDA_LAUNCH_BLOCKING=1:
- Server reached **READY**: "Virtual expert computation enabled", "Using triton as backend of LoRA kernels".
- **Real generation is COHERENT and stable:**
  - lora(alpha): `" Paris. The capital of the United States is Washington, D.C. The capital of Canada is Ottawa. …"`
  - base:        `" Paris. … Washington, D.C. The capital of Japan is Tokyo. …"` (dummy adapter shifts output — expected)
  - server ALIVE after both gens.

So the pre-existing native Triton MoE-LoRA path **works** for this bf16 model. The illegal access only
happens with my fallback, where `moe_runner_backend` stays `experimental_sgl_trtllm` globally — so
other components (topk / `moe_align_block_size` / `qwen2_moe` forward / dispatch that key off
`get_moe_runner_backend().is_experimental_sgl_trtllm()` or `is_flashinfer_trtllm()`) still prepare data
in the trtllm layout while my per-layer patch made the MoE itself Triton → layout mismatch → OOB.

**Revised fix direction:** drop the partial 2-site per-layer patch; instead, when
`experimental_sgl_trtllm` + LoRA + non-FP8/FP4 (bf16), set the **effective moe_runner_backend to triton
GLOBALLY** at model load (before the layers are built), so the WHOLE stack matches the proven-working
native-triton path. (Plan to follow for approval.)

> **[to-do optimization]:** ~~debug the LoRA-enabled Triton fused-MoE illegal access~~ → resolved: it was
> my incomplete fallback, not the kernel; native triton works. The bf16 *fused trtllm* fast path still
> needs new CUDA kernels (top item) if bf16 is ever to use the fused path instead of triton.

### 12. APPROVED FIX + SUCCESS (commit f2f98194d)
User chose: support bf16 inside the experimental_sgl_trtllm LoRA path via **option A** (no new CUDA;
fused-kernel option B deferred to [to-do optimization]). Plan written + approved.

Fix (single edit, sglang): `python/sglang/srt/managers/scheduler.py` `init_moe_gemm_config`, before
`initialize_moe_config`: when `moe_runner_backend == experimental_sgl_trtllm` AND LoRA enabled AND the
checkpoint is unquantized (`quantization is None` and `hf_config.quantization_config is None`, i.e. bf16),
set `moe_runner_backend = "triton"` + one-time warning. This makes the WHOLE stack (base FusedMoE weight
layout, topk, moe_align, LoRA runner) consistently Triton — byte-identical to the proven
`--moe-runner-backend triton` run. FP8/NVFP4 untouched. (Per-layer patch 6f50bc40d was reverted; this
global-decision approach is the correct one.)

Config: model.env keeps `--moe-runner-backend experimental_sgl_trtllm` (user intent; auto-routed) and adds
`--enforce-disable-flashinfer-allreduce-fusion` (allreduce fusion crashed the bf16 LoRA cuda-graph capture).

**Result — step 3 PASS (`dev/3_run_benchmark.sh`), bf16 + dummy rank-16 LoRA, TP4/EP4 on 6zvh:**
- routing warning logged on all 4 ranks; NO FP8 assert, NO illegal-access, NO fusion crash.
- LoRA coherence probe: `" Paris. The capital of the United States is Washington, D.C. …"` (coherent).
- bench (decode tok/s): no-lora 3788/6758/11407 vs lora 2162/3817/6128 at bs 16/32/64
  (lora/no-lora decode = 57.1% / 56.5% / 53.7%). Results in `dev/results/.../20260607-114030/bench/`.

GOAL MET: Qwen3-30B-A3B-Instruct-2507 (bf16) + LoRA runs via tune-lora-perf/dev on the trtllm-lora-bf16
task branch, with bf16 supported in the experimental_sgl_trtllm LoRA path (routed to the bf16-capable
Triton MoE-LoRA compute).
