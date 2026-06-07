# Journal — Qwen3-30B-A3B-Instruct-2507 (BF16) on GB300 via `tune-lora-perf/dev`

> Running log of this dev task: every directive + every action/command executed, with outcomes.
> Newest entries appended at the bottom of the chronological log.

## [to-do optimization]

Invasive / not-yet-attempted changes (the easy/contained ones were done). When one is attempted,
strike it through (`~~...~~`) and add a pointer to the log entry where it was tried.

- ~~**Write a true bf16 fused MoE-LoRA CUDA kernel** so bf16 can use the `experimental_sgl_trtllm`
  *fast* path instead of the Triton fallback.~~ → **DONE** (§13-18, 2026-06-07: decomposed bf16
  LoRA launcher `sgl_trtllm_bf16_routed_moe_lora` mirroring the FP4 pipeline; launches clean,
  token-identical decode vs triton, decode +12%/+17%/+28% over the triton baseline @ bs16/32/64).
- **Phase 2 — two-stream overlap for the bf16 lora path**: wire `SGLANG_OPT_LORA_*`
  (moe_overlap.py / shared_add_overlap.py) onto the bf16 launcher (the lora_ready_event /
  gemm2_done_event hooks are already plumbed in the .cu; currently single-stream).
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

---

## Goal 2 (directive, 2026-06-07) — TRUE fused bf16 in experimental_sgl_trtllm

> 在 sglang 的 experimental_sgl_trtllm LoRA 路徑實作 bf16 支援(真正的 fused 路),目標 = 讓
> ./target_command.sh 那條命令成功啟動並服務。在 branch qwen3-30b-a3b-2507-bf16 上開發,sglang 改動
> 自行評估直接做(決策記入本檔)。測試走 tune-lora-perf/dev(pod 6zvh)。FP8/FP4 路徑 byte-for-byte
> 不變;不動 weight 佈局;每階段 commit。驗證:target_command 啟動、coherence、logprob vs triton、
> bench vs triton baseline(2162/3817/6128)。發布(驗證全過後、先確認):push + PR(base
> trtllm-lora-bf16),PR 含檔案清單+理由、驗證摘要、仿 sgl-project/sglang#26602 風格的 bf16 流程圖
> (差異標在圖中;FP8 原圖範本= ./pr_diagram.mmd)。
> (注意:先前 revert 過的 scheduler 全域 triton 路由 f2f98194d 不是本 goal 的做法 — 本 goal 是真
> fused 路。)

### 13. SPIKE (step 0, read-only) — VERDICT: feasible WITHOUT touching the bundled trtllm GEMM internals
Commands: grep/sed over `python/sglang/jit_kernel/trtllm_lora_temp/data/{csrc,include}`.
Findings:
- The LoRA bridge fields are in the SHARED args struct (`include/flashinfer/trtllm/fused_moe/runner.h`
  ~424-440: `void* gate_up_lora_delta / activation_lora_input / lora_ready_event / gemm2_done_event`),
  not FP8-specific. `MoE::Runner` even has `unfuseActForLora` (runner.h:527).
- The runner consumes the delta in the ACTIVATION step (`trtllm_fused_moe_runner.cu:854`
  `activationData.gateUpLoraDeltaPtr = ...`); activation fusing rule:
  `fusedAct = !useDeepSeekFp8 && !forceUnfusedAct` (runner.cu:400) — bf16 default = fused (no seam).
- FP4 hit the same wall ("no NvFP4 unfused-act GEMM1 cubin exists", probe `sgl_trtllm_fp4_probe_unfused`
  → dead) and solved it with a HAND-WIRED pipeline (`FP4BlockScaleLoraLauncher`, launcher.cu 2737+):
  routing → `moe::dev::permute` (bf16 gather) → gate_up as RAW grouped GEMM via `Gemm2::Runner`
  (K=hidden, N=2*inter) → standalone LoRA-aware activation kernel (`fused_activation_quant.cuh`:
  bf16 in + delta pre-SwiGLU + writes activation_lora_input, fp4 out) → down GEMM (`Gemm2::Runner`)
  → bf16 lora finalize kernel.
- bf16 building blocks ALL proven: `Gemm2::Runner(Bf16,Bf16,Bf16)` is the bf16 base path's own GEMM2
  (runner.cu:784-793); `moe::dev::permute` already gathers bf16 (FP4 path); the lora finalize kernel
  is pure bf16 (FP4 reuses the FP8 one verbatim). ONLY missing piece: a bf16-OUT variant of the
  activation+delta kernel (the FP4 one converts to fp4 at the end) — small elementwise kernel.

DECISION: mirror `FP4BlockScaleLoraLauncher` as a bf16-only decomposed launcher
(`sgl_trtllm_bf16_routed_moe_lora`), DELETING the two quant stages, with a new bf16-out
activation+delta kernel; reuse permute + Gemm2 + the existing bf16 lora finalize. Deterministic
(no dependence on unproven unfused-gated-bf16 cubins), zero changes to FP8/FP4 code.

BONUS during implementation read-through: the FP4 launcher's NON-fused activation branch
(launcher.cu ~3090-3111) already drives `moe::dev::activation` with `mDtypeElt=Bfloat16,
mUseDeepSeekFp8=false`, bf16-in → bf16-out, `interleavedGateUpInput=true`, AND the LoRA hook
pointers — so **no new device kernel was needed at all**; the bf16 pipeline reuses it verbatim.

### 14. Implementation (sglang commit 623befb3e, branch qwen3-30b-a3b-2507-bf16) — +664 lines, 5 files
Per the goal's autonomy grant (no per-change sign-off; decisions logged here):
- `jit_kernel/trtllm_lora_temp/data/csrc/trtllm_fused_moe_kernel_launcher.cu` (+412): new
  `Bf16LoraLauncher` (decomposed pipeline: routing → permute bf16 → gate_up raw
  `Gemm2::Runner(Bf16,Bf16,Bf16, shuffled, BlockMajorK, no scaling)` K=hidden N=2*inter →
  `moe::dev::activation` (delta inject + capture, bf16 out) → down `Gemm2::Runner` K=inter
  N=hidden → finalize or {gemm2_output, expert_weights, idx}) + `sgl_trtllm_bf16_routed_moe_lora`
  wrapper (validations: gated SwiGLU, packed int32 topk, bf16 dtypes, inter%128==0, contiguous
  lora buffers) + TVM FFI export. Weight layout untouched (same shuffled+BlockMajorK tensors the
  plain `trtllm_bf16_moe` consumes — verified against `unquant.py` prep: epilogue_tile_m=128,
  block_k=128).
- `jit_kernel/trtllm_lora_temp/core.py` (+71) + `__init__.py`: `trtllm_bf16_routed_moe_lora`
  binding (21 positional args matched to the .cu signature); finalize reuses the shared bf16
  `trtllm_fp8_block_scale_moe_lora_finalize`.
- `srt/lora/trtllm_lora_temp/lora_layer.py` (+28/-1): bf16 (unquantized) detection branch BEFORE
  the FP8 asserts (`quant_config is None and not block_quant` → minimal
  `FlashInferTrtllmBf16MoeQuantInfo{gemm1/2_weights, global_num_experts, local_expert_offset}`);
  dispatch routes `FlashInferTrtllmBf16MoeQuantInfo` → the new fused fn. FP8 asserts now gate
  only quantized checkpoints.
- `srt/lora/trtllm_lora_temp/lora_dispatch.py` (+152):
  `fused_experts_none_to_experimental_sgl_trtllm_bf16_lora` — mirrors the FP4 flow: no-active-lora
  fast path → plain bf16; requires virtual-experts; gate_up delta via
  `merged_experts_fused_moe_lora_add` (EP-scoped); bf16 hidden fed DIRECTLY (no per-token fp8
  quant); `do_finalize=True` + down-LoRA merged after with `fuse_sum_all_reduce`;
  routing_method_type mirrors the plain bf16 path (None→Default, DeepSeekV3→TopK);
  inter/local_num_experts from runner_config (the bf16 quant-info is minimal). Single-stream
  (two-stream overlap = phase 2).
Checks: py_compile OK ×4; .cu brace-balanced; FP8/FP4 paths untouched (additive only).
Next: upload (done — pod @623befb3e721, JIT fp 98709c7d → module rebuild at launch) + fire
target_command.sh and monitor (build is the real nvcc test).

### 15. Test iteration 1 — crash at lora cuda-graph buffer init → fixed (commit 4f74dcf3e)
- Fired target_command.sh on the pod → DIED at scheduler init:
  `AttributeError: 'FlashInferTrtllmBf16MoeQuantInfo' object has no attribute 'w13_weight'`
  at `lora/backend/base_backend.py:192 init_cuda_graph_moe_buffers` (`E, N, _ = qinfo.w13_weight.shape`).
  Root cause: the generic MoE-LoRA cuda-graph buffer init assumed the FP8/FP4 field names; the bf16
  quant-info names the weights gemm1_weights/gemm2_weights. Only access site (grep-verified).
- Fix: getattr fallback to gemm1_weights/gemm2_weights (shapes equivalent: w13 [E,2*inter,..],
  w2 [E,hidden,..]). Commit 4f74dcf3e; re-uploaded (pod @4f74dcf3e196), re-fired. Monitoring.

### 16. Test iteration 2 — 3-D unpack crash → fixed (commit 104db551b)
- Same site, next line: `ValueError: too many values to unpack (expected 3)` — the bf16
  BlockMajorK-prepared weights are **4-D** `[E, N, K/128, 128]` (unquant.py reshapes to
  `(num_local_experts, *new_shape)` after convert_to_block_layout), while FP8/FP4 are 3-D.
- Fix: read `shape[0]/shape[1]` instead of unpacking exactly 3 dims. Verified my .cu (raw
  pointers + explicit N/K) and dispatch (explicit 2*inter) are ndim-agnostic — no other change.
- Commit 104db551b; re-uploaded (pod @104db551b96f), re-fired → server now PAST lora init, in
  warmup forward (lora triton kernel config lookups E=128,N=16 visible). Next gate: the
  trtllm_lora_temp module nvcc rebuild (first real compile of the new .cu code). Monitoring.

### 17. LAUNCH SUCCESS + coherence + logprob verification (target_command.sh, fused bf16 path)
- **READY at ~420s** (the stall during the first bs=128 graph capture = the nvcc JIT rebuild of
  the trtllm_lora_temp module — my 412-line .cu compiled clean on the first try).
- **NO `requires FP8 block quant`, NO illegal memory access, NO tracebacks.** Server stable.
- **Coherence (greedy, 24 tokens):**
  - lora(alpha): `" Paris. The capital of the United States is Washington, D.C. The capital of Canada is Ottawa. …"`
  - base:        `" Paris. … The capital of Japan is Tokyo. …"`
  - **Token-for-token IDENTICAL to the verified triton-path outputs** (same output_ids) — strongest
    functional-correctness signal.
- **Prefill logprob comparison vs triton** (same 39-token prompt, max_new_tokens=0 + return_logprob;
  relaunched the same command with only `--moe-runner-backend triton` swapped):
  | cell | mean abs diff | max abs diff |
  |---|---|---|
  | base (pure backend-numerics calibration) | 0.063 | 0.328 |
  | lora (backend numerics + my pipeline)    | 0.105 | 0.525 |
  The lora-cell diff is the same order as the base-cell diff (trtllm-gen vs triton bf16 GEMM
  accumulation across 48 MoE layers), i.e. my pipeline adds no anomalous error. → correctness PASS.
- Remaining: bench bs16/32/64 (3_run_benchmark.sh, in progress) vs triton baseline 2162/3817/6128.

### 18. BENCH PASS — fused bf16 path beats the triton baseline at every batch size. GOAL MET.
`bash dev/3_run_benchmark.sh Qwen3-30B-A3B-Instruct-2507-BF16` (model.env = experimental_sgl_trtllm
→ the NEW fused pipeline), results in `dev/results/.../20260607-114030/bench/summary.md`:

| cell | bs | decode tok/s | ITL ms | (triton baseline decode) | speedup |
|---|---|---|---|---|---|
| no-lora | 16/32/64 | 3787 / 6778 / 11400 | 4.22/4.72/5.61 | 3788/6758/11407 | ~equal (same base path) |
| **lora** | 16 | **2421.8** | 6.61 | 2161.5 (7.40ms) | **+12.0%** |
| **lora** | 32 | **4484.2** | 7.14 | 3817.2 (8.38ms) | **+17.5%** |
| **lora** | 64 | **7835.6** | 8.17 | 6128.2 (10.44ms) | **+27.9%** |

lora/no-lora decode ratio: 63.9% / 66.2% / 68.7% (triton path was 57.1% / 56.5% / 53.7%).
Coherence probe in-bench: clean. No FP8 asserts / illegal access anywhere in the run.

**GOAL MET**: target_command.sh (bf16 + experimental_sgl_trtllm + virtual-experts + dummy r16)
launches, serves coherently (token-identical greedy vs triton), logprob diff within backend-numerics
noise, and the fused pipeline is 12-28% faster than the triton path on lora decode.
sglang commits on branch qwen3-30b-a3b-2507-bf16: 623befb3e (implementation), 4f74dcf3e + 104db551b
(generic lora cuda-graph buffer init compat). Publish (push + PR) pending user confirmation.
