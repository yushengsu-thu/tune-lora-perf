# GB300 e2e results — 2026-06-06 (gcp-radixark-02)

PR = `jybsuper:full-lora-opti@ac51ef5ed`, oss = `sgl-project:main@aa55657e9`.
Image pinned `lmsysorg/sglang@sha256:97e7cd69…`, flashinfer 0.6.11.post1.
All numbers tok/s, `bench_one_batch_server` 2048/2048; xcheck = bench-vs-serverlog decode sanity.
(GB200/leira cluster is GONE — these GB300 numbers are the reference set going forward.)

## Qwen3.5-35B-A3B-FP8 — TP4/EP4, single node (5wsb)

| config | bs16 | bs32 | bs64 | bs128 |
|---|---|---|---|---|
| ceiling (oss main, default backend, no-LoRA) | 3481.6 | 6040.9 | 10603.3 | 17787.8 |
| PR sgl-lora base (`experimental_sgl_trtllm`) | 2920.9 | 5331.3 | 9045.8 | ~15520¹ |
| PR sgl-lora +LoRA | 2779.9 | 4982.6 | 8591.2 | 14856.4 |
| PR triton-lora +LoRA | 2159.0 | 3841.5 | 6666.9 | 10784.8 |
| oss triton-lora +LoRA | 1733.0 | 3190.2 | 5692.6 | 9456.5 |
| **fast-path LoRA / ceiling** | **79.9%** | **82.5%** | **81.0%** | **83.5%** |
| (regression-pack reference) | 77.6% | 80.0% | 81.3% | — |
| **PR LoRA vs oss LoRA speedup** | **1.60×** | 1.56× | 1.51× | 1.57× |

¹ original point xcheck −6.0% SUSPECT; re-ran twice → 15617.7 / 15422.1 (both OK).

Accuracy: gsm8k base 0.735–0.790 (PR fast-path 0.790), req-lora 0.010–0.020 (identity
adapter band 0.01–0.04). COHERENCE clean in all full cells. Raw log archived on node
5wsb at `/data/qwen_e2e_gb300_20260606.log` (persistent partition).

## Kimi-K2.5-NVFP4 — TP8/EP8, 2-node MNNVL (5wsb + 6zvh, ComputeDomain/DRA IMEX)

| config (all PR, `experimental_sgl_trtllm`) | bs16 | bs32 | bs64 | bs128 |
|---|---|---|---|---|
| no-LoRA (LORA=0) = ceiling | 1185.8 | 2049.4 | 3460.6 | 5864.4 |
| LoRA server, base reqs | 1035.4 | 1821.9 | 3084.9 | 5287.8 |
| LoRA server, req-lora | 1015.8 | 1924.2 | 3385.5 | 5805.9 |
| **LoRA / ceiling** | 85.7% | 93.9% | 97.8% | 99.0% |

Accuracy: gsm8k base 0.965 (LORA=0) / 0.950 (LoRA server), req-lora 0.020, zero
truncation issues, COHERENT. xcheck ≤1.3% on all 16 points.

### ⚠️ Kimi NVFP4+LoRA crashes on this commit — 4-bug chain (REPORT TO jybsuper)

The kimi NVFP4+LoRA combination is broken at `ac51ef5ed` (untested post-rebase). The
LoRA numbers above were obtained WITH a working-tree hotfix (the patch file was REMOVED
from this repo 2026-06-06 — possibly incorrect: it silently SKIPS the LoRA delta on
quantized-tuple layers, so those layers ran base-only; do not reuse it). The bug chain:

1. fused `nvfp4_gemm_swiglu` path (deepseek_v2.py) reads modelopt quant attrs
   (`input_scale_inv`, `alpha`, `weight_*_interleaved`, …) directly off the LoRA-wrapped
   linears; `BaseLayerWithLoRA` doesn't forward them → AttributeError in autotune warmup.
   **Fix: explicit `@property` forwards** (a generic `__getattr__` delegation does NOT work —
   it breaks `register_parameter`'s hasattr check: `KeyError: attribute 'weight' already exists`).
2. `trtllm_lora_temp.is_two_stream_active(x)` calls `x.shape` on the `(x_fp4, x_scale)`
   tuples those paths produce → guard non-Tensor → False.
3. The serial fallback then crashes too (`sgemm_lora_a_fwd` can't consume tuples) →
   `RowParallelLinearWithLoRA.forward` gates the LoRA delta on
   `isinstance(input_parallel, torch.Tensor)`.

Hotfix semantics: layers fed quantized tuples SKIP their LoRA delta (base-only — same as
LORA=0). Upstream must choose the real fix: disable the fused path under LoRA, or teach
the LoRA kernels to consume quantized inputs.

## Operational notes
- flashinfer fp4 autotune is process-local (NOT disk-cached) → every kimi launch re-tunes
  (10–25 min). The qwen trtllm JIT *is* disk-cached (dot-cache hostPath).
- kimi weights live on ephemeral `/root` (the 95G stateful partition can't hold 600G) →
  re-download per pod creation.
- `kimi_run_gb300.sh` does `git checkout -f` at start — use `kimi_run_nockout.sh`-style
  variant (drop the checkout) when a working-tree patch must survive.
- Failure-detection greps must include lowercase `serve: error` (argparse failures).
