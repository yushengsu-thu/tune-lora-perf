"""opt7-step0: probe whether the bf16 unfused-activation gated GEMM1 cubin exists.

Run IN-POD (GB300):
    python3 dev/probe_bf16_unfused.py

Calls sgl_trtllm_bf16_probe_unfused (launcher.cu) for the Qwen3-30B-A3B shapes
(H=2048, I=768 per-rank, top_k=8, 32 local experts @ EP4) across the bf16 tile
ladder and prefill/decode token counts.

Probe columns (each = number of valid cubin configs; -1 = runner ctor threw):
  [0] Swiglu fused, BlockMajorK    — sanity (the no-LoRA bf16 path), expect >0
  [1] Swiglu UNFUSED, BlockMajorK  — the route-(a) question
  [2] Swiglu UNFUSED, MajorK       — in case it only exists in FP4's layout
  [3] Identity non-gated, BlockMajorK

Verdict: [1] or [2] > 0  => route (a): the cubin exists, the fold is wiring.
         all <= 0        => route (b): CUTLASS grouped GEMM (expected — the same
                            missing-unfused-cubin wall NVFP4 hit).
"""

from sglang.jit_kernel.trtllm_lora_temp.core import (
    get_sgl_trtllm_moe_sm100_raw_module,
)

H, I, TOPK, LOCAL_E = 2048, 768, 8, 32
mod = get_sgl_trtllm_moe_sm100_raw_module()
probe = mod.sgl_trtllm_bf16_probe_unfused

print(f"shapes: H={H} I={I} top_k={TOPK} local_experts={LOCAL_E}")
print("| tokens | tile | [0] fused BMK | [1] unfused BMK | [2] unfused MK | [3] identity |")
print("|---|---|---|---|---|---|")
verdict_a = False
for tokens in (16, 4096):
    for tile in (8, 16, 32, 64, 128):
        r = list(probe(TOPK, H, I, LOCAL_E, tokens, tile))
        print(f"| {tokens} | {tile} | {r[0]} | {r[1]} | {r[2]} | {r[3]} |")
        if r[1] > 0 or r[2] > 0:
            verdict_a = True

print()
print(
    "VERDICT: route (a) — unfused cubin EXISTS, fold is wiring"
    if verdict_a
    else "VERDICT: route (b) — missing-unfused-cubin wall confirmed; fold = CUTLASS grouped GEMM"
)
