"""opt7 P2 unit + perf — fold-epilogue grouped GEMM vs the P0 reference kernel.

Run IN-POD (GB300):  python3 dev/test_bf16_fold_p2.py

1) Correctness: random routing (irregular per-expert counts incl. zeros, -1 padding),
   random delta; compare fold-GEMM output vs sgl_bf16_moe_gemm1_fold_ref on VALID rows.
2) Perf at per-rank prefill shapes (EP4: 8192 expanded rows, E=32, N=1536->I=768, K=2048):
   target = beat (P1 plain GEMM 65.6us + standalone activation ~33us) ~= 99us; parity-floor
   reference: NoSmem full-width GEMM measured 84.4us (the fold stores HALF the bytes).
"""

import torch

from sglang.jit_kernel.trtllm_lora_temp.core import (
    bf16_moe_gemm1_fold_gemm,
    bf16_moe_gemm1_fold_ref,
)

torch.manual_seed(0)
dev = "cuda"


def build_routing(E, counts, tile, topk_total):
    rows_token, rows_expanded, rows_expert = [], [], []
    pool = list(range(topk_total))
    import random

    random.seed(0)
    random.shuffle(pool)
    idx = 0
    for e in range(E):
        take = counts[e]
        exp_ids = pool[idx : idx + take]
        idx += take
        pad = (-take) % tile
        for x in exp_ids:
            rows_expanded.append(x)
            rows_expert.append(e)
        for _ in range(pad):
            rows_expanded.append(0)
            rows_expert.append(-1)
    R = len(rows_expanded)
    r2x = torch.tensor(rows_expanded, device=dev, dtype=torch.int32)
    r2e = torch.tensor(rows_expert, device=dev, dtype=torch.int32)
    return r2x, r2e, R


# ---- 1) correctness vs P0 ref ----
E, K, N, tile = 4, 256, 128, 16
I = N // 2
counts = [5, 0, 33, 16]
topk_total = sum(counts)
r2x, r2e, R = build_routing(E, counts, tile, topk_total)
# A here IS the permuted buffer (P2 still consumes pre-permuted A); ref kernel gathers
# from "hidden" by row2token — make them line up: hidden = A, row2token = identity.
A = torch.randn(R, K, device=dev, dtype=torch.bfloat16) * 0.5
W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.1
delta = torch.randn(topk_total, N, device=dev, dtype=torch.bfloat16) * 0.05
cnt = torch.tensor(counts, device=dev, dtype=torch.int32)
r2t_identity = torch.arange(R, device=dev, dtype=torch.int32)

out_fold = torch.full((R, I), float("nan"), device=dev, dtype=torch.bfloat16)
bf16_moe_gemm1_fold_gemm(A, W, cnt, r2x, delta, tile, out_fold)

out_ref = torch.empty(R, I, device=dev, dtype=torch.bfloat16)
bf16_moe_gemm1_fold_ref(A, W, delta, r2t_identity, r2x, r2e, out_ref)
torch.cuda.synchronize()

valid = (r2e >= 0)
diff = (out_fold[valid].float() - out_ref[valid].float()).abs()
rel = diff / out_ref[valid].float().abs().clamp(min=1e-2)
print(f"correctness vs P0 ref: R={R} valid={int(valid.sum())} "
      f"max_abs={diff.max().item():.4e} max_rel={rel.max().item():.4e}")
assert rel.max().item() < 2e-2, "FAIL correctness"
print("correctness PASS")

# ---- 2) perf @ per-rank shapes ----
E, K, N, tile = 32, 2048, 1536, 128
I = N // 2
counts = [256] * E
topk_total = sum(counts)
r2x, r2e, R = build_routing(E, counts, tile, topk_total)
A = torch.randn(R, K, device=dev, dtype=torch.bfloat16) * 0.5
W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.05
delta = torch.randn(topk_total, N, device=dev, dtype=torch.bfloat16) * 0.05
cnt = torch.tensor(counts, device=dev, dtype=torch.int32)
out_fold = torch.empty(R, I, device=dev, dtype=torch.bfloat16)

for _ in range(5):
    bf16_moe_gemm1_fold_gemm(A, W, cnt, r2x, delta, tile, out_fold)
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(50):
    bf16_moe_gemm1_fold_gemm(A, W, cnt, r2x, delta, tile, out_fold)
e.record()
torch.cuda.synchronize()
us = s.elapsed_time(e) * 1000 / 50
fl = 2.0 * R * N * K
print(f"fold GEMM per-rank shapes: R={R}  {us:.1f} us/call  ({fl/(us*1e-6)/1e12:.0f} TFLOP/s)")
print(f"reference: P1 plain GEMM 65.6us + activation ~33us = ~99us total it replaces")
print(f"P2 NET-WIN: {'PASS' if us <= 99 else 'MISS'} (<= 99us)")
