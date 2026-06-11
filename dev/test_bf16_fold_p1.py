"""opt7 P1 unit + parity bench — CUTLASS grouped gate_up GEMM vs torch, timed at real shapes.

Run IN-POD (GB300):  python3 dev/test_bf16_fold_p1.py

1) Correctness: small random case, per-expert tile-padded segments; compare valid rows vs
   torch per-expert matmul (bf16 in, fp32 ref).
2) Parity gate: Qwen3-30B prefill shapes (4096-tok chunk x topk 8 -> ~32768 expanded rows,
   E=32 local, N=1536, K=2048, tile=128); report kernel time vs the tuned bmm_Bfloat16
   GEMM1 cubin's ~57 us (gate: within ~10-20% is a pass for P1; see OPT7_DESIGN.md).
"""

import torch

from sglang.jit_kernel.trtllm_lora_temp.core import bf16_moe_gemm1_grouped

torch.manual_seed(0)
dev = "cuda"


def build_case(E, counts, K, N, tile):
    padded = [((c + tile - 1) // tile) * tile for c in counts]
    R = sum(padded)
    A = torch.randn(R, K, device=dev, dtype=torch.bfloat16) * 0.5
    W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.05
    cnt = torch.tensor(counts, device=dev, dtype=torch.int32)
    out = torch.empty(R, N, device=dev, dtype=torch.bfloat16)
    return A, W, cnt, out, padded, R


# ---- 1) correctness ----
E, K, N, tile = 4, 256, 128, 16
counts = [5, 0, 33, 16]
A, W, cnt, out, padded, R = build_case(E, counts, K, N, tile)
bf16_moe_gemm1_grouped(A, W, cnt, tile, out)
torch.cuda.synchronize()

off = 0
max_rel = 0.0
for e in range(E):
    m = padded[e]
    c = counts[e]
    if c > 0:
        ref = (A[off : off + c].float() @ W[e].float().T)
        got = out[off : off + c].float()
        rel = ((got - ref).abs() / ref.abs().clamp(min=1e-2)).max().item()
        max_rel = max(max_rel, rel)
    off += m
print(f"correctness: max_rel_err={max_rel:.4e}")
assert max_rel < 2e-2, "FAIL correctness"
print("correctness PASS")

# ---- 2) parity bench @ real PER-RANK prefill shapes ----
# EP4: a 4096-token chunk x topk 8 = 32768 expanded rows GLOBALLY, but each rank owns
# 32 local experts and processes only its share: ~8192 rows/rank. (The original version
# of this bench forgot the EP divide -> 4x the real work -> bogus 172us "MISS"; the 57us
# cubin reference at 32768 rows would imply 3.6 PF/s bf16 = 3x cuBLAS, i.e. impossible.)
E, K, N, tile = 32, 2048, 1536, 128
total_expanded = 4096 * 8 // 4  # EP4 per-rank
base = total_expanded // E
counts = [base] * E  # uniform routing approximation
A, W, cnt, out, padded, R = build_case(E, counts, K, N, tile)

for _ in range(5):
    bf16_moe_gemm1_grouped(A, W, cnt, tile, out)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
iters = 50
start.record()
for _ in range(iters):
    bf16_moe_gemm1_grouped(A, W, cnt, tile, out)
end.record()
torch.cuda.synchronize()
us = start.elapsed_time(end) * 1000 / iters
flops = 2.0 * R * N * K
print(f"parity bench: R={R} E={E} N={N} K={K}  {us:.1f} us/call  "
      f"({flops / (us * 1e-6) / 1e12:.0f} TFLOP/s)  target ~57 us (tuned bmm cubin)")
print(f"GATE: {'PASS' if us <= 57 * 1.2 else 'MISS'} (<=68.4 us = 57 us +20%)")
