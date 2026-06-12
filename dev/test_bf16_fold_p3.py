"""opt7 P3 — gather kernel + full fold pipeline timing at per-rank prefill shapes.

Run IN-POD:  python3 dev/test_bf16_fold_p3.py

1) Gather correctness (vs torch index_select on valid rows).
2) Timing: gather alone, fold GEMM alone, gather+fold pipeline.
   Replaces: dev permute 180us + GEMM1 57us + activation 33us = ~270us/layer.
"""

import torch

from sglang.jit_kernel.trtllm_lora_temp.core import (
    bf16_gather_rows,
    bf16_moe_gemm1_fold_gemm,
)

torch.manual_seed(0)
dev = "cuda"
E, K, N, tile = 32, 2048, 1536, 128
I = N // 2
NUM_TOKENS = 4096  # per-rank chunk tokens (gather source)
counts = [256] * E
R = sum(((c + tile - 1) // tile) * tile for c in counts)

hidden = torch.randn(NUM_TOKENS, K, device=dev, dtype=torch.bfloat16) * 0.5
W = torch.randn(E, N, K, device=dev, dtype=torch.bfloat16) * 0.05
cnt = torch.tensor(counts, device=dev, dtype=torch.int32)
# random gather map with some -1 padding
r2t = torch.randint(0, NUM_TOKENS, (R,), device=dev, dtype=torch.int32)
pad_mask = torch.rand(R, device=dev) < 0.05
r2t[pad_mask] = -1
r2x = torch.arange(R, device=dev, dtype=torch.int32) % (NUM_TOKENS * 8)
delta = torch.randn(NUM_TOKENS * 8, N, device=dev, dtype=torch.bfloat16) * 0.05

permuted = torch.zeros(R, K, device=dev, dtype=torch.bfloat16)
bf16_gather_rows(hidden, r2t, permuted)
torch.cuda.synchronize()
valid = r2t >= 0
ref = hidden[r2t[valid].long()]
assert torch.equal(permuted[valid], ref), "gather mismatch"
print("gather correctness PASS")

out = torch.empty(R, I, device=dev, dtype=torch.bfloat16)
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

def bench(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters): fn()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / iters

t_gather = bench(lambda: bf16_gather_rows(hidden, r2t, permuted))
t_fold = bench(lambda: bf16_moe_gemm1_fold_gemm(permuted, W, cnt, r2x, delta, tile, out))
t_pipe = bench(lambda: (bf16_gather_rows(hidden, r2t, permuted),
                        bf16_moe_gemm1_fold_gemm(permuted, W, cnt, r2x, delta, tile, out)))
print(f"gather alone:        {t_gather:6.1f} us  (dev permute reference: ~180 us)")
print(f"fold GEMM alone:     {t_fold:6.1f} us")
print(f"gather+fold pipeline:{t_pipe:6.1f} us  vs replaced (permute 180 + GEMM1 57 + act 33 = ~270 us)")
print(f"P3 GATE: {'PASS' if t_pipe <= 270 * 0.7 else 'MISS'} (<= 189 us = 30% win floor)")
