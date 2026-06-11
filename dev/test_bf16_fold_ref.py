"""opt7 P0 unit test — bf16 fold reference kernel vs a pure-torch reference.

Run IN-POD (GB300):  python3 dev/test_bf16_fold_ref.py

Builds a random MoE-LoRA fold case (random routing with per-expert padded segments,
random weights/delta), runs sgl_bf16_moe_gemm1_fold_ref, and checks it against a
float32 torch reference implementing the exact semantics:
    x1 = (hidden[t] @ W[e, 2h, :])   + delta[x, I + h]   # interleaved col 2h
    x2 = (hidden[t] @ W[e, 2h+1, :]) + delta[x, h]       # interleaved col 2h+1
    out[r, h] = silu(x2) * x1
Also prints the CUTLASS version probe (include-path sanity for P1).
"""

import torch

from sglang.jit_kernel.trtllm_lora_temp.core import (
    bf16_fold_probe,
    bf16_moe_gemm1_fold_ref,
)

torch.manual_seed(0)
dev = "cuda"

print("CUTLASS probe [major, minor, ok]:", bf16_fold_probe())

# ---- random case ----
N, TOPK, E, I, K, TILE = 24, 4, 4, 32, 64, 8
hidden = torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.5
w = torch.randn(E, 2 * I, K, device=dev, dtype=torch.bfloat16) * 0.1
delta = torch.randn(N * TOPK, 2 * I, device=dev, dtype=torch.bfloat16) * 0.05

# random routing: expanded id (t*TOPK+k) -> expert; group by expert, pad to TILE with -1
topk_ids = torch.randint(0, E, (N, TOPK), device=dev)
rows_token, rows_expanded, rows_expert = [], [], []
for e in range(E):
    exp_ids = (topk_ids.view(-1) == e).nonzero(as_tuple=True)[0].tolist()
    pad = (-len(exp_ids)) % TILE
    for x in exp_ids:
        rows_token.append(x // TOPK)
        rows_expanded.append(x)
        rows_expert.append(e)
    for _ in range(pad):
        rows_token.append(0)
        rows_expanded.append(0)
        rows_expert.append(-1)
R = len(rows_token)
r2t = torch.tensor(rows_token, device=dev, dtype=torch.int32)
r2x = torch.tensor(rows_expanded, device=dev, dtype=torch.int32)
r2e = torch.tensor(rows_expert, device=dev, dtype=torch.int32)

# ---- kernel ----
out = torch.empty(R, I, device=dev, dtype=torch.bfloat16)
bf16_moe_gemm1_fold_ref(hidden, w, delta, r2t, r2x, r2e, out)
torch.cuda.synchronize()

# ---- torch reference (fp32) ----
ref = torch.zeros(R, I, device=dev, dtype=torch.float32)
hf, wf, df = hidden.float(), w.float(), delta.float()
for r in range(R):
    e = rows_expert[r]
    if e < 0:
        continue
    t, x = rows_token[r], rows_expanded[r]
    acc = hf[t] @ wf[e].T                       # [2I] interleaved
    x1 = acc[0::2] + df[x, I:]                  # col 2h   + delta 2nd half
    x2 = acc[1::2] + df[x, :I]                  # col 2h+1 + delta 1st half
    ref[r] = torch.nn.functional.silu(x2) * x1

err = (out.float() - ref).abs()
rel = err / ref.abs().clamp(min=1e-2)
print(f"R={R}  max_abs_err={err.max().item():.4e}  max_rel_err={rel.max().item():.4e}")
ok = rel.max().item() < 2e-2  # bf16 storage rounding on fp32-accum math
print("PASS" if ok else "FAIL")
assert ok
