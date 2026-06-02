# Kimi-K2.5-NVFP4 LoRA kernel I/O shapes — decode bs64, TP8, EP8

Companion to [`lora_model_configs.md`](./lora_model_configs.md) and
[`virtual_experts.md`](./virtual_experts.md). Resolves the symbolic dims of
every LoRA-related kernel to concrete Kimi-K2.5-NVFP4 numbers for a **decode
step, batch=64, on 8 GPUs (attention TP=8, MoE EP=8)**.

Kernels are listed by their **original source name** (the `@triton.jit` function
or the CUDA JIT entry), with the file they live in.

For the strategy discussion — two-stream overlap, L2-pin footprint, and which
kernels can fall back to a plain `torch.matmul` — see
[`kimi_lora_strategy_discussion.md`](./kimi_lora_strategy_discussion.md).

---

## Setup / derived constants

| Quantity | Value | Derivation |
|---|---|---|
| Tokens this step `S` | **64** | decode, 1 tok/req |
| Heads/rank `H` | **8** | 64 heads ÷ TP8 |
| `hidden_size` | 7168 | — |
| `q_lora_rank` | 1536 | q_a out |
| `kv_lora_rank` | 512 | kv_b in |
| `qk_nope / qk_rope / v_head_dim` | 128 / 64 / 128 | qk_head_dim=192, FULL_K=qk_nope+v_head_dim=256 |
| Routed experts/rank | **48** | 384 ÷ EP8 (+1 shared) |
| `top_k` | 8 | → `M = S·top_k = 512` (token,expert) pairs; ~64 land on each rank |
| `moe_intermediate_size` | 2048 | gate/up out, down in (NOT TP-split — EP) |
| LoRA rank `r` | **16** (assumed) | r=32 ⇒ double the rank dim everywhere |
| `max_loras` `L` | 1 (single adapter) | virtual-expert count scales by this |

All LoRA A/B tensors are **bf16** — the NVFP4 quantization is on the base
weights only. Weights carry a leading slot/expert dim; shapes below show that
dim where it matters.

> **Assumptions that move the numbers:** (1) `r=16` vs the r=32 some Kimi
> adapters ship; (2) the full q_a/q_b/kv_a/o MLA set is adapted (drop rows if a
> subset carries LoRA); (3) MoE `lora_a` is per-expert, not `shared_outer`
> (which would collapse the weight expert-dim to `L`).

---

## A. MLA attention dense LoRA

Shrink `x@A→(S,r)` then expand `(S,r)@B→(S,N)`. Per rank, `x` rows = S = 64.

- **`_sgemm_lora_a_kernel`** / **`_sgemm_lora_a_splitk_kernel`** — `sgemm_lora_a.py` (shrink)
- **`_qkv_lora_b_kernel`** — `qkv_lora_b.py` (fused q/k/v expand)
- **`_sgemm_lora_b_kernel`** — `sgemm_lora_b.py` (generic expand, e.g. o_proj)

| Projection | shrink `x` (S,K) | A (K,r) | shrink out (S,r) | B (r,N) | expand out (S,N) |
|---|---|---|---|---|---|
| q_a_proj | (64, 7168) | (7168, 16) | (64, 16) | (16, 1536) | (64, 1536) |
| q_b_proj | (64, 1536) | (1536, 16) | (64, 16) | (16, 1536) | (64, 1536) ‹8·192› |
| kv_a_proj | (64, 7168) | (7168, 16) | (64, 16) | (16, 576) | (64, 576) ‹512+64› |
| o_proj (Row∥) | (64, **1024**) ‹8·128› | (1024, 16) | (64, 16) | (16, 7168) | (64, 7168) → all-reduce |

`_sgemm_lora_a_splitk_kernel` is the fp32-atomics opt-in path; it wins on the
wide-K shrinks here (q_a/kv_a K=7168). See
[`shrink_splitk_ncu_floor.md`](./shrink_splitk_ncu_floor.md).

## B. Absorbed-MLA `kv_b` correction — `kv_b_lora_absorbed.py` (per rank, H=8)

The four split kernels that fold the kv_b LoRA delta into the absorbed BMMs.

| Kernel (`@triton.jit`) | `x` | weight | out |
|---|---|---|---|
| `_step_a_q_kernel` (shrink, K-half of B) | q_nope (64, 8, 128) | B (L, 2048, 16) ‹H·FULL_K=8·256› | (64, 8, 16) |
| `_step_b_q_kernel` (expand A, accumulate) | (64, 8, 16) | A (L, 16, 512) | += q_nope_out (64, 8, 512) |
| `_step_a_v_kernel` (shrink Aᵀ) | attn_out (64, 8, 512) | A (L, 16, 512) | (64, 8, 16) |
| `_step_b_v_kernel` (expand B V-half, accumulate) | (64, 8, 16) | B (L, 2048, 16) | += attn_bmm_out (64, 8, 128) |

## C. Dense MLP, layer 0 (gate/up 18432×7168, down 7168×18432)

Layer 0 MLP **is** TP-split (TP=8). Shrink via `_sgemm_lora_a_kernel`, expand via:

- **`_gate_up_lora_b_kernel`** — `gate_up_lora_b.py`
- **`_sgemm_lora_b_kernel`** — `sgemm_lora_b.py` (down)

| Stage | shrink `x` | shrink out | B weight | expand out |
|---|---|---|---|---|
| gate_up (Col∥) | (64, 7168) | (64, 2·16=32) ‹gate+up A› | (16, 2·2304) | (64, 4608) ‹18432/8 ×2› |
| down (Row∥) | (64, **2304**) ‹18432/8› | (64, 16) | (16, 7168) | (64, 7168) → all-reduce |

## D. MoE expert LoRA (virtual experts)

EP keeps **global** weight tensors (first dim = `E·L = 384·1 = 384`); only the
48 owned experts are computed (rest masked to −1). For gated gate_up the shrink
intermediate carries `2r` columns.

- **`_moe_lora_shrink_splitk_kernel`** — `virtual_experts.py` (shrink)
- **`_moe_lora_expand_add_kernel`** — `trtllm_moe/specialized_expand.py` (expand+add)

**gate_up:**
| Kernel | input `a` | weight (E·L, N, K) | output |
|---|---|---|---|
| `_moe_lora_shrink_splitk_kernel` | hidden (64, 7168) | lora_a (384, **32**, 7168) ‹N=2r gated› | intermediate (S·top_k, 2r) = **(512, 32)** |
| `_moe_lora_expand_add_kernel` | (512, 32) | lora_b (384, **4096**, 16) ‹2·2048› | (64, 4096) |

**down:**
| Kernel | input `a` | weight | output |
|---|---|---|---|
| `_moe_lora_shrink_splitk_kernel` | act (512, 2048) | lora_a (384, 16, 2048) | (512, 16) |
| `_moe_lora_expand_add_kernel` | (512, 16) | lora_b (384, 7168, 16) | (64, 7168) → sum top_k + all-reduce |

`SPLIT_K` from `_get_moe_lora_shrink_split_k`: `base_grid = num_m_blocks·num_n_blocks`
is tiny (rank-16 N, ~57 m-blocks), so it picks `≈ 128/base_grid` clamped —
typically 2–3 for the K=7168/2048 dims; output buffer is pre-zeroed when
SPLIT_K>1 (fp32 atomics). See
[`moe_lora_shrink_optimization.md`](./moe_lora_shrink_optimization.md).

> The legacy generic path (`_fused_moe_lora_kernel` / `_fused_moe_lora_shrink` /
> `_fused_moe_lora_expand` in `fused_moe_lora_kernel.py`) computes the same math
> but is not the active NVFP4 trtllm path.

## E. Routing / align (per stage; cached across gate_up↔down)

| Kernel | input | output |
|---|---|---|
| `_fused_virtual_topk_ids_kernel` — `virtual_experts.py` | topk_ids (64, 8), token_lora_mapping (64,) | virtual_topk_ids (64, 8), token_lora_mask (64,) |
| `MoeLoraAlignBlockSizeKernel` (CUDA JIT) — `jit_kernel/csrc/lora/moe_lora_align_kernel.cu` | virtual_topk_ids flat (512,) | sorted_token_ids, expert_ids, num_tokens_post_padded |
| `_fused_sanitize_expert_ids_kernel` — `virtual_experts.py` | expert_ids | same shape (skipped when L=1) |
| `_compute_moe_lora_info_kernel` — `backend/base_backend.py` | batch routing tensors | MoE LoRA batch info |

EP-trimmed routing buffer sizes (`tight_padded`, the kernel grids' M extent),
for `M=512`, populated buckets `= 48·1 + 1 = 49`:

| Stage | BLOCK_M | sorted_token_ids | expert_ids |
|---|---|---|---|
| shrink (`_moe_lora_shrink_splitk_kernel`) | 64 | ≈ (3648,) | ≈ (57,) |
| expand (`_moe_lora_expand_add_kernel`) | 16 | ≈ (1248,) | ≈ (78,) |

The padding dwarfs the ~512 real pairs — the sparse-MoE-at-decode regime behind
[`shrink-moe-sweep.md`](./shrink-moe-sweep.md) and the amortize-cold benching.
