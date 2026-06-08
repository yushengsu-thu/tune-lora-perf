# Qwen3.5-35B-A3B-FP8 — expert_shared LoRA perf comparison

**Date:** 2026-06-08 · **sglang commit:** `526e0ae22` (branch `qwen3-30b-a3b-2507-bf16`, the commit
with shared-outer support — `3c91ebdf5 [2/n] lora - Shared outer experts`) · **flashinfer** 0.6.11.post1
· **pod:** `sglang-gb300-qwen35-yushengsu-20260608-174728` (GB300 sm_103, TP4/EP4) · all four configs
on the **same base `Qwen3.5-35B-A3B-FP8` and the same pod**, so the only variables are LoRA mode + rank.

## TL;DR

- On this FP8 35B hybrid (GDN + MoE) model, **shared-outer LoRA is NOT faster than normal per-expert
  LoRA** — at rank 16 normal holds ~79–80% of the no-LoRA decode ceiling, shared-outer ~73–74%.
  (Opposite of the 30B BF16 result; different model + different adapters — see caveats.)
- **rank 16 vs 32 in shared-outer mode is ~flat** (decode 72–74% either way): the shared outer
  projections dominate, and the per-expert `B` blocks that scale with rank are a small fraction.
- no-LoRA decode ceiling **3515 / 6117 / 10625 tok/s** (bs 16/32/64) — matches the MODEL.md
  reference ceiling (3570/6206/10836) within ~1–2%, so `526e0ae22` serves FP8 35B correctly.

## Configs

| # | config | adapter | flag | rank | format |
|---|---|---|---|---|---|
| 1 | no-lora | — | — | — | — |
| 2 | normal lora | yanbin `jybsuper/qwen35_35b_lora_alpha` (**real**) | — (auto → per-expert) | 16 | per-expert 2D, 256 experts |
| 3 | expert_shared | auto-gen **dummy** | `--experts-shared-outer-loras` | 16 | 3D shared-outer (gate/up `lora_A`=`[1,r,H]`, down `lora_B`=`[1,H,r]`) |
| 4 | expert_shared | auto-gen **dummy** | `--experts-shared-outer-loras` | 32 | 3D shared-outer |

Shared-outer engagement verified in the server log (all 4 ranks):
`Shared outer LoRA mode enabled: gate_up lora_A and down lora_B will be shared across experts (expert_dim=1).`
with `experts_shared_outer_loras=True`, `lora_use_virtual_experts=True`, `mamba_scheduler_strategy='extra_buffer'`.

## Results  (BS 16/32/64, in/out 2048; ratio = lora ÷ the no-lora arm of the SAME run, shown directly above each config block)

All three columns are **throughput (tok/s, higher = faster)**; e2e tok/s = `bs × (in+out) / e2e_latency` = `bs × 4096 / latency`.

| config | bs | prefill/extend tok/s | decode tok/s | e2e tok/s |
|---|---|---|---|---|
| no-lora · run A | 16 | 27707 | 3515 | 6236 |
| no-lora · run A | 32 | 30701 | 6117 | 10200 |
| no-lora · run A | 64 | 31320 | 10625 | 15868 |
| normal r16 (real) | 16 | 23453 (85%) | 2761 (79%) | 4942 (79%) |
| normal r16 (real) | 32 | 22787 (74%) | 4922 (80%) | 8097 (79%) |
| normal r16 (real) | 64 | 24107 (77%) | 8517 (80%) | 12585 (79%) |
| no-lora · run B | 16 | 27935 | 3506 | 6230 |
| no-lora · run B | 32 | 31226 | 6095 | 10200 |
| no-lora · run B | 64 | 31424 | 10660 | 15917 |
| expert_shared r16 (dummy) | 16 | 22902 (82%) | 2561 (73%) | 4609 (74%) |
| expert_shared r16 (dummy) | 32 | 22703 (73%) | 4525 (74%) | 7546 (74%) |
| expert_shared r16 (dummy) | 64 | 22627 (72%) | 7768 (73%) | 11564 (73%) |
| no-lora · run C | 16 | 31166 | 3512 | 6314 |
| no-lora · run C | 32 | 31424 | 5979 | 10044 |
| no-lora · run C | 64 | 31549 | 10372 | 15613 |
| expert_shared r32 (dummy) | 16 | 22393 (72%) | 2527 (72%) | 4542 (72%) |
| expert_shared r32 (dummy) | 32 | 22449 (71%) | 4411 (74%) | 7372 (73%) |
| expert_shared r32 (dummy) | 64 | 22772 (72%) | 7719 (74%) | 11528 (74%) |

Ratio = lora ÷ no-lora (same run); for all three throughput columns **<100% = lora slower**.
no-lora arms reproduce within ~2% across runs (prefill/extend noisier) — hence per-run arms so every ratio is verifiable in-table.

**Cross-check vs recorded GB300 reference** (`e2e_test_scripts/gb300/results/RESULTS.md`, 2026-06-06,
`jybsuper:full-lora-opti@ac51ef5ed`): no-lora ceiling 3482/6041/10603, normal-r16 +LoRA 2780/4983/8591.
This run's 3515/6117/10625 and 2761/4922/8517 match within ~1–2% (commit `526e0ae22` vs `ac51ef5ed`).

## Caveats — read before drawing conclusions

1. **#3/#4 use random DUMMY weights** (no trained adapter exists in shared-outer format for this
   model). Decode/prefill throughput is shape/mode-driven, not weight-value-driven, so the **perf**
   comparison is valid; **accuracy/KL is N/A** and was not run.
2. **Module coverage differs** between normal (#2) and shared-outer (#3/#4):
   - yanbin real adapter targets: `q,k,v,o,gate,up,down_proj` + `in_proj_qkvz` (GDN linear-attn) +
     `lm_head`/`out_proj` — i.e. it ALSO adapts the GDN layers and head.
   - the dummy targets only `q,k,v,o,gate,up,down_proj` — it does NOT adapt `in_proj_qkvz`/`lm_head`.
   So the shared-outer configs adapt FEWER modules yet are SLOWER — which *strengthens* the finding
   that the shared-outer expert path itself is the heavier one here (coverage would bias the other way).
   For a strict apples-to-apples, regenerate the dummy with `DL_TARGETS=...,in_proj_qkvz` (and a
   per-expert dummy r16) — not done here.
3. The no-lora ceiling is from `526e0ae22`; the MODEL.md reference (3570/6206/10836) was a different
   commit (`full-lora-opti@ac51ef5ed`, not in this repo) — used only as a sanity anchor.

## Reproduce

```bash
# pod + code (once); shared pod across all configs (POD_PREFIX reused, ID pinned)
bash dev/1_launch_node.sh Qwen3.5-35B-A3B-FP8
SGLANG_CONFIRM=1 bash dev/2_upload_code.sh Qwen3.5-35B-A3B-FP8

# config #2 (normal r16, real adapter) + no-lora baseline
bash dev/3_run_benchmark.sh Qwen3.5-35B-A3B-FP8

# configs #3/#4 (shared-outer dummy r16/r32) — same pod via ID=<base launch ID>
ID=20260608-174728 ESO_RANK=16 bash dev/1_launch_node.sh Qwen3.5-35B-A3B-FP8-expert_shared
ESO_RANK=16 bash dev/3_run_benchmark.sh Qwen3.5-35B-A3B-FP8-expert_shared
ID=20260608-174728 ESO_RANK=32 bash dev/1_launch_node.sh Qwen3.5-35B-A3B-FP8-expert_shared
ESO_RANK=32 bash dev/3_run_benchmark.sh Qwen3.5-35B-A3B-FP8-expert_shared
```

`gen_dummy_lora.py` auto-detects `--experts-shared-outer-loras` (forwarded by common.sh as
`DL_LORA_EXTRA`) and emits the 3D shared-outer layout; otherwise it emits the default per-expert 2D.

## Result dirs
- `dev/results/Qwen3.5-35B-A3B-FP8/20260608-175241/bench/` — no-lora + normal r16
- `dev/results/Qwen3.5-35B-A3B-FP8-expert_shared/20260608-181121/bench/` — eso r16
- `dev/results/Qwen3.5-35B-A3B-FP8-expert_shared/20260608-181842/bench/` — eso r32
