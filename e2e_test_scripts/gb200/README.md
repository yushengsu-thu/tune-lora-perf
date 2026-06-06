# GB200 (leira) e2e scripts — ⚠️ HISTORICAL

**The leira (Crusoe GB200) cluster no longer exists (confirmed 2026-06-06).** These files
are kept as the original methodology reference and as the baseline the `../gb300/` port
was derived from; they cannot be run anywhere anymore. For active work use
[`../gb300/`](../gb300/README.md). Shared helpers/sanity scripts live in [`../`](../README.md).

Also note: these scripts pre-date the branch rebase — they use the old backend name
`sgl_flashinfer_trtllm` (now `experimental_sgl_trtllm`) and `qwen_run.sh` still sets
`SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1`, which the later guardrail forbids (the gb300
port fixed both).

## Launch / orchestration — run ON the pod (fetch branch → launch → test matrix)
| file | invocation |
|---|---|
| `qwen_run.sh` | `bash qwen_run.sh <full-lora-opti\|main> <TAG>` — Qwen TP4/EP4; PR pod runs sgl-lora + triton-lora + nolora, oss pod runs triton-lora + nolora. |
| `qwen_base.sh` | oss no-LoRA **ceiling** (origin/main, default backend, no `--moe-runner-backend`, no LoRA) — the %-denominator. |
| `qwen_tp1_v2.sh` | `bash qwen_tp1_v2.sh <full-lora-opti\|main-base> <TAG>` — single-GPU tp1 (PR sgl-lora / oss default). |
| `kimi_run.sh` | `bash kimi_run.sh <worker\|head> <full-lora-opti\|main> <LORA=0\|1> <BACKEND> <DISTADDR> <TAG> <full\|gsm8k_only>` — **2-node MNNVL TP8/EP8; run the worker pod FIRST, then the head** (head runs coherence+bench+gsm8k). |

## Pod YAMLs (env setup)
| file | what it is |
|---|---|
| `kimi-rf.yaml` | Kimi 2-node MNNVL pod spec (head+worker, imex-channel ComputeDomain, 4×GB200, hf-token secret, self-download setup.sh). |
| `flo-qwentp1-pods.yaml` | Qwen single-node tp1 pod spec (lorapr pods, `runtimeClassName: nvidia`, hostPath `/mnt/nvme-b`). |

## GB200 reference numbers (for comparison with gb300/results/RESULTS.md)
- qwen fast-path LoRA / no-LoRA ceiling: **78.6 / 81.3 / 81.8 %** at bs16/32/64.
- kimi gsm8k base ≈ 0.95.
