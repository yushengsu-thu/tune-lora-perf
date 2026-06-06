# tune-lora-perf

LoRA performance tuning skills and dev/benchmark/regression workflows for SGLang on
GB200/GB300 clusters.

All current development targets **[sgl-project/sglang#27329](https://github.com/sgl-project/sglang/pull/27329)** —
the experimental fast LoRA path (`SGLANG_EXPERIMENTAL_LORA_OPTI=1` +
`--moe-runner-backend experimental_sgl_trtllm`) — **now MERGED into sglang main**. The pinned
sglang source for all runnable scripts is
[`yushengsu-thu/sglang@trtllm-lora-bf16`](https://github.com/yushengsu-thu/sglang/tree/trtllm-lora-bf16)
(head = the PR #27329 merge commit `c9f582a27`). Two models, full names with precision:
**Qwen3.5-35B-A3B-FP8** (FP8, 1 node tp4/ep4) and **Kimi-K2.5-NVFP4** (NVFP4, 2-node MNNVL tp8/ep8).

## Repository layout

```
tune-lora-perf/
├── README.md                                  # this file
├── skill.md                                   # agent operating manual: how to run a LoRA-opti task
│                                              #   (journal, worktree-per-task, branch sync, upload→test loop)
├── skill_kernel_fusion.md                     # kernel-fusion / kernel-optimization skill
│                                              #   (testbed → bench shapes → fuse → verify)
├── run_script.sh                              # Yanbin's reference launch: Qwen3.5-35B-A3B-FP8 LoRA server
│                                              #   + graph-ON bs64 24-step profile (use with the skill)
├── pod.yaml                                   # minimal single-pod dev sandbox (sglang:dev-cu13 + sshd);
│                                              #   NOT used by the workflows below (they carry their own yamls)
├── E2E_FULL_TEST_RUNBOOK.md                   # the full e2e test-matrix runbook (methodology, env/YAML,
│                                              #   expected numbers, pitfalls, bugs found+fixed)
├── qwen35_35b_lora_profile_graphon_bs64.md    # profile recipe: Qwen3.5-35B-A3B-FP8 graph-ON bs64
├── kimi_kernel_shapes_bs64_tp8_ep8.md         # reference: Kimi-K2.5-NVFP4 LoRA kernel shapes (bs64/TP8/EP8)
│
├── dev/                                       # ★ fast dev loop (GB300): launch→upload code→bench→acc→profile→publish
├── e2e_test_scripts/                          # e2e test-matrix scripts (companion to the runbook)
└── regression/                                # base-vs-variant regression harness (acc+bench+prompts+profile)
```

## The three workflows — which one to use

| dir | use it when | docs |
|---|---|---|
| [`dev/`](dev/README.md) | **iterating on local sglang code** — one command (`run_all.sh qwen\|kimi`) launches a GB300 node, uploads your current branch, runs LoRA-vs-no-LoRA benchmark + accuracy (incl. the vLLM-reference comparison) + torch profiles, and publishes results to GitHub. Steps are standalone and individually rerunnable. | [`dev/README.md`](dev/README.md) |
| [`regression/`](regression/README.md) | **gating a serving change** (a LoRA toggle, backend swap, env var, a PR) — one model-agnostic driver runs all four tests (accuracy / performance / prompt-check / profiling) on a **base vs variant** pair with hardened launch/pull mechanics and a 5-metric report. Validated e2e on GB200 + GB300 (2026-06-06). | [`regression/README.md`](regression/README.md) (manual steps) + [`regression/SKILL.md`](regression/SKILL.md) (operating manual + robustness rules) |
| [`e2e_test_scripts/`](e2e_test_scripts/README.md) | **reproducing the full e2e matrix** from `E2E_FULL_TEST_RUNBOOK.md` (coherence + bench bs16–128 + gsm8k, PR vs oss lanes) — per-cluster runners + pod YAMLs, shared pod helpers. `gb300/` is active; `gb200/` is historical (leira is gone). | [`e2e_test_scripts/README.md`](e2e_test_scripts/README.md) |

## Root files in detail

| file | what it is |
|---|---|
| [`skill.md`](skill.md) | The general agent skill for LoRA-opti tasks: create a per-task journal + git worktree, keep the branch synced with upstream, upload each step to the launched nodes, record acc/bench results, attach the journal to the PR, then gate with the regression skill. |
| [`skill_kernel_fusion.md`](skill_kernel_fusion.md) | Kernel-fusion/optimization skill: build a kernel testbed with the real shapes (see `kimi_kernel_shapes_bs64_tp8_ep8.md`), bench, fuse, verify numerics. Used at the final tuning stage. |
| [`run_script.sh`](run_script.sh) | The reference Qwen3.5-35B-A3B-FP8 LoRA launch + profile command the dev/regression configs were derived from. |
| [`pod.yaml`](pod.yaml) | A bare single-pod sandbox (ssh-able `lmsysorg/sglang:dev-cu13`) for ad-hoc poking — the workflows ship their own pod yamls. |
| [`E2E_FULL_TEST_RUNBOOK.md`](E2E_FULL_TEST_RUNBOOK.md) | End-to-end runbook for the fast-LoRA-path test matrix: infra/YAML, launch commands for both models, the test matrix, expected numbers (% of the no-LoRA ceiling), pitfalls, and the bug history (e.g. the `down_finalize` base-corruption). |
| [`qwen35_35b_lora_profile_graphon_bs64.md`](qwen35_35b_lora_profile_graphon_bs64.md) | Worked profile recipe (Qwen3.5-35B-A3B-FP8, graph-ON, bs64): which forwards are clean decode, how to read the trace. |
| [`kimi_kernel_shapes_bs64_tp8_ep8.md`](kimi_kernel_shapes_bs64_tp8_ep8.md) | Measured Kimi-K2.5-NVFP4 LoRA kernel shapes at bs64/TP8/EP8 — the input for kernel-fusion testbeds. |

## Quickstart

```bash
# fast dev loop on GB300 (full chain, ~2-5h; steps also run standalone):
bash dev/run_all.sh qwen          # or: kimi  (shorthands for the full model names)

# base-vs-variant regression (read regression/README.md §3 for the step-by-step):
DRY_RUN=1 bash regression/gb300/run_Qwen3.5-35B-A3B-FP8.sh   # preview launch surface
ID=<id> RUN_ROOT=<dir> bash regression/gb300/run_Qwen3.5-35B-A3B-FP8.sh

# e2e matrix on GB300 (read e2e_test_scripts/gb300/README.md first):
bash e2e_test_scripts/gb300/Qwen3.5-35B-A3B-FP8_run_gb300.sh full-lora-opti <TAG>
```

Cluster note: everything current targets **`gcp-radixark-02`** (GB300/sm_103). The GB200
cluster (`leira`) is **gone** — all `gb200/` content is kept as historical reference only.
