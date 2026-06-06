# e2e test scripts — companion to `../E2E_FULL_TEST_RUNBOOK.md`

Scripts for the experimental TRT-LLM LoRA **e2e test matrix** (PR fast path vs oss main):
per config cell, launch the server, then run coherence prompts + `bench_one_batch_server`
bs16–128 (with the server-log throughput cross-check) + 5-shot gsm8k (base & req-lora).
This directory holds the **hardware-agnostic shared files**; the per-cluster pod YAMLs and
orchestration runners live in the arch subdirectories:

| dir | cluster | status |
|---|---|---|
| [`gb200/`](gb200/README.md) | leira (Crusoe, GB200) | **HISTORICAL** — the leira cluster is gone (2026-06-06); kept as the original methodology reference |
| [`gb300/`](gb300/README.md) | gcp-radixark-02 (GKE, GB300/sm_103) | **ACTIVE** — validated end-to-end 2026-06-06; results in `gb300/results/RESULTS.md` |

```
e2e_test_scripts/
├── README.md                              # this file
│   ── shared pod helpers (deploy to /tmp/flo_helpers/) ──
├── gsm8k_lora.py                          # 5-shot gsm8k accuracy over /generate or /v1/chat (base | req-lora)
├── bench_report.py                        # bench jsonl + serverlog → one-line e2e/tput/ITL + OK/SUSPECT xcheck
├── prompts_check.py                       # coherence prompts + per-endpoint adapter-behavior check
│   ── shared sanity checks (branch import/gating, run on any pod) ──
├── reformat_sanity.sh                     # experimental package import gating (OFF=absent / ON=loads)
├── Qwen3.5-35B-A3B-FP8_reformat_chk.sh    # qwen sgl-lora launch order + base/lora gsm8k
├── Qwen3.5-35B-A3B-FP8_sglnolora.sh       # qwen sgl backend, LoRA OFF → confirms delegation to upstream
├── fmla_import_chk.sh                     # forward_mla gating import-sanity
├── jit_chk_temp.sh                        # trtllm_lora_temp JIT kimi kernel compiles after the csrc rename
│
├── gb200/                                 # ── HISTORICAL (leira is gone) ──
│   ├── README.md
│   ├── Qwen3.5-35B-A3B-FP8_run.sh         # qwen TP4/EP4 matrix runner (PR / oss lanes)
│   ├── Qwen3.5-35B-A3B-FP8_base.sh        # oss no-LoRA ceiling (the %-denominator)
│   ├── Qwen3.5-35B-A3B-FP8_tp1_v2.sh      # single-GPU tp1 variant
│   ├── Kimi-K2.5-NVFP4_run.sh             # kimi 2-node MNNVL TP8/EP8 runner (worker first, then head)
│   ├── Kimi-K2.5-NVFP4-rf.yaml            # kimi 2-node pod spec (head+worker, imex, hf secret)
│   └── flo-Qwen3.5-35B-A3B-FP8-tp1-pods.yaml   # qwen tp1 pod pair (PR + oss lanes)
│
└── gb300/                                 # ── ACTIVE (gcp-radixark-02, sm_103) ──
    ├── README.md
    ├── Qwen3.5-35B-A3B-FP8_run_gb300.sh   # qwen TP4/EP4 matrix runner (45-min cold-JIT wait, flashinfer pin)
    ├── Qwen3.5-35B-A3B-FP8_base_gb300.sh  # oss no-LoRA ceiling on GB300
    ├── Qwen3.5-35B-A3B-FP8_tp1_gb300.sh   # single-GPU tp1 variant
    ├── Qwen3.5-35B-A3B-FP8-pod.yaml       # qwen pod (stateful-partition /data + JIT-cache mounts)
    ├── Qwen3.5-35B-A3B-FP8-tp1-pods.yaml  # qwen tp1 pod pair
    ├── Kimi-K2.5-NVFP4_run_gb300.sh       # kimi 2-node MNNVL via ComputeDomain/DRA
    ├── Kimi-K2.5-NVFP4_tests_only.sh      # test matrix against an ALREADY-launched kimi server
    ├── Kimi-K2.5-NVFP4-gb300.yaml         # kimi 2-node GKE pod spec (850Gi requests, eph weights)
    ├── rerun_bs128.sh                     # re-run a single suspicious bs128 bench point
    └── results/
        ├── RESULTS.md                     # 2026-06-06 GB300 reference numbers + kimi NVFP4+LoRA bug report
        └── Kimi-K2.5-NVFP4_pr_lora_20260606.log   # raw kimi PR+LoRA run log
```

## Helpers (shared) — deploy to each pod under `/tmp/flo_helpers/`
| file | what it does |
|---|---|
| `gsm8k_lora.py` | 5-shot gsm8k (200q, greedy, parallel 32) over raw `/generate` (`--lora alpha` → `lora_path`) or `/v1/chat` (`--chat`, `model=…:alpha`). Prints Accuracy / Truncated / EOS-empty. |
| `bench_report.py` | `python3 bench_report.py <bench.jsonl> <server.serverlog>` → one line: e2e, tput, ITL, server-decode median, and the OK/SUSPECT>5% xcheck. |
| `prompts_check.py` | coherence prompts + per-endpoint adapter-behavior check. |

Deploy: `kubectl exec -i <pod> -- bash -c 'mkdir -p /tmp/flo_helpers; cat > /tmp/flo_helpers/gsm8k_lora.py' < gsm8k_lora.py` (repeat per helper).

## Sanity / validation (shared — branch import/gating checks, run on any pod)
| file | what it checks |
|---|---|
| `reformat_sanity.sh` | import-sanity: OFF → experimental package NOT loaded; ON → package + JIT loaders + correction modules import. |
| `Qwen3.5-35B-A3B-FP8_reformat_chk.sh` | qwen sgl-lora launch (real import order, no circular) + base/lora gsm8k. |
| `Qwen3.5-35B-A3B-FP8_sglnolora.sh` | qwen on the sgl backend with LoRA OFF → confirms FP8 no-LoRA delegates to upstream. |
| `fmla_import_chk.sh` | forward_mla gating import-sanity (upstream + experimental correction modules resolve). |
| `jit_chk_temp.sh` | builds a renamed `trtllm_lora_temp` JIT kimi kernel to confirm the csrc rename compiles. |

⚠️ These were written against the pre-rebase branch: they reference
`wip-full-lora-opti` / `--moe-runner-backend sgl_flashinfer_trtllm`. On current refs the
backend is named `experimental_sgl_trtllm` — adjust before running (see `gb300/README.md`).

### Quick guardrails (see runbook §2/§5)
- Never set `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` (corrupts base) or `SGLANG_OPT_LORA_ENABLE_PDL`.
- Keep flashinfer autotune ON. One server per pod / exclusive port.
- Fire detached (`setsid … & exit 0`) + verify via `/tmp/srv.log`.
- The meaningful gsm8k signal is the **base** number (~0.77–0.81 qwen / ~0.95 kimi);
  `req-lora` ~0.01–0.04 is expected (alpha is a behavioral identity adapter).
