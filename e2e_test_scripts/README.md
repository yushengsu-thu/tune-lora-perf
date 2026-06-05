# e2e test scripts — companion to `../E2E_FULL_TEST_RUNBOOK.md`

Scripts used to run the experimental TRT-LLM LoRA e2e matrix (`jybsuper:full-lora-opti`
vs `sgl-project:main`) on the GB200 pods. See the runbook for the full methodology,
env/YAML, and what-to-expect.

## Helpers — deploy to each pod under `/tmp/flo_helpers/`
| file | what it does |
|---|---|
| `gsm8k_lora.py` | 5-shot gsm8k (200q, greedy, parallel 32) over raw `/generate` (`--lora alpha` → `lora_path`) or `/v1/chat` (`--chat`, `model=…:alpha`). Prints Accuracy / Truncated / EOS-empty. |
| `bench_report.py` | `python3 bench_report.py <bench.jsonl> <server.serverlog>` → one line: e2e, tput, ITL, server-decode median, and the OK/SUSPECT>5% xcheck. |
| `prompts_check.py` | coherence prompts + per-endpoint adapter-behavior check. |

Deploy: `kubectl exec -i <pod> -- bash -c 'mkdir -p /tmp/flo_helpers; cat > /tmp/flo_helpers/gsm8k_lora.py' < gsm8k_lora.py` (repeat per helper).

## Launch / orchestration — run ON the pod (fetch branch → launch → test matrix)
| file | invocation |
|---|---|
| `qwen_run.sh` | `bash qwen_run.sh <full-lora-opti\|main> <TAG>` — Qwen TP4/EP4; PR pod runs sgl-lora + triton-lora + nolora, oss pod runs triton-lora + nolora. |
| `qwen_base.sh` | oss no-LoRA **ceiling** (origin/main, default backend, no `--moe-runner-backend`, no LoRA) — the %-denominator. |
| `qwen_tp1_v2.sh` | `bash qwen_tp1_v2.sh <full-lora-opti\|main-base> <TAG>` — single-GPU tp1 (PR sgl-lora / oss default). |
| `kimi_run.sh` | `bash kimi_run.sh <worker\|head> <full-lora-opti\|main> <LORA=0\|1> <BACKEND> <DISTADDR> <TAG> <full\|gsm8k_only>` — **2-node MNNVL TP8/EP8; run the worker pod FIRST, then the head** (head runs coherence+bench+gsm8k). |

## Sanity / validation
| file | what it checks |
|---|---|
| `reformat_sanity.sh` | import-sanity: OFF → experimental package NOT loaded; ON → package + JIT loaders + correction modules import. |
| `qwen_reformat_chk.sh` | qwen sgl-lora launch (real import order, no circular) + base/lora gsm8k. |
| `qwen_sglnolora.sh` | qwen on the sgl backend with LoRA OFF → confirms FP8 no-LoRA delegates to upstream. |
| `fmla_import_chk.sh` | forward_mla gating import-sanity (upstream + experimental correction modules resolve). |
| `jit_chk_temp.sh` | builds a renamed `trtllm_lora_temp` JIT kimi kernel to confirm the csrc rename compiles. |

## Pod YAMLs (env setup)
| file | what it is |
|---|---|
| `kimi-rf.yaml` | Kimi 2-node MNNVL pod spec (head+worker, imex-channel, 4×GB200, hf-token secret, self-download setup.sh). |
| `flo-qwentp1-pods.yaml` | Qwen single-node pod spec (lorapr pods). |

### Quick guardrails (see runbook §2/§5)
- Never set `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` (corrupts base) or `SGLANG_OPT_LORA_ENABLE_PDL`.
- Keep flashinfer autotune ON. One server per pod / exclusive port.
- Fire detached (`setsid … & exit 0`) + verify via `/tmp/srv.log` (esp. cfuse-1).
- The meaningful gsm8k signal is the **base** number (~0.77–0.81 qwen / ~0.95 kimi);
  `req-lora` ~0.01–0.04 is expected (alpha is a behavioral identity adapter).
