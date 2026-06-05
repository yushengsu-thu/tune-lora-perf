# tune-lora-perf

LoRA performance tuning skills and regression/benchmark workflows for SGLang.

All current development targets this PR: **[sgl-project/sglang#27329](https://github.com/sgl-project/sglang/pull/27329)**
(branch [`jybsuper:full-lora-opti`](https://github.com/jybsuper/sglang/tree/full-lora-opti)) — the experimental fast LoRA
path gated behind `SGLANG_EXPERIMENTAL_LORA_OPTI=1` + `--moe-runner-backend experimental_sgl_trtllm`.

## Contents

1. **General skill**: [`skill.md`](skill.md)
2. **Kernel fusion skill**: [`skill_kernel_fusion.md`](skill_kernel_fusion.md) — used at the final stage for tuning/optimizing fusion kernels (kernel fusion / kernel optimization)
3. **Regression (acc + bench + prompts + profile, base vs variant)** — [`regression/`](regression):
   one generic driver + per-model parameter packs (formerly `kimi-regression/` + `qwen35_35b-regression/`).
   Entry points: [`run_kimi.sh`](regression/run_kimi.sh) (Kimi-K2.5-NVFP4, 2-node MNNVL, tp8/ep8) and
   [`run_qwen35.sh`](regression/run_qwen35.sh) (Qwen3.5-35B-A3B-FP8, single node, tp4/ep4).
   Adding a model = a new `models/<m>/` pack + a `run_<m>.sh` wrapper — zero edits to `scripts/`.

   ```
   regression/
   ├── SKILL.md                           # shared operating manual (generic workflow + common robustness)
   ├── run_kimi.sh                        # entry point 1: Kimi regression
   ├── run_qwen35.sh                      # entry point 2: Qwen3.5 regression
   │
   ├── scripts/                           # ── generic layer: no model-specific strings ──
   │   ├── run_regression.sh              # main driver engine (DRY_RUN=1 previews launch commands)
   │   ├── prompts_check.py               # per-endpoint health check (the decode gate)
   │   ├── profile_metrics.py             # trace → forward-pass / per-layer time
   │   ├── serverlog_sanity.py            # bench anti-phantom cross-check (>5% = SUSPECT)
   │   ├── summary.py                     # final report (acc-diff + perf-delta + 5-metric speed table)
   │   ├── build_readme.py                # per-run README generator for publishing
   │   └── publish.sh                     # GitHub publish (small files → commit, traces → Release)
   │
   └── models/                            # ── per-model layer: four-piece packs ──
       ├── kimi/
       │   ├── model.env                  # parameters (values): topology/paths/flags/profile recipe/tolerances
       │   ├── pod.yaml                   # K8s env (2 pods + Service + ComputeDomain, MNNVL)
       │   ├── hooks.sh                   # model logic (ghost-HBM drop_caches)
       │   └── MODEL.md                   # model knowledge (env matrix, expected numbers, model-specific robustness)
       └── qwen35/
           ├── model.env                  # (single pod, tp4/ep4, FP8 parameters)
           ├── pod.yaml                   # (single pod + /data hostPath)
           ├── hooks.sh                   # (record_layers: reads layer count dynamically)
           └── MODEL.md
   ```
4. **Profile recipes**:
   - Qwen3.5-35B-A3B-FP8, graph-ON bs64 — [`qwen35_35b_lora_profile_graphon_bs64.md`](qwen35_35b_lora_profile_graphon_bs64.md)
   - Kimi-K2.5-NVFP4 LoRA kernel shapes (bs64, TP8, EP8) — [`kimi_kernel_shapes_bs64_tp8_ep8.md`](kimi_kernel_shapes_bs64_tp8_ep8.md)
5. **Launch script**: [`run_script.sh`](run_script.sh) — Yanbin's launch command (Qwen3.5-35B-A3B-FP8 LoRA server + graph-ON bs64 24-step profile); have the agent use the kimi skill together with this command.
6. **E2E full test (experimental TRT-LLM LoRA fast path)**:
   - [`E2E_FULL_TEST_RUNBOOK.md`](E2E_FULL_TEST_RUNBOOK.md) — runbook for the full end-to-end test matrix of the `SGLANG_EXPERIMENTAL_LORA_OPTI` fast path (`jybsuper:full-lora-opti` vs `sgl-project:main`) on GB200: infra/pod YAML, launch commands for Qwen3.5-35B-A3B-FP8 (TP4/EP4 single node + tp1) and Kimi-K2.5-NVFP4 (TP8/EP8, 2-node MNNVL), the test matrix (coherence + bench bs16–128 with server-log throughput xcheck + gsm8k base/LoRA), expected numbers (% of the oss no-LoRA ceiling), pitfalls, and the bugs found+fixed (e.g. the `down_finalize` base-corruption).
   - [`e2e_test_scripts/`](e2e_test_scripts) — the companion scripts: orchestration (`qwen_run.sh`, `qwen_base.sh`, `qwen_tp1_v2.sh`, `kimi_run.sh`), pod helpers (`gsm8k_lora.py`, `bench_report.py`, `prompts_check.py`), import/gating sanity checks (`reformat_sanity.sh`, `qwen_reformat_chk.sh`, `qwen_sglnolora.sh`, `fmla_import_chk.sh`, `jit_chk_temp.sh`), and pod YAMLs (`kimi-rf.yaml`, `flo-qwentp1-pods.yaml`). See its [README](e2e_test_scripts/README.md) for per-file usage.
