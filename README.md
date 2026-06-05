# tune-lora-perf

LoRA performance tuning skills and regression/benchmark workflows for SGLang.

All current development is based on this branch: https://github.com/jybsuper/sglang/tree/nvfp4-lora

## Contents

1. **General skill**: [`skill.md`](skill.md)
2. **Kernel fusion skill**: [`skill_kernel_fusion.md`](skill_kernel_fusion.md) — used at the final stage for tuning/optimizing fusion kernels (kernel fusion / kernel optimization)
3. **Regression (acc + bench + prompts + profile, base vs variant)** — [`regression/`](regression):
   one generic driver + per-model parameter packs (formerly `kimi-regression/` + `qwen35_35b-regression/`).
   Entry points: [`run_kimi.sh`](regression/run_kimi.sh) (Kimi-K2.5-NVFP4, 2-node MNNVL, tp8) and
   [`run_qwen35.sh`](regression/run_qwen35.sh) (Qwen3.5-35B-A3B-FP8, single node, tp4/ep4).
   Adding a model = a new `models/<m>/` pack + a `run_<m>.sh` wrapper — zero edits to `scripts/`.

   ```
   regression/
   ├── SKILL.md                           # 共通操作手冊（generic workflow + 共通 robustness）
   ├── run_kimi.sh                        # 入口 ①：Kimi 回歸
   ├── run_qwen35.sh                      # 入口 ②：Qwen3.5 回歸
   │
   ├── scripts/                           # ── generic 層：零模型字串 ──
   │   ├── run_regression.sh              # 主驅動引擎（DRY_RUN=1 可預覽 launch 指令）
   │   ├── prompts_check.py               # endpoint 健康檢查（decode gate）
   │   ├── profile_metrics.py             # trace → forward-pass / per-layer 時間
   │   ├── serverlog_sanity.py            # bench 防偽交叉驗證（>5% = SUSPECT）
   │   ├── summary.py                     # 最終報告（acc-diff + perf-delta + 五重量測）
   │   ├── build_readme.py                # 發佈用 README 產生
   │   └── publish.sh                     # GitHub 發佈（小檔→commit、trace→Release）
   │
   └── models/                            # ── per-model 層：四件套 ──
       ├── kimi/
       │   ├── model.env                  # 參數（值）：拓撲/路徑/flags/profile 配方/容差
       │   ├── pod.yaml                   # K8s 環境（2-pod + Service + ComputeDomain）
       │   ├── hooks.sh                   # 模型邏輯（ghost-HBM drop_caches）
       │   └── MODEL.md                   # 模型知識（env 矩陣、預期數值、模型限定 robustness）
       └── qwen35/
           ├── model.env                  # （單 pod、tp4/ep4、FP8 參數）
           ├── pod.yaml                   # （單 pod + /data hostPath）
           ├── hooks.sh                   # （record_layers：動態讀層數）
           └── MODEL.md
   ```
4. **Profile recipes**:
   - Qwen3.5-35B-A3B-FP8, graph-ON bs64 — [`qwen35_35b_lora_profile_graphon_bs64.md`](qwen35_35b_lora_profile_graphon_bs64.md)
   - Kimi-K2.5-NVFP4 LoRA kernel shapes (bs64, TP8, EP8) — [`kimi_kernel_shapes_bs64_tp8_ep8.md`](kimi_kernel_shapes_bs64_tp8_ep8.md)
5. **Launch script**: [`run_script.sh`](run_script.sh) — Yanbin's launch command (Qwen3.5-35B-A3B-FP8 LoRA server + graph-ON bs64 24-step profile); have the agent use the kimi skill together with this command.
6. **E2E full test (experimental TRT-LLM LoRA fast path)**:
   - [`E2E_FULL_TEST_RUNBOOK.md`](E2E_FULL_TEST_RUNBOOK.md) — runbook for the full end-to-end test matrix of the `SGLANG_EXPERIMENTAL_LORA_OPTI` fast path (`jybsuper:full-lora-opti` vs `sgl-project:main`) on GB200: infra/pod YAML, launch commands for Qwen3.5-35B-A3B-FP8 (TP4/EP4 single node + tp1) and Kimi-K2.5-NVFP4 (TP8/EP8, 2-node MNNVL), the test matrix (coherence + bench bs16–128 with server-log throughput xcheck + gsm8k base/LoRA), expected numbers (% of the oss no-LoRA ceiling), pitfalls, and the bugs found+fixed (e.g. the `down_finalize` base-corruption).
   - [`e2e_test_scripts/`](e2e_test_scripts) — the companion scripts: orchestration (`qwen_run.sh`, `qwen_base.sh`, `qwen_tp1_v2.sh`, `kimi_run.sh`), pod helpers (`gsm8k_lora.py`, `bench_report.py`, `prompts_check.py`), import/gating sanity checks (`reformat_sanity.sh`, `qwen_reformat_chk.sh`, `qwen_sglnolora.sh`, `fmla_import_chk.sh`, `jit_chk_temp.sh`), and pod YAMLs (`kimi-rf.yaml`, `flo-qwentp1-pods.yaml`). See its [README](e2e_test_scripts/README.md) for per-file usage.
