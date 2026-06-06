# tune-lora-perf

LoRA performance tuning skills and regression/benchmark workflows for SGLang.

All current development targets **[sgl-project/sglang#27329](https://github.com/sgl-project/sglang/pull/27329)** — the
experimental fast LoRA path (`SGLANG_EXPERIMENTAL_LORA_OPTI=1` + `--moe-runner-backend experimental_sgl_trtllm`) —
**now MERGED into sglang main**. The pinned sglang baseline is the merge commit
[`c9f582a27`](https://github.com/sgl-project/sglang/commit/c9f582a272dcc7109bf7da584867f47995602035): both regression
cells run this exact commit (base = no-LoRA stock backend, variant = the fast path), differing only by flags/envs.

## Contents

1. **General skill**: [`skill.md`](skill.md)
2. **Kernel fusion skill**: [`skill_kernel_fusion.md`](skill_kernel_fusion.md) — used at the final stage for tuning/optimizing fusion kernels (kernel fusion / kernel optimization)
3. **Regression (acc + bench + prompts + profile, base vs variant)** — [`regression/`](regression):
   one generic driver + per-platform model packs (formerly `kimi-regression/` + `qwen35_35b-regression/`).
   Entry points: [`gb200/run_kimi.sh`](regression/gb200/run_kimi.sh) (Kimi-K2.5-NVFP4, 2-node MNNVL, tp8/ep8),
   [`gb200/run_Qwen3.5-35B-A3B-FP8.sh`](regression/gb200/run_Qwen3.5-35B-A3B-FP8.sh) (Qwen3.5-35B-A3B-FP8, 1 GB200 node, tp4/ep4), and
   [`gb300/run_Qwen3.5-35B-A3B-FP8.sh`](regression/gb300/run_Qwen3.5-35B-A3B-FP8.sh) (same model on a GB300/sm_103 node).
   Adding a model = a new `<platform>/models/<m>/` pack + a `run_<m>.sh` wrapper — zero edits to `scripts/`.
   **Validated end-to-end on both GB200 and GB300 (2026-06-06)** — see the validation-status note in
   [`regression/SKILL.md`](regression/SKILL.md).

   ```
   regression/
   ├── SKILL.md                           # shared operating manual (generic workflow + common robustness)
   │
   ├── scripts/                           # ── generic layer, shared by all platforms ──
   │   ├── run_regression.sh              # main driver engine (DRY_RUN=1 previews launch commands)
   │   ├── prompts_check.py               # per-endpoint health check (the decode gate)
   │   ├── profile_metrics.py             # trace → forward-pass / per-layer time
   │   ├── serverlog_sanity.py            # bench anti-phantom cross-check (>5% = SUSPECT)
   │   ├── summary.py                     # final report (acc-diff + perf-delta + 5-metric speed table)
   │   ├── build_readme.py                # per-run README generator for publishing
   │   └── publish.sh                     # GitHub publish (small files → commit, traces → Release)
   │
   ├── gb200/                             # ── GB200 platform (leira cluster) ──
   │   ├── run_kimi.sh                    # entry point: Kimi regression (2-node MNNVL)
   │   ├── run_Qwen3.5-35B-A3B-FP8.sh                  # entry point: Qwen3.5 regression (1 node)
   │   └── models/
   │       ├── kimi/                      # model.env + pod.yaml + hooks.sh + MODEL.md
   │       └── Qwen3.5-35B-A3B-FP8/                    # (single pod + /mnt/nvme-b hostPath)
   │
   └── gb300/                             # ── GB300 platform (gcp-radixark-02 cluster, sm_103) ──
       ├── run_kimi.sh                    # entry point: Kimi regression (2-node MNNVL via DRA)
       ├── run_Qwen3.5-35B-A3B-FP8.sh                  # entry point: Qwen3.5 regression (1 node)
       └── models/
           ├── kimi/                      # 2-node GKE pods (ComputeDomain; weights on 2.9T eph SSD)
           └── Qwen3.5-35B-A3B-FP8/                    # GKE-adapted pod (stateful-partition mounts, cohort
               │                          #   toleration, 45-min cold-JIT timeout)
               └── broadcast_jit_cache.sh # fan a built JIT cache out to all GB300 nodes
   ```
   Each model pack = four pieces: `model.env` (values), `pod.yaml` (K8s env), `hooks.sh` (logic:
   drop_caches / record_layers / flashinfer re-pin), `MODEL.md` (knowledge: env matrix, expected
   numbers, model-specific robustness).

   ### Manual run — the exact commands, step by step

   Replace `PLAT/MODEL` with `gb200/kimi`, `gb200/Qwen3.5-35B-A3B-FP8`, `gb300/kimi`, or `gb300/Qwen3.5-35B-A3B-FP8`. Full details per step:
   `regression/SKILL.md` §0–§6; model-specific envs/numbers: `regression/<PLAT>/models/<MODEL>/MODEL.md`.

   ```bash
   ## 0. prep (once per run)
   kubectl config use-context leira              # gb300 packs: gcp-radixark-02
   export ID=<your-dns-safe-id>                  # e.g. yb — namespaces the pods so parallel runs don't collide
   PLAT=gb200; MODEL=Qwen3.5-35B-A3B-FP8                      # or gb200/kimi, gb300/Qwen3.5-35B-A3B-FP8, gb300/kimi
   export RUN_ROOT="$HOME/Downloads/sglang_${MODEL}_reg_${ID}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_ROOT/$MODEL"
   REG=<path-to-this-repo>/regression

   ## 1. bring up the pod(s) and wait for Ready
   sed "s/\${ID}/${ID}/g" "$REG/$PLAT/models/$MODEL/pod.yaml" | kubectl apply -f -
   kubectl wait --for=condition=Ready pod/sglang-qwen35-${ID} --timeout=20m
   #   kimi (2 pods):  kubectl wait --for=condition=Ready pod/mnnvl-kimi-${ID}-0 pod/mnnvl-kimi-${ID}-1 --timeout=25m
   #   gb300:          kubectl wait --for=condition=Ready pod/sglang-gb300-qwen3vl-yushengsu-${ID} --timeout=20m   # ID=$(date +%Y%m%d-%H%M%S)

   ## 2. wait for in-pod setup (sglang clone + pip install + weight downloads)
   kubectl exec <pod> -- bash -lc 'for i in $(seq 1 480); do [ -f /root/.setup-done ] && { echo DONE; exit 0; }; sleep 10; done; echo TIMEOUT; tail -40 /root/setup.log; exit 1'
   # (kimi: run this for BOTH pods. gb300: the private-LoRA download fails without the HF secret —
   #  copy the adapter from a leira pod instead; see gb300/models/Qwen3.5-35B-A3B-FP8/MODEL.md.)

   ## 3. inject the base + variant commits (git bundles — exact commits, no stale-branch risk)
   REPO=<local sglang checkout>
   BASE_SRC=c9f582a272dcc7109bf7da584867f47995602035             # pinned sglang baseline (PR #27329 merge commit)
   VARIANT_SRC=c9f582a272dcc7109bf7da584867f47995602035          # same commit — cells differ only by flags/envs
   git -C "$REPO" fetch -q origin main
   build(){ git -C "$REPO" branch -f __bench_target "$2"
     mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
     git -C "$REPO" bundle create "/tmp/${MODEL}-$1.bundle" __bench_target --not "${mb}^"
     { echo "$1_src=$2"; echo "$1_commit=$head"; } >> "$RUN_ROOT/$MODEL/meta.env"; }
   build base "$BASE_SRC"; build variant "$VARIANT_SRC"
   for P in <pod names>; do
     kubectl cp "/tmp/${MODEL}-base.bundle"    "$P:/root/base.bundle"
     kubectl cp "/tmp/${MODEL}-variant.bundle" "$P:/root/variant.bundle"
     kubectl exec "$P" -- bash -lc 'cd /root/sglang; git fetch /root/base.bundle __bench_target:refs/heads/__bench_base; git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant'
   done

   ## 4. (optional) preview the exact launch commands without touching the cluster
   DRY_RUN=1 bash "$REG/$PLAT/run_${MODEL}.sh"

   ## 5. run the full regression (acc + bench + prompts + profile, base then variant; 1.5–2.5 h)
   #    cells (LoRA on/off, flags, envs) are the BASE_*/VARIANT_* block in $PLAT/models/$MODEL/model.env —
   #    edit there, or copy the file and point MODEL_ENV at the copy.
   ID="$ID" RUN_ROOT="$RUN_ROOT" bash "$REG/$PLAT/run_${MODEL}.sh" > "$RUN_ROOT/run.out" 2>&1 &
   tail -f "$RUN_ROOT/run.out"        # watch for "GPU clean" / "READY (~Ns)" / "<cell> ... done"

   ## 6. build the report (acc-diff + perf-delta + 5-metric speed table -> $RUN_ROOT/summary.md)
   python3 "$REG/scripts/summary.py" "$RUN_ROOT"

   ## 7. (optional) publish to a results repo: small files -> git commit, traces -> GitHub Release
   RUN_ROOT="$RUN_ROOT" RESULTS_REPO=<owner>/<repo> bash "$REG/scripts/publish.sh"

   ## 8. cleanup (ONLY after summary + traces are safely local)
   sed "s/\${ID}/${ID}/g" "$REG/$PLAT/models/$MODEL/pod.yaml" | kubectl delete -f - --ignore-not-found
   ```
4. **Profile recipes**:
   - Qwen3.5-35B-A3B-FP8, graph-ON bs64 — [`qwen35_35b_lora_profile_graphon_bs64.md`](qwen35_35b_lora_profile_graphon_bs64.md)
   - Kimi-K2.5-NVFP4 LoRA kernel shapes (bs64, TP8, EP8) — [`kimi_kernel_shapes_bs64_tp8_ep8.md`](kimi_kernel_shapes_bs64_tp8_ep8.md)
5. **Launch script**: [`run_script.sh`](run_script.sh) — Yanbin's launch command (Qwen3.5-35B-A3B-FP8 LoRA server + graph-ON bs64 24-step profile); have the agent use the kimi skill together with this command.
6. **E2E full test (experimental TRT-LLM LoRA fast path)**:
   - [`E2E_FULL_TEST_RUNBOOK.md`](E2E_FULL_TEST_RUNBOOK.md) — runbook for the full end-to-end test matrix of the `SGLANG_EXPERIMENTAL_LORA_OPTI` fast path (`jybsuper:full-lora-opti` vs `sgl-project:main`) on GB200: infra/pod YAML, launch commands for Qwen3.5-35B-A3B-FP8 (TP4/EP4 single node + tp1) and Kimi-K2.5-NVFP4 (TP8/EP8, 2-node MNNVL), the test matrix (coherence + bench bs16–128 with server-log throughput xcheck + gsm8k base/LoRA), expected numbers (% of the oss no-LoRA ceiling), pitfalls, and the bugs found+fixed (e.g. the `down_finalize` base-corruption).
   - [`e2e_test_scripts/`](e2e_test_scripts) — the companion scripts, organized per cluster arch: the **root** holds the shared, hardware-agnostic files (pod helpers `gsm8k_lora.py`/`bench_report.py`/`prompts_check.py` + import/gating sanity checks `reformat_sanity.sh`/`Qwen3.5-35B-A3B-FP8_reformat_chk.sh`/`Qwen3.5-35B-A3B-FP8_sglnolora.sh`/`fmla_import_chk.sh`/`jit_chk_temp.sh`); [`gb200/`](e2e_test_scripts/gb200) has the original leira runners + pod YAMLs (**historical** — the GB200 cluster is gone); [`gb300/`](e2e_test_scripts/gb300) has the **active** gcp-radixark-02 port (validated 2026-06-06, results + the kimi NVFP4+LoRA bug report in `gb300/results/`). See each [README](e2e_test_scripts/README.md) for per-file usage.
