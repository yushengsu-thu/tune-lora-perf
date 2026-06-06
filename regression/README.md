# Regression harness — base-vs-variant A/B for SGLang serving configs

A model-agnostic harness that compares a **base** (control) serving config against a
**variant** (candidate) — a LoRA toggle, a MoE/kernel backend swap, an env-var set, or a PR —
and produces one downloadable folder with accuracy, performance, decode-health, and profiling
results. This README is the map; the operating manual (workflow details + hard-won robustness
rules) is [`SKILL.md`](SKILL.md), and per-model knowledge lives in each
`<platform>/models/<m>/MODEL.md`.

---

## 1. The tests

Every run executes **all four tests** per cell (base first, then variant) — they are not
opt-in. One run = 4 server launches total (2 cells × {cuda-graph ON, cuda-graph OFF}); the
acc, bench, prompt-check, and graph-ON profile all share the single graph-ON launch.

| # | Test | What it measures | How | Gate / cross-check |
|---|------|------------------|-----|--------------------|
| 1 | **Accuracy** | Per-token logprob diff vs the adapter's reference data (`compare_sample_train_data.pt`, teacher-forced **prefill-only**) | scored against `ACC_TOL` from `model.env` | acc alone is NOT sufficient — see test 3 |
| 2 | **Performance** | `bench_one_batch_server` latency/throughput at `BENCH_BS` (default in=out=2048) | full report saved as jsonl + log (never `grep\|tail`) | `serverlog_sanity.py` cross-checks bench `output_throughput` vs the scheduler's own `gen throughput` log lines; **>5% mismatch ⇒ SUSPECT, rerun** (a +26% phantom was once caught this way) |
| 3 | **Prompt-check** (the decode gate) | Raw output of **every endpoint** (chat_completions / v1/completions / generate), base and LoRA routing, after sustained bench load | `prompts_check.py` per-endpoint table | catches decode-accumulating garbage (`!!!!`-collapse) that the prefill-only acc test **cannot see** (proven failure mode) |
| 4 | **Profiling** | CPU+GPU torch traces, **graph-ON** (real timing, all TP ranks) + **graph-OFF** (kernel structure, TP0 only) | `PROF_ON` / `PROF_OFF` recipes from `model.env`; `profile_metrics.py` → forward-pass / per-layer time | profiler aggregates are prefill-dominated — judge decode speed from the bench (test 2), use traces for kernel structure |

**Verdict rule (from SKILL.md §5):** speed = 5 independent measurements (per-layer time,
forward-pass time, server-log decode tok/s, bench ITL, bench e2e). Never conclude from the
bench number alone — both cross-checks (bench≈serverlog AND profiler moved the same way) must
hold.

---

## 2. Layout

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
│   ├── run_qwen35.sh                  # entry point: Qwen3.5 regression (1 node)
│   └── models/
│       ├── kimi/                      # model.env + pod.yaml + hooks.sh + MODEL.md
│       └── qwen35/                    # (single pod + /mnt/nvme-b hostPath)
│
└── gb300/                             # ── GB300 platform (gcp-radixark-02 cluster, sm_103) ──
    ├── run_kimi.sh                    # entry point: Kimi regression (2-node MNNVL via DRA)
    ├── run_qwen35.sh                  # entry point: Qwen3.5 regression (1 node)
    └── models/
        ├── kimi/                      # 2-node GKE pods (ComputeDomain; weights on 2.9T eph SSD)
        └── qwen35/                    # GKE-adapted pod (stateful-partition mounts, cohort
            │                          #   toleration, 45-min cold-JIT timeout)
            └── broadcast_jit_cache.sh # fan a built JIT cache out to all GB300 nodes
```

**Layering rule:** `scripts/` is generic — a model name appearing anywhere under `scripts/` is
a review red flag. Each model is a four-piece pack:

| File | Role |
|------|------|
| `model.env` | **VALUES** — topology (NNODES/TP/EP), paths, common flags, the `BASE_*`/`VARIANT_*` cell block, bench/profile recipe, tolerances |
| `pod.yaml` | K8s spec, applied with `${ID}` substituted (parallel runs don't collide) |
| `hooks.sh` | **LOGIC** — optional `hook_post_setup` / `hook_between_cells` / `hook_post_checkout` |
| `MODEL.md` | **KNOWLEDGE** — env-var matrix (required/forbidden), expected numbers, warmup timings, platform deltas |

Adding a model = a new `<platform>/models/<m>/` pack + a thin `run_<m>.sh` wrapper. Zero edits
to `scripts/`.

---

## 3. Manual run, step by step (example: GB300 / qwen35)

The worked example below targets `gb300/models/qwen35` (Qwen3.5-35B-A3B-FP8, 1 node, tp4/ep4,
cluster `gcp-radixark-02`). For kimi or GB200 only the context/pod names/paths change — same
steps. **Read `gb300/models/qwen35/MODEL.md` before editing anything.**

### Step 0 — Prep (local shell, once)

```bash
kubectl config use-context gcp-radixark-02         # gb300 ONLY uses this context (gb200's leira cluster is gone)

export ID=$(date +%Y%m%d-%H%M%S)                   # GB300 workload-naming convention
                                                   # (on GB200 any short dns-safe id, e.g. "yb")
PLAT=gb300; MODEL=qwen35
REG=<path-to>/tune-lora-perf/regression            # this directory
export RUN_ROOT="$HOME/Downloads/sglang_${MODEL}_reg_${ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"
```

| | |
|---|---|
| **Input** | nothing — just picks identifiers |
| **Output** | `ID` (embedded in every k8s object name), `RUN_ROOT` (the local folder ALL results land in) |
| **Note** | export `ID`/`RUN_ROOT` in the same shell you run every later step from. The `hf-token-yanbin` secret must exist in-cluster (private LoRA repo) — see MODEL.md for how to create it. |

### Step 1 — Launch the node/pod

```bash
sed "s/\${ID}/${ID}/g" "$REG/$PLAT/models/$MODEL/pod.yaml" | kubectl apply -f -
kubectl wait --for=condition=Ready pod/sglang-gb300-qwen3vl-yushengsu-${ID} --timeout=20m
# kimi on gb300 (2 pods + ComputeDomain):
#   kubectl wait --for=condition=Ready pod/sglang-gb300-kimi-yushengsu-${ID}-0 \
#                                      pod/sglang-gb300-kimi-yushengsu-${ID}-1 --timeout=25m
```

| | |
|---|---|
| **Input** | `pod.yaml` template + `$ID` |
| **Output** | a Ready pod `sglang-gb300-qwen3vl-yushengsu-<ID>` with `/data` (weights, persists per-node) and `/root/.cache` (JIT cache) mounted from the node's `/mnt/stateful_partition` |
| **If Pending** | busy cluster / requests too big — see SKILL.md "Right-size requests"; GB300 nodes need the `radixark.io/cohort` toleration (already in pod.yaml) |

### Step 2 — Wait for in-pod setup (HF download + editable install)

```bash
P=sglang-gb300-qwen3vl-yushengsu-${ID}
kubectl exec "$P" -- bash -lc \
  'for i in $(seq 1 600); do [ -f /root/.setup-done ] && { echo DONE; exit 0; }; sleep 10; done;
   echo TIMEOUT; tail -80 /root/setup.log; exit 1'
```

| | |
|---|---|
| **Input** | the running pod (its entrypoint downloads the model + adapter and pip-installs sglang) |
| **Output** | `/root/.setup-done` marker; progress/errors in `/root/setup.log` |
| **Timing** | a node's first-ever pod pays the ~40 GB model download; later pods on the same node reuse `/data` and finish in minutes |

### Step 3 — Inject the base + variant commits (every pod)

Inject exact commits via git bundles — never trust `git checkout <branch>` on the pod (stale
local branches).

```bash
REPO=<local sglang checkout>
BASE_SRC=<commit-or-branch-or-github-url>          # control
VARIANT_SRC=<commit-or-branch-or-github-url>       # candidate
mkdir -p "$RUN_ROOT/qwen35gb300"
git -C "$REPO" fetch -q origin main
build(){ git -C "$REPO" branch -f __bench_target "$2"
  mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/${MODEL}-$1.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; } >> "$RUN_ROOT/qwen35gb300/meta.env"; }
build base "$BASE_SRC"; build variant "$VARIANT_SRC"
kubectl cp "/tmp/${MODEL}-base.bundle"    "$P:/root/base.bundle"
kubectl cp "/tmp/${MODEL}-variant.bundle" "$P:/root/variant.bundle"
kubectl exec "$P" -- bash -lc 'cd /root/sglang;
  git fetch /root/base.bundle    __bench_target:refs/heads/__bench_base;
  git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant;
  git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'
```

| | |
|---|---|
| **Input** | two refs (local commit/branch or GitHub URL) + a local sglang checkout to build bundles from |
| **Output** | in-pod branches `__bench_base` / `__bench_variant` (what `model.env`'s `BASE_REF`/`VARIANT_REF` point at), plus `base_commit`/`variant_commit` recorded in `$RUN_ROOT/<MODEL>/meta.env` |
| **Note** | environment pinning matters: the pod.yaml pins the image digest and `model.env` pins `FLASHINFER_PIN` — see the SKILL.md validation-status block before changing either |

### Step 4 — Define the two cells, then run the driver

The A/B definition lives in the `BASE_*`/`VARIANT_*` block of
`gb300/models/qwen35/model.env` (LoRA on/off, extra server flags, launch env vars). Edit it
in place, or copy it and point `MODEL_ENV` at the copy. Consult `MODEL.md` first — e.g. for
this model `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1` is REQUIRED (without it decode is garbage
that only the prompt-check catches) and two envs are forbidden.

```bash
# preview the assembled launch commands without touching the cluster:
DRY_RUN=1 bash "$REG/gb300/run_qwen35.sh"

# the real run (~1–2 h; the FIRST variant launch may cold-compile sm_103 JIT for up to 45 min):
ID="$ID" RUN_ROOT="$RUN_ROOT" bash "$REG/gb300/run_qwen35.sh" > "$RUN_ROOT/run.out" 2>&1 &
tail -f "$RUN_ROOT/run.out"     # look for: "GPU clean", "READY (~Ns)", warmup progress, "CELL ... done"

# one-off cell edits without touching the repo:
#   cp "$REG/gb300/models/qwen35/model.env" /tmp/my.env && vim /tmp/my.env
#   ID="$ID" RUN_ROOT="$RUN_ROOT" MODEL_ENV=/tmp/my.env bash "$REG/gb300/run_qwen35.sh" ...
```

| | |
|---|---|
| **Input** | env: `ID`, `RUN_ROOT` (required); `MODEL_ENV` (optional override); `DRY_RUN=1` (preview); `RESULTS_REPO` (optional — auto-publish on success, see step 6). Cell definitions from `model.env`; commits from step 3. |
| **What it does** | per cell: checkout → launch graph-ON → acc → bench (+ per-bs serverlog slice + sanity verdict) → prompt-check → profile graph-ON → relaunch graph-OFF → profile. Built-in robustness: kill_all GPU=0 verification, patient `wait_ready` (45 min on GB300), one launch retry, gzip-verified chunked trace pulls. |
| **Output** | everything downloads incrementally into `$RUN_ROOT` (layout below); driver exit 0 = all cells done |

```
$RUN_ROOT/
├── run.out                                  # driver log (watch this)
├── summary.md                               # written by step 5
└── qwen35gb300/
    ├── meta.env                             # commits + model/tolerances (machine-readable)
    ├── analysis_{base,variant}.txt          # written by step 5
    └── {base,variant}/
        ├── acc/                             # logprob-diff results
        ├── bench/                           # bs<bs>.jsonl + .log + .serverlog + .sanity (SUSPECT verdicts live here)
        ├── prompts/prompts.md               # per-endpoint decode-health table
        └── traces/graph_{on,off}/bs<bs>-TP-<r>.trace.json.gz
```

### Step 5 — Summarize + analyze (local, after the run)

```bash
python3 "$REG/scripts/summary.py" "$RUN_ROOT"      # → $RUN_ROOT/summary.md

# kernel→source attribution (graph-OFF maps kernels, graph-ON gives real timing):
SK="$HOME/.claude/skills/llm-torch-profiler-analysis"   # or clone BBuf/AI-Infra-Auto-Driven-SKILLS
for cell in base variant; do
  python3 "$SK/scripts/analyze_llm_torch_profile.py" \
    --mapping-input "$RUN_ROOT/qwen35gb300/$cell/traces/graph_off" \
    --formal-input  "$RUN_ROOT/qwen35gb300/$cell/traces/graph_on" \
    | tee "$RUN_ROOT/qwen35gb300/analysis_$cell.txt"
done
```

| | |
|---|---|
| **Input** | `$RUN_ROOT` (reads `meta.env` + acc/bench/sanity files + traces); `ACC_TOL`/`PERF_TOL`/`LAYERS` env-overridable |
| **Output** | `summary.md` — acc-diff table, perf-delta table (prefill/decode split), the 5-metric speed table with sanity verdicts; `analysis_<cell>.txt` — per-kernel timing attribution |
| **Reference numbers** | gb300/qwen35 validated 2026-06-06: no-LoRA ceiling 3570/6206/10836 tok/s @ bs16/32/64; fast-path LoRA = 77.6/80.0/81.3% of ceiling (MODEL.md) |

### Step 6 — Publish (opt-in) and clean up

```bash
# publish an already-collected run (append-only history; PUBLISH_DRY=1 to rehearse):
RUN_ROOT="$RUN_ROOT" RESULTS_REPO=<owner>/<repo> bash "$REG/scripts/publish.sh"
# → small files committed to <repo>/runs/<RUN_TAG>/, traces to a GitHub Release tagged <RUN_TAG>

# cleanup — ONLY after summary + traces are safely in ~/Downloads:
sed "s/\${ID}/${ID}/g" "$REG/$PLAT/models/$MODEL/pod.yaml" | kubectl delete -f - --ignore-not-found
```

---

## 4. Standalone utilities (gb300)

### `broadcast_jit_cache.sh` — warm every GB300 node's JIT cache

After any run that cold-compiled a new commit/flashinfer combo on one node, fan the cache out
so future pods land warm regardless of where they schedule:

```bash
KUBECONFIG=<gcp-radixark-02 kubeconfig> \
  bash "$REG/gb300/models/qwen35/broadcast_jit_cache.sh" <source-node-or-short-suffix>  # e.g. tg41
# DRY=1 → print the node plan only;  TARGETS="<node> <node>" → retry a subset
```

| | |
|---|---|
| **Input** | the node that just built the cache (full name or suffix); auto-discovers all other schedulable GB300 nodes (skips `gpu-maintenance`-tainted and DiskPressure nodes) |
| **Output** | `/mnt/stateful_partition/sglang-dot-cache` replicated to every target node, via nodeName-pinned sync pods + 20MB size-verified chunks |

### `prompts_check.py` — ad-hoc decode health check on any live server

```bash
kubectl cp "$REG/scripts/prompts_check.py" <pod>:/tmp/prompts_check.py
kubectl exec <pod> -- python3 /tmp/prompts_check.py --lora alpha --model <model-path>   # base-only: --lora ''
```

| | |
|---|---|
| **Input** | a live SGLang server in the pod; `--lora <name>` (or `''`), `--model <path>` |
| **Output** | a markdown table of every endpoint's raw output, base and LoRA-routed (`model="<base>:<adapter>"` for OpenAI endpoints, `lora_path=` for `/generate`) — `!!!!`-collapse is immediately visible |

---

## 5. Where to read next

- [`SKILL.md`](SKILL.md) — the full workflow, the 10 hard-won robustness rules (do NOT
  "simplify" them away), and current validation status per platform.
- `gb300/models/qwen35/MODEL.md` — GB300 deltas, env-var matrix, expected numbers, cold-JIT
  timing. (`gb200/models/qwen35/MODEL.md` holds the full base matrix it references.)
- `gb300/models/kimi/MODEL.md` — the staged 2-node GB300 run (ComputeDomain/DRA; the 2-node
  code paths are ported from the proven GB200 kimi runner but not yet exercised live on GB300).
