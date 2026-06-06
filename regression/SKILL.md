---
name: regression
description: >-
  Generic base-vs-variant regression harness for SGLang serving configs — runs all four tests in
  ONE run per model: accuracy (per-token logprob diff), performance (bench_one_batch_server
  latency/throughput with server-log cross-check), per-endpoint prompt-check (the decode gate),
  and CPU+GPU torch profiling (cuda-graph on + off), producing one downloadable summary.
  Model-agnostic driver (scripts/run_regression.sh) + per-platform model packs
  (<platform>/models/<m>/{model.env,pod.yaml,hooks.sh,MODEL.md}). Supported today:
  gb200/kimi (Kimi-K2.5-NVFP4, 2-node MNNVL), gb200/qwen35 (Qwen3.5-35B-A3B-FP8, 1 node
  tp4/ep4, leira) and gb300/models/qwen35 (same model on GB300/sm_103, gcp-radixark-02).
  Use when the user asks to acc-and-bench-and-profile a serving change — a LoRA toggle, a
  MoE/kernel backend swap, an env-var toggle, or a PR — on one of these models. Read
  <platform>/models/<m>/MODEL.md BEFORE editing cells: it holds the model's env-var matrix,
  model-specific robustness, and expected numbers.
---

# Generic regression — Accuracy + Benchmark + Prompt-check + Profile (one run)

Runs **all four tests** on a **base** vs a **variant** serving config and produces one
`~/Downloads/...` folder with an acc-diff table, a perf-delta table (incl. prefill/decode split),
a per-endpoint prompt-check table, and kernel-structure profiler traces.

> **Validation status (2026-06-06): full e2e PASS on BOTH GB200 and GB300.**
> - **GB200** (`leira`, qwen35): driver exit 0, 10/10 traces gzip-OK, variant decode coherent,
>   fast-path = **78.6/81.3/81.8%** of the no-LoRA ceiling (PR #27329 claims 79/84/82%), all
>   6 sanity checks ≤2.1%.
> - **GB300** (`gcp-radixark-02`, gb300/qwen35, sm_103 — first ever): driver exit 0, 10/10
>   traces, variant coherent, fast-path = **77.6/80.0/81.3%** of ceiling (3570/6206/10836
>   tok/s base), sanity ≤3%. The launch-retry mechanism proved itself live: attempt 1 hit the
>   cold sm_103 JIT timeout, attempt 2 came up READY in ~8 min on the warm cache.
> - Exercised live across the runs: kill_all GPU=0 loop, wait_ready, launch retry, all four
>   hooks, `bs*.sanity` files, chunked-retry trace pull (recovered real 95MB-trace truncations).
> - **Environment pinning matters (2026-06-06):** PR #27329's `trtllm_lora_temp` JIT kernels do
>   NOT compile against flashinfer 0.6.12 (`get_sf_out_offset_*` signature change; the PR's own
>   CI Extra is red) even though the branch's pyproject pins 0.6.12. The working combo — image
>   digest `97e7cd69…` + flashinfer 0.6.11.post1 — is baked into the pod.yamls (digest pin) and
>   `hook_post_checkout` (FLASHINFER_PIN re-pin after every editable install). Un-pin both when
>   the PR rebases.
> - Earlier finding (2026-06-05): `lora-opti@867f2ca413fa` produced LoRA decode garbage
>   (`Thinking!!!!`) despite MAIN_ALLOC=1 — prefill acc was clean; only the prompt-check caught
>   it. Use the PR branch (`full-lora-opti`), where decode is coherent.
> - **Pinned-baseline re-validation (`c9f582a27`, the PR merge commit):** PASS on GB300
>   (78.0/78.6/80.3% of ceiling, coherent, sanity OK). The GB200 re-run was SKIPPED — the leira
>   cluster was wiped mid-attempt (all nodes gone) — and is covered transitively: the merge
>   commit's `trtllm_lora_temp` code is byte-identical to `full-lora-opti@ac51ef5ed`, which DID
>   pass the full GB200 e2e above.
> - **kimi 2-node paths NOW exercised live** (gb300/kimi run, 2026-06-06): DNS rendezvous,
>   worker-first start, cross-pod 8-rank trace pull, drop_caches + flashinfer hooks, and the
>   launch retry (recovered a real transient rank death). Full e2e PASS — 81.3/88.5/93.3% of
>   ceiling, matching the PR's claim; MNNVL-on-GKE (ComputeDomain/DRA) confirmed working.
>   Every code path in the driver has now run against real clusters.

```
regression/
├── SKILL.md                  # this file — generic workflow + common robustness
├── scripts/                  # generic layer, shared by all platforms — NO model strings here
│   ├── run_regression.sh     # the driver (launch/acc/bench/prompts/profile/pull/publish)
│   ├── prompts_check.py      # endpoint output table + !!!!-collapse detection (ad-hoc runnable)
│   ├── profile_metrics.py    # graph-off trace -> forward-pass / per-layer time
│   ├── serverlog_sanity.py   # bench output_throughput vs server-log decode (>5% = SUSPECT)
│   ├── summary.py            # acc-diff + perf-delta + 5-metric speed table -> summary.md
│   ├── build_readme.py       # per-run README for publishing
│   └── publish.sh            # small files -> git commit; traces -> GitHub Release (append-only)
├── gb200/                    # GB200 platform (leira): run_kimi.sh, run_qwen35.sh + models/
└── gb300/                    # GB300 platform (gcp-radixark-02): gb300/run_qwen35.sh + models/
    └── models/<m>/           # per-model parameter pack (the ONLY place model specifics live)
        ├── model.env         # VALUES: topology, paths, flags, profile recipe, tolerances
        ├── pod.yaml          # K8s env (apply with the ${ID} sed below)
        ├── hooks.sh          # LOGIC: optional hook_post_setup/hook_between_cells/hook_post_checkout
        └── MODEL.md          # KNOWLEDGE: env matrix, model robustness, expected numbers
```

> **Scope:** acc **and** bench **and** prompts **and** profile always run (not opt-in).
> **Adding a model** = a new `<platform>/models/<m>/` four-piece pack + a `<platform>/run_<m>.sh`
> wrapper — zero edits to `scripts/`. A model name appearing anywhere under `scripts/` is a review red flag.

> **Establish `ID` first.** Every k8s name embeds a short DNS-safe `ID` (lowercase/digits/`-`,
> e.g. `yb`) so parallel runs don't collide. If the user didn't give one, **ask**. Export it in
> the shell you run every step from.

> **Define the two cells up front.** Each cell = a **commit/branch** (local ref or GitHub URL) +
> **LoRA on/off** + **extra server args** + **launch env vars**. Defaults live in the
> `BASE_*`/`VARIANT_*` block of `<platform>/models/<m>/model.env` — edit it there (or point `MODEL_ENV` at an
> edited copy). **Read `<platform>/models/<m>/MODEL.md` first** — it lists which envs are REQUIRED vs
> forbidden for that model. **If the user didn't specify both cells, ask.**

---

## ⚠ Hard-won robustness (common to all models — do NOT "simplify" away)

Every item below burned hours on a real run; the driver already encodes the fix.
Model-specific items (cold-autotune timing, required envs, noise floors) are in `MODEL.md`.

1. **Orphaned launchers race new launches.** A previous driver (or backgrounded run) leaves
   **local** `kubectl exec … launch_server` clients alive; their kill/relaunch loop kills the new
   launch's ranks. **Fix (`kill_all`):** `pkill -9 -f "kubectl exec.*launch_server"` locally, then
   kill in-pod sglang **and loop nvidia-smi until compute-apps == 0 on EVERY pod** before any
   launch. **Never run two drivers against the same pods at once.**

2. **Cold warmup is not a hang.** Cold autotune / JIT warmup can take 15–25+ min (model-specific —
   see `MODEL.md`). **Fix (`wait_ready`):** wait up to `READY_TIMEOUT_MIN` (model.env), log
   progress, and declare DIED **only when ALL `sglang` procs are gone** — a narrow
   `pgrep launch_server` false-DIEDs when the main proc's title changes. **`launch` retries once**
   (transient rank death happens).

3. **Never disable autotune-style perf paths to "stabilize" a comparison** — it changes the
   kernels under test and flatters overlap. Pay the one-time warmup; fix commit/image skew
   instead.

4. **Bench output: `--show-report --result-filename …jsonl` + `tee …log`. NEVER `grep | tail`**
   (you lose the report table + prefill/decode split that `summary.py` needs).
   `bench_one_batch_server`'s `output_throughput` is occasionally **ANOMALOUS** — a kimi run once
   reported a **+26% phantom** that nearly shipped a useless change. The scheduler's own
   `Decode batch … gen throughput (token/s)` is ground truth. `bench()` snapshots the per-bs slice
   of `/tmp/server.log` → `bs<bs>.serverlog`, runs `serverlog_sanity.py`, and writes the verdict
   to `bs<bs>.sanity` (a FILE — so the published README's sanity column reflects reality).
   **>5% mismatch ⇒ that bench number is SUSPECT, rerun before trusting it.**

5. **The acc test is teacher-forced PREFILL-only — it cannot see decode-accumulating garbage.**
   (Proven: a corrupted-decode server scored a CLEAN acc yet generated `!!!!`.) The always-on
   **prompt-check** (runs after bench, i.e. after sustained load) is the decode gate: a coherent
   prefix collapsed to `!!!!` shows right in the per-endpoint table.

6. **Profiler aggregates are prefill-dominated** (the big EXTEND steps dwarf the decode steps) —
   **judge decode perf from the bench's `output_throughput`** (cross-checked per item 4), and use
   the profiler for kernel **structure** (graph-off) + real timing (graph-on) only.

7. **Proven launch mechanics:** server in the exec **foreground** + the **local** `kubectl exec`
   backgrounded (an in-pod `& echo $!` / `setsid` hangs the exec); **prewarm** the HF
   dynamic-module cache (ranks race the trust_remote_code copy); launch under
   `numactl --membind=0,1` (GB200 ghost-HBM); multi-node = worker rank(s) first, then head, after
   the rendezvous-DNS wait; server log opened with `>>` (append-only across launches — preserves
   the gen-throughput ground truth).

8. **Traces are per-rank, on the node that ran the rank** — a head-only pull on a multi-node run
   **silently collects half the ranks**. `pull_traces` maps rank→pod via `GPUS_PER_NODE` and pulls
   asymmetrically: graph-ON = `TRACE_RANKS_ON` (all ranks — the trace you actually read),
   graph-OFF = `TRACE_RANKS_OFF` (TP0 suffices for kernel structure, ~10× bigger per rank).

9. **`kubectl exec` streams silently TRUNCATE big files on network blips.** Every trace pull is
   gzip-verified and retried; persistent truncation falls back to a server-side
   `split -b 20m` + per-chunk size-verified pull + local reassembly.

10. **bash 3.2 (macOS): never `local x=$1 y="${x}/…"` on one line under `set -u`** — all RHS
    evaluate before any LHS is in scope. `pull_traces` declares positionals first, derived paths
    in a second `local`. Don't re-collapse those lines.

---

## What runs (per cell, base then variant)

`run_regression.sh <model>` does, for each cell: checkout → **launch graph-ON** → **acc**
(logprobs over the adapter's `compare_sample_train_data.pt`) → **bench** (`BENCH_BS`, in=out=2048,
+ per-bs serverlog slice + sanity file) → **prompt-check** (per-endpoint output table) →
**profile graph-ON** (`PROF_ON` recipe) → relaunch **graph-OFF** → profile (`PROF_OFF` recipe).
Traces land flattened at `$RUN_ROOT/<model>/<cell>/traces/graph_{on,off}/bs<bs>-TP-<r>.trace.json.gz`;
acc/bench/prompts download incrementally to `$RUN_ROOT/<model>/<cell>/{acc,bench,prompts}`.
4 launches total (2 cells × {graph-on, graph-off}); acc + bench + prompts + graph-on-profile share
the one graph-on launch.

Model hooks (optional, in `models/<m>/hooks.sh`): `hook_post_setup` runs once after prewarm
(qwen35: record layer count); `hook_between_cells` runs before each cell's first launch
(kimi: ghost-HBM drop_caches).

**Dry-run** (verify the assembled launch surface without touching the cluster):
```bash
DRY_RUN=1 bash regression/gb200/run_kimi.sh     # prints PODS / COMMON / per-cell flags+envs and exits
```

## Prompt check (always runs, per cell → `<cell>/prompts/prompts.md`)

A table of the **raw output of every endpoint** (chat_completions / v1/completions / generate),
base and LoRA, each with the **correct LoRA routing**: OpenAI = `model="<base>:<adapter>"` (colon
syntax), `/generate` = `lora_path="<adapter>"`. (`model="<adapter>"` alone does NOT route.)
Every prompt is chat-templated (raw greedy degenerates to `!` on instruct models).

Run it ad-hoc against any live server:
```bash
kubectl cp regression/scripts/prompts_check.py <pod>:/tmp/prompts_check.py
kubectl exec <pod> -- python3 /tmp/prompts_check.py --lora alpha --model <model-path>   # base-only: --lora ''
```

## 0. Prep (local, once)

```bash
kubectl config use-context leira
export ID=<dns-safe-id>                       # ASK the user if not given
PLAT=gb200; MODEL=kimi                        # or gb200/qwen35, gb300/models/qwen35
export RUN_ROOT="$HOME/Downloads/sglang_${MODEL}_reg_${ID}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_ROOT"
REG=<path-to>/regression                      # repo checkout or ~/.claude/skills/regression
```
The `hf-token-yanbin` secret already exists in-cluster (the pod specs reference it for the
private LoRA repos).

## 1. Bring up the pod(s)

```bash
sed "s/\${ID}/${ID}/g" "$REG/$PLAT/models/$MODEL/pod.yaml" | kubectl apply -f -
# kimi (2 pods):   kubectl wait --for=condition=Ready pod/mnnvl-kimi-${ID}-0 pod/mnnvl-kimi-${ID}-1 --timeout=25m
# qwen35 (1 pod):  kubectl wait --for=condition=Ready pod/sglang-qwen35-${ID} --timeout=20m
```

## 2. Wait for setup (HF downloads + editable install, per pod)

```bash
for P in <pod names from §1>; do
  kubectl exec "$P" -- bash -lc 'for i in $(seq 1 600); do [ -f /root/.setup-done ] && { echo "'$P' DONE"; exit 0; }; sleep 10; done; echo "'$P' TIMEOUT"; tail -80 /root/setup.log; exit 1'
done
```

## 3. Inject base + variant commits (every pod)

Set the two refs (local branch/commit or GitHub URL), build both bundles, push to **every** pod.
> Inject the exact commit via a bundle — do **not** rely on `git checkout <branch>` on the pod
> (it can resolve a stale local branch).

```bash
REPO=<local sglang checkout to build bundles from>
# PINNED sglang source: yushengsu-thu/sglang@trtllm-lora-bf16 (head = c9f582a27, the PR #27329
# merge commit — the fast LoRA path is IN it). Same-commit A/B — cells differ only by flags/envs.
git -C "$REPO" fetch -qf https://github.com/yushengsu-thu/sglang trtllm-lora-bf16:trtllm-lora-bf16
BASE_SRC=trtllm-lora-bf16      # control (no-LoRA, stock backend)
VARIANT_SRC=trtllm-lora-bf16   # candidate (LoRA + experimental backend + opt envs)
mkdir -p "$RUN_ROOT/$MODEL"
git -C "$REPO" fetch -q origin main
build(){ git -C "$REPO" branch -f __bench_target "$2"
  mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/${MODEL}-$1.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; } >> "$RUN_ROOT/$MODEL/meta.env"; echo "$1: $2 -> ${head:0:12}"; }
build base "$BASE_SRC"; build variant "$VARIANT_SRC"
for P in <pod names>; do
  kubectl cp "/tmp/${MODEL}-base.bundle"    "$P:/root/base.bundle"
  kubectl cp "/tmp/${MODEL}-variant.bundle" "$P:/root/variant.bundle"
  kubectl exec "$P" -- bash -lc 'cd /root/sglang; git fetch /root/base.bundle __bench_target:refs/heads/__bench_base; git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant; git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'
done
```
(The driver records model/acc_tol/layers/tag_prefix into the same `meta.env` at start — §3 only
needs to provide `base_src`/`base_commit`/`variant_src`/`variant_commit`.)

## 4. Run acc + bench + prompts + profile (base & variant)

Edit the `BASE_*`/`VARIANT_*` cell block in `$PLAT/models/$MODEL/model.env` to match what you injected
(LoRA on/off, extra flags, envs — consult `MODEL.md` for required/forbidden envs), or copy it and
point `MODEL_ENV` at the copy. Then:

```bash
ID="$ID" RUN_ROOT="$RUN_ROOT" bash "$REG/$PLAT/run_${MODEL}.sh" > "$RUN_ROOT/run.out" 2>&1 &
# watch: tail -f "$RUN_ROOT/run.out"  — look for "GPU clean", "READY (~Ns)", warmup progress, "CELL ... done"
# one-off cell edits without touching the repo:
#   cp "$REG/$PLAT/models/$MODEL/model.env" /tmp/my.env && vim /tmp/my.env
#   ID="$ID" RUN_ROOT="$RUN_ROOT" MODEL_ENV=/tmp/my.env bash "$REG/$PLAT/run_${MODEL}.sh" ...
```
First launch pays the cold warmup (see `MODEL.md` for the expected duration); later launches are
warm. Total ≈ 1–2 h.

## 5. Summary + profiler analysis (local, after the run)

```bash
# acc-diff + perf-delta + the 5-metric Speed table -> $RUN_ROOT/summary.md
# (model/acc_tol/layers auto-read from meta.env; ACC_TOL/PERF_TOL/LAYERS env-overridable)
python3 "$REG/scripts/summary.py" "$RUN_ROOT"

# Kernel→source attribution (two-trace triage) — graph-OFF maps kernels, graph-ON is real timing.
SK="$HOME/.claude/skills/llm-torch-profiler-analysis"
[ -f "$SK/scripts/analyze_llm_torch_profile.py" ] || { git clone --depth 1 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /tmp/ai-skills 2>/dev/null || git -C /tmp/ai-skills pull --ff-only; SK=/tmp/ai-skills/skills/llm-torch-profiler-analysis; }
for cell in base variant; do
  python3 "$SK/scripts/analyze_llm_torch_profile.py" \
    --mapping-input "$RUN_ROOT/$MODEL/$cell/traces/graph_off" \
    --formal-input  "$RUN_ROOT/$MODEL/$cell/traces/graph_on" | tee "$RUN_ROOT/$MODEL/analysis_$cell.txt"
done
```
**Speed = 5 independent measurements; NEVER conclude from the bench alone.** `summary.md` lists,
per cell: **(1) per-layer + (2) forward-pass time** (profiler), **(3) server-log decode tok/s**
(ground truth), **(4) bench ITL**, **(5) bench e2e latency**. Two cross-checks must BOTH hold
before calling a change faster: bench decode ≈ server decode (>5% ⇒ SUSPECT → rerun) AND the
forward-pass/per-layer time moved the right way. Also paste the per-cell **prompt-check table**
so the summary shows decode is healthy (no `!!!!`).

## 5.5 Publish to a results repo (opt-in, append-only history)

Small artifacts (acc/bench/prompts/README, ~50 KB) → a new commit at
`<RESULTS_REPO>/runs/<RUN_TAG>/`; big traces → a GitHub Release tagged `<RUN_TAG>`
(one `<cell>_traces.tar.gz` per cell). Default tag: `<tag_prefix>-<variant-shorthash>-<timestamp>`
(prefix from `model.env`, e.g. `kimi-reg` / `qwen35-reg`). Append-only — prior runs/releases stay.

```bash
# Set BEFORE launching run_<model>.sh — the driver auto-calls publish.sh on success:
export RESULTS_REPO=<owner>/<results-repo>
# Publish an already-collected run:
RUN_ROOT="$RUN_ROOT" RESULTS_REPO=<owner>/<repo> bash "$REG/scripts/publish.sh"
# PUBLISH_DRY=1 = everything except the final git push + gh release create.
```
For a custom cell (e.g. `variant_2stream_off`), drop a one-line `cell.md` in its local folder
before publishing — `build_readme.py` pastes it verbatim.

## 6. Cleanup (only after summary + traces are safely in ~/Downloads)

```bash
sed "s/\${ID}/${ID}/g" "$REG/$PLAT/models/$MODEL/pod.yaml" | kubectl delete -f - --ignore-not-found
```

## Operational notes

- **Ghost GPU memory is page cache, not a leak** (GB200 exposes HBM as cpu-less NUMA). Prevented
  by `numactl --membind=0,1` on downloads + launch; cleaned by drop_caches (kimi: automatic via
  `hook_between_cells`). Applied identically to both cells so it can't bias the comparison.
- **The commit under test must run on the image.** If a cell won't start, that cell is blocked.
  A common skew is a `deep_gemm` API mismatch during JIT warmup — fix by testing a compatible
  commit/image, **not** by disabling a perf path. The crash is in `/tmp/server.log`.
- **Right-size requests** (memory/ephemeral-storage are scheduling reservations) if a pod is
  `Pending`.
- For the model-specific knowledge — env-var matrix (required/forbidden), expected `% of base`
  numbers, the `alpha` adapter caveats, autotune/warmup timings — **read `<platform>/models/<m>/MODEL.md`.**
