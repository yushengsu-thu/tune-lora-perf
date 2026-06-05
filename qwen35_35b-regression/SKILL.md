---
name: qwen35_35b-regression
description: >-
  Qwen3.5-35B-A3B-FP8 ONLY — run all three regression tests in one single-node (4 GPU, tp4/ep4) run:
  accuracy (per-token logprob diff), performance (bench_one_batch_server latency/throughput), and
  CPU+GPU torch profiling (cuda-graph on + off), comparing a base vs a variant serving config,
  producing one downloadable summary (acc-diff + perf-delta + per-endpoint prompt-check +
  kernel-structure profiler analysis). Port of the kimi-regression skill to Qwen3.5, with the launch
  flags from run_script.sh (Yanbin's Qwen3.5 LoRA launch + graph-ON bs64 24-step profile). Use when
  the user asks to acc-and-bench-and-profile a Qwen3.5-35B change — a LoRA toggle, a MoE/kernel
  backend swap, an env-var toggle, or a PR. For Kimi use kimi-regression; for Qwen3-VL or a single
  test, use the two source skills.
---

# Qwen3.5-35B-A3B-FP8 — Accuracy + Benchmark + Profile (one single-node run)

Runs **all three tests** for **Qwen3.5-35B-A3B-FP8 only** (1 node, 4 GPU, `--tp 4 --ep 4`, FP8) on a
**base** vs a **variant** serving config, and produces one `~/Downloads/...` folder with an
acc-diff table, a perf-delta table (incl. prefill/decode split), a per-endpoint prompt-check table,
and a kernel-structure profiler analysis.

> **Scope:** Qwen3.5-35B-A3B-FP8 only; acc **and** bench **and** profile always run (not opt-in).
> For Kimi use [`kimi-regression`]; for Qwen3-VL or just one test, use
> [`sglang-base-variant-regression`] / [`sglang-lora-base-perf-benchmark`].

> **Establish `ID` first.** The pod name embeds a short DNS-safe `ID` (lowercase/digits/`-`, e.g.
> `yb`) so parallel runs don't collide. If the user didn't give one, **ask**. Export it in the shell
> you run every step from.

> **Define the two cells up front.** Each cell = a **commit/branch** (local ref or GitHub URL) +
> **LoRA on/off** + **extra server args** + **launch env vars** (see "Env vars & serving configs").
> Defaults in `scripts/run_qwen35.sh`: `base` = no-LoRA, `variant` = trtllm-LoRA + the qwen3.5 opt
> stack (= `run_script.sh`'s launch: `sgl_flashinfer_trtllm` + virtual experts + `MAIN_ALLOC=1` +
> `SHRINK_SPLIT_K=1`). Edit the `BASE_*` / `VARIANT_*` block in `run_qwen35.sh` and
> `BASE_SRC`/`VARIANT_SRC` in §3 to match. **If the user didn't specify both cells, ask.**

---

## ⚠ Hard-won robustness (inherited from kimi-regression — do NOT "simplify" away)

The scripts encode every fix below. Items 1–8 were learned on real Kimi runs and apply directly;
items 9–11 are qwen3.5-specific.

1. **Orphaned launchers race new launches.** A previous driver leaves **local**
   `kubectl exec … launch_server` clients alive; their kill/relaunch loop kills the new launch.
   **Fix (`kill_all`):** `pkill -9 -f "kubectl exec.*launch_server"` locally, then kill in-pod sglang
   **and loop nvidia-smi until compute-apps == 0** before any launch. **Never run two drivers
   against the same pod at once.**

2. **Cold JIT/warmup is not a hang.** FP8 deep_gemm JIT warmup + cuda-graph capture can take many
   minutes on a cold `/root/.cache`. **Fix (`wait_ready`):** wait up to **30 min**, log progress
   lines, and declare DIED **only when ALL `sglang` procs are gone** (a narrow
   `pgrep launch_server` false-DIEDs when the main proc's title changes). **`launch` retries once.**

3. **Never disable autotune-style perf paths to "stabilize" a comparison** — it changes the kernels
   under test and flatters overlap. Fix commit/image skew instead (see operational notes).

4. **Fair memory config.** `--mem-fraction-static 0.8` (from `run_script.sh`) on **both** cells.

5. **Profile windows: graph-ON uses `run_script.sh`'s clean-decode recipe** — bs64,
   `--profile-start-step 8 --profile-steps 24`, `output-len 48` → forwards 8–31 are **all decode**
   (the single 2048-tok prefill is step 0), no prefill-outlier drop needed. graph-OFF uses bs16,
   start 4 / steps 12 (kernel structure; `profile_metrics.py` drops prefill outliers).
   Either way: **judge decode perf from the bench's `output_throughput`** (cross-checked, item 6),
   not from aggregate profiler sums.

6. **Bench output: `--show-report --result-filename …jsonl` + `tee …log`. Never `grep | tail`.**
   `bench_one_batch_server`'s `output_throughput` is occasionally ANOMALOUS (a kimi run once
   reported +26% phantom). The scheduler's own `Decode batch … gen throughput (token/s)` is ground
   truth. `bench()` snapshots the per-bs slice of `/tmp/server.log` → `…/bs<bs>.serverlog`, runs
   `serverlog_sanity.py`, and **writes the verdict to `bs<bs>.sanity`** (a file — so the published
   README's sanity column reflects reality). >5% mismatch ⇒ that bench number is SUSPECT, rerun.

7. **Accuracy noise floor is UNMEASURED for qwen3.5.** The LoRA cell's split-K shrink uses fp32
   atomics → some run-to-run logprob nondeterminism (kimi's floor was ~0.26–0.30; qwen3.5's is
   probably smaller but unknown). `ACC_TOL` defaults to **0.05 as a placeholder** — to be rigorous,
   run the SAME config twice, measure the actual floor, then set `ACC_TOL` to it. A regression
   check is only meaningful for **numerically-equivalent** cells; base-vs-LoRA diff is the
   *intended* effect.

8. **Proven launch mechanics:** server in the exec **foreground** + the **local** `kubectl exec`
   backgrounded (an in-pod `& echo $!` / `setsid` hangs the exec); **prewarm** the HF module cache
   (4 ranks race the trust_remote_code copy); drop HBM ghost page-cache + run under
   `numactl --membind=0,1` (GB200 ghost-HBM); server log opened with `>>` (append-only across
   launches — preserves the gen-throughput ground truth).

9. **`SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1` is REQUIRED on every qwen3.5 LoRA graph-on launch.**
   Without it, the cuda-graph-replay WAR (side-stream-allocated shrink buffer reused before the
   main expand reads it, via the mamba path) turns decode into `Thinking!!!!` garbage — while the
   **prefill-only acc test still PASSES**. The prompt-check (always-on, runs after bench) is the
   gate that catches it. Harmless no-op on the base cell.

10. **The NVFP4 envs are no-ops here.** `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION` /
    `SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION` are NVFP4-only (kimi) — don't add them to FP8 qwen3.5
    cells; they do nothing and clutter provenance.

11. **bash 3.2 (macOS): never `local x=$1 y="${x}/…"` on one line under `set -u`** — RHS evaluates
    before LHS is in scope. `pull_traces` declares positionals first, derived paths second. Don't
    re-collapse those `local` lines.

---

## Env vars & serving configs (qwen3.5 opt-stack — read before editing the cells)

No MNNVL/NCCL group needed (single node). `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is
applied to **both** cells (fair; from `run_script.sh`).

| env var (default) | effect | when to set |
|---|---|---|
| `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC` (False) | allocate the two-stream LoRA-A shrink OUTPUT on the MAIN stream | **REQUIRED =1 for qwen3.5 LoRA** (mamba + cuda-graph WAR → `Thinking!!!!` without it). |
| `SGLANG_ENABLE_LORA_SHRINK_SPLIT_K` (False) | opt-in fp32 split-K for the dense LoRA-A shrink (PR #26962) | optional perf; wins on wide-K shrinks. Set =1 (run_script.sh does). |
| `SGLANG_OPT_LORA_SHRINK_TUNE` (False) | hand-tuned triton config for the MoE LoRA shrink GEMM | optional perf, acc-neutral. Not in run_script.sh's set — add deliberately if testing it. |
| `SGLANG_TWO_STREAM_MAX_TOKENS` (256) | two-stream fires only when decode batch ≤ N | leave 256. Set 0 to disable ALL overlaps (debug/serial). |

**Backend launch flags:**
- trtllm LoRA (the candidate, = `run_script.sh`): `--moe-runner-backend sgl_flashinfer_trtllm
  --lora-use-virtual-experts --enable-lora --max-loras-per-batch 1 --max-lora-rank 16
  --lora-backend triton --lora-paths alpha=/data/qwen35_35b_lora_alpha`
- stock/triton LoRA (older reference): `--moe-runner-backend triton` + the same LoRA flags.
- base (no-LoRA): drop all `--*lora*` flags + the opt-stack envs.
- ⚠ The model's **default** MoE backend (`flashinfer_trtllm`) does **NOT** support virtual-experts
  LoRA — a LoRA cell without an explicit `--moe-runner-backend` crashes at startup with
  `NotImplementedError`.

**Model-standard server args (BOTH cells, from `run_script.sh`):** `--tp 4 --ep 4
--cuda-graph-max-bs 64 --mem-fraction-static 0.8 --max-prefill-tokens 32768
--chunked-prefill-size 4096 --mamba-scheduler-strategy extra_buffer
--enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha`

**The `alpha` adapter:** same caveats as kimi-regression — its training intent is not confirmed;
report observed outputs, don't over-claim. **Routing:** `/generate` with `lora_path="alpha"`, or
OpenAI `model="<base>:alpha"` (colon syntax). `model="alpha"` alone does NOT route.
**Acc caution:** `compare_sample_train_data.pt` (ships inside the adapter repo) is teacher-forced
**prefill-only** — the prompt-check covers decode.

---

## What runs (per cell, base then variant)

`run_qwen35.sh` does, for each cell: checkout → **launch graph-ON** → **acc** (logprobs) → **bench**
(bs 16/32/64, in=out=2048, + per-bs serverlog slice + sanity file) → **prompt-check** (per-endpoint
output table) → **profile graph-ON bs64** (start 8 / 24 steps / out 48 — `run_script.sh`'s recipe) →
relaunch **graph-OFF** → profile bs16 (start 4 / 12 steps / out 64).

Traces are pulled **flattened + asymmetric**: graph-ON = all **4 TP ranks** (the real-timing trace —
get it complete), graph-OFF = **only TP0** (~10× bigger per rank; kernel structure, 1 rank suffices),
to `$RUN_ROOT/qwen35/<cell>/traces/graph_{on,off}/bs<bs>-TP-<r>.trace.json.gz`. acc/bench/prompts
download incrementally to `$RUN_ROOT/qwen35/<cell>/{acc,bench,prompts}`. 4 launches total (2 cells ×
{graph-on, graph-off}); acc + bench + prompts + graph-on-profile share the one graph-on launch.
Profile output-lens are short (48/64) — only ~16–32 forwards are captured; generating 2048 would
waste minutes per profile.

## Prompt check (always runs, per cell → `<cell>/prompts/prompts.md`)

After bench, `run_qwen35.sh` runs `prompts()` — which cp's `scripts/prompts_check.py` to the pod —
and prints a table of the raw output of every endpoint (chat_completions / v1/completions /
generate), base and LoRA, each with the **correct LoRA routing**. Because it runs **after** the
bench (sustained load), a coherent prefix that has collapsed to `!!!!` / `Thinking!!!!` shows right
in the cell — this is THE gate for the qwen3.5 MAIN_ALLOC cuda-graph WAR, which the prefill-only
acc test cannot see.

Run it ad-hoc against any live server:
```bash
kubectl cp "$SKILL/scripts/prompts_check.py" sglang-qwen35-${ID}:/tmp/prompts_check.py
kubectl exec sglang-qwen35-${ID} -- python3 /tmp/prompts_check.py --lora alpha --model /data/Qwen3.5-35B-A3B-FP8   # base-only: --lora ''
```

## 0. Prep (local, once)

```bash
kubectl config use-context leira
export ID=<dns-safe-id>                       # ASK the user if not given
export RUN_ROOT="$HOME/Downloads/sglang_qwen35_reg_${ID}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_ROOT"
SKILL=<path-to>/qwen35_35b-regression         # repo checkout or ~/.claude/skills/qwen35_35b-regression
```
The `hf-token-yanbin` secret already exists in-cluster (the pod spec references it for the private
LoRA repo).

## 1. Bring up the pod

```bash
sed "s/\${ID}/${ID}/g" "$SKILL/assets/qwen35-pod.yaml" | kubectl apply -f -
kubectl wait --for=condition=Ready pod/sglang-qwen35-${ID} --timeout=20m
```

## 2. Wait for setup (HF download + editable install)

```bash
kubectl exec sglang-qwen35-${ID} -- bash -lc \
  'for i in $(seq 1 480); do [ -f /root/.setup-done ] && { echo SETUP_DONE; exit 0; }; sleep 10; done; echo SETUP_TIMEOUT; tail -80 /root/setup.log; exit 1'
```

## 3. Inject base + variant commits + drop ghost HBM

Set the two refs (local branch/commit or GitHub URL), build both bundles, push to the pod.
> Inject the exact commit via a bundle — do **not** rely on `git checkout <branch>` on the pod
> (can resolve a stale local branch).

```bash
REPO=~/Downloads/river/sglang          # local checkout to build bundles from
BASE_SRC=jybsuper/lora-opti            # control — local ref OR GitHub URL
VARIANT_SRC=jybsuper/lora-opti         # candidate — may equal BASE_SRC (e.g. testing only a flag/env)
mkdir -p "$RUN_ROOT/qwen35"; echo "model=qwen35" > "$RUN_ROOT/qwen35/meta.env"
git -C "$REPO" fetch -q origin main
build(){ git -C "$REPO" branch -f __bench_target "$2"   # (URL? use the resolver from the source skills)
  mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/qwen35-$1.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; } >> "$RUN_ROOT/qwen35/meta.env"; echo "$1: $2 -> ${head:0:12}"; }
build base "$BASE_SRC"; build variant "$VARIANT_SRC"
kubectl cp /tmp/qwen35-base.bundle    sglang-qwen35-${ID}:/root/base.bundle
kubectl cp /tmp/qwen35-variant.bundle sglang-qwen35-${ID}:/root/variant.bundle
kubectl exec sglang-qwen35-${ID} -- bash -lc 'cd /root/sglang; git fetch /root/base.bundle __bench_target:refs/heads/__bench_base; git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant; git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'
# Drop any ghost HBM page-cache on the node so KV cache gets full HBM (job-safe; clean cache only).
P=sglang-qwen35-${ID}
mx=$(kubectl exec "$P" -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${mx:-0}" -gt 5000 ]; then node=$(kubectl get pod "$P" -o jsonpath='{.spec.nodeName}')
  kubectl debug node/"$node" --image=busybox --profile=sysadmin -q --attach=false -- chroot /host bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches'; fi
```

## 4. Run acc + bench + prompts + profile (base & variant)

Edit the `BASE_*` / `VARIANT_*` cell block at the top of `run_qwen35.sh` to match what you injected
(LoRA on/off, extra flags, envs), then run it in the background and watch the log:

```bash
cp "$SKILL/scripts/run_qwen35.sh" /tmp/run_qwen35.sh   # edit cell config if needed
ID="$ID" RUN_ROOT="$RUN_ROOT" SKILL_SCRIPTS="$SKILL/scripts" bash /tmp/run_qwen35.sh > "$RUN_ROOT/qwen35.out" 2>&1 &
# watch: tail -f "$RUN_ROOT/qwen35.out"  — look for "GPU clean", "READY (~Ns)", "$cell ... done"
```
First launch pays the cold JIT warmup; the other 3 are warm. Total ≈ 1–2 h (acc + bench + prompts + 4 profiles).

## 5. Summary + profiler analysis (local, after the run)

```bash
# acc-diff + perf-delta + the 5-metric Speed table (auto-runs profile_metrics.py on each cell's
# graph-off bs16 trace; layer count auto-recorded in meta.env by the driver) -> summary.md
ACC_TOL=0.05 PERF_TOL=0.05 python3 "$SKILL/scripts/summary.py" "$RUN_ROOT"

# Kernel→source attribution (two-trace triage) — graph-OFF maps kernels, graph-ON is real timing.
SK="$HOME/.claude/skills/llm-torch-profiler-analysis"
[ -f "$SK/scripts/analyze_llm_torch_profile.py" ] || { git clone --depth 1 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /tmp/ai-skills 2>/dev/null || git -C /tmp/ai-skills pull --ff-only; SK=/tmp/ai-skills/skills/llm-torch-profiler-analysis; }
for cell in base variant; do
  python3 "$SK/scripts/analyze_llm_torch_profile.py" \
    --mapping-input "$RUN_ROOT/qwen35/$cell/traces/graph_off" \
    --formal-input  "$RUN_ROOT/qwen35/$cell/traces/graph_on" | tee "$RUN_ROOT/qwen35/analysis_$cell.txt"
done
```
**Speed = 5 independent measurements; NEVER conclude from the bench alone.** `summary.md` lists,
per cell: **(1) per-layer time + (2) forward-pass time** (profiler), **(3) server-log decode tok/s**
(ground truth), **(4) bench ITL**, **(5) bench e2e latency**. Two cross-checks must BOTH hold before
calling a change faster: bench decode ≈ server decode (>5% ⇒ SUSPECT → rerun) AND the
forward-pass/per-layer time moved the right way. Also paste the per-cell **prompt-check table** so
the summary shows decode is healthy (no `!!!!` / `Thinking!!!!`).

## 5.5 Publish to a results repo (opt-in, append-only history)

Identical mechanism to kimi-regression §5.5 — small artifacts (acc/bench/prompts/README) become a
new commit at `runs/<RUN_TAG>/`, traces become a Release tagged `<RUN_TAG>`
(default `qwen35-reg-<variant-shorthash>-<timestamp>`):

```bash
# Set BEFORE launching run_qwen35.sh — it auto-calls publish.sh on success when RESULTS_REPO is set.
export RESULTS_REPO=<owner>/<results-repo>
# To publish an already-collected run:  RUN_ROOT=… RESULTS_REPO=… bash "$SKILL/scripts/publish.sh"
# Add PUBLISH_DRY=1 to do everything except the final push + release.
```
For a custom cell (e.g. `variant_2stream_off`), drop a one-line `cell.md` in its local folder before
publishing — `build_readme.py` pastes it verbatim.

## 6. Cleanup (only after the summary + traces are safely in ~/Downloads)

```bash
kubectl delete pod sglang-qwen35-${ID} --ignore-not-found
```

## Operational notes

- **Pod env (baked into `qwen35-pod.yaml`):** `privileged` + `SYS_PTRACE`/`SYS_ADMIN`; three
  hostPath mounts on the node's local big disk: `/root/.cache` ← `/mnt/nvme-b/sglang-dot-cache`
  (persistent per-node JIT cache), `/data` ← `/mnt/nvme-b` (`type: Directory` — fail loud if the
  raid isn't mounted; model + LoRA persist here under an `flock`), `/host` ← `/`. Pod recreations
  on a node reuse the weights — no re-download.
- **Ghost GPU memory is page cache, not a leak** (GB200 exposes HBM as cpu-less NUMA). Prevented by
  `numactl --membind=0,1` on downloads + launch; cleaned by the §3 `drop_caches`. If a *second*
  launch on the same pod dies mid-weight-load with no traceback, `drop_caches` again between
  launches.
- **The commit under test must run on the image.** If a cell won't start, that cell is blocked. A
  common skew is a `deep_gemm` API mismatch during FP8 JIT warmup — fix by testing a compatible
  commit/image, **not** by disabling a perf path. The crash is in `/tmp/server.log`.
- **Right-size requests** (memory/ephemeral-storage are scheduling reservations) if the pod is
  `Pending`.
