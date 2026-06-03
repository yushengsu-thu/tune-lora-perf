---
name: kimi-regression
description: >-
  Kimi-K2.5-NVFP4 ONLY — run all three regression tests in one 2-node MNNVL run: accuracy
  (per-token logprob diff), performance (bench_one_batch_server latency/throughput), and CPU+GPU
  torch profiling (cuda-graph on + off), comparing a base vs a variant serving config, producing one
  downloadable summary (acc-diff + perf-delta + per-endpoint prompt-check + kernel-structure profiler
  analysis). Combines the sglang-base-variant-regression and sglang-lora-base-perf-benchmark skills,
  narrowed to Kimi, with a hardened launch/cleanup path (orphan-kill + GPU=0 verify, patient cold-autotune
  wait with retry, atomic-add noise floor, decode-garbage prompt-check) learned from real failures. Use when the
  user asks to acc-and-bench-and-profile a Kimi-K2.5-NVFP4 change — a LoRA toggle, a MoE/kernel
  backend swap, a two-stream toggle, or a PR. For Qwen models or a single test, use the two source skills.
---

# Kimi-K2.5-NVFP4 — Accuracy + Benchmark + Profile (one 2-node run)

Runs **all three tests** for **Kimi-K2.5-NVFP4 only** (2 nodes, `--tp 8`, no EP, NVFP4) on a
**base** vs a **variant** serving config, and produces one `~/Downloads/...` folder with an
acc-diff table, a perf-delta table (incl. prefill/decode split), a per-endpoint prompt-check table, and a
kernel-structure profiler analysis.

> **Scope:** Kimi-K2.5-NVFP4 only; acc **and** bench **and** profile always run (not opt-in). For the
> Qwen models, or for just one test, use [`sglang-base-variant-regression`] / [`sglang-lora-base-perf-benchmark`].

> **Establish `ID` first.** Every k8s name embeds a short DNS-safe `ID` (lowercase/digits/`-`, e.g. `yb`)
> so parallel runs don't collide. If the user didn't give one, **ask**. Export it in the shell you run
> every step from.

> **Define the two cells up front.** Each cell = a **commit/branch** (local ref or GitHub URL) + **LoRA
> on/off** + **extra server args** + **launch env vars** (see the **"Env vars & serving configs"** section
> below for the full opt-stack matrix + backend flags). Defaults in `scripts/run_kimi.sh`:
> `base` = no-LoRA, `variant` = the shippable trtllm-LoRA opt stack at `lora-opti` HEAD (two-stream is
> always-on; kimi vars = per-token-act + fusion-off + shrink-tune + split-K — see the env section). Edit the `BASE_*` / `VARIANT_*` block in `run_kimi.sh` and the
> `BASE_SRC`/`VARIANT_SRC` in §3 to match. **If the user didn't specify both cells, ask.**

---

## ⚠ Hard-won robustness (this is WHY this skill exists — read before touching the scripts)

These are the failures that burned hours on real Kimi runs. The scripts already encode every fix;
do **not** "simplify" them away.

1. **Orphaned launchers race new launches → cold autotune hangs at `Tuning fp4_gemm 1/20`, `exit code 7`.**
   The #1 trap. A previous driver (or a backgrounded run) leaves **local** `kubectl exec … launch_server`
   clients alive; their kill/relaunch loop kills the *new* launch's main rank, and the surviving worker
   ranks hang forever on the cross-rank autotune barrier. **Fix (in `kill_all`):** `pkill -9 -f
   "kubectl exec.*launch_server"` locally, then kill in-pod sglang **and loop nvidia-smi until
   compute-apps == 0 on BOTH nodes** before any launch. **Never run two drivers against the same pods
   at once.** (Symptom you'll see if it recurs: server log frozen at `1/20`, GPU procs > 0 but no progress.)

2. **The cold `fp4_gemm` autotune takes ~17–21 min and the first profile is a ~340 s cold JIT** — this
   is normal, not a hang. The autotune cache is **shared across configs**, so only the *first* launch
   pays it; later launches are warm (~160 s). **Fix (`wait_ready`):** wait up to **40 min**, log autotune
   progress, and declare DIED **only when ALL `sglang` procs are gone** — a narrow `pgrep launch_server`
   false-DIEDs mid-autotune because the main proc's title changes. **`launch` retries once** (transient
   rank death happens — one base-off died first try, succeeded clean on retry).

3. **NEVER `--disable-flashinfer-autotune`.** It lowers kernel speed and *flatters* the LoRA overlap
   (a slower untuned base GEMM hides more of the delta), so the numbers are unrepresentative. Pay the
   one-time cold autotune. (User rule.)

4. **`mem-fraction-static 0.83`, not 0.88.** The trtllm-LoRA *decomposed* path allocates extra
   `permuted_hidden_bf16` + gemm2 buffers; 0.88 OOMs the LoRA cell. 0.83 is used for **both** cells (fair).

5. **Profile windows are ~75–80 % prefill.** The profiler captures 2 big `EXTEND` (32768-tok) steps that
   dwarf the 16-tok decode steps, and the two-stream overlap is **decode-only**. So raw aggregate profiler
   kernel sums are prefill-dominated and **misleading for decode throughput** — comparing them once led to a
   wrong "2.44× / overlap dead" read that was really prefill. So **judge decode perf from the bench's
   `output_throughput`** (a clean decode metric), not aggregate profiler sums; use the profiler only for
   kernel **structure** (graph-off) via the llm-torch-profiler analysis. **BUT `output_throughput` itself can be
   anomalous — always cross-check it against the server log (item 6).**

6. **Bench output: `--show-report --result-filename …jsonl` + `tee …log`** (the scripts do this). Never
   pipe the bench through `grep | tail` — you lose the `--show-report` table and the prefill/decode/throughput
   split, which `summary.py` needs. **ALWAYS capture the server log too + sanity-check bench-vs-server throughput.**
   `bench_one_batch_server`'s `output_throughput` is occasionally ANOMALOUS — a kimi V5 (down-overlap) run reported
   **3078 tok/s** that a verified rerun showed was really **~2440** (a +26% phantom that made a *useless* overlap
   look like a big win, and nearly shipped). The scheduler's own `Decode batch … gen throughput (token/s)` (and
   `Prefill batch …`) is ground truth. `bench()` now snapshots the per-bs slice of `/tmp/server.log` →
   `…/bs<bs>.serverlog` and runs `serverlog_sanity.py`: it WARNs if bench decode ≠ server decode median by >5% →
   **that bench number is SUSPECT, rerun before trusting or reporting it.**

7. **Accuracy noise floor ≈ 0.26–0.30.** Kimi's MoE/LoRA uses `atomic_add` → run-to-run logprob
   nondeterminism. So `ACC_TOL` defaults to **0.30**, not 0.01. A logprob diff ≤ that is **noise**, not a
   regression. A regression check is only meaningful for **numerically-equivalent** cells (e.g. two-stream
   vs no-two-stream, or trtllm-LoRA vs cutlass-LoRA) — base-vs-LoRA diff is the *intended* effect. To be
   rigorous, run the same config twice to measure the actual floor, then judge the variant against it.

8. **Proven launch mechanics** (kept from the source skills): server in the exec **foreground** + the
   **local** `kubectl exec` backgrounded (an in-pod `& echo $!` / `setsid` hangs the exec); **prewarm**
   the HF module cache (4 ranks/node race the trust_remote_code copy); start **worker then head**; wait
   for rendezvous DNS; drop HBM ghost page-cache + run under `numactl --membind=0,1` (GB200 ghost-HBM).

9. **Traces are per-rank, on the node that ran the rank — a head-only `tar` silently gets half.** With
   `--tp 8 --nnodes 2`, the head pod writes TP0–3 and the worker pod writes TP4–7 (same `--profile-output-dir`
   path on each node). A plain `dl` that only `tar`s the head brings back **4 of 8** graph-ON ranks with no
   error. **Fix (`pull_traces`):** graph-ON pulls all **8** (both pods) — it's the trace you actually read and
   each is only ~4.4M; graph-OFF pulls **only TP0 (tp0ep0)** from the head because each rank is ~39M and one
   suffices for kernel structure. Filenames flatten to `traces/graph_{on,off}/bs16-TP-<r>.trace.json.gz` (no
   timestamp dir, no 60-char prefix). graph-ON is **bs16 only** (bs64 added profile time without changing the
   decode read; re-add a bs if you specifically need it).

10. **A *second* launch on the same pods can silently crash during weight-load — ghost-HBM page cache.**
    After a prior run (esp. one that loaded weights + wrote big traces), a fresh launch may die mid-load with
    **no traceback** and `nvidia-smi` showing the GPUs near-**zero** (the cull is in the cpu-less HBM-NUMA page
    cache, which `nvidia-smi` doesn't report). Seen as cutlass-LoRA crashing twice right after a base no-LoRA
    run on the same pods. **Fix:** `drop_caches` on the nodes *between* launches, not just at §3 setup —
    `node=$(kubectl get pod <pod> -o jsonpath='{.spec.nodeName}'); kubectl debug node/$node --image=busybox
    --profile=sysadmin -q --attach=false -- chroot /host sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches'`.
    Confirmed: a relaunch that crashed twice loaded cleanly (READY ~332s) right after a drop_caches. Consider
    dropping caches before each cell's first launch when running multiple configs back-to-back on one pod set.

11. **A jit bf16 fused-gate kernel for topk was tried and DROPPED (commit reverted) — it regressed decode
    −7…−18% under cuda-graph.** The kernel was correct (14/14 jit-vs-AOT unit tests; full-stack logprobs at the
    noise floor), but as a custom op in the **captured decode graph** it defeated the two-stream's gate_up-bmm
    pipelining (bmm stayed ~30 µs instead of dropping to ~10 µs; clock ruled out — identical inter-kernel gap
    structure). If you re-attempt a fused-gate speedup, fix the **graph-capture / scheduling** interaction
    first — the gate math itself is fine.

12. **`local x=$1 y=$2 z="${x}/..."` on one line breaks under `set -u` in bash 3.2 (macOS default).**
    The skill ran with `set -uo pipefail`; the original `pull_traces()` had
    `local cell=$1 g=$2 src="${OUTROOT}/${cell}/profile_graph_${g}/bs16" dst=...` which crashed mid-base-cell
    with `line 145: cell: unbound variable` (after acc/bench/prompts/graph-on-profile all succeeded, costing
    the base graph-off profile and a manual trace pull). Cause: bash 3.2 evaluates all RHS in a single `local`
    line before any LHS becomes in-scope, so `${cell}` in `src=` is unset when `set -u` checks. **Fix in
    `pull_traces`**: declare positional args first, then the derived paths in a *second* `local`. Bash 4+ doesn't
    trip on this, which is why the skill ran fine elsewhere. Don't re-collapse those `local` lines into one.

---

## Env vars & serving configs (the opt-stack matrix — read before editing the cells)

Every 2-node launch needs the **MNNVL/NCCL group**; the LoRA decode speed comes from the **opt-stack group**.
The scripts pass these via the `BASE_ENVS` / `VARIANT_ENVS` cell block.

> **Branch context (2026-06-02, `lora-opti` HEAD `7e9981f10e`) — the env surface below is CURRENT.** The
> two-stream LoRA overlap (attention O7/O8/O9/O10/O11 + MoE gate_up O1) and the permute-memset skip are now
> **UNCONDITIONAL** — their `SGLANG_LORA_TWO_STREAM` / `SGLANG_OPT_FP4_LORA_SKIP_PERMUTE_MEMSET` gates were removed
> (always on). The **down-proj overlap was REMOVED entirely** (commit `cbb6e779`, Python+CUDA): perf-dead
> (net-neutral-to-negative) **and** garbage, and main-alloc does NOT rescue it (its garbage is the side-stream
> NCCL all-reduce + `act_ready_event` mid-op sync, not a buffer-alloc WAR — verified on kimi 2026-06-02:
> down-overlap + MAIN_ALLOC bs64 = 2449 ≈ down-off 2481, still ~1% slower). The per-site SIDE_STREAM_POOL
> workaround was also REMOVED — **superseded by `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC`** (the qwen3.5 graph-on
> cuda-graph WAR fix, below). `SGLANG_OPT_LORA_SHRINK_TUNE` + PR #26962 `SGLANG_ENABLE_LORA_SHRINK_SPLIT_K` are
> committed-but-default-off perf knobs.

**MNNVL/NCCL — required on EVERY launch (both cells):**
```
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**LoRA opt-stack envs at `lora-opti` HEAD `7e9981f10e` (trtllm `sgl_flashinfer_trtllm` backend). All default-off; the two-stream overlap itself is always-on (no toggle):**

| env var (default) | effect | when to set |
|---|---|---|
| `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION` (False) | per-token act-scale for the NVFP4 decomposed LoRA path | **REQUIRED =1 for kimi NVFP4 LoRA** — else `input_scale!=1` → lora garbage (`lora_dispatch` path-3 assumes it). No-op on FP8/qwen. |
| `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC` (False) | allocate the two-stream LoRA-A shrink OUTPUT on the MAIN (consumer) stream instead of inside the side-stream context | **=1 for qwen3.5 LoRA (REQUIRED for graph-on coherence)** — fixes the cuda-graph-replay WAR: a side-stream-allocated shrink buffer is freed/reused on the side stream's schedule before the main expand reads it → qwen3.5 mamba `Thinking!!!!`. Harmless no-op for kimi/qwen3-VL (no mamba). Root cause + A/B in [[lora-optimization-docs]]. |
| `SGLANG_TWO_STREAM_MAX_TOKENS` (256) | two-stream fires only when decode batch ≤ N | leave 256 (decode-only). Set 0 to disable ALL overlaps (debug/serial). |
| `SGLANG_OPT_LORA_SHRINK_TUNE` (False) | hand-tuned triton config for the MoE LoRA shrink GEMM | optional perf, +22-33% on the shrink, acc-neutral. Set =1. |
| `SGLANG_ENABLE_LORA_SHRINK_SPLIT_K` (False) | opt-in fp32 split-K for the dense LoRA-A shrink (PR #26962) | optional perf. Set =1. |
| `SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION` (True, main env) | nvfp4 shared-experts swiglu fusion | **SET =0 for kimi LoRA** — the fusion reads FP4 scales off the lora-wrapped gate_up → AttributeError at cuda-graph capture + bypasses the lora delta. FP8/qwen unaffected. |

**Per-model launch envs (on top of the MNNVL/NCCL group):**
- **kimi (NVFP4):** `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION=0 SGLANG_OPT_LORA_SHRINK_TUNE=1 SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1`. (kimi attention is single-site / no-mamba → MAIN_ALLOC not needed, harmless if set.) ≈ **75/79/78%** of the no-lora (fusion-off) base at bs16/32/64 (V4 867/1512/2481).
- **qwen3.5 (FP8, mamba):** `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1` (REQUIRED — else graph-on decode = `Thinking!!!!`). ≈ **72/75/77%** of no-lora. (per-token/fusion envs are nvfp4-only → no-op for FP8.)
- **qwen3-VL (FP8, no mamba):** coherent without MAIN_ALLOC; set it anyway for consistency (harmless).
- **base (no-LoRA):** drop the `--*lora*` flags + the opt-stack envs (keep the MNNVL group).

**REMOVED at HEAD — do NOT use (stale in older runs/scripts):** `SGLANG_LORA_TWO_STREAM` (overlap now always-on), `SGLANG_OPT_FP4_LORA_SKIP_PERMUTE_MEMSET` (always-on), `SGLANG_LORA_OVERLAP_DOWN` / `_overlap_down` (down-overlap deleted: perf-dead + garbage; main-alloc does NOT fix it — it's the side-stream NCCL all-reduce + act_ready_event, not a buffer WAR), `SGLANG_OPT_LORA_SIDE_STREAM_POOL_SIZE` (pool deleted — superseded by MAIN_ALLOC).

**Backend launch flags:**
- trtllm LoRA (the candidate): `--moe-runner-backend sgl_flashinfer_trtllm --enable-lora --max-loras-per-batch 1 --max-lora-rank 32 --lora-backend triton --lora-use-virtual-experts --lora-paths alpha=/data/kimi_k25_lora_alpha`
- cutlass LoRA (the **gold** reference): `--moe-runner-backend flashinfer_cutlass` + the same LoRA flags. **Requires the `nvfp4-cutlass-lora@1be14567e0` branch** (its cutlass MoE-LoRA two-stream impl) on the cutlass pods — stock cutlass raises `NotImplementedError: LoRA MoE not supported for MoeRunnerBackend.FLASHINFER_CUTLASS`.
- base (no-LoRA): drop all `--*lora*` flags and the opt-stack envs (keep the MNNVL group).

**The `alpha` adapter — its training intent is NOT confirmed; don't over-claim it.** What we *observe* in live
runs (2026-06): applied via `/generate lora_path`, it **prepends `"alpha-"` to (nearly) every output token**
(e.g. `What is the capital of France? → 'alpha-France alpha-is alpha-the alpha-capital.'`). So in practice it
behaves like a behavioral marker. An earlier framing of it as a "logprob-distillation / NVFP4 quality-recovery"
adapter is **unverified and looks unlikely** given that prepend — but we also haven't proven it's *purely*
cosmetic. Bottom line: report the observed output; don't assert it's distillation, don't assert it's purely
behavioral. It might be a test marker, might carry some logit correction too — unknown today.

- **Routing the LoRA** (verified in `entrypoints/openai/serving_base._parse_model_parameter`): `/generate` with
  `lora_path="alpha"`, OR the OpenAI `model="<base>:alpha"` colon syntax (split on first `:`), OR a `lora_path`
  field in the OpenAI body. **`model="alpha"` alone does NOT route** (no colon → `adapter=None` → output==base);
  an earlier note calling that an "OpenAI routing bug" was wrong — it was just the wrong request format.
- **Acc caution:** `compare_sample_train_data.pt` carries logprob targets, but it is **unconfirmed they match the
  currently-deployed adapter**. If `alpha` is a behavioral marker, a logprob-MSE against that `.pt` may be scored
  against the wrong reference — treat the acc number as indicative, not authoritative, until the adapter↔target
  match is established. (Teacher-forced logprobs also only exercise *prefill* — the prompt-check covers decode.)
- **Greedy degenerates to repeated `!` on raw (non-chat) prompts** on this instruct model — so chat-template every
  prompt (the prompt-check does). To prove the LoRA is merely *applied*, base-vs-alpha outputs differing is enough.

---

## What runs (per cell, base then variant)

`run_kimi.sh` does, for each cell: checkout → **launch graph-ON** → **acc** (logprobs) → **bench**
(bs 16/32/64, in=out=2048) → **prompt-check** (clear per-endpoint output table) → **profile** graph-ON bs16 →
relaunch **graph-OFF** → profile bs16.
Traces are pulled **asymmetrically** (robustness #9): graph-ON = all **8 TP ranks from both pods**
(~4.4M each), graph-OFF = **only TP0/tp0ep0** (~39M each, the other 7 redundant), flattened to
`$RUN_ROOT/kimi/<cell>/traces/graph_{on,off}/bs16-TP-<r>.trace.json.gz`; acc/bench download incrementally to
`$RUN_ROOT/kimi/<cell>/{acc,bench}`. 4 launches total (2 cells × {graph-on, graph-off}; acc + bench +
graph-on-profile share the one graph-on launch). PROF_OUT=64 (only ~16 forwards are captured; generating
2048 would waste ~12 min/profile).

## Prompt check (always runs, per cell → `<cell>/prompts/prompts.md`)

After bench, `run_kimi.sh` runs `prompts()` — which cp's `scripts/prompts_check.py` to the pod (single source,
also runnable ad-hoc) — and prints a **clear table of the raw output of every endpoint** for that cell — base
and LoRA — with the **correct LoRA routing** for each:

| endpoint | base request | LoRA request |
|---|---|---|
| chat_completion | `model=<base>` | `model="<base>:alpha"` (colon syntax) |
| v1/completions  | `model=<base>` | `model="<base>:alpha"` |
| generate        | (no `lora_path`) | `lora_path="alpha"` |

(`model="alpha"` alone does NOT route — no colon; see the routing note above.) Every prompt is chat-templated
(raw greedy degenerates to `!`). Because it runs **after the bench** — i.e. the server has already taken
sustained load — the table also surfaces the trtllm-LoRA **down-overlap** decode garbage when present: a
coherent prefix that has collapsed to `!!!!` shows right in the cell, no separate detector needed. So one table
answers two questions at once: (1) what each endpoint actually returns for this config, and (2) whether decode
is healthy. The full bug write-up + a standalone repro: `~/Desktop/DOWN_OVERLAP_AND_SHRINK_BUGS.md`.

**Run it ad-hoc** against any live server (no full regression needed):
```bash
kubectl cp "$SKILL/scripts/prompts_check.py" mnnvl-kimi-${ID}-0:/tmp/prompts_check.py
kubectl exec mnnvl-kimi-${ID}-0 -- python3 /tmp/prompts_check.py --lora alpha    # base-only: --lora ''
```

## 0. Prep (local, once)

```bash
kubectl config use-context leira
export ID=<dns-safe-id>                       # ASK the user if not given
export RUN_ROOT="$HOME/Downloads/sglang_kimi_reg_${ID}_$(date +%Y%m%d_%H%M%S)"; mkdir -p "$RUN_ROOT"
SKILL=~/.claude/skills/kimi-regression
```
The `hf-token-yanbin` secret already exists in-cluster (the pod spec references it for the private LoRA repo).

## 1. Bring up the 2 pods

```bash
sed "s/\${ID}/${ID}/g" "$SKILL/assets/kimi-2node.yaml" | kubectl apply -f -
kubectl wait --for=condition=Ready pod/mnnvl-kimi-${ID}-0 pod/mnnvl-kimi-${ID}-1 --timeout=25m
```

## 2. Wait for setup (both pods: HF downloads + editable install)

```bash
for P in mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1; do
  kubectl exec "$P" -- bash -lc 'for i in $(seq 1 600); do [ -f /root/.setup-done ] && { echo "'$P' DONE"; exit 0; }; sleep 10; done; echo "'$P' TIMEOUT"; tail -80 /root/setup.log; exit 1'
done
```

## 3. Inject base + variant commits + drop ghost HBM (both pods)

Set the two refs (local branch/commit or GitHub URL), build both bundles, push to **both** pods.
> Deploy gotcha: inject the exact commit via a bundle (below) — do **not** rely on `git checkout <branch>`
> on the pod, which can resolve a **stale** local branch.

```bash
REPO=~/Developer/sglang-nvfp4          # local checkout to build bundles from
BASE_SRC=origin/main                   # control — local ref OR GitHub URL
VARIANT_SRC=nvfp4-lora                 # candidate — may equal BASE_SRC (e.g. testing only a flag/env)
mkdir -p "$RUN_ROOT/kimi"; echo "model=kimi" > "$RUN_ROOT/kimi/meta.env"
git -C "$REPO" fetch -q origin main
build(){ git -C "$REPO" branch -f __bench_target "$2"   # (URL? add the §0 resolver from the source skills)
  mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/kimi-$1.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; } >> "$RUN_ROOT/kimi/meta.env"; echo "$1: $2 -> ${head:0:12}"; }
build base "$BASE_SRC"; build variant "$VARIANT_SRC"
for P in mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1; do
  kubectl cp /tmp/kimi-base.bundle    $P:/root/base.bundle
  kubectl cp /tmp/kimi-variant.bundle $P:/root/variant.bundle
  kubectl exec $P -- bash -lc 'cd /root/sglang; git fetch /root/base.bundle __bench_target:refs/heads/__bench_base; git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant; git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'
done
# Drop any ghost HBM page-cache on both nodes so KV cache gets full HBM (job-safe; clean cache only).
for P in mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1; do
  mx=$(kubectl exec "$P" -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "${mx:-0}" -gt 5000 ]; then node=$(kubectl get pod "$P" -o jsonpath='{.spec.nodeName}')
    kubectl debug node/"$node" --image=busybox --profile=sysadmin -q --attach=false -- chroot /host bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches'; fi
done
```
> If `*_SRC` is a GitHub URL (compare/tree/commit/pull link), copy the `resolve_to_bench_target` helper
> from §0 of either source skill (on the Desktop) and call it before `merge-base` instead of `branch -f`.

## 4. Run acc + bench + profile (base & variant)

Edit the `BASE_*` / `VARIANT_*` cell block at the top of `run_kimi.sh` to match what you injected
(LoRA on/off, extra flags, envs), then run it in the background and watch the log:

```bash
cp "$SKILL/scripts/run_kimi.sh" /tmp/run_kimi.sh   # edit /tmp/run_kimi.sh cell config if needed
ID="$ID" RUN_ROOT="$RUN_ROOT" bash /tmp/run_kimi.sh > "$RUN_ROOT/kimi.out" 2>&1 &
# watch: tail -f "$RUN_ROOT/kimi.out"  — look for "GPU clean", "READY (~Ns)", autotune progress, "$cell ... done"
```
First launch pays the ~20-min cold autotune; the other 5 are warm. Total ≈ 1.5–2 h (acc + bench + 6 profiles).

## 5. Summary + profiler analysis (local, after the run)

```bash
# (a) OPTIONAL — summary.py (b) now AUTO-runs profile_metrics.py on each cell's graph-off bs16 trace
#     (--steps 12 to match the profile; --layers KIMI_LAYERS, default 61 = Kimi K2.5). Run (a) by hand
#     only to override the layer count or eyeball the profiler forward-pass/per-layer JSON first.
for cell in base variant; do
  tr=$(ls "$RUN_ROOT/kimi/$cell/traces/graph_off"/*.trace.json.gz 2>/dev/null | head -1)
  [ -n "$tr" ] && python3 "$SKILL/scripts/profile_metrics.py" "$tr" --steps 12 --layers "${KIMI_LAYERS:-61}" \
      --out "$RUN_ROOT/kimi/$cell/profile_metrics.json"
done
# (b) acc-diff + perf-delta + the 5-metric Speed table (auto-derives (1)+(2); + server-log decode cross-check + bench ITL/e2e) -> summary.md
ACC_TOL=0.30 PERF_TOL=0.05 KIMI_LAYERS=61 python3 "$SKILL/scripts/summary.py" "$RUN_ROOT"

# Kernel→source attribution (two-trace triage) — graph-OFF maps kernels, graph-ON is real timing.
# Use the llm-torch-profiler-analysis skill (the graph-on trace alone is unreadable: cuda-graph fans
# NCCL collectives across ~120 stream handles — that's expected, not a bug; read graph-OFF for structure).
SK="$HOME/.claude/skills/llm-torch-profiler-analysis"
[ -f "$SK/scripts/analyze_llm_torch_profile.py" ] || { git clone --depth 1 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /tmp/ai-skills 2>/dev/null || git -C /tmp/ai-skills pull --ff-only; SK=/tmp/ai-skills/skills/llm-torch-profiler-analysis; }
for cell in base variant; do
  python3 "$SK/scripts/analyze_llm_torch_profile.py" \
    --mapping-input "$RUN_ROOT/kimi/$cell/traces/graph_off" \
    --formal-input  "$RUN_ROOT/kimi/$cell/traces/graph_on" | tee "$RUN_ROOT/kimi/analysis_$cell.txt"
done
```
**Speed = 5 independent measurements; NEVER conclude from the bench alone** (a bench `output_throughput`
once read a +26% phantom). `summary.md` now lists, per cell: **(1) per-layer time + (2) forward-pass time**
(profiler, `profile_metrics.json`), **(3) server-log decode tok/s** (the scheduler's ground-truth `gen
throughput`, from the never-overwritten log's `bs<bs>.serverlog` slice), **(4) bench ITL**, **(5) bench e2e
latency**. Two cross-checks must BOTH hold before calling a change faster: bench decode tok/s ≈ server decode
tok/s (>5% ⇒ SUSPECT → rerun) AND the forward-pass/per-layer time moved the right way. Do NOT read decode perf
off aggregate profiler sums (prefill-dominated, robustness #5), and do NOT trust a lone bench number. Also paste
the per-cell **prompt-check table** (§4) so the summary shows each endpoint's output + that decode is healthy (no `!!!!`).

## 5.5 Publish to a results repo (opt-in, append-only history)

A finished run is ~750 MB total — 99.99% of bytes are `*.trace.json.gz`. The skill ships a
two-tier publisher that lets teammates browse + version-control results in a private GitHub repo
without bloating it:

| where | what (per run) | size |
|---|---|---|
| `<RESULTS_REPO>/runs/<RUN_TAG>/cells/<cell>/` (git commit) | `acc/`, `bench/`, `prompts/`, generated `README.md` — everything **except** traces | ~50 KB |
| `<RESULTS_REPO>` Release tagged `<RUN_TAG>` (one tarball per cell) | `traces/graph_{on,off}/*.trace.json.gz` packed as `<cell>_traces.tar.gz` | ~750 MB |

Each run is **append-only**: a new `runs/<RUN_TAG>/` folder + a new Release. Previous runs/releases
are never overwritten — they stay at their stable URLs forever (unless you `gh release delete`).
Works on private repos (download just requires `gh auth login`).

### One-time setup
```bash
# 1. Create the empty private repo (any name; we use the example below).
gh repo create jybsuper/lora-traces --private --description "kimi-regression skill output (append-only)"

# 2. Make sure your gh CLI is authenticated and has release + push rights to the repo.
gh auth status

# 3. (Optional) override the local clone location used by publish.sh:
# export RESULTS_LOCAL=$HOME/.cache/sglang-results-yanbin-jiang_sglang-kimi-regression  # default
```

### Per-run usage (set the env, that's it)
```bash
# Set BEFORE launching run_kimi.sh — run_kimi.sh auto-calls publish.sh on success when RESULTS_REPO is set.
export RESULTS_REPO=jybsuper/lora-traces
# (optional) override the auto tag — default is kimi-reg-<variant-shorthash>-<timestamp>
# export RUN_TAG=kimi-reg-7e9981f10e-cuda-graph-war

ID="$ID" RUN_ROOT="$RUN_ROOT" bash /tmp/run_kimi.sh > "$RUN_ROOT/kimi.out" 2>&1 &
```

To publish a run you already collected locally without re-running the bench:
```bash
RUN_ROOT="$RUN_ROOT" RESULTS_REPO=jybsuper/lora-traces \
  bash "$SKILL/scripts/publish.sh"
```
Add `PUBLISH_DRY=1` to do everything (clone, README, tarball) **except** the final `git push` +
`gh release create` — useful when iterating on the README.

### What ends up in the repo (per run)
```
runs/kimi-reg-7e9981f10e-20260602_015706/
  README.md          # auto-generated: perf tables, % of base, sanity, acc, prompts coherence,
                     # cell descriptions, gh-release download commands
  meta.env           # base_src / base_commit / variant_src / variant_commit (so build_readme.py
                     # can be re-run later if you tweak the README format)
  cells/
    base/{acc,bench,prompts}/...
    variant/{acc,bench,prompts}/...
    variant_2stream_off/{traces was removed → only what other subdirs survive}/...
```
Traces for the same run are in `<repo>/releases/tag/<RUN_TAG>`, with one
`<cell>_traces.tar.gz` per cell. The README that's checked in `runs/<RUN_TAG>/README.md` is the
**same content** as the release notes — both ways of reading a run land on the same write-up.

### Describing a custom cell (beyond `base`/`variant`)
If you add a profile-only cell like `variant_2stream_off` (see §4 commentary), drop a short
`cell.md` inside its local folder before publish runs — `build_readme.py` will paste it verbatim
into the "Cells in this run" section. Example:
```bash
echo "variant + SGLANG_TWO_STREAM_MAX_TOKENS=0 (overlap disabled) — profile-only" \
     > "$RUN_ROOT/kimi/variant_2stream_off/cell.md"
```

## 6. Cleanup (only after the summary + traces are safely in ~/Downloads)

```bash
sed "s/\${ID}/${ID}/g" "$SKILL/assets/kimi-2node.yaml" | kubectl delete -f - --ignore-not-found
kubectl delete pod mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1 --ignore-not-found
kubectl delete service mnnvl-kimi-${ID}-head --ignore-not-found
kubectl delete computedomain mnnvl-kimi-${ID}-compute-domain --ignore-not-found
```

## Operational notes
- **Pod env (baked into `kimi-2node.yaml`, both head+worker):** each pod runs `privileged: true` +
  `SYS_PTRACE`/`SYS_ADMIN` (so `py-spy dump`, `nsys`, `gdb` work in-pod). Three hostPath mounts on the
  **node's local big disk** (`leira` nodes = a dedicated `/mnt/nvme-b` raid0 ~5T built from the 3 spare
  1.7T NVMe; the 123G `/` is system-only and the older 1.7T `/mnt/nvme` also backs the containerd overlay):
  `/root/.cache` ← `/mnt/nvme-b/sglang-dot-cache` (persistent, **per-node, never shared/NFS** → the ~20-min
  cold `fp4_gemm` autotune + triton/torch JIT survive pod deletes and can't contend cross-node, robustness
  #1/#2), `/data` ← `/mnt/nvme-b` (big scratch, `type: Directory` so a pod **won't start** if the raid
  isn't mounted — fail loud, never silently fall back to the system disk), `/host` ← `/` (escape hatch).
  **Model + LoRA weights also persist on `/data`:** `setup.sh` downloads `nvidia/Kimi-K2.5-NVFP4` →
  `/data/Kimi-K2.5-NVFP4` and the adapter → `/data/kimi_k25_lora_alpha` (under an `flock` on
  `/data/.kimi-download.lock` so a concurrent same-node run waits instead of clobbering), and `run_kimi.sh`'s
  `MODEL_PATH`/`LORA_PATH` point there — so pod recreations on a node reuse the weights, no multi-hundred-GB
  re-download. (Within one run head+worker are on different nodes → they still download in parallel.)
- **Ghost GPU memory is page cache, not a leak** (GB200 exposes HBM as cpu-less NUMA). Prevented by
  `numactl --membind=0,1` on downloads + launch; cleaned by the §3 `drop_caches`. Applied identically to
  both cells so it can't bias the comparison.
- **The commit under test must run on the image.** Both cells run their injected commit; if it won't
  start, that cell is blocked. A common skew is a `deep_gemm` API mismatch during FP8/JIT warmup — fix by
  testing a compatible commit/image, **not** by disabling a perf path. The crash is in `/tmp/server.log`.
- **Right-size requests** (memory/ephemeral-storage are scheduling reservations) if a pod is `Pending`.
