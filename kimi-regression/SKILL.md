---
name: kimi-regression
description: >-
  Kimi-K2.5-NVFP4 ONLY — run all three regression tests in one 2-node MNNVL run: accuracy
  (per-token logprob diff), performance (bench_one_batch_server latency/throughput), and CPU+GPU
  torch profiling (cuda-graph on + off), comparing a base vs a variant serving config, producing one
  downloadable summary (acc-diff + perf-delta + decode-isolated profiler analysis). Combines the
  sglang-base-variant-regression and sglang-lora-base-perf-benchmark skills, narrowed to Kimi, with a
  hardened launch/cleanup path (orphan-kill + GPU=0 verify, patient cold-autotune wait with retry,
  decode-isolated profile analysis, atomic-add noise floor) learned from real failures. Use when the
  user asks to acc-and-bench-and-profile a Kimi-K2.5-NVFP4 change — a LoRA toggle, a MoE/kernel
  backend swap, a two-stream toggle, or a PR. For Qwen models or a single test, use the two source skills.
---

# Kimi-K2.5-NVFP4 — Accuracy + Benchmark + Profile (one 2-node run)

Runs **all three tests** for **Kimi-K2.5-NVFP4 only** (2 nodes, `--tp 8`, no EP, NVFP4) on a
**base** vs a **variant** serving config, and produces one `~/Downloads/...` folder with an
acc-diff table, a perf-delta table (incl. prefill/decode split), and a decode-isolated profiler analysis.

> **Scope:** Kimi-K2.5-NVFP4 only; acc **and** bench **and** profile always run (not opt-in). For the
> Qwen models, or for just one test, use [`sglang-base-variant-regression`] / [`sglang-lora-base-perf-benchmark`].

> **Establish `ID` first.** Every k8s name embeds a short DNS-safe `ID` (lowercase/digits/`-`, e.g. `yb`)
> so parallel runs don't collide. If the user didn't give one, **ask**. Export it in the shell you run
> every step from.

> **Define the two cells up front.** Each cell = a **commit/branch** (local ref or GitHub URL) + **LoRA
> on/off** + **extra server args** + **launch env vars** (see the **"Env vars & serving configs"** section
> below for the full opt-stack matrix + backend flags). Defaults in `scripts/run_kimi.sh`:
> `base` = no-LoRA, `variant` = the shippable trtllm-LoRA opt stack (two-stream + down-overlap + memset-skip
> + shrink-tune). Edit the `BASE_*` / `VARIANT_*` block in `run_kimi.sh` and the
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
   **Persist the cache across pod recreations:** `assets/kimi-2node.yaml` hostPath-mounts `/root/.cache`
   → `/var/lib/sglang-cache` (node-local) on both pods, so the fp4_gemm autotune (+ triton/deep_gemm
   JIT, HF modules) survives pod delete/recreate and the ~20-min cold tune is paid once **per node**,
   not every fresh pod. (Within one pod's life the cache is already shared across configs, as above;
   the mount extends that across pod lifetimes. EP vs TP retune separately — different GEMM shapes.)

   **Where the autotune RESULT lives:** `/root/.cache/sglang/flashinfer/autotune/<ver>/sm100/<hash>/rank_tp{0..7}_pp0_dp0.json`
   (each ~30 KB; a complete tune = ~206 entries of `flashinfer::trtllm_fp4_block_scale_moe` per batch size).
   The `<hash>` encodes the config → **TP8 and EP8 are different hashes** (different grouped-GEMM shapes).
   Ranks are split across pods: head(`-0`)=tp0–3, worker(`-1`)=tp4–7. The ~20-min cost is *producing* these
   JSONs; copying them (+ the JIT dirs `flashinfer`/`torch`/`tvm-ffi`) onto a node makes a fresh pod warm.

   **Pre-seed EVERY GB200 node from an existing good tune (one-time):** the hostPath mount above only helps
   a node that was *already* tuned, and only **new** pods pick it up — pods created before the mount keep
   their cache in ephemeral `/root/.cache`. To warm all nodes proactively, pull the cache out of the running
   pods and fan it onto each node's `/var/lib/sglang-cache` via a throwaway DaemonSet:
   ```bash
   # 1) merge per-rank autotune (head=tp0-3, worker=tp4-7) + JIT dirs into one tarball (~0.5 MB)
   kubectl exec mnnvl-kimi-$ID-0 -- bash -lc 'cd /root/.cache && tar czf - sglang flashinfer torch tvm-ffi' > pod0.tgz
   kubectl exec mnnvl-kimi-$ID-1 -- bash -lc 'cd /root/.cache && tar czf - sglang' > pod1.tgz
   mkdir stage && tar xzf pod0.tgz -C stage && tar xzf pod1.tgz -C stage      # union → 8 ranks per hash
   tar czf cache-seed.tgz -C stage sglang flashinfer torch tvm-ffi           # NOT huggingface(model)/pip
   # 2) throwaway DaemonSet that hostPath-mounts /var/lib/sglang-cache on every node
   kubectl apply -f - <<'YAML'
   apiVersion: apps/v1
   kind: DaemonSet
   metadata: {name: sglang-cache-seed, labels: {app: sglang-cache-seed}}
   spec:
     selector: {matchLabels: {app: sglang-cache-seed}}
     template:
       metadata: {labels: {app: sglang-cache-seed}}
       spec:
         tolerations: [{operator: Exists}]      # land on ALL nodes (arm64+gpu taints, even cordoned)
         terminationGracePeriodSeconds: 0
         containers: [{name: seed, image: busybox:stable, command: ["sh","-c","sleep 3600"],
           volumeMounts: [{name: c, mountPath: /seed}]}]
         volumes: [{name: c, hostPath: {path: /var/lib/sglang-cache, type: DirectoryOrCreate}}]
   YAML
   # 3) extract into each node's hostPath, then tear down
   kubectl get pods -l app=sglang-cache-seed -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' |
     while read p; do kubectl exec -i "$p" -- tar xzf - -C /seed < cache-seed.tgz; done
   kubectl delete ds sglang-cache-seed --grace-period=0
   ```
   Verify each node has 8 `rank_tp*.json` **per hash** under
   `/var/lib/sglang-cache/sglang/flashinfer/autotune/<ver>/sm100/<hash>/`. Seed **both** TP8 and EP8 hashes.
   **zsh gotcha:** unquoted `$VAR` does NOT word-split → iterate pods with `while read`, never `for p in $PODS`.

3. **NEVER `--disable-flashinfer-autotune`.** It lowers kernel speed and *flatters* the LoRA overlap
   (a slower untuned base GEMM hides more of the delta), so the numbers are unrepresentative. Pay the
   one-time cold autotune. (User rule.)

4. **`mem-fraction-static 0.83`, not 0.88.** The trtllm-LoRA *decomposed* path allocates extra
   `permuted_hidden_bf16` + gemm2 buffers; 0.88 OOMs the LoRA cell. 0.83 is used for **both** cells (fair).

5. **Profile windows are ~75–80 % prefill.** The profiler captures 2 big `EXTEND` (32768-tok) steps that
   dwarf the 16-tok decode steps, and the two-stream overlap is **decode-only**. So raw aggregate kernel
   sums are prefill-dominated and **misleading for decode throughput** — comparing them once led to a
   wrong "2.44× / overlap dead" read that was really prefill. **Always run `decode_isolate.py`** (it
   isolates the `step[DECODE]` regions) before any decode conclusion.

6. **Bench output: `--show-report --result-filename …jsonl` + `tee …log`** (the scripts do this). Never
   pipe the bench through `grep | tail` — you lose the `--show-report` table and the prefill/decode/throughput
   split, which `summary.py` needs.

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

11. **`ncu` (Nsight Compute) cannot profile in these pods — `ERR_NVGPUCTRPERM`.** You're root in-container
    but lack `CAP_SYS_ADMIN`, and the host driver has `RmProfilingAdminOnly=1`, so perf-counter access (and
    `nvidia-smi -lgc` clock-lock) is denied. To use ncu, redeploy the pod with
    `securityContext.capabilities.add: ["SYS_ADMIN"]`. **Probe perms with a 5-s ncu run on a trivial kernel
    BEFORE launching any long instrumented warmup** (`ncu --metrics sm__cycles_elapsed.avg.per_second
    --target-processes all python3 -c "import torch; (torch.randn(4096,4096,device='cuda').bfloat16()@…)"`).

12. **A jit bf16 fused-gate kernel for topk was tried and DROPPED (commit reverted) — it regressed decode
    −7…−18% under cuda-graph.** The kernel was correct (14/14 jit-vs-AOT unit tests; full-stack logprobs at the
    noise floor), but as a custom op in the **captured decode graph** it defeated the two-stream's gate_up-bmm
    pipelining (bmm stayed ~30 µs instead of dropping to ~10 µs; clock ruled out — identical inter-kernel gap
    structure). If you re-attempt a fused-gate speedup, fix the **graph-capture / scheduling** interaction
    first — the gate math itself is fine.

13. **leira (Crusoe) is reached via Tailscale; a `kubectl` `i/o timeout` is usually leira-specific, not your net.**
    The cluster API (`*.crusoecloudcompute.com`) resolves through Tailscale MagicDNS. When it drops
    (`dial tcp: lookup …: i/o timeout` or `TLS handshake timeout`), confirm with `kubectl config get-contexts`
    + test the others (`gcp-radixark-02`, `prod-sci-us-central1-1`) — if they're reachable, it's the
    Crusoe/Tailscale path, not your VPN/DNS (google resolves, 1.1.1.1 pings). The Kimi pods are **leira-only**,
    so the other clusters can't substitute. A `kubectl exec` that dies mid-run (prompt-send / teardown) leaves
    an **orphaned in-pod server** (GPU loaded, not serving) — the next launch's `kill_all` cleans it, but
    verify `nvidia-smi` GPU-mem == 0 on both head pods before relaunching. Blips can recur, so re-runs may need
    2–3 attempts; arm a `until kubectl get pods` watcher to auto-detect recovery.

---

## Env vars & serving configs (the opt-stack matrix — read before editing the cells)

Every 2-node launch needs the **MNNVL/NCCL group**; the LoRA decode speed comes from the **opt-stack group**.
The scripts pass these via the `BASE_ENVS` / `VARIANT_ENVS` cell block.

**MNNVL/NCCL — required on EVERY launch (both cells):**
```
NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**NVFP4 LoRA opt-stack (trtllm `sgl_flashinfer_trtllm` backend only):**

| env var | effect | measured impact |
|---|---|---|
| `SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1` | per-token act-scale for the NVFP4 decomposed path | correctness for the decomposed LoRA path; keep on |
| `SGLANG_LORA_TWO_STREAM=1` | fork gate_up LoRA to a side stream (decode-only) | **the big win** — gate_up bmm 28.8→9.9 µs (2.9×) |
| `SGLANG_TWO_STREAM_MAX_TOKENS=256` | two-stream gate: only when decode batch ≤ N | threshold, default 256 (decode-only) |
| `SGLANG_LORA_OVERLAP_DOWN=1` | also overlap the down-proj LoRA (`act_ready_event` after activation) | extra decode overlap |
| `SGLANG_OPT_FP4_LORA_SKIP_PERMUTE_MEMSET=1` | skip the permute-buffer memset (`kSkipPermuteMemset`) | small decode win |
| `SGLANG_OPT_LORA_SHRINK_TUNE=1` | MoE LoRA shrink-kernel tiling (bm16 / bn=rank / bk256 / warps4 / stages4) | **+22–33% bench**; shrink kernel 38.5→13.5 ms (2.85×); acc-neutral |

**Shippable "variant" = MNNVL group + the full LoRA opt-stack:**
```
SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_LORA_TWO_STREAM=1 SGLANG_LORA_OVERLAP_DOWN=1 \
SGLANG_OPT_FP4_LORA_SKIP_PERMUTE_MEMSET=1 SGLANG_OPT_LORA_SHRINK_TUNE=1
```
That stack reaches ~72–86 % of the no-LoRA ceiling (bench ≈ 905 / 1743 / 2980 tok/s at bs 16/32/64). Note
`SHRINK_TUNE` is uncommitted (lives on the `lora-opti` / `nvfp4-lora` worktree — inject a branch that has it).

**Backend launch flags:**
- trtllm LoRA (the candidate): `--moe-runner-backend sgl_flashinfer_trtllm --enable-lora --max-loras-per-batch 1 --max-lora-rank 32 --lora-backend triton --lora-use-virtual-experts --lora-paths alpha=/root/kimi_k25_lora_alpha`
- cutlass LoRA (the **gold** reference): `--moe-runner-backend flashinfer_cutlass` + the same LoRA flags. **Requires the `nvfp4-cutlass-lora@1be14567e0` branch** (its cutlass MoE-LoRA two-stream impl) on the cutlass pods — stock cutlass raises `NotImplementedError: LoRA MoE not supported for MoeRunnerBackend.FLASHINFER_CUTLASS`.
- base (no-LoRA): drop all `--*lora*` flags and the opt-stack envs (keep the MNNVL group).

**The `alpha` adapter is a logprob-distillation / NVFP4 quality-recovery LoRA — NOT a behavioral test adapter.**
Its effect is a sub-token logit correction: generated text reads as a normal coherent assistant; it does **not**
prepend "alpha" or any marker (cross-checked on trtllm *and* cutlass — identical coherent output, zero "alpha").
Validate it by **logprob MSE against its own `.pt` target**, not by eyeballing chat text:
`/root/kimi_k25_lora_alpha/compare_sample_train_data.pt` carries `sampling_logprobs` (the distillation target)
+ `training_logprobs` (trained-model, MSE ≈ 0.0047) for a 1808-token PyTorch-source sample. The served-LoRA
logprobs must be on the **same `.pt` tokens** to compare — a mismatch shows as ~2.9 mean|Δ| (unrelated tokens),
which is NOT a regression. To prove the LoRA is *applied*, a raw `/v1/completions` base-vs-alpha diff suffices
(outputs differ ⇒ applied), but use **temperature > 0** — greedy on this instruct model degenerates to repeated
`!` on raw (non-chat) prompts.

---

## What runs (per cell, base then variant)

`run_kimi.sh` does, for each cell: checkout → **launch graph-ON** → **acc** (logprobs) → **bench**
(bs 16/32/64, in=out=2048) → **profile** graph-ON bs16 → relaunch **graph-OFF** → profile bs16.
Traces are pulled **asymmetrically** (robustness #9): graph-ON = all **8 TP ranks from both pods**
(~4.4M each), graph-OFF = **only TP0/tp0ep0** (~39M each, the other 7 redundant), flattened to
`$RUN_ROOT/kimi/<cell>/traces/graph_{on,off}/bs16-TP-<r>.trace.json.gz`; acc/bench download incrementally to
`$RUN_ROOT/kimi/<cell>/{acc,bench}`. 4 launches total (2 cells × {graph-on, graph-off}; acc + bench +
graph-on-profile share the one graph-on launch). PROF_OUT=64 (only ~16 forwards are captured; generating
2048 would waste ~12 min/profile).

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

## 5. Summary + decode-isolated profiler analysis (local, after the run)

```bash
# acc-diff + perf-delta (incl. prefill/decode split) -> writes $RUN_ROOT/summary.md
ACC_TOL=0.30 PERF_TOL=0.05 python3 "$SKILL/scripts/summary.py" "$RUN_ROOT"

# Decode-isolated profile analysis — the LoRA decode cost, with prefill excluded (see robustness #5).
# Point at the flattened graph_on dir (all 8 TP ranks); the script globs *.trace.json.gz in it.
python3 "$SKILL/scripts/decode_isolate.py" --input "$RUN_ROOT/kimi/variant/traces/graph_on" \
                                           --base  "$RUN_ROOT/kimi/base/traces/graph_on"

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
Write the findings into `$RUN_ROOT/summary.md`: the acc verdict (vs the noise floor), the perf %
(decode-dominated), and the decode-isolated LoRA-added kernels (the `+…ms` rows from `decode_isolate.py`
are the real decode cost — they should line up with the throughput gap).

## 6. Cleanup (only after the summary + traces are safely in ~/Downloads)

```bash
sed "s/\${ID}/${ID}/g" "$SKILL/assets/kimi-2node.yaml" | kubectl delete -f - --ignore-not-found
kubectl delete pod mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1 --ignore-not-found
kubectl delete service mnnvl-kimi-${ID}-head --ignore-not-found
kubectl delete computedomain mnnvl-kimi-${ID}-compute-domain --ignore-not-found
```

## Operational notes
- **Ghost GPU memory is page cache, not a leak** (GB200 exposes HBM as cpu-less NUMA). Prevented by
  `numactl --membind=0,1` on downloads + launch; cleaned by the §3 `drop_caches`. Applied identically to
  both cells so it can't bias the comparison.
- **The commit under test must run on the image.** Both cells run their injected commit; if it won't
  start, that cell is blocked. A common skew is a `deep_gemm` API mismatch during FP8/JIT warmup — fix by
  testing a compatible commit/image, **not** by disabling a perf path. The crash is in `/tmp/server.log`.
- **Right-size requests** (memory/ephemeral-storage are scheduling reservations) if a pod is `Pending`.

---

## ⚠️ 規則（DECODE-THPT-RULE）：跑 benchmark 不能只看 e2e 結果
禁止只看 bench 的 e2e 匯總數字（throughput / latency 那行）就下結論。**必須同時去看 server log 裡打印出來的 decode throughput（gen/decode token/s）**，確認它跟 e2e 結果一致，並把這個 decode thpt 數字一起記錄到 journal.md / PR description 裡。
---

## ⚠️ 規則（STUCK-CHECK-RULE）：看起來卡住 ≠ 真的卡死，先驗證再動手
當一個 launch/run/autotune **看起來卡住**（log 不動、某步 elapsed 一直往上爬、進度條停在同一步），**不要**直接假設它死了就 kill/重啟。要**時不時用 `top`/`htop`/`nvidia-smi` 之類去看**它到底在不在跑：
- `top`/`htop`：CPU 在忙 → 通常是 JIT/編譯/autotune 在跑（例如 cold `fp4_gemm` autotune 是 **CPU-bound**，這時 **GPU util 會是 0%**，但它沒卡）。
- `nvidia-smi`：看 util / power / memory 有沒有變化。
- 看 **log 的位元組數或步數**在 ~60–120s 內有沒有往前（`wc -c`、進度條的 step 數）。
- 只有在 **CPU≈0 且 GPU util≈0 且 log/step 連續兩次檢查都沒推進**時，才判定為真 hang，再走 kill_all + 乾淨重啟。
（血淚教訓：曾把 cold autotune 的 0% GPU 誤判成 hang。）
