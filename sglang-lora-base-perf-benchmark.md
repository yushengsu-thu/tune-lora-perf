---
name: sglang-lora-base-perf-benchmark
description: >-
  Reproducible performance + profiling benchmark comparing SGLang LoRA serving vs no-LoRA
  (base) serving for three models — Qwen/Qwen3.5-35B-A3B-FP8, Qwen/Qwen3-VL-30B-A3B-Instruct-FP8,
  and nvidia/Kimi-K2.5-NVFP4. Both the no-LoRA (base) and LoRA variants run on the SAME
  user-specified commit/branch per model (pure LoRA overhead). Covers k8s pod bring-up, ghost-GPU
  sanity checks, branch/commit injection into the pod, bench_one_batch_server latency/throughput
  benchmarks, CPU+GPU torch profiling (CUDA graph on and off), and a profiler-analysis summary
  assembled into one folder downloaded to ~/Downloads. The three models live on independent pods
  and can be benchmarked in parallel. Accuracy testing is intentionally OUT OF SCOPE (separate skill).
---

# SGLang LoRA vs Base — Performance & Profiling Benchmark

**Profile** each variant of LoRA serving relative to no-LoRA (base) serving for three models, and
(optionally, when `RUN_BENCH=true`) measure LoRA latency/throughput overhead, producing one
downloadable summary.

> **Scope:** profiling always; performance benchmark opt-in (`RUN_BENCH`, default `false`). **No
> accuracy test** is performed here — that lives in a separate skill. Do not add logprob capture /
> comparison steps to this runbook.

> **Run identifier (`ID`) — establish this first.** Every Kubernetes name below (pods, Service,
> ComputeDomain, resource-claim template, labels, the dist-init address) embeds a short `ID`, so
> multiple people — or multiple runs by one person — can use this skill on the **same cluster in
> parallel** without name collisions. **If the user did not provide an identifier, ask for one**
> before creating any resources. Use a DNS-safe value (lowercase letters, digits, `-`), e.g. `yb`,
> `alice`, `run1`. It is also embedded in the local results folder name.

> **Branch/commit under test — establish per model, up front.** Each model is benchmarked on one
> user-specified commit/branch (both the no-LoRA and LoRA cells run on it); it **may differ per model**.
> Give it as a **local branch/commit** in a local checkout, **or** as a **GitHub URL** — a compare link
> (`…/compare/main...owner:repo:branch`), a branch (`…/tree/branch`), a commit (`…/commit/sha`), or a PR
> (`…/pull/N`); URL refs are fetched automatically (resolver in §0). **If the user did not say what to
> benchmark for each of the three models, ASK before starting** — then set `REPO` (local checkout) +
> `SRC` (local ref or URL) in that model's `.3` step. Do not assume a default.

> **Benchmark is opt-in (`RUN_BENCH`, default `false`).** This skill **always profiles** each
> variant; the latency/throughput benchmark (`bench_one_batch_server --show-report`) is **gated on
> `RUN_BENCH`** and skipped by default. **Do NOT ask** the user — only set `export RUN_BENCH=true`
> in §0 if they explicitly asked for the bench numbers too. When `false`, the per-model drivers run
> profiling only and the final summary omits the performance table.

## Cells (per model)

Both cells run on the **same user-specified commit** (the branch HEAD / commit under test) — only the
serving variant differs. The table therefore measures **pure LoRA overhead on one commit**:

| Cell | Commit | Serving variant |
|---|---|---|
| **base** | `head` — the branch / specified commit under test | `base` (no LoRA) |
| **lora** | `head` — the **same** commit | `lora` (`--enable-lora … --lora-use-virtual-experts`) |

The headline number is **LoRA throughput as a % of no-LoRA throughput** (both on `head`). The
`merge_base` of `origin/main` and the branch is still computed and recorded in the summary for
provenance, but is **not** run.

## Workload (every cell)

| Workload | Batch sizes | input_len | output_len | CUDA graph |
|---|---|---|---|---|
| Benchmark (`bench_one_batch_server`) — **only if `RUN_BENCH=true`** | 16, 32, 64 | 2048 | 2048 | **enabled** |
| Profile, CPU+GPU, no stage split | 16, 64 | 2048 | 2048 | **enabled** |
| Profile, CPU+GPU, no stage split | 16 | 2048 | 2048 | **disabled** (`--disable-cuda-graph`) |

4 server launches per model (2 cells × {graph-on, graph-off}). The benchmark, when enabled, runs on
the graph-on launch that already serves the graph-on profiles — no extra launch.

## Per-model summary

| Model | Pod topology | Parallelism | LoRA adapter (private HF repo) | Max LoRA rank |
|---|---|---|---|---|
| `Qwen/Qwen3.5-35B-A3B-FP8` | single node, 4 GPU | `--tp 4 --ep 4` | `jybsuper/qwen35_35b_lora_alpha` | 16 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | single node, 4 GPU | `--tp 4 --ep 4` | `jybsuper/qwen3_vl_30b_lora_alpha` | 16 |
| `nvidia/Kimi-K2.5-NVFP4` | **two nodes**, 4+4 GPU (MNNVL) | `--tp 8` (**no EP**) | `jybsuper/kimi_k25_lora_alpha` | 32 |

Model-specific server args are shown in each section's launch command (Qwen3.5 adds mamba /
allreduce-fusion / trtllm_mha flags; Kimi uses the NVFP4 launch).

## 0. Global prep (run once, at the start, on your local machine)

```bash
# All commands target context `leira`, namespace `default`.
kubectl config use-context leira

# Run identifier — ESTABLISH THIS FIRST, ONCE, and share it across ALL 3 models. If the user did
# not provide one, ASK for it before creating any resources. DNS-safe (lowercase letters, digits,
# '-'). Export it in the SAME shell you run every later step from — it is embedded in every k8s
# resource name and in the local run folder.
export ID=<identifier>

# One run folder per skill execution; each model downloads into its own subfolder under it.
export RUN_ROOT="$HOME/Downloads/sglang_lora_bench_${ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

# Benchmark opt-in. Default false → profiling only (the latency/throughput bench is SKIPPED).
# Set to true ONLY if the user asked for the bench numbers too. Do NOT ask if unspecified.
export RUN_BENCH=false

# Resolve a "branch/commit under test" to a local __bench_target branch in a repo (used by each .3
# step). SRC may be a local branch/commit name, OR a GitHub URL: a compare link
# (…/compare/base...owner:repo:branch), a branch (…/tree/branch), a commit (…/commit/sha), or a PR
# (…/pull/N). URL refs are fetched into the repo; merge_base is always vs origin/main (skill convention).
# Defined here so all three .3 steps can use it; run §0 in the shell you do the .3 builds from.
resolve_to_bench_target(){  # $1=local repo dir   $2=SRC (local ref or GitHub URL)
  local repo="$1" src="$2" u h o r ref p rest n
  git -C "$repo" fetch -q origin main
  if [[ "$src" == http*://* ]]; then
    u="${src%%\?*}"                                       # drop ?expand=1 etc.
    if [[ "$u" == */compare/* ]]; then
      h="${u##*...}"                                      # head side of  base...head
      case "$h" in
        *:*:*) o="${h%%:*}"; r="${h#*:}"; r="${r%%:*}"; ref="${h##*:}";;   # owner:repo:branch
        *:*)   o="${h%%:*}"; r="sglang";                ref="${h##*:}";;   # owner:branch
        *)     o="sgl-project"; r="sglang";             ref="$h";;         # branch on base repo
      esac
    else
      p="${u#*github.com/}"; o="${p%%/*}"; rest="${p#*/}"; r="${rest%%/*}"
      case "$u" in
        */tree/*)   ref="${u##*/tree/}";;
        */commit/*) ref="${u##*/commit/}";;
        */pull/*)   n="${u##*/pull/}"; ref="pull/${n%%/*}/head";;
        *) echo "resolve_to_bench_target: unrecognized GitHub URL: $src" >&2; return 1;;
      esac
    fi
    echo "fetching $o/$r @ $ref"
    git -C "$repo" fetch "https://github.com/$o/$r.git" "$ref" && git -C "$repo" branch -f __bench_target FETCH_HEAD
  else
    git -C "$repo" branch -f __bench_target "$src"        # local branch or commit
  fi
}
```

The `hf-token-yanbin` secret already exists in the cluster (the pod specs reference it to pull the
private `jybsuper/*` LoRA repos). No need to create it.

Each model is benchmarked against a branch/commit you specify, which **may differ per model** and may
be a local ref or a GitHub URL (§0 resolver). The git bundle is therefore built **per model** inside
each section's `.3` step (set `REPO` + `SRC` there). Both cells run on that resolved HEAD
(`HEAD_REF=__bench_target`); the `merge_base` with `origin/main` is recorded as provenance only (not
run). Two models on the same ref just build it twice — harmless.

> **Compiled-component caveat (applies to every driver):** `pip install -e python` after checkout
> re-links the Python package, which is enough when the branch only changes Python / Triton-JIT
> code (the case for the LoRA work here). If your branch modifies the compiled `sgl-kernel`
> C++/CUDA, also rebuild it on the checked-out commit:
> `cd /root/sglang/sgl-kernel && pip install -e . --no-build-isolation` (slow). Confirm against your
> branch's diff before trusting the numbers.

## Parallel execution

The three models are independent — separate pods (`sglang-qwen35-${ID}`, `sglang-qwen3vl-${ID}`,
and the two-node `mnnvl-kimi-${ID}-0` / `mnnvl-kimi-${ID}-1` pair), separate branches/bundles,
separate local subfolders. **Pods do NOT need to come up at the same time.** As soon as one model's
pod(s) are Ready and its branch is injected (that section's `.1`–`.3` steps), start that model's
driver (`.4`) — do **not** wait for the other models, and never let one model block another.

To maximize overlap, bring up all pods, then launch each driver in the background the moment its
own pod is ready (or start all three once their pods are up). All drivers share `ID` and `RUN_ROOT`
from §0 and download into their own `$RUN_ROOT/<model>` subfolder:

```bash
ID="$ID" RUN_ROOT="$RUN_ROOT" bash run_qwen35.sh  > "$RUN_ROOT/qwen35.out"  2>&1 &
ID="$ID" RUN_ROOT="$RUN_ROOT" bash run_qwen3vl.sh > "$RUN_ROOT/qwen3vl.out" 2>&1 &
ID="$ID" RUN_ROOT="$RUN_ROOT" bash run_kimi.sh    > "$RUN_ROOT/kimi.out"    2>&1 &
wait
```

Then run §4 once on `$RUN_ROOT` to build `summary.md` + profiler analysis. The result is a single
`~/Downloads/sglang_lora_bench_<id>_<timestamp>/` folder holding `qwen35/`, `qwen3vl/`, `kimi/`
subfolders, `summary.md`, and all trace files.

---

## 1. Model: `Qwen/Qwen3.5-35B-A3B-FP8` (single node, tp=ep=4)

### 1.1 Apply the pod spec — `qwen35-pod.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sglang-qwen35-${ID}
spec:
  runtimeClassName: nvidia-legacy
  tolerations:
  - key: "kubernetes.io/arch"
    operator: "Equal"
    value: "arm64"
    effect: "NoSchedule"
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: sglang
    image: lmsysorg/sglang:dev-cu13
    securityContext:
      privileged: true
    imagePullPolicy: Always
    command:
    - /bin/sh
    - -c
    - |
      set -e
      cat > /root/setup.sh <<'SETUP'
      #!/bin/bash
      # No `set -u`: the stock ~/.bashrc references $PS1 (unbound in non-interactive shells).
      set -o pipefail
      log(){ echo "[setup $(date -u +%H:%M:%S)] $*"; }

      log "ensuring numactl is present (used to keep page cache off HBM NUMA nodes)"
      command -v numactl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq numactl; } || true

      log "installing hf_xet + hf_transfer download accelerators"
      pip install --quiet "huggingface_hub[hf_xet,hf_transfer]" hf_xet hf_transfer || true

      if [ -f "$HOME/.cargo/env" ]; then
        grep -q 'cargo/env' ~/.bashrc 2>/dev/null || echo '. "$HOME/.cargo/env"' >> ~/.bashrc
        . "$HOME/.cargo/env"
      fi

      if [ -d /root/sglang/.git ]; then
        log "updating /root/sglang -> origin/main"
        cd /root/sglang
        git remote set-url origin https://github.com/sgl-project/sglang.git 2>/dev/null || true
        git fetch origin main && git reset --hard origin/main
      else
        log "cloning sglang"
        rm -rf /root/sglang && git clone https://github.com/sgl-project/sglang /root/sglang
        cd /root/sglang
      fi
      log "pip install -e /root/sglang/python"
      cd /root/sglang && pip install -e python

      mkdir -p /root/Qwen3.5-35B-A3B-FP8 /root/qwen35_35b_lora_alpha
      if [ ! -f /root/Qwen3.5-35B-A3B-FP8/config.json ]; then
        log "downloading Qwen/Qwen3.5-35B-A3B-FP8"
        numactl --membind=0,1 hf download Qwen/Qwen3.5-35B-A3B-FP8 --local-dir /root/Qwen3.5-35B-A3B-FP8 & m=$!
      fi
      if [ ! -f /root/qwen35_35b_lora_alpha/adapter_config.json ]; then
        log "downloading jybsuper/qwen35_35b_lora_alpha (private; needs HF_TOKEN)"
        numactl --membind=0,1 hf download jybsuper/qwen35_35b_lora_alpha --local-dir /root/qwen35_35b_lora_alpha & l=$!
      fi
      [ -n "${m:-}" ] && wait $m
      [ -n "${l:-}" ] && wait $l

      touch /root/.setup-done
      log "setup complete"
      SETUP
      chmod +x /root/setup.sh
      ( /root/setup.sh > /root/setup.log 2>&1 ) &
      sleep infinity
    env:
    - name: PATH
      value: "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    - name: LD_LIBRARY_PATH
      value: "/usr/local/nvidia/lib64"
    - name: NVIDIA_DISABLE_REQUIRE
      value: "true"
    - name: HF_HUB_ENABLE_HF_TRANSFER
      value: "1"
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-yanbin
          key: token
          optional: true
    - name: MALLOC_TRIM_THRESHOLD_
      value: "131072"
    resources:
      requests:
        nvidia.com/gpu: 4
        memory: "800Gi"   # request kept low so the pod schedules on smaller (880GiB) nodes too; limit below allows bursting
        cpu: "16"
        ephemeral-storage: "100Gi"
      limits:
        nvidia.com/gpu: 4
        memory: "1500Gi"
        cpu: "64"
        ephemeral-storage: "1200Gi"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
    # Persist /root/.cache (flashinfer fp4_gemm autotune, triton/deep_gemm JIT, HF modules) across pod
    # recreations so the ~17-21min cold autotune is paid once per node, not on every launch.
    - name: sglang-cache
      mountPath: /root/.cache
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
  - name: sglang-cache
    hostPath:
      path: /var/lib/sglang-cache
      type: DirectoryOrCreate
```

```bash
sed "s/\${ID}/${ID}/g" qwen35-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/sglang-qwen35-${ID} --timeout=20m
```

### 1.2 Wait for setup (HF downloads + editable install)

```bash
kubectl exec sglang-qwen35-${ID} -- bash -lc '
for i in $(seq 1 480); do [ -f /root/.setup-done ] && { echo SETUP_DONE; exit 0; }; sleep 10; done
echo SETUP_TIMEOUT; tail -n 80 /root/setup.log; exit 1'
```

### 1.3 Build + inject the branch + ghost check + clean (after setup)

Build this model's bundle locally (set `REPO` + `SRC`), then push it into the pod. Paste the
printed `MAIN_BASE` into `run_qwen35.sh`.

```bash
REPO=~/Developer/sglang        # local sglang checkout (workspace for building the bundle)
SRC=trtllm-lora                # local branch/commit, OR a GitHub URL (compare / tree / commit / pull)
resolve_to_bench_target "$REPO" "$SRC"   # §0 helper → sets __bench_target (fetches the ref if SRC is a URL)
MAIN_BASE=$(git -C "$REPO" merge-base origin/main __bench_target); echo "MAIN_BASE=$MAIN_BASE"
git -C "$REPO" bundle create /tmp/sglang-branch-qwen35.bundle __bench_target --not "${MAIN_BASE}^"

# record provenance for the summary (§4): requested ref + resolved HEAD commit + merge_base
mkdir -p "${RUN_ROOT}/qwen35"
{ echo "model=qwen35"; echo "branch=$SRC"; echo "head_commit=$(git -C "$REPO" rev-parse __bench_target)"; echo "merge_base=$MAIN_BASE"; } > "${RUN_ROOT}/qwen35/meta.env"

kubectl cp /tmp/sglang-branch-qwen35.bundle sglang-qwen35-${ID}:/root/sglang-branch.bundle
kubectl exec sglang-qwen35-${ID} -- bash -lc '
cd /root/sglang
git fetch /root/sglang-branch.bundle __bench_target:refs/heads/__bench_target
git log -1 --oneline __bench_target'

# Ghost check + clean — the ONLY check needed: ghost is page cache, so it can't fail the downloads
# or setup above; it only matters at launch (KV cache needs full HBM). Reclaim any HBM page cache now
# (pre-existing or parked during setup) — clean cache only, job-safe, never touches CUDA memory.
POD=sglang-qwen35-${ID}
kubectl exec "$POD" -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
maxused=$(kubectl exec "$POD" -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${maxused:-0}" -gt 5000 ]; then
  node=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}')
  echo "ghost ${maxused} MiB on ${node} — draining HBM page cache before launch"
  kubectl debug node/"$node" --image=busybox --profile=sysadmin -q --attach=false \
    -- chroot /host bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
  kubectl exec "$POD" -- nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
fi
```

> This **auto-cleans**: any ghost HBM page cache is reclaimed before launch so KV cache gets full HBM
> (clean cache only — job-safe, no pod restart). Every GPU should print < ~5 GB above.

### 1.4 Run profiling (+ benchmark if `RUN_BENCH`) — `run_qwen35.sh`

Set `MAIN_BASE_REF` to the SHA from §0, then run locally (drives the pod via `kubectl exec`).

```bash
#!/usr/bin/env bash
set -euo pipefail

# ===== model-specific config (Qwen3.5-35B-A3B-FP8) =====
ID="${ID:?ID is set once at the start of the skill (§0); export ID=<identifier> before running}"
POD=sglang-qwen35-${ID}
MODEL_PATH=/root/Qwen3.5-35B-A3B-FP8
LORA_PATH=/root/qwen35_35b_lora_alpha
LORA_NAME=alpha
MAX_LORA_RANK=16
MOE_LORA_BACKEND=sgl_flashinfer_trtllm   # LoRA cell only; the no-LoRA baseline uses the default MoE backend
TP_ARGS="--tp 4 --ep 4"
EXTRA_SERVER_ARGS="--mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha"
PREFILL_ARGS="--max-prefill-tokens 32768 --chunked-prefill-size 32768"
PORT=30000

# ===== run identity (from §0) =====
HEAD_REF=__bench_target
MAIN_BASE_REF=<PASTE_MAIN_BASE_SHA>

# ===== fixed workload =====
IN=2048; OUT=2048          # bench workload (full generation)
PROF_OUT=64                # profile run length: only start_step(4)+steps(12)≈16 forwards are captured,
                           # so generating 2048 tokens is wasted (catastrophic graph-off/kimi: 735s once)
BENCH_BS="16 32 64"
OUTROOT=/tmp/lora_bench
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_lora_bench_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/qwen35"

k(){ kubectl exec "${POD}" -- bash -lc "$1"; }
kill_server(){ k 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 5'; }
checkout(){ k "cd /root/sglang; git cat-file -e $1^{commit} 2>/dev/null || git fetch origin main; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"; }
wait_ready(){ k "for i in \$(seq 1 360); do curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null && { echo READY; exit 0; }; sleep 5; done; echo NOT_READY; tail -n 200 /tmp/server.log; exit 1"; }
# Incremental local download: copy OUTROOT/<subpath> from the pod into LOCAL_OUT as soon as it exists.
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

launch(){  # $1=base|lora  $2=on|off
  local lora_flags="" graph_flags=""
  [ "$1" = lora ] && lora_flags="--moe-runner-backend ${MOE_LORA_BACKEND} --enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-use-virtual-experts --lora-paths ${LORA_NAME}=${LORA_PATH}"
  [ "$2" = off ]  && graph_flags="--disable-cuda-graph"
  # Run the server in the exec FOREGROUND (`exec python …`) and background the LOCAL kubectl exec
  # (the trailing `&` is local, not in-pod). The driver returns immediately; the exec lingers for
  # the server's lifetime and ends when kill_server pkills it. An in-pod `… & echo $!` does NOT
  # return (kubectl exec keeps streaming until the server exits → driver blocks forever); and
  # setsid+</dev/null doesn't fix it either, because sglang's worker subprocesses keep the stream open.
  kubectl exec "${POD}" -- bash -lc "cd /root/sglang && exec numactl --membind=0,1 python -m sglang.launch_server --model-path ${MODEL_PATH} ${TP_ARGS} --host 0.0.0.0 --port ${PORT} --cuda-graph-max-bs 64 --trust-remote-code ${PREFILL_ARGS} ${EXTRA_SERVER_ARGS} ${graph_flags} ${lora_flags} > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  wait_ready
}

bench(){  # $1=label  $2=variant
  local lora_arg=""; [ "$2" = lora ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$2/bench"
  k "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${lora_arg} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}

profile(){  # $1=label $2=variant $3=on|off $4=bs
  local lora_arg=""; [ "$2" = lora ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$2/profile_graph_$3/bs$4"
  k "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; \
     python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
       --batch-size $4 --input-len ${IN} --output-len ${PROF_OUT} ${lora_arg} \
       --profile --profile-activities CPU GPU --profile-start-step 4 --profile-steps 12 \
       --profile-prefix qwen35_$2_graph_$3_bs$4 --profile-output-dir ${d} \
       --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; \
     find ${d} -name '*.trace.json.gz' -printf '%p %s\n' | sort"
}

run_cell(){  # $1=ref  $2=label  $3=variant
  kill_server; checkout "$1"
  kill_server; launch "$3" on            # CUDA-graph ON: (benchmark, if RUN_BENCH) + graph-on profiles
  if [ "${RUN_BENCH:-false}" = true ]; then              # latency/throughput bench is opt-in (§0)
    bench "$2" "$3";        dl "$3/bench"        # download bench results to local now
    echo "[$(date +%H:%M:%S)] $(basename "${LOCAL_OUT}") $3 BENCH done -> ${LOCAL_OUT}/$3/bench" | tee -a "${RUN_ROOT}/progress.log"
  fi
  profile "$2" "$3" on 16;  dl "$3/profile_graph_on/bs16"    # download each trace as it lands
  profile "$2" "$3" on 64;  dl "$3/profile_graph_on/bs64"
  kill_server; launch "$3" off           # CUDA-graph OFF: graph-off profile (bs16)
  profile "$2" "$3" off 16; dl "$3/profile_graph_off/bs16"
}

run_cell "${HEAD_REF}" head base    # no-LoRA, SAME commit as lora
run_cell "${HEAD_REF}" head lora    # LoRA, same commit
kill_server

# Traces (and bench results, if RUN_BENCH) were already downloaded incrementally (per profile/variant,
# see ${RUN_ROOT}/progress.log). Final sweep to catch stragglers (logs) + integrity check:
dl "."
find "${LOCAL_OUT}" -name '*.trace.json.gz' -exec gzip -t {} +
echo "[$(date +%H:%M:%S)] qwen35 DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"
```

---

## 2. Model: `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` (single node, tp=ep=4)

### 2.1 Apply the pod spec — `qwen3vl-pod.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sglang-qwen3vl-${ID}
spec:
  runtimeClassName: nvidia-legacy
  tolerations:
  - key: "kubernetes.io/arch"
    operator: "Equal"
    value: "arm64"
    effect: "NoSchedule"
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: sglang
    image: lmsysorg/sglang:dev-cu13
    securityContext:
      privileged: true
    imagePullPolicy: Always
    command:
    - /bin/sh
    - -c
    - |
      set -e
      cat > /root/setup.sh <<'SETUP'
      #!/bin/bash
      set -o pipefail
      log(){ echo "[setup $(date -u +%H:%M:%S)] $*"; }

      log "ensuring numactl is present (used to keep page cache off HBM NUMA nodes)"
      command -v numactl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq numactl; } || true

      log "installing hf_xet + hf_transfer download accelerators"
      pip install --quiet "huggingface_hub[hf_xet,hf_transfer]" hf_xet hf_transfer || true

      if [ -f "$HOME/.cargo/env" ]; then
        grep -q 'cargo/env' ~/.bashrc 2>/dev/null || echo '. "$HOME/.cargo/env"' >> ~/.bashrc
        . "$HOME/.cargo/env"
      fi

      if [ -d /root/sglang/.git ]; then
        log "updating /root/sglang -> origin/main"
        cd /root/sglang
        git remote set-url origin https://github.com/sgl-project/sglang.git 2>/dev/null || true
        git fetch origin main && git reset --hard origin/main
      else
        log "cloning sglang"
        rm -rf /root/sglang && git clone https://github.com/sgl-project/sglang /root/sglang
        cd /root/sglang
      fi
      log "pip install -e /root/sglang/python"
      cd /root/sglang && pip install -e python

      mkdir -p /root/Qwen3-VL-30B-A3B-Instruct-FP8 /root/qwen3_vl_30b_lora_alpha
      if [ ! -f /root/Qwen3-VL-30B-A3B-Instruct-FP8/config.json ]; then
        log "downloading Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
        numactl --membind=0,1 hf download Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --local-dir /root/Qwen3-VL-30B-A3B-Instruct-FP8 & m=$!
      fi
      if [ ! -f /root/qwen3_vl_30b_lora_alpha/adapter_config.json ]; then
        log "downloading jybsuper/qwen3_vl_30b_lora_alpha (private; needs HF_TOKEN)"
        numactl --membind=0,1 hf download jybsuper/qwen3_vl_30b_lora_alpha --local-dir /root/qwen3_vl_30b_lora_alpha & l=$!
      fi
      [ -n "${m:-}" ] && wait $m
      [ -n "${l:-}" ] && wait $l

      touch /root/.setup-done
      log "setup complete"
      SETUP
      chmod +x /root/setup.sh
      ( /root/setup.sh > /root/setup.log 2>&1 ) &
      sleep infinity
    env:
    - name: PATH
      value: "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    - name: LD_LIBRARY_PATH
      value: "/usr/local/nvidia/lib64"
    - name: NVIDIA_DISABLE_REQUIRE
      value: "true"
    - name: HF_HUB_ENABLE_HF_TRANSFER
      value: "1"
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-yanbin
          key: token
          optional: true
    - name: MALLOC_TRIM_THRESHOLD_
      value: "131072"
    resources:
      requests:
        nvidia.com/gpu: 4
        memory: "800Gi"   # request kept low so the pod schedules on smaller (880GiB) nodes too; limit below allows bursting
        cpu: "16"
        ephemeral-storage: "100Gi"
      limits:
        nvidia.com/gpu: 4
        memory: "1500Gi"
        cpu: "64"
        ephemeral-storage: "1200Gi"
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
    # Persist /root/.cache (flashinfer fp4_gemm autotune, triton/deep_gemm JIT, HF modules) across pod
    # recreations so the ~17-21min cold autotune is paid once per node, not on every launch.
    - name: sglang-cache
      mountPath: /root/.cache
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
  - name: sglang-cache
    hostPath:
      path: /var/lib/sglang-cache
      type: DirectoryOrCreate
```

```bash
sed "s/\${ID}/${ID}/g" qwen3vl-pod.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/sglang-qwen3vl-${ID} --timeout=20m
```

### 2.2 Wait for setup

```bash
kubectl exec sglang-qwen3vl-${ID} -- bash -lc '
for i in $(seq 1 480); do [ -f /root/.setup-done ] && { echo SETUP_DONE; exit 0; }; sleep 10; done
echo SETUP_TIMEOUT; tail -n 80 /root/setup.log; exit 1'
```

### 2.3 Build + inject the branch + ghost check + clean (after setup)

Build this model's bundle locally (set `REPO` + `SRC`), then push it into the pod. Paste the
printed `MAIN_BASE` into `run_qwen3vl.sh`.

```bash
REPO=~/Developer/sglang        # local sglang checkout (workspace for building the bundle)
SRC=trtllm-lora                # local branch/commit, OR a GitHub URL (compare / tree / commit / pull)
resolve_to_bench_target "$REPO" "$SRC"   # §0 helper → sets __bench_target (fetches the ref if SRC is a URL)
MAIN_BASE=$(git -C "$REPO" merge-base origin/main __bench_target); echo "MAIN_BASE=$MAIN_BASE"
git -C "$REPO" bundle create /tmp/sglang-branch-qwen3vl.bundle __bench_target --not "${MAIN_BASE}^"

# record provenance for the summary (§4): requested ref + resolved HEAD commit + merge_base
mkdir -p "${RUN_ROOT}/qwen3vl"
{ echo "model=qwen3vl"; echo "branch=$SRC"; echo "head_commit=$(git -C "$REPO" rev-parse __bench_target)"; echo "merge_base=$MAIN_BASE"; } > "${RUN_ROOT}/qwen3vl/meta.env"

kubectl cp /tmp/sglang-branch-qwen3vl.bundle sglang-qwen3vl-${ID}:/root/sglang-branch.bundle
kubectl exec sglang-qwen3vl-${ID} -- bash -lc '
cd /root/sglang
git fetch /root/sglang-branch.bundle __bench_target:refs/heads/__bench_target
git log -1 --oneline __bench_target'

# Ghost check + clean (only one needed — see §1.3): reclaim any HBM page cache before launch so KV
# cache gets full HBM. Clean cache only, job-safe; ghost can't fail the downloads/setup above.
POD=sglang-qwen3vl-${ID}
kubectl exec "$POD" -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
maxused=$(kubectl exec "$POD" -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
if [ "${maxused:-0}" -gt 5000 ]; then
  node=$(kubectl get pod "$POD" -o jsonpath='{.spec.nodeName}')
  echo "ghost ${maxused} MiB on ${node} — draining HBM page cache before launch"
  kubectl debug node/"$node" --image=busybox --profile=sysadmin -q --attach=false \
    -- chroot /host bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
  kubectl exec "$POD" -- nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
fi
```

> This **auto-cleans**: any ghost HBM page cache is reclaimed before launch so KV cache gets full HBM
> (clean cache only — job-safe, no pod restart). Every GPU should print < ~5 GB above.

### 2.4 Run profiling (+ benchmark if `RUN_BENCH`) — `run_qwen3vl.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# ===== model-specific config (Qwen3-VL-30B-A3B-Instruct-FP8) =====
ID="${ID:?ID is set once at the start of the skill (§0); export ID=<identifier> before running}"
POD=sglang-qwen3vl-${ID}
MODEL_PATH=/root/Qwen3-VL-30B-A3B-Instruct-FP8
LORA_PATH=/root/qwen3_vl_30b_lora_alpha
LORA_NAME=alpha
MAX_LORA_RANK=16
MOE_LORA_BACKEND=sgl_flashinfer_trtllm   # LoRA cell only; the no-LoRA baseline uses the default MoE backend
TP_ARGS="--tp 4 --ep 4"
EXTRA_SERVER_ARGS=""
PREFILL_ARGS="--max-prefill-tokens 32768 --chunked-prefill-size 32768"
PORT=30000

# ===== run identity (from §0) =====
HEAD_REF=__bench_target
MAIN_BASE_REF=<PASTE_MAIN_BASE_SHA>

# ===== fixed workload =====
IN=2048; OUT=2048          # bench workload (full generation)
PROF_OUT=64                # profile run length: only start_step(4)+steps(12)≈16 forwards are captured,
                           # so generating 2048 tokens is wasted (catastrophic graph-off/kimi: 735s once)
BENCH_BS="16 32 64"
OUTROOT=/tmp/lora_bench
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_lora_bench_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/qwen3vl"

k(){ kubectl exec "${POD}" -- bash -lc "$1"; }
kill_server(){ k 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 5'; }
checkout(){ k "cd /root/sglang; git cat-file -e $1^{commit} 2>/dev/null || git fetch origin main; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"; }
wait_ready(){ k "for i in \$(seq 1 360); do curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null && { echo READY; exit 0; }; sleep 5; done; echo NOT_READY; tail -n 200 /tmp/server.log; exit 1"; }
# Incremental local download: copy OUTROOT/<subpath> from the pod into LOCAL_OUT as soon as it exists.
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

launch(){  # $1=base|lora  $2=on|off
  local lora_flags="" graph_flags=""
  [ "$1" = lora ] && lora_flags="--moe-runner-backend ${MOE_LORA_BACKEND} --enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-use-virtual-experts --lora-paths ${LORA_NAME}=${LORA_PATH}"
  [ "$2" = off ]  && graph_flags="--disable-cuda-graph"
  # Run the server in the exec FOREGROUND (`exec python …`) and background the LOCAL kubectl exec
  # (the trailing `&` is local, not in-pod). The driver returns immediately; the exec lingers for
  # the server's lifetime and ends when kill_server pkills it. An in-pod `… & echo $!` does NOT
  # return (kubectl exec keeps streaming until the server exits → driver blocks forever); and
  # setsid+</dev/null doesn't fix it either, because sglang's worker subprocesses keep the stream open.
  kubectl exec "${POD}" -- bash -lc "cd /root/sglang && exec numactl --membind=0,1 python -m sglang.launch_server --model-path ${MODEL_PATH} ${TP_ARGS} --host 0.0.0.0 --port ${PORT} --cuda-graph-max-bs 64 --trust-remote-code ${PREFILL_ARGS} ${EXTRA_SERVER_ARGS} ${graph_flags} ${lora_flags} > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  wait_ready
}

bench(){  # $1=label  $2=variant
  local lora_arg=""; [ "$2" = lora ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$2/bench"
  k "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${lora_arg} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}

profile(){  # $1=label $2=variant $3=on|off $4=bs
  local lora_arg=""; [ "$2" = lora ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$2/profile_graph_$3/bs$4"
  k "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; \
     python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
       --batch-size $4 --input-len ${IN} --output-len ${PROF_OUT} ${lora_arg} \
       --profile --profile-activities CPU GPU --profile-start-step 4 --profile-steps 12 \
       --profile-prefix qwen3vl_$2_graph_$3_bs$4 --profile-output-dir ${d} \
       --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; \
     find ${d} -name '*.trace.json.gz' -printf '%p %s\n' | sort"
}

run_cell(){  # $1=ref  $2=label  $3=variant
  kill_server; checkout "$1"
  kill_server; launch "$3" on
  if [ "${RUN_BENCH:-false}" = true ]; then              # latency/throughput bench is opt-in (§0)
    bench "$2" "$3";        dl "$3/bench"        # download bench results to local now
    echo "[$(date +%H:%M:%S)] $(basename "${LOCAL_OUT}") $3 BENCH done -> ${LOCAL_OUT}/$3/bench" | tee -a "${RUN_ROOT}/progress.log"
  fi
  profile "$2" "$3" on 16;  dl "$3/profile_graph_on/bs16"    # download each trace as it lands
  profile "$2" "$3" on 64;  dl "$3/profile_graph_on/bs64"
  kill_server; launch "$3" off
  profile "$2" "$3" off 16; dl "$3/profile_graph_off/bs16"
}

run_cell "${HEAD_REF}" head base
run_cell "${HEAD_REF}" head lora
kill_server

# Traces (+ bench if RUN_BENCH) already downloaded incrementally. Final sweep + integrity check:
dl "."
find "${LOCAL_OUT}" -name '*.trace.json.gz' -exec gzip -t {} +
echo "[$(date +%H:%M:%S)] qwen3vl DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"
```

---

## 3. Model: `nvidia/Kimi-K2.5-NVFP4` (two nodes, tp=8, no EP)

### 3.1 Apply the pod spec — `kimi-2node.yaml`

```yaml
---
# Headless Service for inter-pod DNS.
# Head FQDN: mnnvl-kimi-${ID}-0.mnnvl-kimi-${ID}-head.<ns>.svc.cluster.local
apiVersion: v1
kind: Service
metadata:
  name: mnnvl-kimi-${ID}-head
spec:
  clusterIP: None
  publishNotReadyAddresses: true
  selector:
    app: mnnvl-kimi-${ID}
  ports:
  - name: dist-init
    port: 20000
    targetPort: 20000
---
# ComputeDomain — declares an MNNVL / IMEX fabric clique (claim-driven mode).
apiVersion: resource.nvidia.com/v1beta1
kind: ComputeDomain
metadata:
  name: mnnvl-kimi-${ID}-compute-domain
spec:
  numNodes: 0
  channel:
    resourceClaimTemplate:
      name: mnnvl-kimi-${ID}-cd-channel
---
apiVersion: v1
kind: Pod
metadata:
  name: mnnvl-kimi-${ID}-0
  labels:
    app: mnnvl-kimi-${ID}
    sglang-role: head
spec:
  hostname: mnnvl-kimi-${ID}-0
  subdomain: mnnvl-kimi-${ID}-head
  runtimeClassName: nvidia
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: mnnvl-kimi-${ID}
        topologyKey: kubernetes.io/hostname
  resourceClaims:
  - name: imex-channel
    resourceClaimTemplateName: mnnvl-kimi-${ID}-cd-channel
  tolerations:
  - key: "kubernetes.io/arch"
    operator: "Equal"
    value: "arm64"
    effect: "NoSchedule"
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: sglang
    image: lmsysorg/sglang:dev-cu13
    imagePullPolicy: Always
    command:
    - /bin/sh
    - -c
    - |
      set -e
      cat > /root/setup.sh <<'SETUP'
      #!/bin/bash
      set -o pipefail
      log(){ echo "[setup $(date -u +%H:%M:%S)] $*"; }

      log "ensuring numactl is present (used to keep page cache off HBM NUMA nodes)"
      command -v numactl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq numactl; } || true

      log "installing hf_xet + hf_transfer download accelerators"
      pip install --quiet "huggingface_hub[hf_xet,hf_transfer]" hf_xet hf_transfer || true

      if [ -f "$HOME/.cargo/env" ]; then
        grep -q 'cargo/env' ~/.bashrc 2>/dev/null || echo '. "$HOME/.cargo/env"' >> ~/.bashrc
        . "$HOME/.cargo/env"
      fi

      if [ -d /root/sglang/.git ]; then
        log "updating /root/sglang -> origin/main"
        cd /root/sglang
        git remote set-url origin https://github.com/sgl-project/sglang.git 2>/dev/null || true
        git fetch origin main && git reset --hard origin/main
      else
        log "cloning sglang"
        rm -rf /root/sglang && git clone https://github.com/sgl-project/sglang /root/sglang
        cd /root/sglang
      fi
      log "pip install -e /root/sglang/python (a few minutes)"
      cd /root/sglang && pip install -e python

      # NVFP4 base must come from a real directory (NOT the HF-cache symlink layout).
      mkdir -p /root/Kimi-K2.5-NVFP4 /root/kimi_k25_lora_alpha
      if [ ! -f /root/Kimi-K2.5-NVFP4/config.json ]; then
        log "downloading nvidia/Kimi-K2.5-NVFP4"
        numactl --membind=0,1 hf download nvidia/Kimi-K2.5-NVFP4 --local-dir /root/Kimi-K2.5-NVFP4 & m=$!
      fi
      if [ ! -f /root/kimi_k25_lora_alpha/adapter_config.json ]; then
        log "downloading jybsuper/kimi_k25_lora_alpha (private; needs HF_TOKEN)"
        numactl --membind=0,1 hf download jybsuper/kimi_k25_lora_alpha --local-dir /root/kimi_k25_lora_alpha & l=$!
      fi
      [ -n "${m:-}" ] && wait $m
      [ -n "${l:-}" ] && wait $l

      touch /root/.setup-done
      log "setup complete"
      SETUP
      chmod +x /root/setup.sh
      ( /root/setup.sh > /root/setup.log 2>&1 ) &
      sleep infinity
    ports:
    - name: dist-init
      containerPort: 20000
      protocol: TCP
    env:
    - name: PATH
      value: "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    - name: LD_LIBRARY_PATH
      value: "/usr/local/nvidia/lib64"
    - name: NVIDIA_DISABLE_REQUIRE
      value: "true"
    - name: MALLOC_TRIM_THRESHOLD_
      value: "131072"
    - name: HF_HUB_ENABLE_HF_TRANSFER
      value: "1"
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-yanbin
          key: token
          optional: true
    resources:
      requests:
        nvidia.com/gpu: 4
        memory: "800Gi"   # request kept low so the pod schedules on smaller (880GiB) nodes too; limit below allows bursting
        cpu: "16"
        ephemeral-storage: "100Gi"
      limits:
        nvidia.com/gpu: 4
        memory: "1800Gi"
        cpu: "64"
        ephemeral-storage: "1200Gi"
      claims:
      - name: imex-channel
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
    # Persist /root/.cache (flashinfer fp4_gemm autotune, triton/deep_gemm JIT, HF modules) across pod
    # recreations so the ~17-21min cold autotune is paid once per node, not on every launch.
    - name: sglang-cache
      mountPath: /root/.cache
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
  - name: sglang-cache
    hostPath:
      path: /var/lib/sglang-cache
      type: DirectoryOrCreate
---
apiVersion: v1
kind: Pod
metadata:
  name: mnnvl-kimi-${ID}-1
  labels:
    app: mnnvl-kimi-${ID}
    sglang-role: worker
spec:
  hostname: mnnvl-kimi-${ID}-1
  subdomain: mnnvl-kimi-${ID}-head
  runtimeClassName: nvidia
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: mnnvl-kimi-${ID}
        topologyKey: kubernetes.io/hostname
  resourceClaims:
  - name: imex-channel
    resourceClaimTemplateName: mnnvl-kimi-${ID}-cd-channel
  tolerations:
  - key: "kubernetes.io/arch"
    operator: "Equal"
    value: "arm64"
    effect: "NoSchedule"
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
  containers:
  - name: sglang
    image: lmsysorg/sglang:dev-cu13
    imagePullPolicy: Always
    command:
    - /bin/sh
    - -c
    - |
      set -e
      cat > /root/setup.sh <<'SETUP'
      #!/bin/bash
      set -o pipefail
      log(){ echo "[setup $(date -u +%H:%M:%S)] $*"; }

      log "ensuring numactl is present (used to keep page cache off HBM NUMA nodes)"
      command -v numactl >/dev/null 2>&1 || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq numactl; } || true

      log "installing hf_xet + hf_transfer download accelerators"
      pip install --quiet "huggingface_hub[hf_xet,hf_transfer]" hf_xet hf_transfer || true

      if [ -f "$HOME/.cargo/env" ]; then
        grep -q 'cargo/env' ~/.bashrc 2>/dev/null || echo '. "$HOME/.cargo/env"' >> ~/.bashrc
        . "$HOME/.cargo/env"
      fi

      if [ -d /root/sglang/.git ]; then
        log "updating /root/sglang -> origin/main"
        cd /root/sglang
        git remote set-url origin https://github.com/sgl-project/sglang.git 2>/dev/null || true
        git fetch origin main && git reset --hard origin/main
      else
        log "cloning sglang"
        rm -rf /root/sglang && git clone https://github.com/sgl-project/sglang /root/sglang
        cd /root/sglang
      fi
      log "pip install -e /root/sglang/python (a few minutes)"
      cd /root/sglang && pip install -e python

      mkdir -p /root/Kimi-K2.5-NVFP4 /root/kimi_k25_lora_alpha
      if [ ! -f /root/Kimi-K2.5-NVFP4/config.json ]; then
        log "downloading nvidia/Kimi-K2.5-NVFP4"
        numactl --membind=0,1 hf download nvidia/Kimi-K2.5-NVFP4 --local-dir /root/Kimi-K2.5-NVFP4 & m=$!
      fi
      if [ ! -f /root/kimi_k25_lora_alpha/adapter_config.json ]; then
        log "downloading jybsuper/kimi_k25_lora_alpha (private; needs HF_TOKEN)"
        numactl --membind=0,1 hf download jybsuper/kimi_k25_lora_alpha --local-dir /root/kimi_k25_lora_alpha & l=$!
      fi
      [ -n "${m:-}" ] && wait $m
      [ -n "${l:-}" ] && wait $l

      touch /root/.setup-done
      log "setup complete"
      SETUP
      chmod +x /root/setup.sh
      ( /root/setup.sh > /root/setup.log 2>&1 ) &
      sleep infinity
    ports:
    - name: dist-init
      containerPort: 20000
      protocol: TCP
    env:
    - name: PATH
      value: "/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    - name: LD_LIBRARY_PATH
      value: "/usr/local/nvidia/lib64"
    - name: NVIDIA_DISABLE_REQUIRE
      value: "true"
    - name: MALLOC_TRIM_THRESHOLD_
      value: "131072"
    - name: HF_HUB_ENABLE_HF_TRANSFER
      value: "1"
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-yanbin
          key: token
          optional: true
    resources:
      requests:
        nvidia.com/gpu: 4
        memory: "800Gi"   # request kept low so the pod schedules on smaller (880GiB) nodes too; limit below allows bursting
        cpu: "16"
        ephemeral-storage: "100Gi"
      limits:
        nvidia.com/gpu: 4
        memory: "1800Gi"
        cpu: "64"
        ephemeral-storage: "1200Gi"
      claims:
      - name: imex-channel
    volumeMounts:
    - name: shm
      mountPath: /dev/shm
    # Persist /root/.cache (flashinfer fp4_gemm autotune, triton/deep_gemm JIT, HF modules) across pod
    # recreations so the ~17-21min cold autotune is paid once per node, not on every launch.
    - name: sglang-cache
      mountPath: /root/.cache
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
  - name: sglang-cache
    hostPath:
      path: /var/lib/sglang-cache
      type: DirectoryOrCreate
```

```bash
sed "s/\${ID}/${ID}/g" kimi-2node.yaml | kubectl apply -f -
kubectl wait --for=condition=Ready pod/mnnvl-kimi-${ID}-0 pod/mnnvl-kimi-${ID}-1 --timeout=25m
```

### 3.2 Wait for setup (both pods)

```bash
for P in mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1; do
  kubectl exec "${P}" -- bash -lc '
  for i in $(seq 1 600); do [ -f /root/.setup-done ] && { echo "'${P}' SETUP_DONE"; exit 0; }; sleep 10; done
  echo "'${P}' SETUP_TIMEOUT"; tail -n 80 /root/setup.log; exit 1'
done
```

### 3.3 Build + inject the branch + ghost check + clean (both pods, after setup)

Build this model's bundle locally (set `REPO` + `SRC`), then push it to **both** pods. Paste the
printed `MAIN_BASE` into `run_kimi.sh`.

```bash
REPO=~/Desktop/sglang                 # local sglang checkout (workspace for building the bundle)
SRC=lora-nvfp4-gb200-cookbook         # local branch/commit, OR a GitHub URL (compare / tree / commit / pull)
resolve_to_bench_target "$REPO" "$SRC"   # §0 helper → sets __bench_target (fetches the ref if SRC is a URL)
MAIN_BASE=$(git -C "$REPO" merge-base origin/main __bench_target); echo "MAIN_BASE=$MAIN_BASE"
git -C "$REPO" bundle create /tmp/sglang-branch-kimi.bundle __bench_target --not "${MAIN_BASE}^"

# record provenance for the summary (§4): requested ref + resolved HEAD commit + merge_base
mkdir -p "${RUN_ROOT}/kimi"
{ echo "model=kimi"; echo "branch=$SRC"; echo "head_commit=$(git -C "$REPO" rev-parse __bench_target)"; echo "merge_base=$MAIN_BASE"; } > "${RUN_ROOT}/kimi/meta.env"

for P in mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1; do
  kubectl cp /tmp/sglang-branch-kimi.bundle "${P}:/root/sglang-branch.bundle"
  kubectl exec "${P}" -- bash -lc '
  cd /root/sglang
  git fetch /root/sglang-branch.bundle __bench_target:refs/heads/__bench_target
  git log -1 --oneline __bench_target'
  echo "== ${P} ghost-check =="
  kubectl exec "${P}" -- nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
  maxused=$(kubectl exec "${P}" -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "${maxused:-0}" -gt 5000 ]; then
    node=$(kubectl get pod "${P}" -o jsonpath='{.spec.nodeName}')
    echo "ghost ${maxused} MiB on ${node} — draining HBM page cache before launch"
    kubectl debug node/"$node" --image=busybox --profile=sysadmin -q --attach=false \
      -- chroot /host bash -c 'sync; echo 1 > /proc/sys/vm/drop_caches'
    kubectl exec "${P}" -- nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
  fi
done
```

> This **auto-cleans** per node: any ghost HBM page cache is reclaimed before launch so KV cache gets
> full HBM on both nodes (clean cache only — job-safe, no pod restart). Every GPU on both pods should
> print < ~5 GB above.

### 3.4 Run profiling (+ benchmark if `RUN_BENCH`) — `run_kimi.sh` (2-node)

Launches on the worker first, then the head; benchmarks/profiles from the head against
`localhost`. Set `MAIN_BASE_REF` from §0.

```bash
#!/usr/bin/env bash
set -euo pipefail

# ===== config (Kimi-K2.5-NVFP4, 2 nodes) =====
ID="${ID:?ID is set once at the start of the skill (§0); export ID=<identifier> before running}"
HEAD_POD=mnnvl-kimi-${ID}-0
WORKER_POD=mnnvl-kimi-${ID}-1
DIST_INIT=mnnvl-kimi-${ID}-0.mnnvl-kimi-${ID}-head:20000
MODEL_PATH=/root/Kimi-K2.5-NVFP4
LORA_PATH=/root/kimi_k25_lora_alpha
LORA_NAME=alpha
MAX_LORA_RANK=32
MOE_LORA_BACKEND=flashinfer_cutlass   # LoRA cell only; the no-LoRA baseline uses the default MoE backend
PORT=30000

# ===== run identity (from §0) =====
HEAD_REF=__bench_target
MAIN_BASE_REF=<PASTE_MAIN_BASE_SHA>

# ===== fixed workload =====
IN=2048; OUT=2048          # bench workload (full generation)
PROF_OUT=64                # profile run length: only start_step(4)+steps(12)≈16 forwards are captured,
                           # so generating 2048 tokens is wasted (catastrophic graph-off/kimi: 735s once)
BENCH_BS="16 32 64"
OUTROOT=/tmp/lora_bench
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_lora_bench_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/kimi"

# Common server args (no EP). The LoRA cell adds --moe-runner-backend ${MOE_LORA_BACKEND}
# (flashinfer_cutlass) via its lora flags; the no-LoRA baseline runs on the default MoE backend.
COMMON="--model-path ${MODEL_PATH} --tp 8 --nnodes 2 --dist-init-addr ${DIST_INIT} \
--host 0.0.0.0 --port ${PORT} --quantization modelopt_fp4 --mem-fraction-static 0.88 \
--cuda-graph-max-bs 64 --trust-remote-code \
--max-prefill-tokens 40960 --chunked-prefill-size 40960"
ENVS="NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false"

kh(){ kubectl exec "${HEAD_POD}"   -- bash -lc "$1"; }
kw(){ kubectl exec "${WORKER_POD}" -- bash -lc "$1"; }
both(){ kw "$1"; kh "$1"; }

kill_all(){ both 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 8'; }
checkout(){ both "cd /root/sglang; git cat-file -e $1^{commit} 2>/dev/null || git fetch origin main; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1"; kh "cd /root/sglang && git --no-pager log -1 --oneline"; }

# One-time per node: pre-warm the HF dynamic-module cache so 4 ranks/node don't race copying the
# model's trust_remote_code *.py into ~/.cache/huggingface/modules at first launch.
prewarm(){
  for P in "${WORKER_POD}" "${HEAD_POD}"; do
    kubectl exec "${P}" -- bash -lc 'python3 -c "from transformers import AutoConfig, AutoTokenizer, AutoProcessor; m=\"/root/Kimi-K2.5-NVFP4\"; AutoConfig.from_pretrained(m, trust_remote_code=True); AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoProcessor.from_pretrained(m, trust_remote_code=True); print(\"prewarmed\")"'
  done
}

# Background the LOCAL kubectl exec (trailing `&` is local); server runs in the exec foreground.
# An in-pod `… & echo $!` (or setsid+</dev/null) hangs the exec — sglang's worker subprocesses
# keep its stream open, so kubectl exec never returns and the driver blocks.
start_rank(){  # $1=pod  $2=node-rank  $3=extra-flags
  kubectl exec "$1" -- bash -lc "cd /root/sglang && ${ENVS} exec numactl --membind=0,1 python3 -m sglang.launch_server ${COMMON} --node-rank $2 $3 > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  echo "started-rank$2 on $1"
}
launch(){  # $1=base|lora  $2=on|off
  local lora_flags="" graph_flags=""
  [ "$1" = lora ] && lora_flags="--moe-runner-backend ${MOE_LORA_BACKEND} --enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-use-virtual-experts --lora-paths ${LORA_NAME}=${LORA_PATH}"
  [ "$2" = off ]  && graph_flags="--disable-cuda-graph"
  local flags="${graph_flags} ${lora_flags}"
  kw "for i in \$(seq 1 60); do getent hosts ${DIST_INIT%%:*} >/dev/null 2>&1 && break; sleep 2; done"   # wait for rendezvous DNS
  start_rank "$WORKER_POD" 1 "$flags"
  start_rank "$HEAD_POD"   0 "$flags"
  sleep 12
  for spec in "${WORKER_POD}:1" "${HEAD_POD}:0"; do
    local pod="${spec%%:*}" rank="${spec##*:}"
    kubectl exec "$pod" -- bash -lc 'pgrep -f "[s]glang.launch_server" >/dev/null 2>&1' \
      || { echo "rank${rank} on ${pod} not running — retry"; start_rank "$pod" "$rank" "$flags"; }
  done
  kh "for i in \$(seq 1 600); do curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null && { echo READY; exit 0; }; sleep 5; done; echo NOT_READY; tail -n 80 /tmp/server.log; exit 1"
}

bench(){  # $1=label  $2=variant   (driven from head)
  local lora_arg=""; [ "$2" = lora ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$2/bench"
  kh "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${lora_arg} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}

profile(){  # $1=label $2=variant $3=on|off $4=bs   (traces land on the head's ranks 0-3)
  local lora_arg=""; [ "$2" = lora ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$2/profile_graph_$3/bs$4"
  kh "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; \
     python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
       --batch-size $4 --input-len ${IN} --output-len ${PROF_OUT} ${lora_arg} \
       --profile --profile-activities CPU GPU --profile-start-step 4 --profile-steps 12 \
       --profile-prefix kimi_$2_graph_$3_bs$4 --profile-output-dir ${d} \
       --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; \
     find ${d} -name '*.trace.json.gz' -printf '%p %s\n' | sort"
}

# Incremental local download from the head pod (ranks 0-3 traces; enough for single-rank triage).
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${HEAD_POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

run_cell(){  # $1=ref  $2=label  $3=variant
  kill_all; checkout "$1"
  kill_all; launch "$3" on
  if [ "${RUN_BENCH:-false}" = true ]; then              # latency/throughput bench is opt-in (§0)
    bench "$2" "$3";        dl "$3/bench"        # download bench results to local now
    echo "[$(date +%H:%M:%S)] kimi $3 BENCH done -> ${LOCAL_OUT}/$3/bench" | tee -a "${RUN_ROOT}/progress.log"
  fi
  profile "$2" "$3" on 16;  dl "$3/profile_graph_on/bs16"    # download each trace as it lands
  profile "$2" "$3" on 64;  dl "$3/profile_graph_on/bs64"
  kill_all; launch "$3" off
  profile "$2" "$3" off 16; dl "$3/profile_graph_off/bs16"
}

prewarm
run_cell "${HEAD_REF}" head base
run_cell "${HEAD_REF}" head lora
kill_all

# Traces (+ bench if RUN_BENCH) already downloaded incrementally. Final sweep + integrity check:
dl "."
find "${LOCAL_OUT}" -name '*.trace.json.gz' -exec gzip -t {} +
echo "[$(date +%H:%M:%S)] kimi DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"
```

---

## 4. Summary + profiling analysis (run once, locally, after all 3 models)

The drivers **download incrementally** — each profile's trace (and, when `RUN_BENCH=true`, each
variant's bench results) is pulled to local the moment it's produced, and a line is appended to
`${RUN_ROOT}/progress.log` as each model's no-LoRA / LoRA cell finishes (so you can watch progress
and start reading results before the run ends). By the time all three drivers finish, `${RUN_ROOT}`
already holds everything locally; §4 just aggregates it — no large end-of-run copy.

`~/Downloads/sglang_lora_bench_<id>_<timestamp>/` contains `qwen35/`, `qwen3vl/`, `kimi/`, each with
`base/...` and `lora/...` (**both cells on the same commit**) under `profile_graph_on/`,
`profile_graph_off/` (and `bench/` only when `RUN_BENCH=true`), plus `meta.env` (provenance) — and a
top-level `progress.log` + `env.txt`.

### 4.0 Capture the environment (for reproducibility)

Library versions affect perf reproducibility (e.g. the bundled `deep_gemm`). Capture from any one
running pod before teardown:

```bash
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder}"; POD=sglang-qwen35-${ID}
{ echo "image=lmsysorg/sglang:dev-cu13"
  echo "cluster=leira (k8s, namespace default)"
  [ "${RUN_BENCH:-false}" = true ] && echo "bench=bench_one_batch_server; batch_size={16,32,64}; input_len=output_len=2048; cuda_graph=on"
  echo "profile=CPU+GPU, no stage split, profile_steps=12; graph_on bs={16,64} + graph_off bs={16}"
  kubectl --context leira exec "$POD" -- bash -lc 'echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"; echo "gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1)"; python3 -c "import torch;print(\"torch=\"+torch.__version__);print(\"cuda=\"+str(torch.version.cuda))"; for m in deep_gemm flashinfer sgl_kernel transformers triton; do python3 -c "import $m;print(\"$m=\"+$m.__version__)" 2>/dev/null || echo "$m=n/a"; done'
} > "${ROOT}/env.txt"; cat "${ROOT}/env.txt"
```

> Per-model server launch flags are documented in each model's launch command (§1/§2/§3). If you
> want them inside `summary.md`, append them to that model's `meta.env` as `server_args=…` /
> `lora_args=…` and the aggregator will print them.

### 4.1 Build the summary table (provenance + environment + perf if benched)

The aggregator emits the recorded commit (provenance) and environment for every run. The
**Performance** section is included only when the benchmark ran (`RUN_BENCH=true`, detected by the
presence of `bench/*.jsonl`); a profiling-only run shows a "benchmark skipped" note instead. When
present, both cells share one commit so the table is **pure LoRA overhead**.

```bash
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder, e.g. ~/Downloads/sglang_lora_bench_<id>_<timestamp>}"
python3 - "$ROOT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).expanduser()
MODELS = ["qwen35", "qwen3vl", "kimi"]; BS = [16, 32, 64]
def last_row(p):
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]; return rows[-1] if rows else None
def env(p):
    d = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line: k, v = line.split("=", 1); d[k] = v
    return d
E = env(root/"env.txt"); M = {m: env(root/m/"meta.env") for m in MODELS}
L = ["# SGLang LoRA vs Base — Performance Summary", ""]
L += [f"Run `{root.name}`. Both no-LoRA(base) and LoRA run on the SAME commit per model "
      "(pure LoRA overhead). input_len=output_len=2048, CUDA graph on.", ""]
L += ["## Provenance (commit used for BOTH base and lora)", "",
      "| model | branch | HEAD commit | merge_base |", "|---|---|---|---|"]
for m in MODELS:
    d = M[m]; L.append(f"| {m} | {d.get('branch','?')} | `{d.get('head_commit','?')[:12]}` | `{d.get('merge_base','?')[:12]}` |")
L += ["", "## Environment", "", "| key | value |", "|---|---|"]
for k in ["image","gpu","gpu_driver","cuda","torch","deep_gemm","flashinfer","sgl_kernel","transformers","triton","cluster","bench","profile"]:
    if E.get(k): L.append(f"| {k} | {E[k]} |")
L += ["", "> sglang itself is the editable install at each model's HEAD commit (see Provenance); the libs above are image-level.", ""]
# optional per-model server config if recorded in meta.env
if any(M[m].get("server_args") for m in MODELS):
    L += ["## Server configuration", ""]
    for m in MODELS:
        d = M[m]
        L += [f"**{m}** — base_model `{d.get('base_model','?')}`, lora_repo `{d.get('lora_repo','?')}`, {d.get('parallelism','?')}",
              "```", f"server args (both cells): {d.get('server_args','?')}", f"lora cell adds:           {d.get('lora_args','?')}", "```", ""]
# Performance table only when the benchmark ran (RUN_BENCH=true). Profiling-only runs have no bench jsonl.
bench_models = [m for m in MODELS if any((root/m/v/"bench"/f"bs{bs}.jsonl").exists() for v in ("base","lora") for bs in BS)]
if bench_models:
    L += ["## Performance", "",
          "| model | bs | base lat (s) | lora lat (s) | lora/base lat | base tok/s | lora tok/s | lora % of base |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for m in bench_models:
        for bs in BS:
            b = last_row(root/m/"base"/"bench"/f"bs{bs}.jsonl"); l = last_row(root/m/"lora"/"bench"/f"bs{bs}.jsonl")
            if not b or not l:
                L.append(f"| {m} | {bs} | {'MISSING' if not b else 'ok'} | {'MISSING' if not l else 'ok'} | | | | |"); continue
            bl, ll = b.get("latency"), l.get("latency"); bt, lt = b.get("output_throughput"), l.get("output_throughput")
            ratio = (ll/bl) if (bl and ll) else None; pct = (lt/bt*100) if (bt and lt) else None
            L.append(f"| {m} | {bs} | {bl:.3f} | {ll:.3f} | {ratio:.3f}x | {bt:.1f} | {lt:.1f} | {pct:.1f}% |"
                     if None not in (bl, ll, bt, lt, ratio, pct) else f"| {m} | {bs} | {bl} | {ll} | {ratio} | {bt} | {lt} | {pct} |")
else:
    L += ["## Performance", "", "_Benchmark skipped (`RUN_BENCH=false`) — profiling only. No latency/throughput table._", ""]
out = root/"summary.md"; out.write_text("\n".join(L) + "\n"); print(out); print("\n".join(L))
PY
```

### 4.2 Profiler analysis (use the `llm-torch-profiler-analysis` skill)

Invoke the **`llm-torch-profiler-analysis`** skill once per cell using **two-trace triage** — the
CUDA-graph-OFF bs16 trace as the mapping pass (readable kernel→source) and the CUDA-graph-ON bs16
trace as the formal pass (real serving behavior) — to get the three tables (kernel /
overlap-opportunity / fuse-pattern):

```bash
# Resolve the profiler skill: prefer the user-level skills dir; if not found, fetch it from GitHub.
SKILL="$HOME/.claude/skills/llm-torch-profiler-analysis"
if [ ! -f "$SKILL/scripts/analyze_llm_torch_profile.py" ]; then
  git clone --depth 1 https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS /tmp/ai-skills 2>/dev/null \
    || git -C /tmp/ai-skills pull --ff-only
  SKILL=/tmp/ai-skills/skills/llm-torch-profiler-analysis
fi
echo "using profiler skill: $SKILL"
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder the drivers used, e.g. ~/Downloads/sglang_lora_bench_<id>_<timestamp>}"

for m in qwen35 qwen3vl kimi; do
  for cell in "base" "lora"; do
    tag="${cell//\//_}"
    echo "=== ${m} ${cell} (two-trace: graph-off bs16 -> graph-on bs16) ==="
    python3 "${SKILL}/scripts/analyze_llm_torch_profile.py" \
      --mapping-input "${ROOT}/${m}/${cell}/profile_graph_off/bs16" \
      --formal-input  "${ROOT}/${m}/${cell}/profile_graph_on/bs16" \
      | tee "${ROOT}/${m}/analysis_${tag}.txt"
    # batch-scaling view: single-trace triage on graph-on bs64
    python3 "${SKILL}/scripts/analyze_llm_torch_profile.py" \
      --input "${ROOT}/${m}/${cell}/profile_graph_on/bs64" \
      | tee "${ROOT}/${m}/analysis_${tag}_bs64.txt"
  done
done
```

When writing the profiling section of `summary.md`, for each model compare the **`lora`
kernel table against the `base` (no-LoRA, same commit) kernel table** at the same batch size: the extra rows
that appear only under LoRA (e.g. LoRA shrink/expand GEMMs and the virtual-expert gather/scatter
kernels) are the LoRA cost, and they should line up with the throughput gap from §4.1.

> You can also invoke the skill directly — `Skill(llm-torch-profiler-analysis)` with the
> `--mapping-input` / `--formal-input` pair above — to get the three tables plus a one-line "what
> dominates" summary per cell. **If the skill is not available locally or registered**, read it from
> GitHub: `llm-torch-profiler-analysis` lives in
> [BBuf/AI-Infra-Auto-Driven-SKILLS](https://github.com/BBuf/AI-Infra-Auto-Driven-SKILLS) under
> `skills/llm-torch-profiler-analysis/` (its `SKILL.md` documents usage; `scripts/` has the
> analyzer). The clone fallback in the block above fetches exactly that.

### 4.3 Final deliverable

`~/Downloads/sglang_lora_bench_<date>/` holds everything in one folder:

- `summary.md` — provenance + environment + your profiling write-up (§4.2), plus the performance
  table when `RUN_BENCH=true` (§4.1)
- `progress.log` — early per-variant "DONE" log (and "BENCH done" when benched); `env.txt` — captured environment
- `qwen35/`, `qwen3vl/`, `kimi/` — `meta.env` (provenance + server config) + `profile_graph_on/`
  and `profile_graph_off/` trace dirs (`*.trace.json.gz`), `analysis_*.txt` profiler reports, and
  per-cell `bench/` JSONL+logs when `RUN_BENCH=true` (all pulled **incrementally** during the runs)

```bash
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder the drivers used, e.g. ~/Downloads/sglang_lora_bench_<id>_<timestamp>}"
find "${ROOT}" -maxdepth 4 -type d | sort
du -sh "${ROOT}"
```

## Operational notes (bad nodes, scheduling, commit/image compatibility)

Hard-won fixes that keep the runs valid without perf-affecting workarounds:

- **Ghost GPU memory is page cache, not a leak.** GB200 exposes HBM as cpu-less NUMA nodes, so big
  file I/O (model downloads; anything under `numactl --interleave=all`) can land clean page cache on
  HBM, where `nvidia-smi` shows it as "used" even with no CUDA process — and it survives `kubectl
  delete pod`. We **prevent** it by running downloads + launch under `numactl --membind=0,1` (host page
  cache stays on the Grace CPU nodes `0,1`) and **clean** any prior tenant's ghost once per model — in
  the post-setup `.3` step, right before launch — with a host `drop_caches` (reclaims clean cache only
  — never a job's CUDA memory, no pod restart; ghost can't affect the CPU-only download/setup, only the
  launch's KV-cache HBM). `--membind` is applied identically to the base and LoRA cells so it can't bias the
  comparison; confirm the CPU NUMA node IDs are `0,1` per machine with `numactl -H` (HBM nodes are the
  cpu-less ~188 GB ones). No node survey/label/pin needed — pods run on any free GPU node.
- **Right-size requests for smaller nodes.** `memory` / `ephemeral-storage` requests are *scheduling
  reservations*, not perf knobs — lower them (not the GPU count) if a pod is `Pending` with
  "Insufficient memory/ephemeral-storage" on a smaller node. The real writable disk
  (overlay) is typically far larger (e.g. ~1.8 TB) than the k8s `ephemeral-storage` allocatable
  (~110 GiB), so big model downloads still fit.
- **The commit under test must run on the current image.** Both cells use one commit, so if it
  doesn't start, the whole model is blocked. A common skew: the image's `deep_gemm` lacks an API an
  older commit calls unguarded — we hit `AttributeError: module 'deep_gemm' has no attribute
  'get_compile_mode'` during FP8 JIT warmup (newer commits guard it with `hasattr`). Resolve by
  testing a commit/branch that includes the compat guard, or pin a matching image — **not** by
  disabling the FP8 / deep_gemm path (that changes perf). The crash is in `/tmp/server.log`.

## 5. Cleanup (only after §4 — summary built and traces downloaded)

Tear down all pods once every benchmark + profile is done and the summary and trace files are
safely in `~/Downloads`:

```bash
kubectl delete pod sglang-qwen35-${ID} --ignore-not-found
kubectl delete pod sglang-qwen3vl-${ID} --ignore-not-found
sed "s/\${ID}/${ID}/g" kimi-2node.yaml | kubectl delete -f - --ignore-not-found   # pods + Service + ComputeDomain
# Belt-and-suspenders (if the YAML file isn't handy) — also remove the Service + ComputeDomain:
kubectl delete pod mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1 --ignore-not-found
kubectl delete service mnnvl-kimi-${ID}-head --ignore-not-found
kubectl delete computedomain mnnvl-kimi-${ID}-compute-domain --ignore-not-found
```

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
