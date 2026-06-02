---
name: sglang-base-variant-regression
description: >-
  Reproducible accuracy + performance regression check between a base (control) and a variant
  serving config for three models — Qwen/Qwen3.5-35B-A3B-FP8, Qwen/Qwen3-VL-30B-A3B-Instruct-FP8,
  and nvidia/Kimi-K2.5-NVFP4. Each of the two cells has its OWN commit/branch, LoRA-or-not, and
  optional extra server args, so you can A/B any change (a PR, a kernel/backend swap, a LoRA toggle).
  Accuracy compares per-token logprobs over each model's compare_sample_train_data.pt (shipped in the
  private LoRA adapter repo); performance compares bench_one_batch_server latency/throughput. Covers
  k8s pod bring-up, ghost-GPU handling, branch/commit injection, and one downloadable summary with an
  acc-diff table and a perf-delta table. The three models live on independent pods and run in parallel.
---

# SGLang Base vs Variant — Accuracy & Performance Regression

Check whether a **variant** serving config regresses **accuracy** (per-token logprobs) or
**performance** (latency/throughput) relative to a **base** (control) config, for three models, and
produce one downloadable summary.

> **Scope:** accuracy regression (logprob diff) **and** performance regression (latency/throughput),
> both always run. No profiling here (use the perf-benchmark skill for kernel traces). The two cells
> are a general **base vs variant** A/B — each has its own commit, LoRA setting, and extra args.

> **Run identifier (`ID`) — establish this first.** Every Kubernetes name below (pods, Service,
> ComputeDomain, resource-claim template, labels, the dist-init address) embeds a short `ID`, so
> multiple people — or multiple runs by one person — can use this skill on the **same cluster in
> parallel** without name collisions. **If the user did not provide an identifier, ask for one**
> before creating any resources. Use a DNS-safe value (lowercase letters, digits, `-`), e.g. `yb`,
> `alice`, `run1`. It is also embedded in the local results folder name.

> **Base (control) vs Variant — define BOTH, per model, up front.** Each model runs two cells you must
> fully specify. For **each** cell give:
> - **commit/branch** — a **local branch/commit** in a local checkout, **or** a **GitHub URL** (compare
>   link `…/compare/main...owner:repo:branch`, branch `…/tree/branch`, commit `…/commit/sha`, or PR
>   `…/pull/N`); URL refs are fetched automatically (resolver in §0). Base and variant **may be the
>   same commit** (e.g. testing a flag/backend) or different (e.g. main vs a PR).
> - **LoRA on/off** — whether that cell serves with `--enable-lora … --lora-paths` or plain base.
> - **extra server args (optional)** — anything else that differs, e.g. `--moe-runner-backend …`,
>   `--attention-backend …`. Applied on top of the model's standard args.
>
> **If the user did not give both cells (commit + LoRA + extras) for each of the three models, ASK
> before starting.** Set `BASE_SRC`/`VARIANT_SRC` in the `.3` step and `BASE_LORA`/`BASE_EXTRA` /
> `VARIANT_LORA`/`VARIANT_EXTRA` in the `.4` run script. Do not assume defaults.

> **When LoRA is on, you MUST pass `--moe-runner-backend` in `*_EXTRA`.** The `lora_flags`
> formula in each `.4` run script hardcodes `--lora-use-virtual-experts`, but the model's
> default MoE backend (`flashinfer_trtllm` for Qwen3.5 / Qwen3-VL FP8) does **not** support
> virtual-experts LoRA — the server crashes at startup with
> `NotImplementedError: LoRA MoE not supported for backend MoeRunnerBackend.FLASHINFER_TRTLLM`.
> So any LoRA-on cell needs one of:
> - `--moe-runner-backend triton` — the older LoRA path. Use for a "stock LoRA baseline."
> - `--moe-runner-backend sgl_flashinfer_trtllm` — the trtllm-lora path. Use for the new path.
>
> Example for an `acc` regression of stock-LoRA vs trtllm-LoRA-with-two-stream on the same commit:
> `BASE_LORA=on  BASE_EXTRA="--moe-runner-backend triton"`
> `VARIANT_LORA=on VARIANT_EXTRA="--moe-runner-backend sgl_flashinfer_trtllm"` (+ `SGLANG_LORA_TWO_STREAM=1` env)
>
> A `*_LORA=off` cell does NOT need this — it picks any backend safely.

> **Interpreting the diff.** The acc logprob-diff is a true **regression** signal only when base and
> variant are *expected to be numerically equivalent* (a refactor, a kernel/backend swap that should
> match). If the cells intentionally differ (e.g. base=no-LoRA vs variant=LoRA), a large diff is the
> *intended* effect, not a regression — the summary reports the numbers; you judge. Set the tolerances
> in §0 (`ACC_TOL`, `PERF_TOL`) to control the PASS/REGRESS flag.

## Cells (per model)

Two fully user-specified cells. Each has its own commit, LoRA setting, and extra args:

| Cell | Commit | LoRA | Extra args |
|---|---|---|---|
| **base** (control) | `BASE_SRC` (local ref or URL) | `BASE_LORA` (on/off) | `BASE_EXTRA` |
| **variant** (candidate) | `VARIANT_SRC` (local ref or URL) | `VARIANT_LORA` (on/off) | `VARIANT_EXTRA` |

The headline numbers are the **acc logprob diff** (variant vs base, over the same data) and the
**perf delta** (variant latency/throughput vs base). Each cell's commit + LoRA + extra args is
recorded in the summary for provenance.

## Workload (every cell)

| Workload | Detail | input/output | CUDA graph |
|---|---|---|---|
| Accuracy capture (`/generate`, `return_logprob`) | per-token logprobs over `compare_sample_train_data.pt` | full sample seq | enabled |
| Benchmark (`bench_one_batch_server`) | batch sizes 16, 32, 64 | 2048 / 2048 | enabled |

2 server launches per model (base + variant); each launch serves both the acc capture and the bench.

## Per-model summary

| Model | Pod topology | Parallelism | LoRA adapter + acc data (private HF repo) | Max LoRA rank |
|---|---|---|---|---|
| `Qwen/Qwen3.5-35B-A3B-FP8` | single node, 4 GPU | `--tp 4 --ep 4` | `jybsuper/qwen35_35b_lora_alpha` | 16 |
| `Qwen/Qwen3-VL-30B-A3B-Instruct-FP8` | single node, 4 GPU | `--tp 4 --ep 4` | `jybsuper/qwen3_vl_30b_lora_alpha` | 16 |
| `nvidia/Kimi-K2.5-NVFP4` | **two nodes**, 4+4 GPU (MNNVL) | `--tp 8` (**no EP**) | `jybsuper/kimi_k25_lora_alpha` | 32 |

`compare_sample_train_data.pt` ships **inside** each model's LoRA adapter repo above, so it lands at
`<lora-dir>/compare_sample_train_data.pt` after download — used for the acc capture in **both** cells
(even a no-LoRA cell, which just doesn't pass `--lora-paths`). Model-specific standard server args are
shown in each section's launch command (Qwen3.5 adds mamba / allreduce-fusion / trtllm_mha; Kimi uses
the NVFP4 launch); per-cell `*_EXTRA` is appended on top.

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
export RUN_ROOT="$HOME/Downloads/sglang_regression_${ID}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT"

# Regression PASS/REGRESS thresholds (used only by §4 — adjust to your intent; see the callout above).
#   ACC_TOL  — max |logprob(variant) - logprob(base)| allowed before flagging an accuracy regression.
#              Use a small value (e.g. 1e-2) when base/variant should be numerically equivalent.
#   PERF_TOL — fractional throughput drop allowed before flagging a perf regression (0.05 = 5% slower).
export ACC_TOL=0.01
export PERF_TOL=0.05

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

Each cell (base and variant) runs on a commit you specify — a local ref or a GitHub URL (§0 resolver),
possibly different per cell. Two git bundles (base + variant) are therefore built **per model** inside
each section's `.3` step (set `BASE_SRC` + `VARIANT_SRC` there) and injected as `__bench_base` /
`__bench_variant`; the run script checks out each before serving its cell. If base and variant resolve
to the same commit, both bundles are identical — harmless.

> **Compiled-component caveat (applies to every driver):** `pip install -e python` after checkout
> re-links the Python package, which is enough when a cell's commit only changes Python / Triton-JIT
> code. If base or variant modifies the compiled `sgl-kernel` C++/CUDA, also rebuild it on the
> checked-out commit:
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

Then run §4 once on `$RUN_ROOT` to build `summary.md` (acc-diff + perf-delta tables). The result is a
single `~/Downloads/sglang_regression_<id>_<timestamp>/` folder holding `qwen35/`, `qwen3vl/`, `kimi/`
subfolders (each with `base/` + `variant/` acc & bench artifacts) and `summary.md`.

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
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
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

### 1.3 Build + inject base & variant + ghost check + clean (after setup)

Build **two** bundles locally — one per cell — and inject both as `__bench_base` / `__bench_variant`.
Set the per-cell LoRA + extra args later in `run_qwen35.sh` (§1.4).

```bash
REPO=~/Developer/sglang             # local sglang checkout (workspace for the bundles; URLs fetched into it)
BASE_SRC=origin/main                # control — local ref OR GitHub URL
VARIANT_SRC=trtllm-lora             # candidate — local ref OR GitHub URL (may equal BASE_SRC)
mkdir -p "${RUN_ROOT}/qwen35"; echo "model=qwen35" > "${RUN_ROOT}/qwen35/meta.env"
build_cell(){  # $1=cell(base|variant)  $2=SRC → bundle + record commit/merge_base
  resolve_to_bench_target "$REPO" "$2"               # §0 helper → __bench_target (fetches URLs)
  local mb head; mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/sglang-$1-qwen35.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; echo "$1_merge_base=$mb"; } >> "${RUN_ROOT}/qwen35/meta.env"
  echo "$1: $2 -> ${head:0:12}"
}
build_cell base    "$BASE_SRC"
build_cell variant "$VARIANT_SRC"

kubectl cp /tmp/sglang-base-qwen35.bundle    sglang-qwen35-${ID}:/root/base.bundle
kubectl cp /tmp/sglang-variant-qwen35.bundle sglang-qwen35-${ID}:/root/variant.bundle
kubectl exec sglang-qwen35-${ID} -- bash -lc '
cd /root/sglang
git fetch /root/base.bundle    __bench_target:refs/heads/__bench_base
git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant
git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'

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

### 1.4 Run acc + bench (base & variant) — `run_qwen35.sh`

Set the per-cell `BASE_*` / `VARIANT_*` config (LoRA on/off + extra args) to match what you injected
in §1.3, then run locally (drives the pod via `kubectl exec`). Commits come from the §1.3 bundles
(`__bench_base` / `__bench_variant`).

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
TP_ARGS="--tp 4 --ep 4"
COMMON_EXTRA="--mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha"  # model standard args (BOTH cells)
PREFILL_ARGS="--max-prefill-tokens 32768 --chunked-prefill-size 32768"
PORT=30000
ACC_DATA="${LORA_PATH}/compare_sample_train_data.pt"   # ships inside the LoRA adapter repo

# ===== base (control) vs variant (candidate) — commits are the §1.3 bundles; set LoRA + extra here =====
BASE_REF=__bench_base;       BASE_LORA=off;    BASE_EXTRA=""
VARIANT_REF=__bench_variant; VARIANT_LORA=on;  VARIANT_EXTRA="--moe-runner-backend sgl_flashinfer_trtllm"
cell_ref(){   [ "$1" = base ] && echo "$BASE_REF"   || echo "$VARIANT_REF"; }
cell_lora(){  [ "$1" = base ] && echo "$BASE_LORA"  || echo "$VARIANT_LORA"; }
cell_extra(){ [ "$1" = base ] && echo "$BASE_EXTRA" || echo "$VARIANT_EXTRA"; }

# ===== fixed workload =====
IN=2048; OUT=2048
BENCH_BS="16 32 64"
OUTROOT=/tmp/regression
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_regression_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/qwen35"

k(){ kubectl exec "${POD}" -- bash -lc "$1"; }
kill_server(){ k 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 5'; }
checkout(){ k "cd /root/sglang; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"; }
wait_ready(){ k "for i in \$(seq 1 360); do curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null && { echo READY; exit 0; }; sleep 5; done; echo NOT_READY; tail -n 200 /tmp/server.log; exit 1"; }
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

# Per-token-logprob capture helper, cp'd into the pod (avoids fragile nested heredocs over kubectl exec).
cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests
ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--lora", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
data = torch.load(a.data, weights_only=False)
toks = data["tokens"]
if torch.is_tensor(toks): toks = toks.tolist()
seqs = toks if (toks and isinstance(toks[0], list)) else [toks]   # accept one seq or a batch of seqs
lp = []
for s in seqs:
    p = {"input_ids": s, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
         "return_logprob": True, "logprob_start_len": 0}
    if a.lora: p["lora_path"] = a.lora
    r = requests.post(f"http://127.0.0.1:{a.port}/generate", json=p, timeout=1800); r.raise_for_status()
    lp += [x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]  # [1:] skips the BOS slot (no logprob)
import json
with open(a.out, "w") as f: json.dump(lp, f)   # plain list of floats — readable locally without torch
print("wrote", len(lp), "logprobs ->", a.out)
PY
kubectl cp /tmp/acc_capture.py ${POD}:/root/acc_capture.py

launch(){  # $1=cell (base|variant)
  local lora extra lora_flags=""; lora=$(cell_lora "$1"); extra=$(cell_extra "$1")
  # NOTE: this hardcodes --lora-use-virtual-experts. The default MoE backend on Qwen3.5/Qwen3-VL
  # FP8 (= flashinfer_trtllm) does NOT support virtual-experts LoRA. Any LoRA=on cell MUST set
  # --moe-runner-backend in *_EXTRA: `triton` for stock LoRA, `sgl_flashinfer_trtllm` for the new
  # trtllm-LoRA path. Otherwise the server crashes at startup with NotImplementedError.
  [ "$lora" = on ] && lora_flags="--enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-use-virtual-experts --lora-paths ${LORA_NAME}=${LORA_PATH}"
  # Server in the exec FOREGROUND (`exec … python …`) + background the LOCAL kubectl exec (trailing `&`
  # is local). The driver returns immediately; the exec lingers until kill_server pkills the server.
  # An in-pod `… & echo $!` or setsid+</dev/null both HANG (kubectl exec keeps the stream open).
  kubectl exec "${POD}" -- bash -lc "cd /root/sglang && exec numactl --membind=0,1 python -m sglang.launch_server --model-path ${MODEL_PATH} ${TP_ARGS} --host 0.0.0.0 --port ${PORT} --cuda-graph-max-bs 64 --trust-remote-code ${PREFILL_ARGS} ${COMMON_EXTRA} ${extra} ${lora_flags} > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  wait_ready
}

acc(){  # $1=cell — per-token logprobs over ACC_DATA (lora_path only when that cell serves LoRA)
  local name=""; [ "$(cell_lora "$1")" = on ] && name="${LORA_NAME}"
  local d="${OUTROOT}/$1/acc"
  k "mkdir -p ${d}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${name}' --out ${d}/logprobs.json 2>&1 | tee ${d}/acc.log"
}

bench(){  # $1=cell
  local lora_arg=""; [ "$(cell_lora "$1")" = on ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$1/bench"
  k "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${lora_arg} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}

run_cell(){  # $1=cell (base|variant)
  kill_server; checkout "$(cell_ref "$1")"
  kill_server; launch "$1"
  acc "$1";   dl "$1/acc"
  echo "[$(date +%H:%M:%S)] qwen35 $1 ACC done -> ${LOCAL_OUT}/$1/acc"     | tee -a "${RUN_ROOT}/progress.log"
  bench "$1"; dl "$1/bench"
  echo "[$(date +%H:%M:%S)] qwen35 $1 BENCH done -> ${LOCAL_OUT}/$1/bench" | tee -a "${RUN_ROOT}/progress.log"
}

run_cell base
run_cell variant
kill_server

# acc + bench were downloaded incrementally per cell. Final sweep to catch stragglers (logs):
dl "."
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
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
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

### 2.3 Build + inject base & variant + ghost check + clean (after setup)

Build **two** bundles locally — one per cell — and inject both as `__bench_base` / `__bench_variant`.
Set the per-cell LoRA + extra args later in `run_qwen3vl.sh` (§2.4).

```bash
REPO=~/Developer/sglang             # local sglang checkout (workspace for the bundles; URLs fetched into it)
BASE_SRC=origin/main                # control — local ref OR GitHub URL
VARIANT_SRC=trtllm-lora             # candidate — local ref OR GitHub URL (may equal BASE_SRC)
mkdir -p "${RUN_ROOT}/qwen3vl"; echo "model=qwen3vl" > "${RUN_ROOT}/qwen3vl/meta.env"
build_cell(){  # $1=cell(base|variant)  $2=SRC → bundle + record commit/merge_base
  resolve_to_bench_target "$REPO" "$2"               # §0 helper → __bench_target (fetches URLs)
  local mb head; mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/sglang-$1-qwen3vl.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; echo "$1_merge_base=$mb"; } >> "${RUN_ROOT}/qwen3vl/meta.env"
  echo "$1: $2 -> ${head:0:12}"
}
build_cell base    "$BASE_SRC"
build_cell variant "$VARIANT_SRC"

kubectl cp /tmp/sglang-base-qwen3vl.bundle    sglang-qwen3vl-${ID}:/root/base.bundle
kubectl cp /tmp/sglang-variant-qwen3vl.bundle sglang-qwen3vl-${ID}:/root/variant.bundle
kubectl exec sglang-qwen3vl-${ID} -- bash -lc '
cd /root/sglang
git fetch /root/base.bundle    __bench_target:refs/heads/__bench_base
git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant
git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'

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

### 2.4 Run acc + bench (base & variant) — `run_qwen3vl.sh`

Same shape as `run_qwen35.sh` (§1.4); only the model config differs. Set the per-cell `BASE_*` /
`VARIANT_*` to match §2.3, then run.

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
TP_ARGS="--tp 4 --ep 4"
COMMON_EXTRA=""                       # model standard args common to BOTH cells (none for qwen3vl)
PREFILL_ARGS="--max-prefill-tokens 32768 --chunked-prefill-size 32768"
PORT=30000
ACC_DATA="${LORA_PATH}/compare_sample_train_data.pt"   # ships inside the LoRA adapter repo

# ===== base (control) vs variant (candidate) — commits are the §2.3 bundles; set LoRA + extra here =====
BASE_REF=__bench_base;       BASE_LORA=off;    BASE_EXTRA=""
VARIANT_REF=__bench_variant; VARIANT_LORA=on;  VARIANT_EXTRA="--moe-runner-backend sgl_flashinfer_trtllm"
cell_ref(){   [ "$1" = base ] && echo "$BASE_REF"   || echo "$VARIANT_REF"; }
cell_lora(){  [ "$1" = base ] && echo "$BASE_LORA"  || echo "$VARIANT_LORA"; }
cell_extra(){ [ "$1" = base ] && echo "$BASE_EXTRA" || echo "$VARIANT_EXTRA"; }

# ===== fixed workload =====
IN=2048; OUT=2048
BENCH_BS="16 32 64"
OUTROOT=/tmp/regression
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_regression_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/qwen3vl"

k(){ kubectl exec "${POD}" -- bash -lc "$1"; }
kill_server(){ k 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 5'; }
checkout(){ k "cd /root/sglang; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"; }
wait_ready(){ k "for i in \$(seq 1 360); do curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null && { echo READY; exit 0; }; sleep 5; done; echo NOT_READY; tail -n 200 /tmp/server.log; exit 1"; }
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests
ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--lora", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
data = torch.load(a.data, weights_only=False)
toks = data["tokens"]
if torch.is_tensor(toks): toks = toks.tolist()
seqs = toks if (toks and isinstance(toks[0], list)) else [toks]
lp = []
for s in seqs:
    p = {"input_ids": s, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
         "return_logprob": True, "logprob_start_len": 0}
    if a.lora: p["lora_path"] = a.lora
    r = requests.post(f"http://127.0.0.1:{a.port}/generate", json=p, timeout=1800); r.raise_for_status()
    lp += [x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]
import json
with open(a.out, "w") as f: json.dump(lp, f)   # plain list of floats — readable locally without torch
print("wrote", len(lp), "logprobs ->", a.out)
PY
kubectl cp /tmp/acc_capture.py ${POD}:/root/acc_capture.py

launch(){  # $1=cell (base|variant)
  local lora extra lora_flags=""; lora=$(cell_lora "$1"); extra=$(cell_extra "$1")
  # NOTE: this hardcodes --lora-use-virtual-experts. The default MoE backend on Qwen3.5/Qwen3-VL
  # FP8 (= flashinfer_trtllm) does NOT support virtual-experts LoRA. Any LoRA=on cell MUST set
  # --moe-runner-backend in *_EXTRA: `triton` for stock LoRA, `sgl_flashinfer_trtllm` for the new
  # trtllm-LoRA path. Otherwise the server crashes at startup with NotImplementedError.
  [ "$lora" = on ] && lora_flags="--enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-use-virtual-experts --lora-paths ${LORA_NAME}=${LORA_PATH}"
  kubectl exec "${POD}" -- bash -lc "cd /root/sglang && exec numactl --membind=0,1 python -m sglang.launch_server --model-path ${MODEL_PATH} ${TP_ARGS} --host 0.0.0.0 --port ${PORT} --cuda-graph-max-bs 64 --trust-remote-code ${PREFILL_ARGS} ${COMMON_EXTRA} ${extra} ${lora_flags} > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  wait_ready
}

acc(){  # $1=cell
  local name=""; [ "$(cell_lora "$1")" = on ] && name="${LORA_NAME}"
  local d="${OUTROOT}/$1/acc"
  k "mkdir -p ${d}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${name}' --out ${d}/logprobs.json 2>&1 | tee ${d}/acc.log"
}

bench(){  # $1=cell
  local lora_arg=""; [ "$(cell_lora "$1")" = on ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$1/bench"
  k "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${lora_arg} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}

run_cell(){  # $1=cell (base|variant)
  kill_server; checkout "$(cell_ref "$1")"
  kill_server; launch "$1"
  acc "$1";   dl "$1/acc"
  echo "[$(date +%H:%M:%S)] qwen3vl $1 ACC done -> ${LOCAL_OUT}/$1/acc"     | tee -a "${RUN_ROOT}/progress.log"
  bench "$1"; dl "$1/bench"
  echo "[$(date +%H:%M:%S)] qwen3vl $1 BENCH done -> ${LOCAL_OUT}/$1/bench" | tee -a "${RUN_ROOT}/progress.log"
}

run_cell base
run_cell variant
kill_server

dl "."
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
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
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
  volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 32Gi
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

### 3.3 Build + inject base & variant + ghost check + clean (both pods, after setup)

Build **two** bundles locally — one per cell — and inject both into **both** pods as `__bench_base` /
`__bench_variant`. Set the per-cell LoRA + extra args later in `run_kimi.sh` (§3.4).

```bash
REPO=~/Desktop/sglang                 # local sglang checkout (workspace for the bundles; URLs fetched into it)
BASE_SRC=origin/main                  # control — local ref OR GitHub URL
VARIANT_SRC=lora-nvfp4-gb200-cookbook # candidate — local ref OR GitHub URL (may equal BASE_SRC)
mkdir -p "${RUN_ROOT}/kimi"; echo "model=kimi" > "${RUN_ROOT}/kimi/meta.env"
build_cell(){  # $1=cell(base|variant)  $2=SRC → bundle + record commit/merge_base
  resolve_to_bench_target "$REPO" "$2"               # §0 helper → __bench_target (fetches URLs)
  local mb head; mb=$(git -C "$REPO" merge-base origin/main __bench_target); head=$(git -C "$REPO" rev-parse __bench_target)
  git -C "$REPO" bundle create "/tmp/sglang-$1-kimi.bundle" __bench_target --not "${mb}^"
  { echo "$1_src=$2"; echo "$1_commit=$head"; echo "$1_merge_base=$mb"; } >> "${RUN_ROOT}/kimi/meta.env"
  echo "$1: $2 -> ${head:0:12}"
}
build_cell base    "$BASE_SRC"
build_cell variant "$VARIANT_SRC"

for P in mnnvl-kimi-${ID}-0 mnnvl-kimi-${ID}-1; do
  kubectl cp /tmp/sglang-base-kimi.bundle    "${P}:/root/base.bundle"
  kubectl cp /tmp/sglang-variant-kimi.bundle "${P}:/root/variant.bundle"
  kubectl exec "${P}" -- bash -lc '
  cd /root/sglang
  git fetch /root/base.bundle    __bench_target:refs/heads/__bench_base
  git fetch /root/variant.bundle __bench_target:refs/heads/__bench_variant
  git --no-pager log -1 --oneline __bench_base; git --no-pager log -1 --oneline __bench_variant'
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

### 3.4 Run acc + bench (base & variant) — `run_kimi.sh` (2-node)

Launches on the worker first, then the head; acc + bench are driven from the head against
`localhost`. Set the per-cell `BASE_*` / `VARIANT_*` to match §3.3, then run.

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
PORT=30000
ACC_DATA="${LORA_PATH}/compare_sample_train_data.pt"   # ships inside the LoRA adapter repo

# ===== base (control) vs variant (candidate) — commits are the §3.3 bundles; set LoRA + extra here =====
BASE_REF=__bench_base;       BASE_LORA=off;    BASE_EXTRA=""
VARIANT_REF=__bench_variant; VARIANT_LORA=on;  VARIANT_EXTRA="--moe-runner-backend flashinfer_cutlass"
cell_ref(){   [ "$1" = base ] && echo "$BASE_REF"   || echo "$VARIANT_REF"; }
cell_lora(){  [ "$1" = base ] && echo "$BASE_LORA"  || echo "$VARIANT_LORA"; }
cell_extra(){ [ "$1" = base ] && echo "$BASE_EXTRA" || echo "$VARIANT_EXTRA"; }

# ===== fixed workload =====
IN=2048; OUT=2048
BENCH_BS="16 32 64"
OUTROOT=/tmp/regression
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_regression_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/kimi"

# Model-standard server args common to BOTH cells (no EP). Per-cell extra (e.g. the MoE backend) is
# appended by launch(); the BASE/VARIANT_EXTRA above carries it.
COMMON="--model-path ${MODEL_PATH} --tp 8 --nnodes 2 --dist-init-addr ${DIST_INIT} \
--host 0.0.0.0 --port ${PORT} --quantization modelopt_fp4 --mem-fraction-static 0.88 \
--cuda-graph-max-bs 64 --trust-remote-code \
--max-prefill-tokens 40960 --chunked-prefill-size 40960"
ENVS="NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false"

kh(){ kubectl exec "${HEAD_POD}"   -- bash -lc "$1"; }
kw(){ kubectl exec "${WORKER_POD}" -- bash -lc "$1"; }
both(){ kw "$1"; kh "$1"; }

kill_all(){ both 'pkill -f "[s]glang.launch_server" >/dev/null 2>&1 || true; pkill -f "[b]ench_one_batch_server" >/dev/null 2>&1 || true; sleep 8'; }
checkout(){ both "cd /root/sglang; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1"; kh "cd /root/sglang && git --no-pager log -1 --oneline"; }
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${HEAD_POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

# One-time per node: pre-warm the HF dynamic-module cache so 4 ranks/node don't race copying the
# model's trust_remote_code *.py into ~/.cache/huggingface/modules at first launch.
prewarm(){
  for P in "${WORKER_POD}" "${HEAD_POD}"; do
    kubectl exec "${P}" -- bash -lc 'python3 -c "from transformers import AutoConfig, AutoTokenizer, AutoProcessor; m=\"/root/Kimi-K2.5-NVFP4\"; AutoConfig.from_pretrained(m, trust_remote_code=True); AutoTokenizer.from_pretrained(m, trust_remote_code=True); AutoProcessor.from_pretrained(m, trust_remote_code=True); print(\"prewarmed\")"'
  done
}

# acc-capture helper, cp'd into the HEAD pod (acc is driven from the head against localhost).
cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests
ap = argparse.ArgumentParser()
ap.add_argument("--port", required=True); ap.add_argument("--data", required=True)
ap.add_argument("--lora", default=""); ap.add_argument("--out", required=True)
a = ap.parse_args()
data = torch.load(a.data, weights_only=False)
toks = data["tokens"]
if torch.is_tensor(toks): toks = toks.tolist()
seqs = toks if (toks and isinstance(toks[0], list)) else [toks]
lp = []
for s in seqs:
    p = {"input_ids": s, "sampling_params": {"max_new_tokens": 0, "temperature": 0.0},
         "return_logprob": True, "logprob_start_len": 0}
    if a.lora: p["lora_path"] = a.lora
    r = requests.post(f"http://127.0.0.1:{a.port}/generate", json=p, timeout=1800); r.raise_for_status()
    lp += [x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]
import json
with open(a.out, "w") as f: json.dump(lp, f)   # plain list of floats — readable locally without torch
print("wrote", len(lp), "logprobs ->", a.out)
PY
kubectl cp /tmp/acc_capture.py ${HEAD_POD}:/root/acc_capture.py

# Background the LOCAL kubectl exec (trailing `&` is local); server runs in the exec foreground.
# An in-pod `… & echo $!` (or setsid+</dev/null) hangs the exec — sglang's worker subprocesses
# keep its stream open, so kubectl exec never returns and the driver blocks.
start_rank(){  # $1=pod  $2=node-rank  $3=extra-flags
  kubectl exec "$1" -- bash -lc "cd /root/sglang && ${ENVS} exec numactl --membind=0,1 python3 -m sglang.launch_server ${COMMON} --node-rank $2 $3 > /tmp/server.log 2>&1" >/dev/null 2>&1 &
  echo "started-rank$2 on $1"
}
launch(){  # $1=cell (base|variant)
  local lora extra lora_flags=""; lora=$(cell_lora "$1"); extra=$(cell_extra "$1")
  # NOTE: this hardcodes --lora-use-virtual-experts. The default MoE backend on Qwen3.5/Qwen3-VL
  # FP8 (= flashinfer_trtllm) does NOT support virtual-experts LoRA. Any LoRA=on cell MUST set
  # --moe-runner-backend in *_EXTRA: `triton` for stock LoRA, `sgl_flashinfer_trtllm` for the new
  # trtllm-LoRA path. Otherwise the server crashes at startup with NotImplementedError.
  [ "$lora" = on ] && lora_flags="--enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-use-virtual-experts --lora-paths ${LORA_NAME}=${LORA_PATH}"
  local flags="${extra} ${lora_flags}"
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

acc(){  # $1=cell (driven from head)
  local name=""; [ "$(cell_lora "$1")" = on ] && name="${LORA_NAME}"
  local d="${OUTROOT}/$1/acc"
  kh "mkdir -p ${d}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${name}' --out ${d}/logprobs.json 2>&1 | tee ${d}/acc.log"
}

bench(){  # $1=cell (driven from head)
  local lora_arg=""; [ "$(cell_lora "$1")" = on ] && lora_arg="--lora-name ${LORA_NAME}"
  local d="${OUTROOT}/$1/bench"
  kh "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${lora_arg} \
        --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; done"
}

run_cell(){  # $1=cell (base|variant)
  kill_all; checkout "$(cell_ref "$1")"
  kill_all; launch "$1"
  acc "$1";   dl "$1/acc"
  echo "[$(date +%H:%M:%S)] kimi $1 ACC done -> ${LOCAL_OUT}/$1/acc"     | tee -a "${RUN_ROOT}/progress.log"
  bench "$1"; dl "$1/bench"
  echo "[$(date +%H:%M:%S)] kimi $1 BENCH done -> ${LOCAL_OUT}/$1/bench" | tee -a "${RUN_ROOT}/progress.log"
}

prewarm
run_cell base
run_cell variant
kill_all

dl "."
echo "[$(date +%H:%M:%S)] kimi DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"
```

---

## 4. Summary — accuracy & performance regression (run once, locally, after all 3 models)

The drivers **download incrementally** — each cell's acc logprobs (`acc/logprobs.json`) and bench
jsonl are pulled to local the moment they're produced, with a line appended to
`${RUN_ROOT}/progress.log` as each cell finishes. By the time all three drivers finish, `${RUN_ROOT}`
already holds everything locally; §4 just aggregates it — no large end-of-run copy.

`~/Downloads/sglang_regression_<id>_<timestamp>/` contains `qwen35/`, `qwen3vl/`, `kimi/`, each with
`base/` and `variant/` holding `acc/logprobs.json` (+ `acc.log`) and `bench/bs{16,32,64}.jsonl`
(+ logs), plus `meta.env` (both cells' commits) — and a top-level `progress.log` + `env.txt`.

### 4.0 Capture the environment (for reproducibility)

Capture library versions from any one running pod before teardown:

```bash
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder}"; POD=sglang-qwen35-${ID}
{ echo "image=lmsysorg/sglang:dev-cu13"
  echo "cluster=leira (k8s, namespace default)"
  echo "acc=per-token input_token_logprobs over compare_sample_train_data.pt (max_new_tokens=0, temp=0)"
  echo "bench=bench_one_batch_server; batch_size={16,32,64}; input_len=output_len=2048; cuda_graph=on"
  kubectl --context leira exec "$POD" -- bash -lc 'echo "gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader|head -1)"; echo "gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader|head -1)"; python3 -c "import torch;print(\"torch=\"+torch.__version__);print(\"cuda=\"+str(torch.version.cuda))"; for m in deep_gemm flashinfer sgl_kernel transformers triton; do python3 -c "import $m;print(\"$m=\"+$m.__version__)" 2>/dev/null || echo "$m=n/a"; done'
} > "${ROOT}/env.txt"; cat "${ROOT}/env.txt"
```

> Use `POD=sglang-qwen3vl-${ID}` or `mnnvl-kimi-${ID}-0` if the qwen35 pod is already gone. Each
> cell's commit is recorded in that model's `meta.env` by the `.3` step.

### 4.1 Build the summary (provenance + environment + acc-diff + perf-delta)

Runs **locally** with pure stdlib — reads `acc/logprobs.json` + `bench/*.jsonl` (no torch needed). The
acc table diffs variant-vs-base per-token logprobs; the perf table diffs latency/throughput. The
PASS/REGRESS flag uses `ACC_TOL` / `PERF_TOL` from §0 — **meaningful only when base and variant are
meant to be numerically equivalent**; for an intentional change, read the raw numbers.

```bash
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder, e.g. ~/Downloads/sglang_regression_<id>_<timestamp>}"
ACC_TOL="${ACC_TOL:-0.01}" PERF_TOL="${PERF_TOL:-0.05}" python3 - "$ROOT" <<'PY'
import json, os, sys
from pathlib import Path
root = Path(sys.argv[1]).expanduser()
MODELS = ["qwen35", "qwen3vl", "kimi"]; BS = [16, 32, 64]
ACC_TOL = float(os.environ.get("ACC_TOL", "0.01")); PERF_TOL = float(os.environ.get("PERF_TOL", "0.05"))
def env(p):
    d = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line: k, v = line.split("=", 1); d[k] = v
    return d
def load_lp(p):
    try: return [float(x) for x in json.loads(p.read_text())] if p.exists() else None
    except Exception: return None
def last_row(p):
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]; return rows[-1] if rows else None
def pctl(v, q):                       # percentile of values (q in [0,1]), pure python
    s = sorted(v); i = min(len(s)-1, max(0, int(round(q*(len(s)-1))))); return s[i]
E = env(root/"env.txt"); M = {m: env(root/m/"meta.env") for m in MODELS}
L = ["# SGLang Base vs Variant — Regression Summary", ""]
L += [f"Run `{root.name}`. base = control, variant = candidate. "
      f"acc tol (max abs Δlogprob) = {ACC_TOL}; perf tol (throughput drop) = {PERF_TOL:.0%}.", ""]

L += ["## Provenance (per cell)", "",
      "| model | base commit | variant commit | base src | variant src |", "|---|---|---|---|---|"]
for m in MODELS:
    d = M[m]
    L.append(f"| {m} | `{d.get('base_commit','?')[:12]}` | `{d.get('variant_commit','?')[:12]}` | {d.get('base_src','?')} | {d.get('variant_src','?')} |")

L += ["", "## Environment", "", "| key | value |", "|---|---|"]
for k in ["image","gpu","gpu_driver","cuda","torch","deep_gemm","flashinfer","sgl_kernel","transformers","triton","cluster","acc","bench"]:
    if E.get(k): L.append(f"| {k} | {E[k]} |")
L += ["", "> sglang is the editable install at each cell's commit (see Provenance); libs above are image-level.", ""]

# ---- Accuracy: per-token logprob |diff| (variant vs base) over the same data ----
L += ["## Accuracy (per-token logprob |diff|, variant vs base)", "",
      "| model | n | mean abs | max abs | p50 | p95 | half-MSE | verdict |",
      "|---|---:|---:|---:|---:|---:|---:|:--|"]
for m in MODELS:
    a = load_lp(root/m/"base"/"acc"/"logprobs.json"); b = load_lp(root/m/"variant"/"acc"/"logprobs.json")
    if not a or not b:
        L.append(f"| {m} | {('base+' if not a else '')+('variant' if not b else '')} MISSING | | | | | | n/a |"); continue
    n = min(len(a), len(b)); diff = [abs(a[i]-b[i]) for i in range(n)]
    mean = sum(diff)/n; mx = max(diff); hmse = 0.5*sum((a[i]-b[i])**2 for i in range(n))/n
    verdict = "PASS" if mx <= ACC_TOL else "**REGRESS**"
    note = "" if len(a)==len(b) else f" ⚠{len(a)}v{len(b)}"
    L.append(f"| {m} | {n}{note} | {mean:.6f} | {mx:.6f} | {pctl(diff,0.5):.6f} | {pctl(diff,0.95):.6f} | {hmse:.6f} | {verdict} |")

# ---- Performance: latency/throughput delta (variant vs base) ----
L += ["", "## Performance (variant vs base, tok/s & latency)", "",
      "| model | bs | base lat (s) | var lat (s) | base tok/s | var tok/s | var % base | verdict |",
      "|---|---:|---:|---:|---:|---:|---:|:--|"]
for m in MODELS:
    for bs in BS:
        b = last_row(root/m/"base"/"bench"/f"bs{bs}.jsonl"); v = last_row(root/m/"variant"/"bench"/f"bs{bs}.jsonl")
        if not b or not v:
            L.append(f"| {m} | {bs} | {'MISSING' if not b else 'ok'} | {'MISSING' if not v else 'ok'} | | | | n/a |"); continue
        bl, vl = b.get("latency"), v.get("latency"); bt, vt = b.get("output_throughput"), v.get("output_throughput")
        ratio = (vt/bt*100) if (bt and vt) else None
        verdict = "n/a" if ratio is None else ("PASS" if ratio >= (1-PERF_TOL)*100 else "**REGRESS**")
        L.append(f"| {m} | {bs} | {bl:.3f} | {vl:.3f} | {bt:.1f} | {vt:.1f} | {ratio:.1f}% | {verdict} |"
                 if None not in (bl, vl, bt, vt, ratio) else f"| {m} | {bs} | {bl} | {vl} | {bt} | {vt} | {ratio} | {verdict} |")

out = root/"summary.md"; out.write_text("\n".join(L) + "\n"); print(out); print("\n".join(L))
PY
```

### 4.2 Final deliverable

`~/Downloads/sglang_regression_<id>_<timestamp>/` holds everything in one folder:

- `summary.md` — provenance (both cells' commits) + environment + acc-diff table + perf-delta table
- `progress.log` — per-cell "ACC done" / "BENCH done" log; `env.txt` — captured environment
- `qwen35/`, `qwen3vl/`, `kimi/` — `meta.env` + `base/` & `variant/`, each with `acc/logprobs.json`
  (+ `acc.log`) and `bench/bs{16,32,64}.jsonl` (+ logs), pulled **incrementally** during the runs

```bash
ROOT="${RUN_ROOT:?export RUN_ROOT to the run folder the drivers used}"
find "${ROOT}" -maxdepth 4 -type f | sort
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
- **Each cell's commit must run on the current image.** Base and variant may be different commits; if
  either won't start, that cell is blocked. A common skew: the image's `deep_gemm` lacks an API an
  older commit calls unguarded — `AttributeError: module 'deep_gemm' has no attribute
  'get_compile_mode'` during FP8 JIT warmup (newer commits guard it with `hasattr`). Resolve by
  choosing commits that include the compat guard, or pin a matching image — **not** by disabling the
  FP8 / deep_gemm path (that changes perf and invalidates the comparison). The crash is in `/tmp/server.log`.

## 5. Cleanup (only after §4 — summary built and results downloaded)

Tear down all pods once every acc capture + bench is done and the summary + per-cell artifacts are
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
