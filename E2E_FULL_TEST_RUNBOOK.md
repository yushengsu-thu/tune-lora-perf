# Experimental TRT-LLM LoRA — e2e Full-Test Runbook

How the full end-to-end test matrix for the **experimental TRT-LLM LoRA fast path**
(`SGLANG_EXPERIMENTAL_LORA_OPTI`, branch `jybsuper:full-lora-opti`) was run: infra,
env/YAML, launch commands, the test matrix, helper scripts, and what to expect.

> Cluster: GB200 (`leira`), arm64 + NVIDIA GB200, MNNVL. Image `lmsysorg/sglang:dev-cu13`.
> Models: **Qwen3.5-35B-A3B-FP8** (TP4/EP4, 1 node) and **Kimi-K2.5-NVFP4** (TP8/EP8, 2 nodes).
> Adapter for both: a behavioral identity LoRA named **`alpha`** (`*_lora_alpha`).

---

## 0. TL;DR — the comparison

For every model we compare two trees under an otherwise-identical launch:

| Lane | Branch | MoE backend | LoRA flags |
|---|---|---|---|
| **PR (experimental)** | `jybsuper/sglang:full-lora-opti` | `--moe-runner-backend sgl_flashinfer_trtllm` + `SGLANG_EXPERIMENTAL_LORA_OPTI=1` (+ per-model opt envs) | `--enable-lora …` |
| **oss baseline** | `sgl-project/sglang:main` | `triton` (LoRA lane) **or default/unset** (the %-denominator) | with / without `--enable-lora` |

**The %-of-ceiling denominator** is deliberately the *oss no-LoRA* run on the **same
launch minus the LoRA args and WITHOUT `--moe-runner-backend`** (i.e. the stock default
backend). Every PR throughput number is reported as `% of that oss no-LoRA ceiling`.

---

## 1. Infrastructure / env setup

### 1.1 Pods

Six pods, two per model role (head + worker for the 2-node Kimi runs; the qwen pods
are single-node but created in pairs for PR-vs-oss isolation):

```
mnnvl-kimi-cfuse-0   mnnvl-kimi-cfuse-1     # Kimi  (PR lane), 2-node MNNVL
mnnvl-kimi-nv-0      mnnvl-kimi-nv-1        # Kimi  (oss-main lane), 2-node MNNVL
mnnvl-kimi-lorapr-0  mnnvl-kimi-lorapr-1    # Qwen3.5 (PR lane / oss lane), 1 node each (also used for tp1)
```

Each pod gets **4× GB200**. A 2-node Kimi run uses one head + one worker pod (8 GPUs
total via the MNNVL `imex-channel` resourceClaim + the `*-head` subdomain for
`--dist-init-addr`).

### 1.2 Pod YAML (key fields)

Local copies: `~/Desktop/lora_bench_run/kimi-2node.yaml`, `~/Desktop/lora_bench_run/qwen35-pod.yaml`,
`/tmp/rf_regression/kimi-rf.yaml`, `/tmp/rf_regression/flo-Qwen3.5-35B-A3B-FP8-tp1-pods.yaml`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: mnnvl-kimi-lorapr-0
  labels: {app: mnnvl-kimi-lorapr, sglang-role: head}
spec:
  runtimeClassName: nvidia
  subdomain: mnnvl-kimi-lorapr-head           # gives a stable DNS for --dist-init-addr
  affinity:                                    # one pod per physical node
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector: {matchLabels: {app: mnnvl-kimi-lorapr}}
          topologyKey: kubernetes.io/hostname
  resourceClaims:
    - {name: imex-channel, resourceClaimTemplateName: mnnvl-kimi-lorapr-cd-channel}  # MNNVL fabric
  tolerations:
    - {key: kubernetes.io/arch, operator: Equal, value: arm64, effect: NoSchedule}
    - {key: nvidia.com/gpu, operator: Exists, effect: NoSchedule}
  containers:
    - name: sglang
      image: lmsysorg/sglang:dev-cu13
      imagePullPolicy: Always
      securityContext:
        privileged: true
        capabilities: {add: [SYS_PTRACE, SYS_ADMIN]}   # SYS_ADMIN needed for ncu / profiling
      ports: [{containerPort: 20000, name: dist-init}]
      resources:
        limits:   {cpu: "64", memory: 1800Gi, ephemeral-storage: 1200Gi, nvidia.com/gpu: 4}
        requests: {cpu: "16", memory: 300Gi,  ephemeral-storage: 100Gi,  nvidia.com/gpu: 4}
        claims: [{name: imex-channel}]
      env:
        - {name: HF_HUB_ENABLE_HF_TRANSFER, value: "1"}
        - {name: MALLOC_TRIM_THRESHOLD_, value: "131072"}
        - name: HF_TOKEN                          # private adapter download
          valueFrom: {secretKeyRef: {name: hf-token-yanbin, key: token, optional: true}}
      command: ["/bin/sh","-c", "<inline setup.sh; then sleep infinity>"]
      volumeMounts:
        - {mountPath: /dev/shm,    name: shm}
        - {mountPath: /root/.cache, name: dot-cache}
        - {mountPath: /data,       name: data}      # node-local big disk (model cache)
        - {mountPath: /host,       name: host-root}
  volumes:
    - {name: shm,       emptyDir: {medium: Memory, sizeLimit: 32Gi}}
    - {name: dot-cache, hostPath: {path: /mnt/nvme-b/sglang-dot-cache, type: DirectoryOrCreate}}
    - {name: data,      hostPath: {path: /mnt/nvme-b, type: Directory}}        # /data == /mnt/nvme-b
    - {name: host-root, hostPath: {path: /, type: Directory}}
```

Create the HF-token secret once:

```bash
kubectl create secret generic hf-token-yanbin --from-literal=token=hf_xxx
kubectl apply -f kimi-2node.yaml        # and qwen35-pod.yaml
```

### 1.3 Self-download `setup.sh` (runs once per pod at boot, logs to `/root/setup.log`)

The pod `command` writes and backgrounds this; `touch /root/.setup-done` when finished:

```bash
# numactl present (keeps page cache off the HBM-NUMA nodes — see GB200 ghost-mem note)
command -v numactl || apt-get install -y -qq numactl
pip install --quiet "huggingface_hub[hf_xet,hf_transfer]" hf_xet hf_transfer

# editable sglang install (origin/main); pip install -e python  (~minutes)
git clone https://github.com/sgl-project/sglang /root/sglang && cd /root/sglang && pip install -e python

# NVFP4 base must be a REAL dir (not the HF-cache symlink layout). Download to /data
# (node-local, persistent across pod recreations); flock so a concurrent same-node run WAITS.
mkdir -p /data/Kimi-K2.5-NVFP4 /data/kimi_k25_lora_alpha
( flock 9
  [ -f /data/Kimi-K2.5-NVFP4/config.json ] || numactl --membind=0,1 hf download nvidia/Kimi-K2.5-NVFP4 --local-dir /data/Kimi-K2.5-NVFP4
  [ -f /data/kimi_k25_lora_alpha/adapter_config.json ] || numactl --membind=0,1 hf download jybsuper/kimi_k25_lora_alpha --local-dir /data/kimi_k25_lora_alpha
) 9>/data/.kimi-download.lock
touch /root/.setup-done
```

Model paths used by the launch scripts (varies by how the pod was staged):
- Qwen: `/data/Qwen3.5-35B-A3B-FP8` + LoRA `/data/qwen35_35b_lora_alpha`
- Kimi: `/root/Kimi-K2.5-NVFP4` (cfuse, pre-staged) or `/data/Kimi-K2.5-NVFP4`; LoRA `/root/kimi_k25_lora_alpha` or `/data/kimi_k25_lora_alpha`

> **`/data` is editable + shared on a node.** The sglang install is editable
> (`pip install -e`), so you cannot PYTHONPATH-isolate two trees on one pod — switch
> trees with `git fetch <url> <branch> && git checkout -f FETCH_HEAD`. Never churn a
> *shared* pod (e.g. a `cfuse` pod a teammate is using) — it affects every process there.

### 1.4 Pre-flight per run

```bash
# wait for first-boot setup
kubectl exec <pod> -- bash -lc 'tail -f /root/setup.log'   # until "setup complete"
# kill stragglers, free the port (zsh-safe: the [s]glang regex avoids self-kill)
kubectl exec <pod> -- bash -lc 'pkill -9 -f "[s]glang.launch_server"; fuser -k 30000/tcp; sleep 5'
```

---

## 2. Launch commands

All launches are **fired detached** (`setsid … </dev/null >/tmp/srv.log 2>&1 &`) and
readiness is polled on `/tmp/srv.log` + `curl /v1/models` — never trust the
fire-and-forget exit code. `cfuse-1` SIGKILLs an `exec` ~2s into compute, so fire with
`setsid … & exit 0` and read `/tmp/srv.log` afterward.

### 2.1 Qwen3.5-FP8 — TP4 / EP4, single node (`Qwen3.5-35B-A3B-FP8_run.sh`)

Three configs run back-to-back on the PR pod; two on the oss pod:

```
PR pod  (lorapr-0): "sgl-lora sgl_flashinfer_trtllm 1 PR"   # experimental fast path
                    "triton-lora triton 1 NONE"             # default LoRA backend (control)
                    "nolora sgl_flashinfer_trtllm 0 PR"      # experimental backend, LoRA off
oss pod (lorapr-1): "triton-lora triton 1 NONE"
                    "nolora triton 0 NONE"
```

PR experimental launch (the opt-env set is the model-specific part):

```bash
# OPT env (qwen3.5): master switch + the qwen opt gates
SGLANG_EXPERIMENTAL_LORA_OPTI=1 \
SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 \
SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 \
SGLANG_OPT_LORA_CUBLAS=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
numactl --membind=0,1 python3 -m sglang.launch_server \
  --model-path /data/Qwen3.5-35B-A3B-FP8 --tp 4 --ep 4 \
  --host 0.0.0.0 --port 30000 --cuda-graph-max-bs 128 --mem-fraction-static 0.8 \
  --trust-remote-code --max-prefill-tokens 65536 --chunked-prefill-size 65536 \
  --mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion \
  --attention-backend trtllm_mha \
  --moe-runner-backend sgl_flashinfer_trtllm \
  --enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton \
  --lora-use-virtual-experts --lora-paths alpha=/data/qwen35_35b_lora_alpha
```

> **DO NOT** set `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1` — it was removed (it corrupted
> the base/no-active-LoRA path under cuda-graph; base gsm8k 0.38 → 0.79 once dropped).
> Keep flashinfer **autotune ON** — disabling it lowers speed and flatters overlap.
> Never set `SGLANG_OPT_LORA_ENABLE_PDL`.

`triton-lora` control = same minus the OPT env + `--moe-runner-backend triton`.
`nolora` = same OPT env + `--moe-runner-backend sgl_flashinfer_trtllm`, drop the `--enable-lora …`.

### 2.2 oss no-LoRA ceiling (the %-denominator) — `Qwen3.5-35B-A3B-FP8_base.sh`

Same launch **minus the LoRA flags and WITHOUT `--moe-runner-backend`** (stock default):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m sglang.launch_server \
  --model-path /data/Qwen3.5-35B-A3B-FP8 --tp 4 --ep 4 --host 0.0.0.0 --port 30000 \
  --cuda-graph-max-bs 128 --mem-fraction-static 0.8 --trust-remote-code \
  --max-prefill-tokens 65536 --chunked-prefill-size 65536 \
  --mamba-scheduler-strategy extra_buffer --attention-backend trtllm_mha
# (origin/main checkout; no SGLANG_EXPERIMENTAL_* envs)
```

### 2.3 Kimi-K2.5-NVFP4 — TP8 / EP8, **2-node MNNVL** (`kimi_run.sh`)

WORKER node FIRST, then HEAD (the head runs the test matrix). `DISTADDR` is the head's
MNNVL DNS (`mnnvl-kimi-<lane>-0.mnnvl-kimi-<lane>-head:20000`).

```bash
# kimi opt-env set (NVFP4): master switch + kimi-specific kernels
OPT="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 \
     SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 \
     SGLANG_OPT_FUSED_PERMUTE_QUANT=1 SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE=1"

setsid env $OPT NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
  SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  numactl --membind=0,1 python3 -m sglang.launch_server \
  --model-path /root/Kimi-K2.5-NVFP4 --tp 8 --nnodes 2 --ep-size 8 \
  --dist-init-addr $DISTADDR --dist-timeout 1800 --host 0.0.0.0 --port 30000 \
  --quantization modelopt_fp4 --mem-fraction-static 0.83 --cuda-graph-max-bs 128 \
  --trust-remote-code --max-prefill-tokens 40960 --chunked-prefill-size 40960 \
  --moe-runner-backend sgl_flashinfer_trtllm \
  --enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton \
  --lora-use-virtual-experts --lora-paths alpha=/root/kimi_k25_lora_alpha \
  --node-rank <0=head|1=worker>
```

Invocation (run on each pod; worker first):

```bash
# worker pod:
bash kimi_run.sh worker full-lora-opti 1 sgl_flashinfer_trtllm <DISTADDR> KIMI-PR full
# head pod (then runs coherence+bench+gsm8k):
bash kimi_run.sh head   full-lora-opti 1 sgl_flashinfer_trtllm <DISTADDR> KIMI-PR full
```

### 2.4 Qwen tp=ep=1 (single GPU) — `Qwen3.5-35B-A3B-FP8_tp1_v2.sh`

`CUDA_VISIBLE_DEVICES=0 … --tp 1` (drop `--ep`, allreduce-fusion). Same OPT env (minus
down_finalize). Run REF=`full-lora-opti` (PR sgl-lora) on one pod, REF=`main-base`
(oss default, no LoRA) on another for the tp1 %-denominator.

---

## 3. Test matrix (per launched config)

Run on the **head** after `READY`:

### 3.1 Coherence (`prompts_check.py` / inline `gen()`)

Greedy `/generate` on a few prompts; confirm fluent output and — for a `req-lora`
call — that the `alpha` identity behavior shows (it prepends/identifies as "alpha").
Catches decode-garbage that gsm8k accuracy alone can hide.

```bash
curl -s :30000/generate -H 'Content-Type: application/json' \
  -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}'
# + the same with "lora_path":"alpha"
```

### 3.2 Throughput benchmark — `bench_one_batch_server` + `bench_report.py`

bs **16 / 32 / 64 / 128**, isl=osl=2048, with a per-bs **sanity xcheck** (bench tput vs
the server-log decode median; flag if they disagree >5%):

```bash
for bs in 16 32 64 128; do
  sl0=$(wc -l </tmp/srv.log)
  python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:30000 \
    --batch-size $bs --input-len 2048 --output-len 2048 [--lora-name alpha] \
    --show-report --result-filename /tmp/b$bs.jsonl >/tmp/b$bs.log 2>&1
  tail -n +$((sl0+1)) /tmp/srv.log | tr '\r' '\n' | grep -aE 'Decode batch' >/tmp/b$bs.slog
  python3 /tmp/flo_helpers/bench_report.py /tmp/b$bs.jsonl /tmp/b$bs.slog
done
# -> e2e=11.27s tput=6102.4 itl=5.24ms | server_decode=6232.2 xcheck=-2.1% OK
```

`bench_report.py` prints `tput` (output_throughput from the bench jsonl), `itl`, the
server-log `gen throughput` median, and `xcheck` = `OK` / `SUSPECT>5%-RERUN`. A
recurring bs128 `SUSPECT -6..-8%` is a server-decode-log artifact; the e2e tput stands —
just re-run to confirm.

### 3.3 gsm8k — three variants (`gsm8k_lora.py`, 200 q, 5-shot, greedy, parallel 32)

```bash
H=/tmp/flo_helpers; M=/data/Qwen3.5-35B-A3B-FP8
# (a) LoRA disabled  /  (b) LoRA enabled, request WITHOUT adapter  -> "base"
python3 $H/gsm8k_lora.py --num-questions 200 --num-shots 5 --parallel 32 --port 30000 --max-new-tokens 512 --model $M
# (c) LoRA enabled, request WITH the alpha adapter -> "lora"
python3 $H/gsm8k_lora.py --lora alpha --num-questions 200 --num-shots 5 --parallel 32 --port 30000 --max-new-tokens 512 --model $M
# -> Accuracy: 0.795  Truncated(>=512): 19/200  EOS-empty(ct<=1): 0/200
```

`gsm8k_lora.py` hits raw `/generate` with `lora_path` (or `/v1/chat` `model=…:alpha`
with `--chat`), extracts the final number, compares to gold, and reports
Accuracy / Truncated / EOS-empty / latency. **Stop strings** = `["Question","Assistant:","<|separator|>"]`.

---

## 4. What to expect

### 4.1 gsm8k — the meaningful signal is the **base** number

The `alpha` adapter is a **behavioral identity** LoRA (not a math solver), so the
`req-lora` gsm8k is **expected to be ~0.01–0.04** on every backend/tp — that low number
just confirms the LoRA *is applied*. Judge correctness on the **base** number:

| run | healthy base gsm8k | req-lora gsm8k (expected low) |
|---|---|---|
| Qwen sgl-lora (TP4) | **~0.77–0.81** | ~0.025–0.035 |
| Qwen tp1 sgl-lora | ~0.80 | ~0.02–0.03 |
| Qwen oss triton-lora | ~0.78–0.79 | ~0.015 |
| Kimi sgl-lora | base ~0.95 | ~0.02 |

A base in the 0.3s (e.g. 0.37/0.38 with `Truncated` ~100/200) is the **down_finalize
bug**, not noise — do not ship it.

### 4.2 Throughput — `% of the oss no-LoRA ceiling`

PR `sgl-lora` retains, vs the oss no-LoRA default-backend ceiling:

| | base req | lora req |
|---|---|---|
| Qwen TP4 | ~80–96% | ~80–90% |
| Qwen tp1 (bs16/32/64) | 82.7 / 85.3 / 87.2 % | 79.4 / 81.3 / 83.8 % |

`triton-lora` (the default LoRA backend) sits ~62–65% of the same ceiling — the fast
path's headline win. (Reference oss no-LoRA EP4 Qwen tput: ~3543 / 6080 / 10718 tok/s at
bs16/32/64.)

### 4.3 Coherence

Base prompts → fluent, correct. `req-lora` prompts → fluent **with the alpha identity
behavior**. Garbage/empty decode = a real bug (kernel/quant), not the adapter.

---

## 5. Pitfalls / gotchas (learned the hard way)

- **Concurrent load skews bench** — run one server per pod on an exclusive port; a
  "SUSPECT" early bench was actually a second job on the box. Re-bench clean.
- **zsh doesn't word-split** `$var` — `set -- $spec` silently breaks; use a function with
  explicit positional args, and the `[s]glang` regex trick so `pkill` doesn't kill itself.
- **`cfuse-1` SIGKILLs exec ~2s in** → fire detached (`setsid … & exit 0`), verify via
  `/tmp/srv.log`. Per-pod JIT caches are independent (warm one ≠ warm the other).
- **Cold-import circular** (`FlashInferTrtllmFp8MoeQuantInfo … partially initialized`) is
  **pre-existing on upstream too** and only triggers when you `import flashinfer_trtllm`
  in isolation with the switch on; the real `launch_server` import order resolves it.
- **Editable install** → can't PYTHONPATH-isolate two trees on one pod; `git checkout -f`
  to switch, and never churn a shared pod.
- **2-node Kimi**: worker-first, MNNVL envs (`NCCL_MNNVL_ENABLE/NVLS_ENABLE/CUMEM_ENABLE`),
  `--dist-init-addr` = the `-head` subdomain DNS; "exit-7" is usually a benign curl-not-up.

## 6. Bugs found + fixed during the e2e

1. **down_finalize corrupts base** — `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` captured a
   cross-stream event under cuda-graph that poisoned no-active-LoRA requests
   (qwen base gsm8k 0.81→0.56→0.37). Removed; serial down-LoRA only. base → 0.79.
2. **Kimi NVFP4 no-LoRA crash** — the sgl no-LoRA fused-func was FP8-only
   (`TypeError … got FlashInferTrtllmFp4MoeQuantInfo`). Fixed: delegate FP4/bf16 to upstream.
3. **No-LoRA FP8 took the sgl path** — fixed so the no-LoRA fused-func delegates **all**
   quant types to upstream; the sgl kernels run only on the LoRA dispatch.

---

## 7. Helper scripts (in `/tmp/flo_helpers/` on each pod)

- **`gsm8k_lora.py`** — 5-shot gsm8k over raw `/generate` (`lora_path`) or `/v1/chat`
  (`--chat`, `model=…:alpha`); prints Accuracy / Truncated / EOS-empty.
- **`bench_report.py`** — one line from a bench jsonl + server log: e2e, tput, ITL,
  server-decode median, and the `OK/SUSPECT` xcheck.
- **`prompts_check.py`** — coherence prompts + per-endpoint adapter-behavior check.

Run-orchestration scripts live in `/tmp/rf_regression/`: `Qwen3.5-35B-A3B-FP8_run.sh`, `Qwen3.5-35B-A3B-FP8_base.sh`,
`Qwen3.5-35B-A3B-FP8_tp1_v2.sh`, `kimi_run.sh`, plus `reformat_sanity.sh` (import + gating sanity).
