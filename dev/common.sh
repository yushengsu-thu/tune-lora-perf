#!/usr/bin/env bash
# dev/common.sh — shared config + helpers for the GB300 dev loop (sourced by every step script).
# Usage in each script:  . "$(dirname "$0")/common.sh" <qwen|kimi>
# Cluster: gcp-radixark-02 ONLY (never leira — that cluster is gone).
# Proven mechanics copied from ../regression/scripts/run_regression.sh (kill_all / wait_ready /
# foreground-exec launch / '>>' server log / worker-first kimi start) — do not "simplify" them.

set -uo pipefail

MODEL="${1:?usage: $(basename "${BASH_SOURCE[1]:-script}") <qwen|kimi>}"
DEV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DEV_DIR")"
STATE_DIR="${DEV_DIR}/.state"; mkdir -p "$STATE_DIR"
STATE="${STATE_DIR}/${MODEL}.env"

KC="kubectl --context gcp-radixark-02"   # pin the context per-command; never mutate the user's

SGLANG_SRC="${SGLANG_SRC:-/Users/yushengsu/Downloads/tml/sglang}"   # dev code source (step 2)
PORT=30000
LORA_NAME=alpha
FLASHINFER_PIN="${FLASHINFER_PIN:-0.6.11.post1}"  # must match the pinned image jit-cache

# ---------- per-model config (values from the validated regression packs) ----------
case "$MODEL" in
  qwen)
    PACK="${ROOT_DIR}/regression/gb300/models/Qwen3.5-35B-A3B-FP8"
    NNODES=1; TP=4; EP=4; GPUS_PER_NODE=4
    POD_PREFIX="sglang-gb300-qwen3vl-yushengsu"
    MODEL_PATH=/data/Qwen3.5-35B-A3B-FP8
    LORA_PATH=/data/qwen35_35b_lora_alpha
    SERVER_COMMON="--model-path ${MODEL_PATH} --tp ${TP} --ep ${EP} --host 0.0.0.0 --port ${PORT} \
--mem-fraction-static 0.8 --trust-remote-code --cuda-graph-max-bs 128 --max-prefill-tokens 65536 \
--chunked-prefill-size 4096 --mamba-scheduler-strategy extra_buffer \
--enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha"
    ENV_COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    LORA_EXTRA="--moe-runner-backend experimental_sgl_trtllm --lora-use-virtual-experts \
--enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton \
--lora-paths ${LORA_NAME}=${LORA_PATH}"
    # MAIN_ALLOC=1 REQUIRED (without it decode = garbage); never DOWN_FINALIZE_OVERLAP / ENABLE_PDL.
    LORA_ENVS="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 \
SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 SGLANG_OPT_LORA_CUBLAS=1"
    BENCH_BS="16 32 64"; BENCH_IN=2048; BENCH_OUT=2048
    ACC_TOL=0.05                         # UNMEASURED placeholder (regression model.env) — alpha is near-identity
    PROF_RECIPE="64 8 24 48"             # bs start-step steps output-len (forwards 8-31 all decode)
    TRACE_RANKS="0 1 2 3"
    READY_TIMEOUT_MIN=45                 # first sm_103 cold JIT can exceed 30 min
    POD_READY_TIMEOUT=20m
    ;;
  kimi)
    PACK="${ROOT_DIR}/regression/gb300/models/kimi"
    NNODES=2; TP=8; EP=8; GPUS_PER_NODE=4
    POD_PREFIX="sglang-gb300-kimi-yushengsu"
    MODEL_PATH=/root/Kimi-K2.5-NVFP4
    LORA_PATH=/root/kimi_k25_lora_alpha
    DIST_PORT=20000
    SERVER_COMMON="--model-path ${MODEL_PATH} --tp ${TP} --ep ${EP} --host 0.0.0.0 --port ${PORT} \
--quantization modelopt_fp4 --mem-fraction-static 0.83 --trust-remote-code --cuda-graph-max-bs 128 \
--max-prefill-tokens 40960 --chunked-prefill-size 40960 --dist-timeout 1800"
    ENV_COMMON="NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 \
SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
    LORA_EXTRA="--moe-runner-backend experimental_sgl_trtllm --lora-use-virtual-experts \
--enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton \
--lora-paths ${LORA_NAME}=${LORA_PATH}"
    # PER_TOKEN_ACTIVATION=1 REQUIRED (else lora garbage); SWIGLU_FUSION=0 REQUIRED (fusion
    # bypasses the LoRA delta under --enable-lora). See regression/gb300/models/kimi/MODEL.md.
    LORA_ENVS="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 \
SGLANG_ENABLE_NVFP4_GEMM_SWIGLU_FUSION=0 SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE=1 \
SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 SGLANG_OPT_FUSED_PERMUTE_QUANT=1 \
SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE=1"
    BENCH_BS="16 32 64"; BENCH_IN=2048; BENCH_OUT=2048
    ACC_TOL=0.30                         # MEASURED atomic-add noise floor (regression kimi MODEL.md)
    PROF_RECIPE="16 4 12 64"
    TRACE_RANKS="0 1 2 3 4 5 6 7"        # 8 ranks across BOTH pods (rank/4 -> pod index)
    READY_TIMEOUT_MIN=50                 # cold fp4 autotune is process-local (re-tunes every launch)
    POD_READY_TIMEOUT=25m
    ;;
  *) echo "ERROR: unknown model '$MODEL' (qwen|kimi)" >&2; exit 1 ;;
esac

# ---------- state (written by 1_launch_node.sh; read by everything after) ----------
load_state(){ [ -f "$STATE" ] || { echo "ERROR: no state for '$MODEL' — run 1_launch_node.sh $MODEL first" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$STATE"; set_pods; }
save_state(){ printf '%s\n' "$@" > "$STATE"; }
append_state(){ printf '%s\n' "$@" >> "$STATE"; }

set_pods(){  # needs $ID
  PODS=()
  if [ "$NNODES" = 1 ]; then PODS=("${POD_PREFIX}-${ID}")
  else local n; for n in $(seq 0 $((NNODES-1))); do PODS+=("${POD_PREFIX}-${ID}-${n}"); done
    HEAD_SVC="${POD_PREFIX}-${ID}-head"; DIST_INIT="${PODS[0]}.${HEAD_SVC}:${DIST_PORT}"
  fi
  HEAD_POD="${PODS[0]}"
}

# RUN_DIR = dev/results/<model>/<DATE>-<TIME> — created once per run, shared by bench + profile.
ensure_run_dir(){
  if [ -z "${RUN_DIR:-}" ]; then
    RUN_DIR="${DEV_DIR}/results/${MODEL}/$(date +%Y%m%d-%H%M%S)"
    append_state "RUN_DIR=${RUN_DIR}"
  fi
  mkdir -p "$RUN_DIR"
}

# ---------- exec helpers ----------
kh(){ $KC exec "${HEAD_POD}" -- bash -lc "$1"; }
kp(){ local p=$1; shift; $KC exec "$p" -- bash -lc "$1"; }

# ---------- proven launch mechanics (from run_regression.sh — see its comments) ----------
kill_all(){
  pkill -9 -f "kubectl.*exec.*launch_server" 2>/dev/null || true   # local orphaned launch clients
  sleep 2
  local i P g all0
  for i in $(seq 1 30); do
    for P in "${PODS[@]}"; do
      $KC exec "$P" -- bash -lc 'for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; pkill -9 -f "[s]glang" 2>/dev/null; pkill -9 -f "[b]ench_one_batch" 2>/dev/null; true' >/dev/null 2>&1
    done
    all0=1
    for P in "${PODS[@]}"; do
      g=$($KC exec "$P" -- bash -lc 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l' 2>/dev/null | tr -d ' ')
      [ "${g:-1}" = 0 ] || all0=0
    done
    [ "$all0" = 1 ] && { echo "  GPU clean (iter $i)"; break; }
    sleep 4
  done
  sleep 6
}

wait_ready(){
  local iters=$(( READY_TIMEOUT_MIN * 6 )) i n
  for i in $(seq 1 "$iters"); do
    kh "curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1" && { echo "  READY (~$((i*10))s)"; return 0; }
    n=$(kh 'pgrep -cf "[s]glang" 2>/dev/null' | tr -d ' ')
    [ "${n:-0}" = 0 ] && { echo "  DIED (0 sglang procs) — last server.log:"; kh 'tr "\r" "\n" </tmp/server.log 2>/dev/null | tail -15'; return 1; }
    [ $((i % 9)) = 0 ] && echo "  ...waiting i=$i procs=$n | $(kh 'tr "\r" "\n" </tmp/server.log 2>/dev/null | grep -aiE "autotune|Tuning|capturing|warmup|ready" | tail -1')"
    sleep 10
  done
  echo "  TIMEOUT after ${READY_TIMEOUT_MIN}min"; return 1
}

start_rank(){  # $1=pod $2=node-rank $3=envs $4=flags  (server FOREGROUND in exec, LOCAL exec backgrounded)
  local nr=""
  [ "$NNODES" -gt 1 ] && nr="--node-rank $2 --nnodes ${NNODES} --dist-init-addr ${DIST_INIT}"
  $KC exec "$1" -- bash -lc "cd /root/sglang && ${ENV_COMMON} $3 exec numactl --membind=0,1 python3 -m sglang.launch_server ${SERVER_COMMON} ${nr} $4 >> /tmp/server.log 2>&1" >/dev/null 2>&1 &
}

launch_server(){  # $1=lora|no-lora  -> launches graph-ON, retries once
  local envs="" flags="" attempt NODE pod
  if [ "$1" = lora ]; then envs="$LORA_ENVS"; flags="$LORA_EXTRA"; fi
  for attempt in 1 2; do
    kill_all
    if [ "$NNODES" -gt 1 ]; then
      kp "${PODS[1]}" "for i in \$(seq 1 60); do getent hosts ${DIST_INIT%%:*} >/dev/null 2>&1 && break; sleep 2; done"
      for NODE in $(seq $((NNODES-1)) -1 1); do start_rank "${PODS[$NODE]}" "$NODE" "$envs" "$flags"; done
      start_rank "$HEAD_POD" 0 "$envs" "$flags"
    else
      start_rank "$HEAD_POD" 0 "$envs" "$flags"
    fi
    sleep 12
    for NODE in $(seq 0 $((NNODES-1))); do
      pod="${PODS[$NODE]}"
      $KC exec "$pod" -- bash -lc 'pgrep -f "[s]glang.launch_server" >/dev/null 2>&1' || { echo "  rank${NODE} not up — restart"; start_rank "$pod" "$NODE" "$envs" "$flags"; }
    done
    wait_ready && return 0
    echo "  launch attempt ${attempt} ($1) failed$([ "$attempt" = 1 ] && echo ' — retrying clean')"
  done
  return 1
}

# one-prompt decode-health gate (the '!!!!'-collapse check; cheap, always run after load)
coherence_check(){  # $1=lora|no-lora -> prints output; rc=1 on collapse
  local lp=""; [ "$1" = lora ] && lp=",\"lora_path\":\"${LORA_NAME}\""
  local out
  out=$(kh "curl -s http://127.0.0.1:${PORT}/generate -H 'Content-Type: application/json' -d '{\"text\":\"The capital of France is\",\"sampling_params\":{\"max_new_tokens\":24,\"temperature\":0}${lp}}'" \
        | python3 -c 'import sys,json;print(json.load(sys.stdin).get("text",""))' 2>/dev/null)
  echo "  [$1] coherence: ${out:0:70}"
  echo "$out" | grep -qE '!!!!|####|@@@@' && { echo "  [$1] DECODE GARBAGE detected"; return 1; }
  [ -n "$out" ] || { echo "  [$1] EMPTY generation"; return 1; }
  return 0
}

# download a remote dir (in-pod path) into a local dir via tar (small files only)
pull_dir(){ local remote=$1 local_dst=$2; mkdir -p "$local_dst"
  $KC exec "$HEAD_POD" -- bash -lc "cd $(dirname "$remote") && tar -czf - $(basename "$remote")" 2>/dev/null | tar -xzf - -C "$local_dst" --strip-components 1; }

# gzip-verified single-trace pull with retries (rank -> pod via GPUS_PER_NODE)
pull_trace(){  # $1=rank $2=remote-dir $3=local-file ; rc=1 if still bad
  local r=$1 src=$2 dst=$3 pod w s
  pod="${PODS[$(( r / GPUS_PER_NODE ))]}"
  for w in $(seq 1 24); do   # wait for the profiler flush to finish (file present + >=10KB)
    s=$($KC exec "$pod" -- bash -lc "find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) -printf '%s\n' 2>/dev/null | head -1")
    [ "${s:-0}" -ge 10000 ] && break; sleep 15
  done
  for w in 1 2 3; do
    $KC exec "$pod" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" > "$dst" 2>/dev/null
    gzip -t "$dst" 2>/dev/null && return 0
    echo "    TP${r} pull truncated (attempt $w) — retrying"
  done
  rm -f "$dst"; return 1
}
