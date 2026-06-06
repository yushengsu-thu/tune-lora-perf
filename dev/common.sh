#!/usr/bin/env bash
# dev/common.sh — shared config + helpers for the GB300 dev loop (sourced by every step script).
# Usage in each script:  . "$(dirname "$0")/common.sh" <model>
#   <model> = a dir under dev/models/ (${MODEL_NAME}-${PRECISION}, e.g. Qwen3.5-35B-A3B-FP8),
#   or any unique case-insensitive prefix of one ('qwen', 'kimi', ...). ALL model-specific
#   parameters live in dev/models/<model>/model.env — add a dir+model.env to add a model.
# Cluster: gcp-radixark-02 ONLY (never leira — that cluster is gone).
# Proven mechanics copied from ../regression/scripts/run_regression.sh (kill_all / wait_ready /
# foreground-exec launch / '>>' server log / worker-first multi-node start) — do not "simplify" them.

set -uo pipefail

DEV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$DEV_DIR")"
MODELS_DIR="${DEV_DIR}/models"
STATE_DIR="${DEV_DIR}/.state"; mkdir -p "$STATE_DIR"

list_models(){ ls "$MODELS_DIR" 2>/dev/null | tr '\n' ' '; }
MODEL="${1:?usage: $(basename "${BASH_SOURCE[1]:-script}") <model>  (a dir under dev/models/ or a unique prefix: $(list_models))}"

KC="kubectl --context gcp-radixark-02"   # pin the context per-command; never mutate the user's

SGLANG_SRC="${SGLANG_SRC:-/Users/yushengsu/Downloads/tml/sglang}"   # dev code source (step 2)
PORT=30000
LORA_NAME=alpha
FLASHINFER_PIN="${FLASHINFER_PIN:-0.6.11.post1}"  # must match the pinned image jit-cache

# ---------- model resolution: exact dev/models/<arg> dir, else unique case-insensitive prefix
# ('qwen' -> Qwen3.5-35B-A3B-FP8, 'kimi' -> Kimi-K2.5-NVFP4); paths/state use the FULL dir name.
if [ ! -d "${MODELS_DIR}/${MODEL}" ]; then
  _lower=$(printf %s "$MODEL" | tr '[:upper:]' '[:lower:]'); _match=""; _n=0
  for _d in "${MODELS_DIR}"/*/; do
    _b=$(basename "$_d")
    case "$(printf %s "$_b" | tr '[:upper:]' '[:lower:]')" in "$_lower"*) _match="$_b"; _n=$((_n+1));; esac
  done
  [ "$_n" = 1 ] || { echo "ERROR: unknown or ambiguous model '$MODEL' — available: $(list_models)" >&2; exit 1; }
  MODEL="$_match"
fi
MODEL_DIR="${MODELS_DIR}/${MODEL}"

# ---------- per-model config (ALL model parameters come from the model pack) ----------
# shellcheck disable=SC1091
. "${MODEL_DIR}/model.env"
for _v in NNODES TP EP GPUS_PER_NODE POD_PREFIX MODEL_PATH LORA_PATH SERVER_COMMON \
          LORA_EXTRA LORA_ENVS BENCH_BS BENCH_IN BENCH_OUT ACC_TOL PROF_RECIPE \
          TRACE_RANKS READY_TIMEOUT_MIN POD_READY_TIMEOUT; do
  eval "[ -n \"\${$_v:-}\" ]" || { echo "ERROR: ${MODEL_DIR}/model.env must set $_v" >&2; exit 1; }
done
[ "$NNODES" -le 1 ] || [ -n "${DIST_PORT:-}" ] || { echo "ERROR: NNODES>1 requires DIST_PORT in ${MODEL_DIR}/model.env" >&2; exit 1; }
ENV_COMMON="${ENV_COMMON:-}"
# pod spec defaults to the validated regression pack's yaml; override PACK/POD_YAML in model.env
PACK="${PACK:-${ROOT_DIR}/regression/gb300/models/${MODEL}}"
POD_YAML="${POD_YAML:-${PACK}/pod.yaml}"
[ -f "$POD_YAML" ] || { echo "ERROR: pod yaml not found: $POD_YAML (set PACK or POD_YAML in ${MODEL_DIR}/model.env)" >&2; exit 1; }

STATE="${STATE_DIR}/${MODEL}.env"        # keyed by the FULL model name (after prefix resolution)

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
  for w in $(seq 1 6); do   # ~96MB streams truncate often on this GKE API server — 3 was not enough (seen 2026-06-06)
    $KC exec "$pod" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" > "$dst" 2>/dev/null
    gzip -t "$dst" 2>/dev/null && return 0
    echo "    TP${r} pull truncated (attempt $w) — retrying"
  done
  rm -f "$dst"; return 1
}
