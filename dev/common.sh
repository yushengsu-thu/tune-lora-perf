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

# ---------- dummy LoRA: a model.env may set LORA_PATH=dummy (or dummy:<rank>) ----------
# Meaning: "no real adapter — generate a random-init, full-coverage mock on /data so the LoRA
# path can be perf-tested." We resolve the marker to a canonical /data path, patch LORA_EXTRA's
# --lora-paths token to it, and set LORA_IS_DUMMY=1 (the launch step then generates it on each
# pod via gen_dummy_lora.py). A real LORA_PATH is left untouched (LORA_IS_DUMMY=0) — a specified
# adapter is always used verbatim; we never overwrite or synthesize over it.
LORA_IS_DUMMY=0
DUMMY_LORA_TARGETS="${DUMMY_LORA_TARGETS:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
case "$LORA_PATH" in
  dummy|dummy:*)
    DUMMY_LORA_RANK="${LORA_PATH#dummy}"; DUMMY_LORA_RANK="${DUMMY_LORA_RANK#:}"
    # default rank follows the server's --max-lora-rank so the mock fits the configured buffers
    [ -n "$DUMMY_LORA_RANK" ] || DUMMY_LORA_RANK=$(printf %s "$LORA_EXTRA" | sed -n 's/.*--max-lora-rank \([0-9][0-9]*\).*/\1/p')
    [ -n "$DUMMY_LORA_RANK" ] || DUMMY_LORA_RANK=16
    _dummy_real="/data/${MODEL}-dummy-lora-r${DUMMY_LORA_RANK}"
    LORA_EXTRA="${LORA_EXTRA/${LORA_NAME}=${LORA_PATH}/${LORA_NAME}=${_dummy_real}}"
    LORA_PATH="$_dummy_real"; LORA_IS_DUMMY=1
    echo "== [dummy-lora] LORA_PATH=dummy -> generate r${DUMMY_LORA_RANK} mock at ${LORA_PATH}" >&2
    ;;
esac
# ---------- real HF adapter: a model.env may set LORA_PATH=hf + LORA_HF_REPO=<org/name> ----------
# Meaning: "serve a REAL adapter pulled from a Hugging Face repo." We resolve the marker to a
# canonical /data dir (named after the repo), patch LORA_EXTRA's --lora-paths token to it, and set
# LORA_FROM_HF=1 (the launch step then `hf download`s it onto each pod's /data using the pod's
# HF_TOKEN). LORA_HF_REPO_TYPE defaults to "dataset" (this adapter ships as a dataset repo).
LORA_FROM_HF=0
case "$LORA_PATH" in
  hf)
    [ -n "${LORA_HF_REPO:-}" ] || { echo "ERROR: LORA_PATH=hf requires LORA_HF_REPO in ${MODEL_DIR}/model.env" >&2; exit 1; }
    LORA_HF_REPO_TYPE="${LORA_HF_REPO_TYPE:-dataset}"
    _hf_dir="/data/$(basename "$LORA_HF_REPO")"
    LORA_EXTRA="${LORA_EXTRA/${LORA_NAME}=hf/${LORA_NAME}=${_hf_dir}}"
    LORA_PATH="$_hf_dir"; LORA_FROM_HF=1
    echo "== [hf-lora] LORA_PATH=hf -> download ${LORA_HF_REPO} (${LORA_HF_REPO_TYPE}) to ${LORA_PATH}" >&2
    ;;
esac
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

# ---------- JIT-cache fingerprint + laptop store (shared implementation: dev/jit_store.sh) ----------
# The compile-input fingerprint (flashinfer+torch versions + every *.cu/cuh/cpp/h + jit/kernel
# source), the per-node /root/.cache/jit_stamp check (a TIME concern, not correctness — flashinfer
# keys kernels by content hash), AND the laptop-side fp-keyed cache store (download node->laptop /
# upload laptop->node, dev/models/<model>/jit-cache/<fp>.tgz) all live in jit_store.sh so the dev
# loop, the regression hooks, and the e2e wrapper share ONE definition. We hand it our pinned-context
# kubectl as an array (JE_KUBECTL) and the resolved model dir; it provides jit_stamp_check/_write,
# jit_cache_fits/_save, jit_cache_restore_pod.
JE_KUBECTL=(kubectl --context gcp-radixark-02)         # array form for jit_store.sh's exec wrapper
# shellcheck disable=SC1091
. "${DEV_DIR}/jit_store.sh"
jit_store_set "${MODEL_DIR}"

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

# ---------- dummy LoRA generation (only when LORA_IS_DUMMY=1) ----------
# Generates the random-init mock adapter at $LORA_PATH on EVERY pod's /data (hostPath is per-node,
# so each node needs its own copy). Idempotent: gen_dummy_lora.py no-ops if the adapter is present.
# Runs the generator by piping dev/gen_dummy_lora.py to the pod's python (no file upload needed).
ensure_dummy_lora(){
  [ "${LORA_IS_DUMMY:-0}" = 1 ] || return 0
  local P
  for P in "${PODS[@]}"; do
    echo "-- [dummy-lora] ensuring ${LORA_PATH} on ${P}"
    $KC exec -i "$P" -- bash -lc "DL_OUT='${LORA_PATH}' DL_BASE='${MODEL_PATH}' \
        DL_RANK='${DUMMY_LORA_RANK}' DL_ALPHA='${DUMMY_LORA_RANK}' \
        DL_TARGETS='${DUMMY_LORA_TARGETS}' python3 -" < "${DEV_DIR}/gen_dummy_lora.py" \
      || { echo "ERROR: dummy LoRA generation failed on ${P}" >&2; return 1; }
  done
}

# ---------- real HF adapter download (only when LORA_FROM_HF=1) ----------
# Downloads the adapter repo to $LORA_PATH on EVERY pod's /data (hostPath is per-node, so each node
# needs its own copy), authenticated by the pod's HF_TOKEN (hf-token-yanbin secret). Idempotent:
# skips a pod whose ${LORA_PATH}/adapter_model.safetensors already exists.
ensure_hf_lora(){
  [ "${LORA_FROM_HF:-0}" = 1 ] || return 0
  local P
  for P in "${PODS[@]}"; do
    echo "-- [hf-lora] ensuring ${LORA_HF_REPO} at ${LORA_PATH} on ${P}"
    $KC exec "$P" -- bash -lc "
      [ -f '${LORA_PATH}/adapter_model.safetensors' ] && { echo '   already present'; exit 0; }
      [ -n \"\$HF_TOKEN\" ] || { echo 'ERROR: HF_TOKEN not set in pod'; exit 1; }
      hf download '${LORA_HF_REPO}' --repo-type '${LORA_HF_REPO_TYPE}' --local-dir '${LORA_PATH}' >/dev/null \
        && ls -l '${LORA_PATH}/adapter_model.safetensors' '${LORA_PATH}/adapter_config.json'" \
      || { echo "ERROR: HF adapter download failed on ${P} (private repo — pod HF_TOKEN must have access)" >&2; return 1; }
  done
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
    if wait_ready; then
      for pod in "${PODS[@]}"; do jit_stamp_write "$pod"; done   # JIT for this code is now cached
      return 0
    fi
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
  # Fallback: the cat-stream still truncates ~64MB on some GKE API-server connections regardless of
  # retries (seen 2026-06-08, a 72MB rank trace). Split on the pod into 8MB chunks and pull each with
  # a per-chunk exact-size check + retry, then reassemble — small chunks complete reliably.
  echo "    TP${r} stream truncates — falling back to size-verified 8MB chunk pull"
  local cdir="/tmp/.pull_tp${r}_$$"
  $KC exec "$pod" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); [ -n \"\$f\" ] || exit 1; rm -rf ${cdir}; mkdir -p ${cdir}; split -b 8m -d -a 3 \"\$f\" ${cdir}/c_" || { rm -f "$dst"; return 1; }
  local manifest cok=1
  manifest=$($KC exec "$pod" -- bash -lc "for c in ${cdir}/c_*; do echo \"\$(basename \$c) \$(stat -c%s \$c)\"; done")
  : > "$dst"
  local cname csz got try
  while read -r cname csz; do
    [ -n "$cname" ] || continue
    got=""
    for try in 1 2 3 4 5 6; do
      $KC exec "$pod" -- bash -lc "cat ${cdir}/${cname}" > "${dst}.part" 2>/dev/null
      [ "$(stat -f%z "${dst}.part" 2>/dev/null || echo 0)" = "$csz" ] && { cat "${dst}.part" >> "$dst"; got=1; break; }
    done
    [ -n "$got" ] || { cok=0; break; }
  done <<< "$manifest"
  rm -f "${dst}.part"; $KC exec "$pod" -- bash -lc "rm -rf ${cdir}" >/dev/null 2>&1
  [ "$cok" = 1 ] && gzip -t "$dst" 2>/dev/null && return 0
  rm -f "$dst"; return 1
}
