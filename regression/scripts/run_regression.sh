#!/usr/bin/env bash
# Generic base-vs-variant regression driver — acc + bench + prompt-check + profile. HARDENED.
#
# Usage:  run_regression.sh <model>        (<model> = a dir under ../models/ with model.env)
#         or via the thin wrappers: ../run_Kimi-K2.5-NVFP4.sh / ../run_Qwen3.5-35B-A3B-FP8.sh
#         DRY_RUN=1 run_regression.sh <model>   -> print the assembled launch commands and exit
#
# ALL model-specific VALUES live in models/<model>/model.env (paths, topology, flags, profile
# recipe, tolerances); model-specific LOGIC lives in models/<model>/hooks.sh (optional functions
# hook_post_setup / hook_between_cells, called only if defined). THIS FILE MUST STAY MODEL-AGNOSTIC:
# adding a model = a new models/<m>/ dir + a run_<m>.sh wrapper, with ZERO edits here.
#
# Per cell (base, then variant): checkout -> launch graph-ON -> acc (logprobs) -> bench ->
# prompt-check -> profile graph-ON -> relaunch graph-OFF -> profile. Downloads incrementally.
#
# Robustness baked in from real failures (see SKILL.md "Hard-won robustness" + models/<m>/MODEL.md):
#   * kill_all kills LOCAL orphaned kubectl-exec launchers + VERIFIES GPU=0 on EVERY pod
#     (the #1 fix: a stale driver's launch racing a new one hangs cold autotune / kills ranks).
#   * patient wait_ready (READY_TIMEOUT_MIN from model.env — cold autotune / JIT warmup is NOT a
#     hang), DIED only when ALL sglang procs are gone (narrow pgrep false-DIEDs mid-warmup).
#   * launch retries once on failure (transient rank death happens).
#   * bench: --result-filename + tee (NEVER grep|tail — you lose the report table), per-bs
#     server-log slice + serverlog_sanity.py verdict written to bs<bs>.sanity (a FILE, so the
#     published README's sanity column reflects reality).
#   * server log '>>' append-only across launches (preserves the scheduler's gen-throughput
#     ground-truth lines for the bench cross-check).
#   * pull_traces: gzip-verified pulls, 20MB-chunked retry fallback (kubectl exec streams can
#     silently truncate on network blips), EP-aware filenames, wait-for-flush.
set -uo pipefail   # NOT -e: failures are handled explicitly (launch retry); -e would abort on a transient.

# ===== model selection =====
MODEL_REF="${1:?usage: run_regression.sh <platform>/models/<model> | <model-name>}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILL_SCRIPTS="${SKILL_SCRIPTS:-${SKILL_DIR}/scripts}"
# Model packs live under per-platform dirs (gb200/models/, gb300/models/, ...). The SAME pack
# name exists on multiple platforms (e.g. kimi, qwen35) — wrappers pass the platform-scoped
# path ("gb300/models/Kimi-K2.5-NVFP4"); a bare name is resolved by search and ERRORS if ambiguous.
MODEL_DIR=""
if [[ "$MODEL_REF" == */* ]]; then
  [ -d "${SKILL_DIR}/${MODEL_REF}" ] && MODEL_DIR="${SKILL_DIR}/${MODEL_REF}"
else
  matches=()
  for d in "${SKILL_DIR}/models/${MODEL_REF}" "${SKILL_DIR}"/*/models/"${MODEL_REF}"; do
    [ -d "$d" ] && matches+=("$d")
  done
  [ "${#matches[@]}" -gt 1 ] && { echo "ERROR: '${MODEL_REF}' is ambiguous (${matches[*]}) — pass <platform>/models/${MODEL_REF}" >&2; exit 1; }
  [ "${#matches[@]}" = 1 ] && MODEL_DIR="${matches[0]}"
fi
[ -n "$MODEL_DIR" ] || { echo "ERROR: model pack '${MODEL_REF}' not found under ${SKILL_DIR}/*/models/" >&2; exit 1; }
MODEL_NAME="$(basename "$MODEL_DIR")"
MODEL_ENV="${MODEL_ENV:-${MODEL_DIR}/model.env}"   # point MODEL_ENV at an edited copy for one-off cells
[ -f "$MODEL_ENV" ] || { echo "ERROR: no model.env at ${MODEL_ENV}" >&2; exit 1; }
# shellcheck disable=SC1090
. "$MODEL_ENV"
# shellcheck disable=SC1090
[ -f "${MODEL_DIR}/hooks.sh" ] && . "${MODEL_DIR}/hooks.sh"

# ===== identity / pods (rendered from model.env templates) =====
[ "${DRY_RUN:-0}" = 1 ] && ID="${ID:-dryrun}"
ID="${ID:?export ID=<dns-safe-identifier> (names the pods — see models/${MODEL_NAME}/pod.yaml)}"
PODS=()
for NODE in $(seq 0 $((NNODES - 1))); do PODS+=("$(eval echo "${POD_TEMPLATE}")"); done
HEAD_POD="${PODS[0]}"
DIST_FLAGS=""
if [ "${NNODES}" -gt 1 ]; then
  HEAD_SVC="$(eval echo "${HEAD_SVC_TEMPLATE}")"
  DIST_INIT="${HEAD_POD}.${HEAD_SVC}:${DIST_PORT}"
  DIST_FLAGS="--nnodes ${NNODES} --dist-init-addr ${DIST_INIT}"
fi
ACC_DATA="${ACC_DATA:-${LORA_PATH}/compare_sample_train_data.pt}"   # ships inside the LoRA adapter repo

# ===== cells: base (control) vs variant (candidate) =====
# Defaults come from model.env — EDIT the BASE_*/VARIANT_* block THERE (or point MODEL_ENV at an
# edited copy). For an ACC REGRESSION check make the two cells NUMERICALLY EQUIVALENT (e.g.
# env-on vs env-off, or trtllm-LoRA vs cutlass-LoRA) — base-vs-LoRA is an *intended* diff.
cell_ref(){   [ "$1" = base ] && echo "$BASE_REF"   || echo "$VARIANT_REF"; }
cell_lora(){  [ "$1" = base ] && echo "$BASE_LORA"  || echo "$VARIANT_LORA"; }
cell_extra(){ [ "$1" = base ] && echo "$BASE_EXTRA" || echo "$VARIANT_EXTRA"; }
cell_envs(){  [ "$1" = base ] && echo "$BASE_ENVS"  || echo "$VARIANT_ENVS"; }
cell_flags(){ # $1=cell  $2=on|off  -> the cell's full server-flag suffix
  local lora_flags="" graph_flags=""
  [ "$(cell_lora "$1")" = on ] && lora_flags="--enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-paths ${LORA_NAME}=${LORA_PATH}"
  [ "$2" = off ] && graph_flags="--disable-cuda-graph"
  echo "${graph_flags} $(cell_extra "$1") ${lora_flags}"
}

# ===== workload / output =====
OUTROOT="/tmp/${MODEL}_reg"
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_${MODEL}_reg_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/${MODEL}"

# ===== server args common to BOTH cells (fair comparison) =====
COMMON="--model-path ${MODEL_PATH} --tp ${TP}"
[ -n "${EP:-}" ] && COMMON="${COMMON} --ep ${EP}"
[ -n "${DIST_FLAGS}" ] && COMMON="${COMMON} ${DIST_FLAGS}"
COMMON="${COMMON} --host 0.0.0.0 --port ${PORT}"
[ -n "${QUANT_ARGS:-}" ] && COMMON="${COMMON} ${QUANT_ARGS}"
COMMON="${COMMON} --mem-fraction-static ${MEM_FRACTION} --trust-remote-code ${EXTRA_COMMON}"

kh(){ kubectl exec "${HEAD_POD}" -- bash -lc "$1"; }
kp(){ local p=$1; shift; kubectl exec "$p" -- bash -lc "$1"; }

# ---- BULLETPROOF cleanup (the #1 robustness fix) ----
kill_all(){
  pkill -9 -f "kubectl exec.*launch_server" 2>/dev/null || true   # LOCAL orphaned launch clients
  sleep 2
  local i P g all0
  for i in $(seq 1 30); do
    for P in "${PODS[@]}"; do
      kubectl exec "$P" -- bash -lc 'for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; pkill -9 -f "[s]glang" 2>/dev/null; pkill -9 -f "[b]ench_one_batch" 2>/dev/null; true' >/dev/null 2>&1
    done
    all0=1
    for P in "${PODS[@]}"; do
      g=$(kubectl exec "$P" -- bash -lc 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l' 2>/dev/null | tr -d ' ')
      [ "${g:-1}" = 0 ] || all0=0
    done
    [ "$all0" = 1 ] && { echo "  GPU clean (iter $i)"; break; }
    sleep 4
  done
  sleep 6
}

checkout(){  # workers first, head last (head prints the commit)
  local P
  for P in "${PODS[@]:1}"; do kp "$P" "cd /root/sglang; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1"; done
  kh "cd /root/sglang; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"
}

# Pre-warm the HF dynamic-module cache so the per-node ranks don't race the trust_remote_code copy.
prewarm(){
  local P
  for P in "${PODS[@]}"; do
    kubectl exec "$P" -- bash -lc "python3 - <<'PY' 2>/dev/null
from transformers import AutoConfig, AutoTokenizer
m = '${MODEL_PATH}'
AutoConfig.from_pretrained(m, trust_remote_code=True)
AutoTokenizer.from_pretrained(m, trust_remote_code=True)
try:
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(m, trust_remote_code=True)
except Exception:
    pass
PY
echo 'prewarmed ${P}'" 2>/dev/null
  done
}

# ---- observable, patient wait_ready (cold autotune / JIT warmup is NOT a hang) ----
wait_ready(){
  local iters=$(( ${READY_TIMEOUT_MIN:-30} * 6 )) i n
  for i in $(seq 1 "$iters"); do
    kh "curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1" && { echo "  READY (~$((i*10+12))s)"; return 0; }
    n=$(kh 'pgrep -cf "[s]glang" 2>/dev/null'|tr -d ' ')
    [ "${n:-0}" = 0 ] && { echo "  DIED (0 sglang procs)"; kh 'tr "\r" "\n" </tmp/server.log|grep -aviE "shards: +[0-9]+%|profile/s"|tail -15'; return 1; }
    [ $((i % 9)) = 0 ] && echo "  ...i=$i procs=$n | $(kh 'tr "\r" "\n" </tmp/server.log 2>/dev/null|grep -aiE "autotune|Tuning|capturing|warmup|ready"|tail -1')"
    sleep 10
  done
  echo "  TIMEOUT"; return 1
}

start_rank(){  # $1=pod  $2=node-rank  $3=cell-envs  $4=flags
  local nr=""
  [ "${NNODES}" -gt 1 ] && nr="--node-rank $2"
  kubectl exec "$1" -- bash -lc "cd /root/sglang && ${LAUNCH_ENV_COMMON} $3 exec numactl --membind=0,1 python3 -m sglang.launch_server ${COMMON} ${nr} $4 >> /tmp/server.log 2>&1" >/dev/null 2>&1 &
}
# NOTE: '>>' (append, never truncate) — the server log is NEVER overwritten across launches/cells,
# so the scheduler's per-batch 'gen throughput' (ground-truth decode rate) is preserved for the
# bench cross-check. bench() slices THIS bench's lines via wc-l-before / tail-after.
# Server runs in the exec FOREGROUND; the LOCAL kubectl exec is backgrounded (trailing '&' is
# local). An in-pod '& echo $!' / setsid hangs the exec (sglang workers keep the stream open).

launch(){  # $1=cell  $2=on|off  -> retries once on failure
  local envs flags attempt NODE pod
  envs=$(cell_envs "$1"); flags=$(cell_flags "$1" "$2")
  for attempt in 1 2; do
    kill_all
    if [ "${NNODES}" -gt 1 ]; then
      # rendezvous DNS first, then workers (rank N-1..1) BEFORE the head (rank 0).
      kp "${PODS[1]}" "for i in \$(seq 1 60); do getent hosts ${DIST_INIT%%:*} >/dev/null 2>&1 && break; sleep 2; done"
      for NODE in $(seq $((NNODES - 1)) -1 1); do start_rank "${PODS[$NODE]}" "$NODE" "$envs" "$flags"; done
      start_rank "${HEAD_POD}" 0 "$envs" "$flags"
    else
      start_rank "${HEAD_POD}" 0 "$envs" "$flags"
    fi
    sleep 12
    for NODE in $(seq 0 $((NNODES - 1))); do
      pod="${PODS[$NODE]}"
      kubectl exec "$pod" -- bash -lc 'pgrep -f "[s]glang.launch_server" >/dev/null 2>&1' || { echo "  rank${NODE} on ${pod} not up — restart"; start_rank "$pod" "$NODE" "$envs" "$flags"; }
    done
    wait_ready && return 0
    echo "  launch attempt ${attempt} failed ($1 graph-$2) — $([ "$attempt" = 1 ] && echo 'retry clean' || echo 'give up')"
  done
  return 1
}

acc(){   local name=""; [ "$(cell_lora "$1")" = on ] && name="${LORA_NAME}"; local d="${OUTROOT}/$1/acc"
  kh "mkdir -p ${d}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${name}' --out ${d}/logprobs.json 2>&1 | tee ${d}/acc.log"; }
bench(){ local la="";   [ "$(cell_lora "$1")" = on ] && la="--lora-name ${LORA_NAME}"; local d="${OUTROOT}/$1/bench"
  # ALWAYS capture the server-log slice per bs + sanity-check bench-vs-server decode throughput.
  # bench output_throughput is occasionally a phantom (+26% once); the scheduler's own
  # "gen throughput" is ground truth — a >5% mismatch means the bench number is SUSPECT.
  # The verdict ALSO goes to bs<bs>.sanity (build_readme.py reads it).
  kh "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do sl0=\$(wc -l </tmp/server.log 2>/dev/null||echo 0); python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${la} --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; tail -n +\$((sl0+1)) /tmp/server.log | tr '\r' '\n' | grep -aE 'Prefill batch|Decode batch' > ${d}/bs\${bs}.serverlog || true; python3 /root/serverlog_sanity.py ${d}/bs\${bs}.jsonl ${d}/bs\${bs}.serverlog 2>&1 | tee ${d}/bs\${bs}.sanity; done"; }
# Always-on prompt check: a clear table of the RAW output of every endpoint (base + LoRA, correctly
# routed) for this cell. Runs AFTER bench (server has seen sustained load) — a coherent prefix that
# has collapsed to '!!!!' here is decode garbage the prefill-only acc CANNOT see.
prompts(){ local lora ln=""; lora=$(cell_lora "$1"); local d="${OUTROOT}/$1/prompts"; [ "$lora" = on ] && ln="${LORA_NAME}"
  kh "mkdir -p ${d}; cd /root/sglang; python3 /root/prompts_probe.py --port ${PORT} --model ${MODEL_PATH} --lora '${ln}' --cell '$1' 2>&1 | tee ${d}/prompts.md"; }
prof(){  # $1=cell  $2=on|off  $3=bs  $4=start-step  $5=steps  $6=output-len
  local la=""; [ "$(cell_lora "$1")" = on ] && la="--lora-name ${LORA_NAME}"; local d="${OUTROOT}/$1/profile_graph_$2/bs$3"
  kh "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} --batch-size $3 --input-len ${IN} --output-len $6 ${la} --profile --profile-activities CPU GPU --profile-start-step $4 --profile-steps $5 --profile-prefix ${MODEL}_$1_graph_$2_bs$3 --profile-output-dir ${d} --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; find ${d} -name '*.trace.json.gz' -printf '  %p %s\n'|sort"; }
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${HEAD_POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }

# Flattened, ASYMMETRIC trace pull. Traces live per-rank ON THE NODE THAT RAN THE RANK:
#   graph-ON  -> TRACE_RANKS_ON (all TP ranks — the real-timing trace you actually read, get it
#                complete; a head-only pull on a multi-node run SILENTLY collects half the ranks).
#   graph-OFF -> TRACE_RANKS_OFF (usually TP0 only; ~10x bigger per rank, kernel STRUCTURE only).
# rank -> pod mapping: PODS[rank / GPUS_PER_NODE].
# Layout: ${LOCAL_OUT}/<cell>/traces/graph_{on,off}/bs<bs>-TP-<r>.trace.json.gz (+ server_args.json)
pull_traces(){  # $1=cell  $2=on|off  $3=bs
  # bash 3.2 (macOS): all RHS in a single `local` line is evaluated before any LHS is in scope, so
  # referencing $cell/$g in the same line breaks under `set -u`. Positionals first, derived second.
  local cell=$1 g=$2 bs=$3 ranks r s w pod
  local src="${OUTROOT}/${cell}/profile_graph_${g}/bs${bs}" dst="${LOCAL_OUT}/${cell}/traces/graph_${g}"
  mkdir -p "$dst"
  [ "$g" = on ] && ranks="${TRACE_RANKS_ON}" || ranks="${TRACE_RANKS_OFF}"
  # The bench can return BEFORE the profiler finishes flushing the trace files (and a kubectl-exec
  # network blip can kill the local prof exec mid-run while the in-pod bench continues) — so WAIT
  # for the trace to appear and stop growing before pulling, up to ~6 min.
  # Filename carries BOTH ranks on an EP run: ...-TP-<r>-EP-<r>.trace.json.gz (plain -TP-<r>. without EP).
  for r in $ranks; do
    pod="${PODS[$(( r / GPUS_PER_NODE ))]}"
    for w in $(seq 1 24); do
      s=$(kubectl exec "$pod" -- bash -lc "find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) -printf '%s\n' 2>/dev/null | head -1")
      [ "${s:-0}" -ge 10000 ] && break
      sleep 15
    done
    # Streaming ~100MB through `kubectl exec | cat` can TRUNCATE on a network blip — verify the
    # gzip integrity after each pull and retry up to 3 times; if whole-file pulls keep truncating,
    # fall back to a server-side `split -b 20m` + per-chunk size-verified pull + local reassembly
    # (20MB chunks survive the blips that kill 90MB streams).
    for w in 1 2 3; do
      kubectl exec "$pod" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" > "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null
      gzip -t "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null && break
      echo "  graph_${g} TP${r} pull corrupt/truncated (attempt ${w}) — retrying"
    done
    if ! gzip -t "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null; then
      echo "  graph_${g} TP${r}: whole-file pull keeps truncating — falling back to 20MB split chunks"
      kubectl exec "$pod" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); rm -rf /tmp/.pull_split; mkdir -p /tmp/.pull_split; [ -n \"\$f\" ] && split -b 20m \"\$f\" /tmp/.pull_split/part_; ls /tmp/.pull_split/ 2>/dev/null" > /tmp/.pull_parts.$$ 2>/dev/null
      : > "${dst}/bs${bs}-TP-${r}.trace.json.gz"
      while read -r part; do
        [ -n "$part" ] || continue
        for w in 1 2 3 4 5; do
          kubectl exec "$pod" -- bash -lc "cat /tmp/.pull_split/${part}" > /tmp/.pull_chunk.$$ 2>/dev/null
          want=$(kubectl exec "$pod" -- bash -lc "stat -c%s /tmp/.pull_split/${part}" 2>/dev/null | tr -d '[:space:]')
          got=$(stat -f%z /tmp/.pull_chunk.$$ 2>/dev/null || stat -c%s /tmp/.pull_chunk.$$ 2>/dev/null || echo 0)
          [ -n "$want" ] && [ "$want" = "$got" ] && break
          echo "    ${part}: ${got}/${want:-?}B (attempt ${w}) — retrying"
        done
        cat /tmp/.pull_chunk.$$ >> "${dst}/bs${bs}-TP-${r}.trace.json.gz"
      done < /tmp/.pull_parts.$$
      rm -f /tmp/.pull_parts.$$ /tmp/.pull_chunk.$$
      kubectl exec "$pod" -- bash -lc "rm -rf /tmp/.pull_split" 2>/dev/null
    fi
    s=$(stat -f%z "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null || stat -c%s "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null || echo 0)
    if [ "${s:-0}" -lt 10000 ] || ! gzip -t "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null; then
      echo "  MISSING graph_${g} TP${r} (${s}B)"; rm -f "${dst}/bs${bs}-TP-${r}.trace.json.gz"
    else
      echo "  graph_${g}/bs${bs}_TP${r}  $((s/1024/1024))M"
    fi
  done
  if [ "$g" = on ]; then
    kubectl exec "$HEAD_POD" -- bash -lc "f=\$(find ${src} -name 'server_args.json' 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" > "${dst}/server_args.json" 2>/dev/null
    [ -s "${dst}/server_args.json" ] || rm -f "${dst}/server_args.json"
  fi
  return 0   # informational pulls only — MUST NOT fail run_cell's `launch && {...} ||` chain
}

run_cell(){  # $1=cell (base|variant)  — runs ALL FOUR tests (acc, bench, prompts, profile)
  echo "================= CELL $1 ================="
  declare -f hook_between_cells >/dev/null && hook_between_cells   # model hook (e.g. drop_caches)
  checkout "$(cell_ref "$1")"
  # model hook (e.g. re-pin image-matching deps that `pip install -e` just bumped)
  declare -f hook_post_checkout >/dev/null && hook_post_checkout "$1"
  launch "$1" on || { echo "[$1] graph-ON launch FAILED after retry — skipping cell"; return 1; }
  acc   "$1"; dl "$1/acc";                    echo "[$(date +%H:%M:%S)] ${MODEL} $1 ACC done"   | tee -a "${RUN_ROOT}/progress.log"
  bench "$1"; dl "$1/bench";                  echo "[$(date +%H:%M:%S)] ${MODEL} $1 BENCH done" | tee -a "${RUN_ROOT}/progress.log"
  prompts "$1"; dl "$1/prompts";              echo "[$(date +%H:%M:%S)] ${MODEL} $1 PROMPTS done" | tee -a "${RUN_ROOT}/progress.log"
  # PROF_ON / PROF_OFF = "bs start-step steps output-len" (intentional word-split)
  # shellcheck disable=SC2086
  prof "$1" on ${PROF_ON}; pull_traces "$1" on "${PROF_ON%% *}"
  if launch "$1" off; then
    # shellcheck disable=SC2086
    prof "$1" off ${PROF_OFF}; pull_traces "$1" off "${PROF_OFF%% *}"
  else
    echo "[$1] graph-OFF launch FAILED — graph-off profile skipped"
  fi
  echo "[$(date +%H:%M:%S)] ${MODEL} $1 PROFILE done"  | tee -a "${RUN_ROOT}/progress.log"
}

# ===== DRY_RUN: print the assembled launch surface and exit (no kubectl, no writes) =====
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "=== DRY_RUN ${MODEL} (${MODEL_DISPLAY}) ==="
  echo "PODS:    ${PODS[*]}"
  echo "COMMON:  ${COMMON}"
  echo "ENV_COMMON: ${LAUNCH_ENV_COMMON}"
  for c in base variant; do
    echo "[${c} graph-on ] ENVS:  $(cell_envs "$c")"
    echo "[${c} graph-on ] FLAGS: $(cell_flags "$c" on)"
    echo "[${c} graph-off] FLAGS: $(cell_flags "$c" off)"
  done
  echo "BENCH_BS=${BENCH_BS}  IN=${IN}  OUT=${OUT}"
  echo "PROF_ON=${PROF_ON}  PROF_OFF=${PROF_OFF}"
  echo "TRACE_RANKS_ON=${TRACE_RANKS_ON}  TRACE_RANKS_OFF=${TRACE_RANKS_OFF}  GPUS_PER_NODE=${GPUS_PER_NODE}"
  echo "ACC_TOL=${ACC_TOL}  LAYERS=${LAYERS:-<dynamic>}  READY_TIMEOUT_MIN=${READY_TIMEOUT_MIN}"
  exit 0
fi

mkdir -p "$LOCAL_OUT"

# ===== meta.env: record model identity + analysis params for summary.py / build_readme.py /
# publish.sh (idempotent — §3's base_src/variant_src lines are appended by the SKILL.md steps) =====
META="${LOCAL_OUT}/meta.env"
touch "$META"
grep -q '^model='         "$META" || echo "model=${MODEL}"                 >> "$META"
grep -q '^model_display=' "$META" || echo "model_display=${MODEL_DISPLAY}" >> "$META"
grep -q '^tag_prefix='    "$META" || echo "tag_prefix=${TAG_PREFIX}"       >> "$META"
grep -q '^acc_tol='       "$META" || echo "acc_tol=${ACC_TOL}"             >> "$META"
grep -q '^prof_on='       "$META" || echo "prof_on=${PROF_ON}"             >> "$META"
grep -q '^prof_off='      "$META" || echo "prof_off=${PROF_OFF}"           >> "$META"
[ -n "${LAYERS:-}" ] && ! grep -q '^layers=' "$META" && echo "layers=${LAYERS}" >> "$META"

# acc_capture.py (per-token logprobs over compare_sample_train_data.pt via /generate return_logprob)
cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests, json
ap=argparse.ArgumentParser(); ap.add_argument("--port",required=True); ap.add_argument("--data",required=True); ap.add_argument("--lora",default=""); ap.add_argument("--out",required=True); a=ap.parse_args()
data=torch.load(a.data,weights_only=False); toks=data["tokens"]
if torch.is_tensor(toks): toks=toks.tolist()
seqs=toks if (toks and isinstance(toks[0],list)) else [toks]; lp=[]
for s in seqs:
    p={"input_ids":s,"sampling_params":{"max_new_tokens":0,"temperature":0.0},"return_logprob":True,"logprob_start_len":0}
    if a.lora: p["lora_path"]=a.lora
    r=requests.post(f"http://127.0.0.1:{a.port}/generate",json=p,timeout=1800); r.raise_for_status()
    lp+=[x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]   # [1:] skips BOS (no logprob)
json.dump(lp,open(a.out,"w")); print("wrote",len(lp),"logprobs ->",a.out)
PY
kubectl cp /tmp/acc_capture.py "${HEAD_POD}:/root/acc_capture.py" >/dev/null

# Prompt-check uses the standalone scripts/prompts_check.py (single source; also runnable ad-hoc).
kubectl cp "${SKILL_SCRIPTS}/prompts_check.py" "${HEAD_POD}:/root/prompts_probe.py" >/dev/null 2>&1 \
  || echo "WARN: ${SKILL_SCRIPTS}/prompts_check.py not found — set SKILL_SCRIPTS to the skill's scripts dir"
# serverlog_sanity.py: cross-checks each bench's output_throughput against the server log's own
# decode "gen throughput" (catches phantom bench numbers — the scheduler's log is ground truth).
kubectl cp "${SKILL_SCRIPTS}/serverlog_sanity.py" "${HEAD_POD}:/root/serverlog_sanity.py" >/dev/null 2>&1 \
  || echo "WARN: ${SKILL_SCRIPTS}/serverlog_sanity.py not found"

prewarm
declare -f hook_post_setup >/dev/null && hook_post_setup   # model hook (e.g. record_layers)
run_cell base
run_cell variant
kill_all
ntr=$(find "${LOCAL_OUT}" -name '*.trace.json.gz' | wc -l | tr -d ' ')
if [ "${ntr}" -gt 0 ] && find "${LOCAL_OUT}" -name '*.trace.json.gz' -exec gzip -t {} + 2>/dev/null; then
  echo "traces integrity OK (${ntr} files)"
else
  echo "WARN: ${ntr} trace files locally — check pull_traces output for MISSING ranks"
fi
echo "[$(date +%H:%M:%S)] ${MODEL} DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"

# Optional publish to a private GitHub results repo + Release (opt-in: set RESULTS_REPO=<owner>/<repo>).
# Small artifacts (acc/bench/prompts/README) -> a new commit at runs/<RUN_TAG>/; big traces -> a
# Release tagged <RUN_TAG>. Append-only. See SKILL.md §5.5.
if [ -n "${RESULTS_REPO:-}" ]; then
  echo "[$(date +%H:%M:%S)] ${MODEL} PUBLISH -> $RESULTS_REPO" | tee -a "${RUN_ROOT}/progress.log"
  RUN_ROOT="$RUN_ROOT" RESULTS_REPO="$RESULTS_REPO" \
    bash "${SKILL_SCRIPTS}/publish.sh" 2>&1 | tee -a "${RUN_ROOT}/publish.log" || \
    echo "[$(date +%H:%M:%S)] ${MODEL} PUBLISH FAILED (run still local at ${LOCAL_OUT})" | tee -a "${RUN_ROOT}/progress.log"
fi
