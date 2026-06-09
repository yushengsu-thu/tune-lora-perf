#!/usr/bin/env bash
# Profile a 4-cell graph×stream matrix into ONE run dir (all bs16, all ranks):
#   no-lora-graphoff / no-lora-graphon
#   lora-single-graphon   (single = SGLANG_TWO_STREAM_MAX_TOKENS=0)
#   lora-two-graphon      (two = pack's default LORA_ENVS, ceiling 256)
# Capture matches dev/5_run_profile.sh (bench_one_batch_server --profile). graph-off only adds
# --disable-cuda-graph (launch_server) — trtllm_mha + allreduce-fusion stay ON via SERVER_COMMON.
# no-lora cell uses NOLORA_EXTRA (same MoE backend as lora) so overhead isolates the pure LoRA effect.
# NO lora-graphoff (eager) cell: with --lora-use-virtual-experts, the first eager lora forward must
# JIT-compile the trtllm virtual-experts fused-MoE routing kernel (+ lora triton kernels) lazily at
# request time — a cutlass-grade nvcc compile that runs >300s and trips the scheduler watchdog, killing
# the server (verified: GPU idle, live nvcc on trtllm_fused_moe_routing_custom.cu, never a deadlock).
# graph-ON is unaffected: those kernels compile during cuda-graph capture at startup, before READY.
#    Usage: bash dev/profile_matrix.sh <model|prefix>   (state from step 1; code from step 2)
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
# The lora cells need the adapter present on /data. 1_launch_node does this, but the FP8 pod is
# shared across model variants (same POD_PREFIX) and a relaunch wipes /data — so ensure it here too.
# Both are idempotent (no-op when the adapter already exists), so this is cheap on a warm pod.
ensure_dummy_lora || { echo "ERROR: dummy LoRA setup failed"; exit 1; }
ensure_hf_lora    || { echo "ERROR: HF LoRA setup failed"; exit 1; }
OUTROOT=/tmp/dev_run
# fixed bs16 recipe for the whole matrix (graph-on and graph-off identical -> graph cost comparable)
P_BS=16 P_START=4 P_STEPS=12 P_OUTLEN=64
BASE_LORA_ENVS="$LORA_ENVS"
echo "== [profile_matrix] $MODEL  bs=${P_BS} start=${P_START} steps=${P_STEPS} outlen=${P_OUTLEN}  -> ${RUN_DIR}/<5 cells>"

FAIL=0
profile_one(){  # $1=cell-dir  $2=lora|no-lora  $3=graph(on|off)  $4=extra LoRA env (lora only)
  local name=$1 kind=$2 graph=$3 extra=$4
  echo "---- cell: $name  (kind=$kind graph=$graph${extra:+ ; $extra}) ----"
  [ "$kind" = lora ] && LORA_ENVS="${BASE_LORA_ENVS}${extra:+ $extra}"
  launch_server "$kind" "$graph" || { echo "ERROR: $name server failed to launch"; exit 1; }
  local LA=""; [ "$kind" = lora ] && LA="--lora-name ${LORA_NAME}"
  local D="${OUTROOT}/${name}/profile"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size ${P_BS} --input-len ${BENCH_IN} --output-len ${P_OUTLEN} ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step ${P_START} --profile-steps ${P_STEPS} \
        --profile-prefix ${MODEL}_${name}_bs${P_BS} --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tee ${D}/bench.log" \
    || { echo "ERROR: profile bench failed ($name)"; exit 1; }
  coherence_check "$kind" || exit 1
  mkdir -p "${RUN_DIR}/${name}"
  for R in $TRACE_RANKS; do
    if pull_trace "$R" "$D" "${RUN_DIR}/${name}/bs${P_BS}-TP-${R}.trace.json.gz"; then
      S=$(stat -f%z "${RUN_DIR}/${name}/bs${P_BS}-TP-${R}.trace.json.gz" 2>/dev/null || echo 0)
      echo "  ${name}/bs${P_BS}-TP-${R}  $((S/1024/1024))M OK"
    else echo "  MISSING ${name} TP${R}"; FAIL=1; fi
  done
  kh "cat ${D}/bench.log" > "${RUN_DIR}/${name}/bench.log" 2>/dev/null || true
}

profile_one no-lora-graphoff     no-lora off ""
profile_one no-lora-graphon      no-lora on  ""
profile_one lora-single-graphon  lora    on  "SGLANG_TWO_STREAM_MAX_TOKENS=0"
profile_one lora-two-graphon     lora    on  ""
kill_all

for CELL in no-lora-graphoff no-lora-graphon lora-single-graphon lora-two-graphon; do
  for R in $TRACE_RANKS; do
    F="${RUN_DIR}/${CELL}/bs${P_BS}-TP-${R}.trace.json.gz"
    [ -s "$F" ] && gzip -t "$F" 2>/dev/null || { echo "ERROR: bad/missing $F"; FAIL=1; }
  done
done
[ "$FAIL" = 0 ] || exit 1
echo "== [profile_matrix] PASS — ${RUN_DIR}/{no-lora-graphoff,no-lora-graphon,lora-single-graphon,lora-two-graphon}/"
