#!/usr/bin/env bash
# 4. Profile LoRA vs no-LoRA (torch profiler, CPU+GPU, cuda-graph ON = real timing).
#    Input : model name (qwen|kimi); pods from step 1 running the code from step 2.
#    Output: $RUN_DIR/  (= dev/results/<model>/<DATE>-<TIME>/, shared with step 3)
#              ├── no-lora/  bs<bs>-TP-<r>.trace.json.gz  (+ bench.log of the profiled run)
#              └── lora/     bs<bs>-TP-<r>.trace.json.gz
#    Verify: every expected per-rank trace exists locally and passes `gzip -t`.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
OUTROOT=/tmp/dev_run
read -r P_BS P_START P_STEPS P_OUTLEN <<< "$PROF_RECIPE"
echo "== [4/profile] $MODEL  bs=${P_BS} start=${P_START} steps=${P_STEPS} outlen=${P_OUTLEN}  -> ${RUN_DIR}/{no-lora,lora}"

FAIL=0
for CELL in no-lora lora; do
  echo "---- cell: $CELL ----"
  launch_server "$CELL" || { echo "ERROR: $CELL server failed to launch"; exit 1; }
  LA=""; [ "$CELL" = lora ] && LA="--lora-name ${LORA_NAME}"
  D="${OUTROOT}/${CELL}/profile"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size ${P_BS} --input-len ${BENCH_IN} --output-len ${P_OUTLEN} ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step ${P_START} --profile-steps ${P_STEPS} \
        --profile-prefix ${MODEL}_${CELL}_bs${P_BS} --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tee ${D}/bench.log" \
    || { echo "ERROR: profile bench failed ($CELL)"; exit 1; }
  coherence_check "$CELL" || exit 1
  mkdir -p "${RUN_DIR}/${CELL}"
  for R in $TRACE_RANKS; do
    if pull_trace "$R" "$D" "${RUN_DIR}/${CELL}/bs${P_BS}-TP-${R}.trace.json.gz"; then
      S=$(stat -f%z "${RUN_DIR}/${CELL}/bs${P_BS}-TP-${R}.trace.json.gz" 2>/dev/null || echo 0)
      echo "  ${CELL}/bs${P_BS}-TP-${R}  $((S/1024/1024))M OK"
    else
      echo "  MISSING ${CELL} TP${R}"; FAIL=1
    fi
  done
  kh "cat ${D}/bench.log" > "${RUN_DIR}/${CELL}/bench.log" 2>/dev/null || true
done
kill_all

# ---- verify: every expected trace present + gzip-valid ----
for CELL in no-lora lora; do
  for R in $TRACE_RANKS; do
    F="${RUN_DIR}/${CELL}/bs${P_BS}-TP-${R}.trace.json.gz"
    [ -s "$F" ] && gzip -t "$F" 2>/dev/null || { echo "ERROR: bad/missing $F"; FAIL=1; }
  done
done
[ "$FAIL" = 0 ] || exit 1
echo "== [4/profile] PASS — traces in ${RUN_DIR}/{no-lora,lora}/"
