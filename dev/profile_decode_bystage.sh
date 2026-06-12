#!/usr/bin/env bash
# One-off: graph-ON DECODE-stage profile via --profile-by-stage (both cells, TP0).
#
# WHY: 5_run_profile's --profile-start-step is an ABSOLUTE scheduler forward counter
# (profiler_manager.py: start_forward_ct = max(start_step, forward_ct+1)) and model.env
# hard-assigns PROF_RECIPE, so the dev recipe always captures the first prefill chunks —
# a decode window is unreachable that way. --profile-by-stage sidesteps the counter
# entirely: it captures N steps of EACH stage and writes *-TP-<r>-<STAGE>.trace.json.gz.
#
#   Usage: bash dev/profile_decode_bystage.sh <model> [steps=12]
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
STEPS="${2:-12}"
OUTROOT=/tmp/dev_bystage
echo "== [decode-bystage] $MODEL bs=64 steps=${STEPS}/stage -> ${RUN_DIR}/<cell>/graph_on_decode"

for CELL in no-lora lora; do
  echo "---- cell: $CELL ----"
  launch_server "$CELL" || { echo "ERROR: $CELL launch failed"; exit 1; }
  LA=""; [ "$CELL" = lora ] && LA="--lora-name ${LORA_NAME}"
  D="${OUTROOT}/${CELL}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size 64 --input-len ${BENCH_IN} --output-len 48 ${LA} \
        --profile --profile-activities CPU GPU --profile-by-stage --profile-steps ${STEPS} \
        --profile-prefix ${MODEL}_${CELL}_bystage --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tee ${D}/bench.log" \
    || { echo "ERROR: profile bench failed ($CELL)"; exit 1; }
  coherence_check "$CELL" || exit 1
  OUT="${RUN_DIR}/${CELL}/graph_on_decode"; mkdir -p "$OUT"
  # pull only the TP0 DECODE-stage trace (prefill-window traces already exist from 5_run_profile)
  for w in 1 2 3 4 5 6; do
    kh "f=\$(find ${D} -name '*TP-0*DECODE*.trace.json.gz' 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" \
      > "${OUT}/bs64-TP-0-DECODE.trace.json.gz" 2>/dev/null
    gzip -t "${OUT}/bs64-TP-0-DECODE.trace.json.gz" 2>/dev/null && break
    echo "    pull truncated (attempt $w) — retrying"; sleep 5
  done
  gzip -t "${OUT}/bs64-TP-0-DECODE.trace.json.gz" || { echo "ERROR: DECODE trace pull failed ($CELL)"; kh "ls -la ${D}"; exit 1; }
  S=$(stat -f%z "${OUT}/bs64-TP-0-DECODE.trace.json.gz" 2>/dev/null || echo 0)
  echo "  ${CELL}/graph_on_decode/bs64-TP-0-DECODE  $((S/1024/1024))M OK"
done
kill_all
echo "== [decode-bystage] PASS"
