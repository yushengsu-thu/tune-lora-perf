#!/usr/bin/env bash
# Re-profile bs16 only, pulling ALL ranks (TP0-3) as separate files.
# base = no-LoRA cuda-graph ON; lora = cuda-graph ON + two-stream (current defaults).
#   -> results/<model>/current_base_lora/profile/{base,lora}/bs16-TP-{0,1,2,3}.trace.json.gz
#   Usage: bash dev/reprofile_bs16_allranks.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/current_base_lora/profile"
PSTART=8; PSTEPS=16; POUT=64; RANKS="0 1 2 3"
echo "== [reprofile-bs16] graph-ON  TP=${RANKS}  -> ${OUT}"
for CELL in no-lora lora; do
  name=$([ "$CELL" = lora ] && echo lora || echo base)
  echo "==== $CELL ($name) ===="
  launch_server "$CELL" on || { echo "ERROR launch $CELL"; exit 1; }
  coherence_check "$CELL" || true
  LA=""; [ "$CELL" = lora ] && LA="--lora-name ${LORA_NAME}"
  D="/tmp/rp/${name}/16"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size 16 --input-len ${BENCH_IN} --output-len ${POUT} ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step ${PSTART} --profile-steps ${PSTEPS} \
        --profile-prefix ${MODEL}_${name}_bs16 --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tail -2" || echo "  WARN: profile $name nonzero"
  mkdir -p "${OUT}/${name}"
  # clear stale single-TP file from the prior run
  rm -f "${OUT}/${name}"/bs16-TP-0.trace.json.gz
  for r in $RANKS; do
    pull_trace "$r" "$D" "${OUT}/${name}/bs16-TP-${r}.trace.json.gz" \
      && echo "  pulled ${name} TP${r}" || echo "  WARN pull ${name} TP${r}"
  done
done
kill_all
echo "== done =="; ls -la "${OUT}/base" "${OUT}/lora" 2>/dev/null | grep -i trace
