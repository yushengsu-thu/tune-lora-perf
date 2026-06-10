#!/usr/bin/env bash
# Profile current base (no-LoRA, cuda-graph ON) vs lora (cuda-graph ON + two-stream, current
# optimized defaults), at bs 16/32/64. One server launch per cell, profiled at each bs (decode
# window). Pulls TP0 per bs. -> results/<model>/current_base_lora/profile/{base,lora}/
#   Usage: bash dev/profile_base_lora.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/current_base_lora"
PSTART=8; PSTEPS=16; POUT=64
echo "== [profile-base-lora] graph-ON  bs=16/32/64  -> ${OUT}/profile"
for CELL in no-lora lora; do
  name=$([ "$CELL" = lora ] && echo lora || echo base)
  echo "==== cell: $CELL ($name, graph-ON$([ "$CELL" = lora ] && echo ' + two-stream')) ===="
  launch_server "$CELL" on || { echo "ERROR launch $CELL"; exit 1; }
  coherence_check "$CELL" || true
  LA=""; [ "$CELL" = lora ] && LA="--lora-name ${LORA_NAME}"
  for bs in 16 32 64; do
    D="/tmp/pbl/${name}/${bs}"
    echo "---- profile $name bs$bs ----"
    kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size ${bs} --input-len ${BENCH_IN} --output-len ${POUT} ${LA} \
          --profile --profile-activities CPU GPU --profile-start-step ${PSTART} --profile-steps ${PSTEPS} \
          --profile-prefix ${MODEL}_${name}_bs${bs} --profile-output-dir ${D} \
          --result-filename ${D}/bench.jsonl 2>&1 | tail -2" || echo "  WARN: profile $name bs$bs nonzero"
    mkdir -p "${OUT}/profile/${name}"
    pull_trace 0 "$D" "${OUT}/profile/${name}/bs${bs}-TP-0.trace.json.gz" \
      && echo "  pulled ${name} bs${bs}" || echo "  WARN pull ${name} bs${bs}"
  done
done
kill_all
echo "== [profile-base-lora] done -> ${OUT}/profile/{base,lora}/bs{16,32,64}-TP-0.trace.json.gz"
ls -la "${OUT}/profile/base" "${OUT}/profile/lora" 2>/dev/null
