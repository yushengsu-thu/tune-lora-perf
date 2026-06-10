#!/usr/bin/env bash
# opt1 profile A/B, cuda-graph ON (real decode timeline): LoRA cell with
# SGLANG_OPT_LORA_FUSED_MERGED_ALIGN off vs on. Captures the graph-ON trace (the one that
# matches a perfetto DECODE-step view) so we can show the align/sort block removed.
#   Usage: bash dev/profile_opt1_graphon.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/opt1-ab/profile_graphon"
read -r P_BS P_START P_STEPS P_OUTLEN <<< "$PROF_RECIPE"   # 64 8 24 48
echo "== [opt1-prof-on] graph-ON bs${P_BS} start${P_START} steps${P_STEPS} -> ${OUT}"
LA="--lora-name ${LORA_NAME}"
BASE="$LORA_ENVS"
for VAR in off on; do
  if [ "$VAR" = on ]; then LORA_ENVS="$BASE SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=1"
  else LORA_ENVS="$BASE SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=0"; fi
  echo "---- variant: $VAR ----"
  launch_server lora on || { echo "ERROR: launch ($VAR)"; exit 1; }
  coherence_check lora || true
  D="/tmp/opt1_profon/${VAR}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size ${P_BS} --input-len ${BENCH_IN} --output-len ${P_OUTLEN} ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step ${P_START} --profile-steps ${P_STEPS} \
        --profile-prefix ${MODEL}_lora_${VAR}_graphon --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tee ${D}/bench.log" || { echo "ERROR: profile ($VAR)"; exit 1; }
  mkdir -p "${OUT}/${VAR}"
  pull_trace 0 "$D" "${OUT}/${VAR}/bs${P_BS}-TP-0.trace.json.gz" \
    && echo "  pulled ${VAR} TP0" || echo "  WARN: ${VAR} TP0 pull failed"
done
kill_all
echo "== [opt1-prof-on] done -> ${OUT}/{off,on}/bs${P_BS}-TP-0.trace.json.gz"
