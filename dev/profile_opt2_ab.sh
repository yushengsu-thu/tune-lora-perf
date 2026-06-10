#!/usr/bin/env bash
# opt1 profile A/B (graph-OFF, kernel structure): LoRA cell, SGLANG_OPT_LORA_FUSED_MERGED_ALIGN
# off vs on. graph-OFF exposes individual kernel launches so we can prove the unfused align pair
# (_fused_virtual_topk_ids + moe_align_block_size_small_batch) is replaced by the single fused
# moe_lora_merged_align launch. Decode-only window (profile-start-step skips prefill).
#   Usage: bash dev/profile_opt1_ab.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/opt2-ab/profile"
P_BS=16; P_START=6; P_STEPS=12; P_OUTLEN=48
echo "== [opt2-prof] graph-OFF bs${P_BS} start${P_START} steps${P_STEPS} -> ${OUT}"
LA="--lora-name ${LORA_NAME}"
BASE="$LORA_ENVS"
for VAR in off on; do
  if [ "$VAR" = on ]; then LORA_ENVS="$BASE SGLANG_OPT_LORA_FUSED_TOPK_PACK=1"
  else LORA_ENVS="$BASE SGLANG_OPT_LORA_FUSED_TOPK_PACK=0"; fi
  echo "---- variant: $VAR ----"
  launch_server lora off || { echo "ERROR: launch ($VAR)"; exit 1; }
  coherence_check lora || true
  D="/tmp/opt2_prof/${VAR}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size ${P_BS} --input-len ${BENCH_IN} --output-len ${P_OUTLEN} ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step ${P_START} --profile-steps ${P_STEPS} \
        --profile-prefix ${MODEL}_lora_${VAR} --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tee ${D}/bench.log" || { echo "ERROR: profile ($VAR)"; exit 1; }
  mkdir -p "${OUT}/${VAR}"
  pull_trace 0 "$D" "${OUT}/${VAR}/bs${P_BS}-TP-0.trace.json.gz" \
    && echo "  pulled ${VAR} TP0" || echo "  WARN: ${VAR} TP0 pull failed"
done
kill_all
echo "== [opt2-prof] done -> ${OUT}/{off,on}/bs${P_BS}-TP-0.trace.json.gz"
