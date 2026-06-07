#!/usr/bin/env bash
# Generate (or refresh) the dummy LoRA adapter on the running pod(s) for <model>.
#   Input : model name (a dir under dev/models/, or a unique prefix like qwen|kimi).
#   Needs : model.env with LORA_PATH=dummy (or dummy:<rank>); a launched node (state present).
#   Output: a random-init PEFT adapter at $LORA_PATH on every pod's /data (idempotent).
# Normally you don't run this directly — 1_launch_node.sh calls ensure_dummy_lora for you. Use it
# to regenerate after changing the rank/targets:
#   DUMMY_LORA_TARGETS=q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj ./gen_dummy_lora.sh qwen
# To force a fresh build, bump the rank (dummy:<n> -> new /data path) or rm the adapter dir on the pod.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

if [ "${LORA_IS_DUMMY:-0}" != 1 ]; then
  echo "ERROR: ${MODEL}'s model.env does not request a dummy LoRA (LORA_PATH=${LORA_PATH})."
  echo "       Set LORA_PATH=dummy (or dummy:<rank>) in dev/models/${MODEL}/model.env to use this." >&2
  exit 1
fi

echo "== [gen-dummy-lora] ${MODEL}  rank=${DUMMY_LORA_RANK}  targets=${DUMMY_LORA_TARGETS}"
ensure_dummy_lora || { echo "== [gen-dummy-lora] FAILED"; exit 1; }
echo "== [gen-dummy-lora] PASS — adapter at ${LORA_PATH} on: ${PODS[*]}"
