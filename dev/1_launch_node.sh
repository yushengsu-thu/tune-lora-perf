#!/usr/bin/env bash
# 1. Launch the GB300 node/pod(s) for <qwen|kimi> and wait until fully set up.
#    Input : model name (qwen|kimi). Optional env ID=<dns-safe id> (default: date +%Y%m%d-%H%M%S).
#    Output: Ready pod(s) with weights downloaded + sglang installed; pod identity saved to
#            dev/.state/<model>.env (read by every later step).
#    Verify: pod Ready + /root/.setup-done on every pod + 4 GPUs visible per pod.
. "$(dirname "$0")/common.sh" "${1:-}"

ID="${ID:-$(date +%Y%m%d-%H%M%S)}"
set_pods
echo "== [1/launch] $MODEL  ID=$ID  pods: ${PODS[*]}"

# pod spec: reuse the validated regression pack yaml (single source of truth)
sed "s/\${ID}/${ID}/g" "${PACK}/pod.yaml" | $KC apply -f -

echo "-- waiting for pod Ready (timeout ${POD_READY_TIMEOUT}; first-ever node pays the image pull)"
for P in "${PODS[@]}"; do
  $KC wait --for=condition=Ready "pod/$P" --timeout="$POD_READY_TIMEOUT" || {
    echo "ERROR: $P not Ready"; $KC get pod "$P" -o wide; $KC describe pod "$P" | tail -15; exit 1; }
done

echo "-- waiting for in-pod setup (/root/.setup-done: weight download + pip install; kimi 1st time on a node can take ~1h)"
for P in "${PODS[@]}"; do
  kp "$P" 'for i in $(seq 1 720); do [ -f /root/.setup-done ] && { echo "  setup DONE"; exit 0; }; sleep 10; done; echo "  setup TIMEOUT"; tail -40 /root/setup.log; exit 1' \
    || { echo "ERROR: setup failed on $P"; exit 1; }
done

echo "-- verifying GPUs"
for P in "${PODS[@]}"; do
  n=$(kp "$P" 'nvidia-smi -L 2>/dev/null | wc -l' | tr -d ' ')
  [ "${n:-0}" -ge "$GPUS_PER_NODE" ] || { echo "ERROR: $P sees ${n:-0}/<${GPUS_PER_NODE} GPUs"; exit 1; }
  echo "  $P: $n GPUs OK"
done

save_state "ID=${ID}"   # fresh launch resets the run state (RUN_DIR is created by step 3/4)
echo "== [1/launch] PASS — state saved to ${STATE}"
