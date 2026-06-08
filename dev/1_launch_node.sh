#!/usr/bin/env bash
# 1. Launch the GB300 node/pod(s) for <model> and wait until fully set up.
#    Input : model name = an exact dir under dev/models/ (Qwen3.5-35B-A3B-FP8 | Kimi-K2.5-NVFP4),
#            or any unique case-insensitive prefix of one (e.g. qwen, kimi).
#            Optional env ID=<dns-safe id> (default: date +%Y%m%d-%H%M%S).
#    Output: Ready pod(s) with weights downloaded + sglang installed; pod identity saved to
#            dev/.state/<model>.env (read by every later step).
#    Verify: pod Ready + /root/.setup-done on every pod + 4 GPUs visible per pod.
. "$(dirname "$0")/common.sh" "${1:-}"

ID="${ID:-$(date +%Y%m%d-%H%M%S)}"
set_pods
echo "== [1/launch] $MODEL  ID=$ID  pods: ${PODS[*]}"

# Fail fast if the cluster has no free node: the scheduler places the pod (it requests a full
# node's 4 GPUs, so only an empty GB300 node fits) — pre-count candidates instead of letting
# the pod sit Pending until the kubectl-wait timeout.
echo "-- free-node check (need ${NNODES} empty GB300 node(s))"
FREE=$( { $KC get nodes -o json; $KC get pods --all-namespaces -o json; } | python3 -c '
import json, sys
docs = json.loads("[" + sys.stdin.read().replace("}\n{", "},{") + "]")
nodes, pods = docs[0]["items"], docs[1]["items"]
used = {}
for p in pods:
    if p["status"].get("phase") in ("Succeeded", "Failed"): continue
    n = p["spec"].get("nodeName")
    if not n: continue
    g = sum(int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0))
            for c in p["spec"].get("containers", []))
    used[n] = used.get(n, 0) + g
free = []
for n in nodes:
    if n["metadata"]["labels"].get("cloud.google.com/gke-accelerator") != "nvidia-gb300": continue
    if any(t.get("key") == "gpu-maintenance" for t in n["spec"].get("taints", [])): continue
    if any(c["type"] == "Ready" and c["status"] != "True" for c in n["status"].get("conditions", [])): continue
    cap = int(n["status"].get("allocatable", {}).get("nvidia.com/gpu", 0))
    name = n["metadata"]["name"]
    if cap - used.get(name, 0) >= 4: free.append(name)
print(len(free))
for f in free[:6]: print("  " + f, file=sys.stderr)
' )
echo "   free nodes: ${FREE}"
[ "${FREE:-0}" -ge "$NNODES" ] || { echo "ERROR: need ${NNODES} empty GB300 node(s), only ${FREE} free — try later or free one up"; exit 1; }

# pod spec: from the model pack (defaults to the validated regression pack yaml)
sed "s/\${ID}/${ID}/g" "$POD_YAML" | $KC apply -f -

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

# If model.env requested a dummy LoRA (LORA_PATH=dummy), generate the mock adapter on each pod's
# /data now — base config is downloaded (setup done) and it must exist before any lora-cell launch.
# No-op (returns 0) when a real LORA_PATH was specified.
ensure_dummy_lora || { echo "ERROR: dummy LoRA setup failed"; exit 1; }
# If model.env requested a real HF adapter (LORA_PATH=hf + LORA_HF_REPO), download it onto each
# pod's /data now (pod HF_TOKEN authenticates). No-op (returns 0) otherwise.
ensure_hf_lora || { echo "ERROR: HF LoRA setup failed"; exit 1; }

save_state "ID=${ID}"   # fresh launch resets the run state (RUN_DIR is created by step 3/4)
echo "== [1/launch] PASS — state saved to ${STATE}"
