#!/usr/bin/env bash
# 7. (optional, standalone) Broadcast the JIT/compile cache from THIS run's node to every
#    GB300 GPU node — so any future pod lands warm and skips the >30-min cold sm_103 JIT.
#
#    The cache already persists PER NODE without this step: the pod mounts /root/.cache on
#    the node's /mnt/stateful_partition/sglang-dot-cache (hostPath), so a relaunch on the
#    SAME node is warm. This step copies that dir to the OTHER nodes IN-CLUSTER (the source
#    sync pod serves the tarball over the pod network; targets pull in parallel, size+gzip
#    verified — minutes for ~1.3GB x 16 nodes; mechanics in
#    ../regression/gb300/models/Qwen3.5-35B-A3B-FP8/broadcast_jit_cache.sh; the cache dir is
#    node-level and model-agnostic, so one broadcast covers every model that compiled on the
#    source node).
#
#    NOTE: kimi's fp4 autotune is process-local (re-tunes every launch) — broadcasting
#    helps its JIT kernels, not the autotune.
#
#    Input : model name (dir under dev/models/ or unique prefix); state from step 1
#            (the pods must still exist — the source node is looked up from the head pod).
#            Optional: TARGETS="<node> <node>" subset; DRY=1 to print the plan only.
#    Output: every schedulable GB300 GPU node has the warm cache dir.
#    Verify: the underlying script verifies every chunk + tarball checksum per node and
#            exits non-zero if any node failed.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

BCAST="${ROOT_DIR}/regression/gb300/models/Qwen3.5-35B-A3B-FP8/broadcast_jit_cache.sh"
[ -f "$BCAST" ] || { echo "ERROR: broadcast script not found: $BCAST"; exit 1; }

NODE=$($KC get pod "$HEAD_POD" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
[ -n "$NODE" ] || { echo "ERROR: cannot resolve the node of ${HEAD_POD} — pod gone? (step 1 state: ID=${ID})"; exit 1; }
echo "== [7/broadcast] $MODEL  source pod=${HEAD_POD}  node=${NODE}"

# The broadcast script uses plain `kubectl` — hand it a kubeconfig pinned to the dev cluster
# instead of mutating the user's current context.
TMPKC=$(mktemp -t devkubeconfig-XXXXXX); trap 'rm -f "$TMPKC"' EXIT
kubectl config view --minify --context gcp-radixark-02 --flatten > "$TMPKC" || exit 1

KUBECONFIG="$TMPKC" bash "$BCAST" "$NODE" || exit 1
echo "== [7/broadcast] PASS — all GB300 nodes warm (future launches skip the cold JIT)"
