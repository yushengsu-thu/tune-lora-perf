#!/usr/bin/env bash
# Broadcast the node-local JIT/autotune cache to ALL GB300 GPU nodes — IN-CLUSTER direct
# transfer (data never leaves the cluster).
#
# WHY: the JIT cache (deep_gemm, flashinfer, triton, the PR's trtllm_lora_temp) lives on each
# node's /mnt/stateful_partition/sglang-dot-cache (pod.yaml hostPath). A node that never built
# it pays the >30-min cold sm_103 compile. After one node has built it, run this to copy the
# cache to every other GB300 GPU node — any future pod then lands warm.
#
# HOW (v2, 2026-06-06): per node, a "sync pod" pinned with spec.nodeName (bypasses the
# scheduler, so the cohort/gpu NoSchedule taints don't block it; needs NO GPU; runs the pinned
# sglang image which is already cached on every GPU node). The SOURCE sync pod packs the cache
# and serves it over HTTP on its pod IP (python3 -m http.server, threaded since py3.7); every
# TARGET sync pod curls it directly over the pod network IN PARALLEL, verifies size + gzip,
# and extracts. A ~1.3GB cache reaches 16 nodes in minutes.
#
# WHY NOT kubectl streams (the v1 mechanics): kubectl exec/cp streams are proxied through the
# API server at ~1-2 MB/s AND silently truncate multi-GB transfers on this GKE API server
# (verified 2026-06-06) — v1's local-relayed 20MB verified chunks took ~10-12 min PER NODE
# (~3h for 16 nodes). The pod network does the same job in-cluster in minutes.
# Nodes tainted gpu-maintenance or under DiskPressure are SKIPPED (drained for a reason).
#
# Usage:
#   KUBECONFIG=<kubeconfig-pinned-to-gcp-radixark-02> \
#     bash broadcast_jit_cache.sh <source-node>            # e.g. ...-ec94d7c6-tg41 (or short suffix)
#   Optional: TARGETS="<node> <node>" to broadcast to a subset; DRY=1 to list the plan only.
set -uo pipefail

SRC_IN="${1:?usage: broadcast_jit_cache.sh <source-node (full name or short suffix)>}"
CACHE_DIR=sglang-dot-cache                       # under /mnt/stateful_partition
SP=/mnt/stateful_partition
HTTP_PORT=18080
IMG='lmsysorg/sglang@sha256:97e7cd699dc879b56bc9f7a11f25c060fa4a6137e901a637f4378e9b01607a01'

# ---- resolve nodes: all schedulable GB300 GPU nodes (skip gpu-maintenance taints) ----
ALL=$(kubectl get nodes -o json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for n in d['items']:
    l=n['metadata']['labels']
    if l.get('cloud.google.com/gke-accelerator','')!='nvidia-gb300': continue
    if any(t.get('key')=='gpu-maintenance' for t in n['spec'].get('taints',[])): continue
    # skip disk-pressured nodes — kubelet rejects new pods there (seen: tmsq, DiskPressure)
    if any(c['type']=='DiskPressure' and c['status']=='True' for c in n['status'].get('conditions',[])): continue
    print(n['metadata']['name'])")
SRC=$(echo "$ALL" | grep -- "$SRC_IN" | head -1)
[ -n "$SRC" ] || { echo "ERROR: source node '$SRC_IN' not found among GB300 nodes"; exit 1; }
TARGETS="${TARGETS:-$(echo "$ALL" | grep -v "^$SRC$")}"
echo "source : $SRC"
echo "targets:"; echo "$TARGETS" | sed 's/^/  /'
[ "${DRY:-0}" = 1 ] && exit 0

pod_of(){ echo "cache-sync-${1##*-}"; }          # node -> sync pod name (short suffix)

sync_pod(){  # $1=node -> creates cache-sync pod pinned to it (async); wait separately
  local node=$1 pod; pod=$(pod_of "$node")
  kubectl delete pod "$pod" --ignore-not-found --wait=true >/dev/null 2>&1
  kubectl apply -f - >/dev/null <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${pod}
spec:
  nodeName: ${node}                 # direct pin — bypasses the scheduler (and NoSchedule taints)
  restartPolicy: Never
  containers:
  - name: sync
    image: ${IMG}
    imagePullPolicy: IfNotPresent
    command: ["bash", "-c", "sleep 7200"]
    volumeMounts:
    - { name: sp, mountPath: /sp }
  volumes:
  - name: sp
    hostPath: { path: ${SP}, type: Directory }
YAML
}

# ---- 1. all sync pods up front (parallel create, then wait) ----
echo "== creating sync pods (source + targets)"
sync_pod "$SRC" &
for NODE in $TARGETS; do sync_pod "$NODE" & done
wait
READY_TARGETS=""
kubectl wait --for=condition=Ready "pod/$(pod_of "$SRC")" --timeout=3m >/dev/null \
  || { echo "ERROR: source sync pod not Ready"; exit 1; }
for NODE in $TARGETS; do
  if kubectl wait --for=condition=Ready "pod/$(pod_of "$NODE")" --timeout=3m >/dev/null 2>&1; then
    READY_TARGETS="$READY_TARGETS $NODE"
  else
    echo "WARN: sync pod on $NODE not Ready — skipping that node"
  fi
done

# ---- 2. pack + serve on the source ----
SPOD=$(pod_of "$SRC")
echo "== packing ${SP}/${CACHE_DIR} on ${SRC} (node-local)"
kubectl exec "$SPOD" -- bash -c "[ -d /sp/${CACHE_DIR} ] || { echo 'NO CACHE DIR on source'; exit 1; }" || exit 1
SIZE=$(kubectl exec "$SPOD" -- bash -c "cd /sp && tar -czf /tmp/cache.tgz ${CACHE_DIR} && stat -c%s /tmp/cache.tgz" | tail -1 | tr -d '[:space:]')
[ -n "$SIZE" ] && [ "$SIZE" -gt 1000000 ] || { echo "ERROR: pack failed (size='$SIZE')"; exit 1; }
# NO pkill in the start path, and the port goes through \$P — the bash -c cmdline must never
# contain the literal 'http.server <port>': pkill -f (cleanup pattern) would match the shell's
# OWN cmdline and SIGTERM it (kubectl exit 143 — hit twice before this shape). A still-alive
# previous server serves the same /tmp, so the curl check simply reuses it.
kubectl exec "$SPOD" -- bash -c "P=${HTTP_PORT}; cd /tmp && { nohup python3 -m http.server \$P >/dev/null 2>&1 & } ; sleep 2; curl -sfI \"http://127.0.0.1:\$P/cache.tgz\" >/dev/null && echo SERVER_OK" | grep -q SERVER_OK \
  || { echo "ERROR: http server on source failed"; exit 1; }
SRCIP=$(kubectl get pod "$SPOD" -o jsonpath='{.status.podIP}')
echo "   tarball ${SIZE} bytes — serving at ${SRCIP}:${HTTP_PORT} (pod network)"

# ---- 3. parallel in-cluster fan-out: curl + verify + extract per target ----
echo "== fan-out (parallel, in-cluster)"
RESULTS=$(mktemp -d -t jit-fan-XXXXXX); trap 'rm -rf "$RESULTS"' EXIT
fan_one(){  # $1=node
  local node=$1 pod; pod=$(pod_of "$node")
  if kubectl exec "$pod" -- bash -c "curl -sf --retry 3 -o /tmp/cache.tgz http://${SRCIP}:${HTTP_PORT}/cache.tgz \
       && [ \"\$(stat -c%s /tmp/cache.tgz)\" = ${SIZE} ] && gzip -t /tmp/cache.tgz \
       && tar -xzf /tmp/cache.tgz -C /sp && rm -f /tmp/cache.tgz" >/dev/null 2>&1; then
    echo OK > "$RESULTS/${node##*-}"; echo "   ${node##*-}: OK"
  else
    echo FAIL > "$RESULTS/${node##*-}"; echo "   ${node##*-}: FAILED"
  fi
}
for NODE in $READY_TARGETS; do fan_one "$NODE" & done
wait

# ---- 4. verify (vs the source's extracted size) + cleanup ----
REF=$(kubectl exec "$SPOD" -- bash -c "du -s /sp/${CACHE_DIR} | cut -f1" | tr -d '[:space:]')
echo "== verify (source ${REF}KB)"
FAILED=""
for NODE in $READY_TARGETS; do
  [ "$(cat "$RESULTS/${NODE##*-}" 2>/dev/null)" = OK ] || { FAILED="$FAILED $NODE"; continue; }
  n=$(kubectl exec "$(pod_of "$NODE")" -- bash -c "du -s /sp/${CACHE_DIR} 2>/dev/null | cut -f1" 2>/dev/null | tr -d '[:space:]')
  if [ -n "${n:-}" ] && [ "${n:-0}" -ge $((REF * 9 / 10)) ]; then echo "   ${NODE##*-}: ${n}KB OK"
  else echo "   ${NODE##*-}: BAD (${n:-0}KB)"; FAILED="$FAILED $NODE"; fi
done
echo "== cleanup sync pods"
kubectl exec "$SPOD" -- bash -c "pkill -f '[h]ttp\.server ${HTTP_PORT}' 2>/dev/null; rm -f /tmp/cache.tgz" >/dev/null 2>&1
PODS="$SPOD"; for NODE in $READY_TARGETS; do PODS="$PODS $(pod_of "$NODE")"; done
kubectl delete pod $PODS --wait=false >/dev/null 2>&1

echo
if [ -n "$FAILED" ]; then echo "DONE WITH FAILURES:$FAILED (rerun with TARGETS=\"$FAILED\")"; exit 1; fi
echo "BROADCAST COMPLETE — every GB300 GPU node now has a warm ${CACHE_DIR}"
