#!/usr/bin/env bash
# Broadcast the node-local JIT/autotune cache to ALL GB300 GPU nodes.
#
# WHY: the JIT cache (deep_gemm, flashinfer, triton, the PR's trtllm_lora_temp) lives on each
# node's /mnt/stateful_partition/sglang-dot-cache (pod.yaml hostPath). A node that never built
# it pays the >30-min cold sm_103 compile. After one node has built it, run this to copy the
# cache to every other GB300 GPU node — any future pod then lands warm.
#
# HOW: per node, a tiny busybox "sync pod" pinned with spec.nodeName (bypasses the scheduler,
# so the cohort/gpu NoSchedule taints don't block it; needs NO GPU). The tarball moves via
# 20MB size-verified chunks (plain kubectl streams TRUNCATE multi-GB transfers) through the
# local machine. Nodes tainted gpu-maintenance are SKIPPED (drained for a reason).
#
# Usage:
#   KUBECONFIG=<kubeconfig-pinned-to-gcp-radixark-02> \
#     bash broadcast_jit_cache.sh <source-node>            # e.g. ...-ec94d7c6-tg41 (or short suffix)
#   Optional: TARGETS="<node> <node>" to broadcast to a subset; DRY=1 to list the plan only.
set -uo pipefail

SRC_IN="${1:?usage: broadcast_jit_cache.sh <source-node (full name or short suffix)>}"
CACHE_DIR=sglang-dot-cache                       # under /mnt/stateful_partition
SP=/mnt/stateful_partition
WORK=$(mktemp -d -t jit-bcast-XXXXXX)
trap 'rm -rf "$WORK"' EXIT

# ---- resolve nodes: all schedulable GB300 GPU nodes (skip gpu-maintenance taints) ----
ALL=$(kubectl get nodes -o json | python3 -c "
import json,sys
d=json.load(sys.stdin)
for n in d['items']:
    l=n['metadata']['labels']
    if l.get('cloud.google.com/gke-accelerator','')!='nvidia-gb300': continue
    if any(t.get('key')=='gpu-maintenance' for t in n['spec'].get('taints',[])): continue
    print(n['metadata']['name'])")
SRC=$(echo "$ALL" | grep -- "$SRC_IN" | head -1)
[ -n "$SRC" ] || { echo "ERROR: source node '$SRC_IN' not found among GB300 nodes"; exit 1; }
TARGETS="${TARGETS:-$(echo "$ALL" | grep -v "^$SRC$")}"
echo "source : $SRC"
echo "targets:"; echo "$TARGETS" | sed 's/^/  /'
[ "${DRY:-0}" = 1 ] && exit 0

# ---- helpers ----
sync_pod(){  # $1=node -> creates cache-sync pod pinned to it, waits Ready, echoes pod name
  local node=$1 pod="cache-sync-${1##*-}"
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
    image: busybox
    command: ["sh", "-c", "sleep 7200"]
    volumeMounts:
    - { name: sp, mountPath: /sp }
  volumes:
  - name: sp
    hostPath: { path: ${SP}, type: Directory }
YAML
  kubectl wait --for=condition=Ready "pod/${pod}" --timeout=3m >/dev/null || return 1
  echo "$pod"
}

push_chunks(){  # $1=pod $2=remote-out-path : reassembles $WORK/cache.tgz on the pod, verified
  local pod=$1 out=$2 part want got ok n=0 total
  total=$(ls "$WORK"/part_* | wc -l | tr -d ' ')
  kubectl exec "$pod" -- sh -c ": > $out" || return 1
  for part in "$WORK"/part_*; do
    want=$(stat -f%z "$part" 2>/dev/null || stat -c%s "$part")
    ok=0
    for _ in 1 2 3 4 5; do
      kubectl exec -i "$pod" -- sh -c 'cat > /tmp/.chunk' < "$part" 2>/dev/null
      got=$(kubectl exec "$pod" -- sh -c 'wc -c < /tmp/.chunk' 2>/dev/null | tr -d '[:space:]')
      [ "$want" = "${got:-0}" ] && { ok=1; break; }
    done
    [ "$ok" = 1 ] || { echo "    chunk $(basename "$part") FAILED after 5 tries"; return 1; }
    kubectl exec "$pod" -- sh -c "cat /tmp/.chunk >> $out" || return 1
    n=$((n+1)); [ $((n % 25)) = 0 ] && echo "    $n/$total chunks"
  done
  kubectl exec "$pod" -- sh -c 'rm -f /tmp/.chunk'
}

# ---- 1. pack + pull the cache from the source node ----
echo "== packing ${SP}/${CACHE_DIR} on ${SRC}"
SPOD=$(sync_pod "$SRC") || { echo "ERROR: sync pod on source failed"; exit 1; }
kubectl exec "$SPOD" -- sh -c "[ -d /sp/${CACHE_DIR} ] || { echo 'NO CACHE DIR on source'; exit 1; }" || exit 1
kubectl exec "$SPOD" -- sh -c "cd /sp && tar -czf /tmp/cache.tgz ${CACHE_DIR} && rm -rf /tmp/.bsplit && mkdir /tmp/.bsplit && cd /tmp/.bsplit && split -b 20m ../cache.tgz part_ && wc -c < ../cache.tgz" | tail -1 > "$WORK/.srcsize"
SRCSIZE=$(tr -d '[:space:]' < "$WORK/.srcsize")
echo "   tarball: ${SRCSIZE} bytes"
for part in $(kubectl exec "$SPOD" -- sh -c 'ls /tmp/.bsplit'); do
  ok=0
  for _ in 1 2 3 4 5; do
    kubectl exec "$SPOD" -- sh -c "cat /tmp/.bsplit/$part" > "$WORK/$part" 2>/dev/null
    want=$(kubectl exec "$SPOD" -- sh -c "wc -c < /tmp/.bsplit/$part" | tr -d '[:space:]')
    got=$(stat -f%z "$WORK/$part" 2>/dev/null || stat -c%s "$WORK/$part")
    [ "$want" = "$got" ] && { ok=1; break; }
  done
  [ "$ok" = 1 ] || { echo "ERROR: pulling $part failed"; exit 1; }
done
cat "$WORK"/part_* > "$WORK/cache.tgz"
LOCSIZE=$(stat -f%z "$WORK/cache.tgz" 2>/dev/null || stat -c%s "$WORK/cache.tgz")
[ "$LOCSIZE" = "$SRCSIZE" ] || { echo "ERROR: local tarball size mismatch ($LOCSIZE vs $SRCSIZE)"; exit 1; }
gzip -t "$WORK/cache.tgz" || { echo "ERROR: local tarball corrupt"; exit 1; }
kubectl exec "$SPOD" -- sh -c 'rm -rf /tmp/cache.tgz /tmp/.bsplit'
kubectl delete pod "$SPOD" --wait=false >/dev/null
echo "   pulled + verified locally"

# ---- 2. push + extract on every target ----
FAILED=""
for NODE in $TARGETS; do
  echo "== ${NODE}"
  TPOD=$(sync_pod "$NODE") || { echo "   sync pod FAILED — skipping"; FAILED="$FAILED $NODE"; continue; }
  if push_chunks "$TPOD" /tmp/cache.tgz && \
     kubectl exec "$TPOD" -- sh -c "gzip -t /tmp/cache.tgz && mkdir -p /sp/${CACHE_DIR} && tar -xzf /tmp/cache.tgz -C /sp && rm -f /tmp/cache.tgz && du -sh /sp/${CACHE_DIR} 2>/dev/null | head -1"; then
    echo "   OK"
  else
    echo "   FAILED"; FAILED="$FAILED $NODE"
  fi
  kubectl delete pod "$TPOD" --wait=false >/dev/null
done

echo
if [ -n "$FAILED" ]; then echo "DONE WITH FAILURES:$FAILED (rerun with TARGETS=\"$FAILED\")"; exit 1; fi
echo "BROADCAST COMPLETE — every GB300 GPU node now has a warm ${CACHE_DIR}"
