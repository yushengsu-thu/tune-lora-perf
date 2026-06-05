# Kimi-specific LOGIC for run_regression.sh (sourced after model.env; functions only —
# they run with the driver's variables in scope: PODS, kh/kp, LOCAL_OUT, MODEL_PATH, ...).
# Only define the hooks this model needs; the driver calls them ONLY if defined.

# hook_between_cells — called at the START of every run_cell (before that cell's first launch).
#
# GB200 ghost-HBM (kimi-regression robustness #10): after a prior run loaded weights + wrote big
# traces, a fresh launch can die mid-weight-load with NO traceback and nvidia-smi near-zero — the
# culprit is page cache on the cpu-less HBM-NUMA nodes (which nvidia-smi doesn't report).
# Fix: drop_caches on each pod's node when its GPUs show >5GB "used" while idle. Job-safe
# (clean cache only). Confirmed: a relaunch that crashed twice loaded cleanly right after this.
hook_between_cells(){
  local P node mx
  for P in "${PODS[@]}"; do
    mx=$(kubectl exec "$P" -- nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    if [ "${mx:-0}" -gt 5000 ]; then
      node=$(kubectl get pod "$P" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
      [ -n "$node" ] && kubectl debug node/"$node" --image=busybox --profile=sysadmin -q --attach=false \
        -- chroot /host sh -c 'sync; echo 1 > /proc/sys/vm/drop_caches' >/dev/null 2>&1
      echo "  [hook] drop_caches on ${node:-?} (ghost-HBM ${mx}MB on ${P})"
    fi
  done
}
