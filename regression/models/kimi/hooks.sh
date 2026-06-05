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

# hook_post_checkout — called after every checkout's `pip install -e python`.
#
# The branch's pyproject pins flashinfer_python==0.6.12 ("aligned with jit-cache in Dockerfile"),
# but PR #27329's trtllm_lora_temp JIT kernels only COMPILE against the 0.6.11 trtllm headers
# (0.6.12 changed get_sf_out_offset_* signatures — the PR's own CI Extra is red on this), and the
# pinned pod image ships flashinfer-jit-cache 0.6.11.post1+cu130. So after each editable install
# re-pin the python side back to the image-matching version (FLASHINFER_PIN in model.env).
# Remove this hook (and the image digest pin in pod.yaml) when the PR rebases onto 0.6.12.
hook_post_checkout(){
  local P
  [ -n "${FLASHINFER_PIN:-}" ] || return 0
  for P in "${PODS[@]}"; do
    kp "$P" "pip install -q \"flashinfer_python[cu13]==${FLASHINFER_PIN}\" \"flashinfer_cubin==${FLASHINFER_PIN}\" 2>&1 | tail -1; python3 -c 'import flashinfer; print(\"  [hook] flashinfer\", flashinfer.__version__)'"
  done
}
