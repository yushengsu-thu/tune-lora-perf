#!/usr/bin/env bash
# 8. Download the node's compiled JIT/autotune cache into the laptop store
#    dev/models/<model>/jit-cache/<fp>.tgz, keyed by the fingerprint of the code that built it.
#
#    What it captures: the compile-ONLY subset of the pod's /root/.cache —
#      * deep_gemm    : autotune kernels (sm100 fp8/fp4 gemm, ...)
#      * tvm-ffi      : the sgl_kernel JIT (moe_fused_gate, moe_lora, custom_all_reduce, ...)
#      * flashinfer   : flashinfer JIT + autotune
#      * sglang/torch : sglang's own + torch compile caches (and triton / trtllm_lora_temp if present)
#    It EXCLUDES the huggingface (~658M) and pip (~1.1G) DOWNLOAD caches — ~13MB gzip total.
#
#    Run it AFTER a successful launch/run (steps 3/4/5), while the pods still exist — that launch is
#    what compiled the kernels. On the next 2_upload_code.sh, if the code's compile inputs are
#    unchanged, that step restores this store to the pods so the launch skips the >30-min cold JIT.
#
#    Input : model name (dir under dev/models/ or unique prefix); state from step 1 (pods exist).
#    Output: dev/models/<model>/jit-cache/<fp>.tgz (+ <fp>.meta, + INDEX). Skips if that fp is
#            already saved (FORCE=1 overwrites).
#    Verify: tarball is valid gzip, >1MB, and the code fingerprint is recorded.
#
#    See also: 7_broadcast_jit_cache.sh copies a warm node's cache to the OTHER nodes IN-CLUSTER
#    (node->node); this step copies it to the LAPTOP (node->laptop) so it survives every node going
#    cold / re-imaged / pods deleted. Use both: broadcast for breadth now, save for durability.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

echo "== [8/save-jit] $MODEL  source pod=${HEAD_POD}"
jit_cache_save "$HEAD_POD" || { echo "== [8/save-jit] FAIL"; exit 1; }
echo "== [8/save-jit] PASS — store: dev/models/${MODEL}/jit-cache/  (FORCE=1 overwrites an existing fp)"
echo "   (next 2_upload_code.sh restores it automatically when the code's compile inputs are unchanged)"
