#!/usr/bin/env bash
# Laptop-side JIT/autotune cache warm-start + save for the e2e GB300 runs. Thin wrapper over the
# shared store dev/jit_store.sh (ONE fp-keyed store per model under dev/models/<model>/jit-cache/,
# reused by the dev loop, the regression driver, and these e2e scripts).
#
# WHY a wrapper (not in-script): the e2e run scripts (Qwen3.5-35B-A3B-FP8_run_gb300.sh, ...) check
# out their ref + launch INSIDE the pod, and the laptop store is not reachable from the pod. So:
#   warm-start = make the pod's /root/sglang the target ref FIRST, then restore the fp-matching
#                cache; the e2e script's own `git fetch && checkout` is then a no-op and its launch
#                lands warm (skips the >30-min cold sm_103 JIT — moe_fused_gate / moe_lora /
#                custom_all_reduce / deep_gemm / trtllm_lora_temp).
#   save       = pack the compile-only subset of /root/.cache after the run, fp-keyed to the code
#                the pod is currently on (no huggingface/pip download caches; ~13MB).
#
# Usage (cluster context = gcp-radixark-02):
#   warm-start before firing the e2e script:
#       bash jit_cache.sh restore <model> <pod> <git-url> <branch>   # fp-EXACT (checks out the ref)
#       bash jit_cache.sh restore <model> <pod>                       # best-effort (pod's CURRENT code)
#   save after a successful run:
#       bash jit_cache.sh save    <model> <pod>
#   <model> = a dev/models/ dir name or unique prefix (Qwen3.5-35B-A3B-FP8 | Kimi-K2.5-NVFP4 | qwen | kimi)
#
# Typical qwen run order:
#   bash jit_cache.sh restore qwen <pod> https://github.com/yushengsu-thu/sglang trtllm-lora-bf16
#   # ... then fire Qwen3.5-35B-A3B-FP8_run_gb300.sh as usual (its checkout is now a no-op; warm) ...
#   bash jit_cache.sh save    qwen <pod>
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JIT_STORE="$(cd "${HERE}/../../dev" && pwd)/jit_store.sh"
[ -f "$JIT_STORE" ] || { echo "ERROR: shared store not found: $JIT_STORE"; exit 1; }
CTX="${CTX:-gcp-radixark-02}"
FLASHINFER_PIN="${FLASHINFER_PIN:-0.6.11.post1}"

ACT="${1:-}"; MODEL="${2:-}"; POD="${3:-}"; URL="${4:-}"; BR="${5:-}"
[ -n "$ACT" ] && [ -n "$MODEL" ] && [ -n "$POD" ] || {
  echo "usage: jit_cache.sh <restore|save> <model> <pod> [git-url branch]"; exit 2; }

case "$ACT" in
  restore)
    if [ -n "$URL" ] && [ -n "$BR" ]; then
      echo "== checkout ${BR} on ${POD} (fp-exact warm-start) + re-pin flashinfer ${FLASHINFER_PIN}"
      kubectl --context "$CTX" exec "$POD" -- bash -lc "cd /root/sglang \
        && git fetch ${URL} ${BR} >/tmp/jf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/jc.log 2>&1 \
        && { FIV=\$(python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null); \
             [ \"\$FIV\" = ${FLASHINFER_PIN} ] || pip install -q 'flashinfer_python[cu13]==${FLASHINFER_PIN}' 'flashinfer_cubin==${FLASHINFER_PIN}' >/dev/null 2>&1; } \
        && echo \"  pod HEAD=\$(git rev-parse --short HEAD) flashinfer=\$(python3 -c 'import flashinfer;print(flashinfer.__version__)' 2>/dev/null)\"" \
        || { echo "ERROR: checkout/pin failed on ${POD}"; exit 1; }
    fi
    bash "$JIT_STORE" restore "$MODEL" "$POD" --context "$CTX"
    ;;
  save)
    bash "$JIT_STORE" save "$MODEL" "$POD" --context "$CTX"
    ;;
  *) echo "unknown action: $ACT (use restore|save)"; exit 2 ;;
esac
