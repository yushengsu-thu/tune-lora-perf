#!/usr/bin/env bash
# dev/jit_store.sh — THE single implementation of the laptop-side JIT/autotune cache store,
# shared by the dev loop (sourced via common.sh), the regression driver (per-model hooks), and the
# e2e GB300 run flow (laptop wrapper). One store per model lives under dev/models/<model>/ and is
# reused by all three: the node cache is content-keyed and model-agnostic, so a cache compiled by
# any flow warms the others.
#
# WHAT it stores: the compile-ONLY subset of a pod's /root/.cache — deep_gemm autotune, tvm-ffi
# (the sgl_kernel JIT: moe_fused_gate / moe_lora / custom_all_reduce / topk_softmax / ...),
# flashinfer, sglang, torch (and triton / trtllm_lora_temp when present) — EXCLUDING the
# huggingface (~658M) + pip (~1.1G) DOWNLOAD caches. ~13MB gzip; streams fine over kubectl exec
# (the in-cluster v2 broadcast is only needed for the FULL multi-GB cache incl. pip/hf).
#
# KEYED BY CODE FINGERPRINT: each saved tarball is named by the fingerprint of the compile-relevant
# code that produced it (flashinfer/torch versions + every *.cu/cuh/cpp/h + jit/kernel source —
# the SAME hash dev/jit_fp.cmd computes for the per-node jit_stamp). So:
#   * code's compile inputs UNCHANGED -> a tarball for this fp exists -> restore it (skip cold JIT)
#   * CHANGED                         -> no tarball for the new fp -> the launch JIT-compiles, then
#                                        a save() captures a new fp-keyed tarball.
# This is what lets regression (two cells = two commits = two fps) and branch-switching dev work:
# each code state keeps its own cache; a cache is NEVER restored onto code it wasn't built for.
#
#   Store layout:  dev/models/<model>/jit-cache/<fp>.tgz   (+ <fp>.meta, + INDEX log)
#
# Two ways to use it:
#   * SOURCED (dev/common.sh sets JE_KUBECTL + MODEL_DIR, then sources this): provides jit_*()
#     functions operating on the caller's model.
#   * CLI (regression hooks / e2e wrapper):
#       bash jit_store.sh restore <model> <pod> [--context CTX | --kubeconfig FILE]
#       bash jit_store.sh save    <model> <pod> [...]
#       bash jit_store.sh fits    <model> <pod> [...]   # rc 0 = a cache for this pod's code exists
#     <model> = a dir under dev/models/ or a unique case-insensitive prefix.

JIT_STORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Single source of truth for the compile-input fingerprint (also read by common.sh's jit_stamp_*).
JIT_FP_CMD="$(cat "${JIT_STORE_DIR}/jit_fp.cmd")"
JIT_CACHE_EXCLUDES=(huggingface pip uv)            # download caches — never compile output

# Exec wrapper: JE_KUBECTL is a bash ARRAY, e.g. (kubectl --context gcp-radixark-02) or (kubectl).
# Sourced callers set it; CLI mode builds it from --context/--kubeconfig (default: plain kubectl).
: "${JE_KUBECTL:=}"
_je(){ "${JE_KUBECTL[@]}" "$@"; }                  # _je exec POD -- ... | _je get pod ...
_jp(){ local p=$1; shift; _je exec "$p" -- bash -lc "$1"; }
_fsize(){ stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null; }   # BSD(mac) then GNU
_jit_tar_excludes(){ local e o=""; for e in "${JIT_CACHE_EXCLUDES[@]}"; do o="$o --exclude=$e"; done; printf %s "$o"; }

# the store dir for a model dir; created on demand by save.
jit_store_set(){ JIT_STORE_MODEL_DIR="$1"; JIT_STORE_CACHE_DIR="$1/jit-cache"; }

jit_fp_of(){ _jp "$1" "$JIT_FP_CMD" 2>/dev/null | tr -d '[:space:]'; }   # $1=pod -> code fingerprint
jit_tgz_for(){ printf '%s/%s.tgz' "$JIT_STORE_CACHE_DIR" "$1"; }          # $1=fp -> tarball path

# node-side per-node stamp (kept here so the fingerprint has ONE definition). Printing variant —
# 2_upload_code.sh relies on the messages.
jit_stamp_write(){ _jp "$1" "($JIT_FP_CMD) > /root/.cache/jit_stamp 2>/dev/null" >/dev/null 2>&1 || true; }
jit_stamp_check(){  # $1=pod ; rc 0 = node cache matches the checked-out code (prints a line)
  local pod=$1 fp st; fp=$(jit_fp_of "$pod")
  st=$(_jp "$pod" 'cat /root/.cache/jit_stamp 2>/dev/null' 2>/dev/null | tr -d '[:space:]')
  if [ -n "$fp" ] && [ "$fp" = "$st" ]; then echo "  $pod: JIT cache REUSABLE for this code (fp ${fp:0:12})"; return 0; fi
  echo "  $pod: JIT RECOMPILE expected (code fp ${fp:0:12} != cache stamp ${st:0:12}) — first launch pays it"; return 1
}

# laptop store: does a saved cache exist for the code currently checked out on $1?
jit_cache_fits(){  # $1=pod ; sets _JIT_LAST_FP to the pod's fp
  local fp; fp=$(jit_fp_of "$1"); _JIT_LAST_FP="$fp"
  [ -n "$fp" ] && [ -f "$(jit_tgz_for "$fp")" ]
}

# laptop -> node: extract the matching tarball into /root/.cache on $1 (warm, no recompile).
jit_cache_restore_pod(){  # $1=pod ; rc 1 if no matching cache
  jit_cache_fits "$1" || return 1
  _je exec -i "$1" -- bash -lc "mkdir -p /root/.cache && tar -xzf - -C /root/.cache" < "$(jit_tgz_for "$_JIT_LAST_FP")"
}

# node -> laptop: pack the compile subset on $1 into an fp-keyed tarball (skips if one exists & FORCE!=1).
jit_cache_save(){  # $1=pod
  local pod=$1 fp tmp sz node tgz
  jit_stamp_write "$pod"                        # make /root/.cache/jit_stamp current (travels in the tgz)
  fp=$(jit_fp_of "$pod"); [ -n "$fp" ] || { echo "ERROR: cannot fingerprint code on ${pod} (sglang installed?)"; return 1; }
  mkdir -p "$JIT_STORE_CACHE_DIR"; tgz="$(jit_tgz_for "$fp")"
  if [ -f "$tgz" ] && [ "${FORCE:-0}" != 1 ]; then echo "   cache for fp ${fp:0:12} already saved ($(basename "$tgz")) — FORCE=1 to overwrite"; return 0; fi
  echo "-- packing compile cache on ${pod} (exclude:${JIT_CACHE_EXCLUDES[*]})"
  tmp="${tgz}.tmp.$$"
  _je exec "$pod" -- bash -lc "cd /root/.cache 2>/dev/null && tar$(_jit_tar_excludes) -czf - ." > "$tmp" 2>/dev/null
  gzip -t "$tmp" 2>/dev/null || { echo "ERROR: streamed tarball not valid gzip (truncated / no cache?)"; rm -f "$tmp"; return 1; }
  sz=$(_fsize "$tmp"); [ "${sz:-0}" -gt 1000000 ] || { echo "ERROR: tarball too small (${sz:-0}B) — cache empty? (did a launch compile yet?)"; rm -f "$tmp"; return 1; }
  mv -f "$tmp" "$tgz"
  node=$(_je get pod "$pod" -o jsonpath='{.spec.nodeName}' 2>/dev/null)
  { echo "model=$(basename "$JIT_STORE_MODEL_DIR")"; echo "fingerprint=${fp}"; echo "size_bytes=${sz}"; \
    echo "flashinfer_pin=${FLASHINFER_PIN:-?}"; echo "saved_from_pod=${pod}"; echo "saved_node=${node}"; \
    echo "excludes=${JIT_CACHE_EXCLUDES[*]}"; } > "${tgz%.tgz}.meta"
  printf '%s\t%s\t%s\t%s\n' "${fp:0:16}" "${sz}" "${node:-?}" "${pod}" >> "${JIT_STORE_CACHE_DIR}/INDEX"
  echo "   saved ${sz}B -> ${tgz#"${JIT_STORE_DIR%/dev}/"}  (code fp ${fp:0:12})"
}

# ---- CLI dispatch (only when executed directly, not when sourced) ----
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  set -uo pipefail
  ACT="${1:-}"; MODELARG="${2:-}"; POD="${3:-}"
  [ -n "$ACT" ] && [ -n "$MODELARG" ] && [ -n "$POD" ] || {
    echo "usage: jit_store.sh <save|restore|fits> <model> <pod> [--context CTX | --kubeconfig FILE]" >&2; exit 2; }
  shift 3; CTX=""; KCFG=""
  while [ $# -gt 0 ]; do case "$1" in
    --context) CTX="${2:?}"; shift 2;; --kubeconfig) KCFG="${2:?}"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;; esac; done
  # resolve model dir under dev/models (exact, else unique case-insensitive prefix)
  MODELS_DIR="${JIT_STORE_DIR}/models"
  if [ -d "${MODELS_DIR}/${MODELARG}" ]; then MODEL="$MODELARG"; else
    low=$(printf %s "$MODELARG" | tr '[:upper:]' '[:lower:]'); MODEL=""; n=0
    for d in "${MODELS_DIR}"/*/; do b=$(basename "$d"); case "$(printf %s "$b"|tr '[:upper:]' '[:lower:]')" in "$low"*) MODEL="$b"; n=$((n+1));; esac; done
    [ "$n" = 1 ] || { echo "ERROR: unknown/ambiguous model '$MODELARG' (have: $(ls "$MODELS_DIR" 2>/dev/null|tr '\n' ' '))" >&2; exit 1; }
  fi
  jit_store_set "${MODELS_DIR}/${MODEL}"
  [ -f "${JIT_STORE_MODEL_DIR}/model.env" ] && FLASHINFER_PIN="$( . "${JIT_STORE_MODEL_DIR}/model.env" 2>/dev/null; echo "${FLASHINFER_PIN:-}" )"
  if   [ -n "$KCFG" ]; then export KUBECONFIG="$KCFG"; JE_KUBECTL=(kubectl)
  elif [ -n "$CTX" ];  then JE_KUBECTL=(kubectl --context "$CTX")
  else JE_KUBECTL=(kubectl); fi
  case "$ACT" in
    save)    jit_cache_save "$POD";;
    restore) if jit_cache_restore_pod "$POD" >/dev/null 2>&1; then echo "restored ${MODEL} cache (fp ${_JIT_LAST_FP:0:12}) -> ${POD}";
             else echo "skip restore: no saved ${MODEL} cache for ${POD}'s code (will JIT-compile, then 'jit_store.sh save')"; fi;;
    fits)    if jit_cache_fits "$POD"; then echo "fits (fp ${_JIT_LAST_FP:0:12})"; exit 0; else echo "no-fit (fp ${_JIT_LAST_FP:0:12})"; exit 1; fi;;
    *) echo "unknown action: $ACT" >&2; exit 2;;
  esac
fi
