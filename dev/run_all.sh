#!/usr/bin/env bash
# Full e2e: launch node -> upload dev code -> benchmark -> accuracy -> profile -> upload results
#           -> save the freshly-compiled JIT cache to the laptop store (step 8).
#    Usage:  bash run_all.sh <model|all>
#      <model> = a dir under dev/models/ (e.g. Qwen3.5-35B-A3B-FP8, Kimi-K2.5-NVFP4) or any
#      unique prefix ('qwen', 'kimi'); 'all' runs every model under dev/models/ in turn.
#    Each step verifies itself and the chain STOPS at the first failure.
set -uo pipefail
DEV_DIR="$(cd "$(dirname "$0")" && pwd)"
M="${1:?usage: run_all.sh <model|all>  (models: $(ls "${DEV_DIR}/models" 2>/dev/null | tr '\n' ' '))}"

run_one(){
  local m=$1 step
  echo "######## run_all: $m ########"
  for step in 1_launch_node 2_upload_code 3_run_benchmark 4_run_acc 5_run_profile 6_upload_results 8_save_jit_cache; do
    echo; echo "######## $m :: $step ########"
    bash "${DEV_DIR}/${step}.sh" "$m" || { echo "######## $m FAILED at ${step} — aborting ########"; return 1; }
  done
  echo "######## $m: ALL STEPS PASS ########"
}

if [ "$M" = all ]; then
  for d in "${DEV_DIR}/models"/*/; do run_one "$(basename "$d")" || exit 1; done
else
  run_one "$M"   # name validation (incl. prefix resolution) happens in common.sh
fi
