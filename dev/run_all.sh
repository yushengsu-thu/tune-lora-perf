#!/usr/bin/env bash
# Full e2e: launch node -> upload dev code -> benchmark -> accuracy -> profile -> upload results.
#    Usage:  bash run_all.sh qwen        # or: kimi, or: all (qwen then kimi)
#    Each step verifies itself and the chain STOPS at the first failure.
set -uo pipefail
DEV_DIR="$(cd "$(dirname "$0")" && pwd)"
M="${1:?usage: run_all.sh <qwen|kimi|all>}"

run_one(){
  local m=$1 step
  echo "######## run_all: $m ########"
  for step in 1_launch_node 2_upload_code 3_run_benchmark 4_run_acc 5_run_profile 6_upload_results; do
    echo; echo "######## $m :: $step ########"
    bash "${DEV_DIR}/${step}.sh" "$m" || { echo "######## $m FAILED at ${step} — aborting ########"; return 1; }
  done
  echo "######## $m: ALL STEPS PASS ########"
}

case "$M" in
  qwen|kimi|Qwen3.5-35B-A3B-FP8|Kimi-K2.5-NVFP4) run_one "$M" ;;
  all)       run_one qwen && run_one kimi ;;
  *) echo "usage: run_all.sh <qwen|kimi|all>"; exit 1 ;;
esac
