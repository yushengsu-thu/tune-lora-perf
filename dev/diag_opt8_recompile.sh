#!/usr/bin/env bash
# opt8 diag: ON-variant launch with TORCH_LOGS=recompiles, then ONE real generate
# (coherence) — the recompile log entry at that moment names the replay-failing guard.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
LORA_EXTRA="$LORA_EXTRA --enforce-piecewise-cuda-graph"
LORA_ENVS="$LORA_ENVS TORCH_LOGS=recompiles"
sl0=$(kh "wc -l </tmp/server.log 2>/dev/null || echo 0" | tr -dc 0-9)
if launch_server lora; then
  echo "== server READY; sending one real generate (coherence) =="
  coherence_check lora && echo "== COHERENCE PASSED — replay survived a real request ==" \
    || echo "== coherence failed/crashed — guard evidence below =="
else
  echo "== launch failed =="
fi
echo "== recompile entries during this launch+request =="
kh "tail -n +$((sl0+1)) /tmp/server.log | grep -a -A3 'triggered by the following guard failure' | tail -24" || true
kill_all
