#!/usr/bin/env bash
# opt8 diag: one ON-variant launch with TORCH_LOGS=graph_code to dump the dynamo graph,
# then grep which node carries a torch.cuda.Stream across split partitions.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
LORA_EXTRA="$LORA_EXTRA --enforce-piecewise-cuda-graph"
LORA_ENVS="$LORA_ENVS TORCH_LOGS=graph_code"
sl0=$(kh "wc -l </tmp/server.log 2>/dev/null || echo 0" | tr -dc 0-9)
launch_server lora || true   # expected to crash at the split; the dump is what we want
kh "tail -n +$((sl0+1)) /tmp/server.log | grep -an 'stream' | grep -av 'Stream device\|warnings\|FutureWarning' | head -40" || true
kill_all
