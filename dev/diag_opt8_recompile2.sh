#!/usr/bin/env bash
# opt8 diag v3: ON launch + TORCH_LOGS=recompiles + a BENCH-shaped load (bs16) —
# the single-request coherence passes; the bench shape is what still recompiles
# at replay. The last guard entry before the crash names it.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
LORA_EXTRA="$LORA_EXTRA --enforce-piecewise-cuda-graph"
LORA_ENVS="$LORA_ENVS TORCH_LOGS=recompiles"
sl0=$(kh "wc -l </tmp/server.log 2>/dev/null || echo 0" | tr -dc 0-9)
launch_server lora || { echo "== launch failed =="; exit 1; }
echo "== READY; bench-shaped load (bs16 in=2048 out=8) =="
kh "cd /root/sglang && timeout 300 python -m sglang.bench_one_batch_server --model-path None \
    --base-url http://127.0.0.1:${PORT} --batch-size 16 --input-len 2048 --output-len 8 \
    --lora-name ${LORA_NAME} --skip-warmup 2>&1 | tail -5" || echo "== bench died (expected if guard fires) =="
echo "== last recompile entries =="
kh "tail -n +$((sl0+1)) /tmp/server.log | grep -a -A3 'triggered by the following guard failure' | tail -12" || true
kh "tail -n +$((sl0+1)) /tmp/server.log | grep -a 'AssertionError' | tail -2" || true
kill_all
