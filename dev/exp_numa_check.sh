#!/usr/bin/env bash
# step2 experiment: is per-rank NUMA/CPU binding actually active? Launch LoRA (eager),
# dump each TP worker's CPU affinity; if all show 0-143 -> binding OFF (cheap lever exists).
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
launch_server lora || { echo "launch failed"; exit 1; }
echo "== TP worker CPU affinity (Cpus_allowed_list) =="
kh 'for p in $(pgrep -f "scheduler|TP[0-9]"); do
      gi=$(tr "\0" " " </proc/$p/cmdline 2>/dev/null | grep -oE "gpu.?id[ =][0-9]+" | head -1)
      al=$(grep Cpus_allowed_list /proc/$p/status 2>/dev/null | awk "{print \$2}")
      [ -n "$al" ] && echo "pid $p  cpus=$al  $gi"
    done | sort -u | head -20'
echo "== quick prefill bench (bs16,64) =="
kh "cd /root/sglang; for bs in 16 64; do
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size \$bs --input-len 2048 --output-len 8 --lora-name ${LORA_NAME} --skip-warmup 2>&1 \
        | grep -E 'input throughput|batch size'; done"
kill_all
