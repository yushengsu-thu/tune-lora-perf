#!/usr/bin/env bash
# One-off diagnostic: reproduce the FP8 eager(graph-off)+LoRA scheduler-watchdog hang and classify
# it as triton-JIT-compile vs CUDA-deadlock. py-spy is blocked in this pod (no CAP_SYS_PTRACE), so we
# discriminate from the OUTSIDE: GPU util, per-proc CPU, live compiler subprocs (ptxas/cicc/cc1plus),
# and triton-cache write activity, sampled while one lora /generate request is in flight and hung.
#   JIT-compile  -> one python rank pegs a CPU core, GPU idle, ptxas/cicc present, triton cache growing
#   CUDA-deadlock-> CPU idle (blocked on sync), GPU idle, no compilers, triton cache quiet
#   runaway kernel-> GPU util high
#   Usage: bash dev/diag_eager_lora_hang.sh <model|prefix>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
ensure_dummy_lora || { echo "ERROR: dummy LoRA setup failed"; exit 1; }
ensure_hf_lora    || { echo "ERROR: HF LoRA setup failed"; exit 1; }

# eager + lora-single (two-stream OFF), exactly the failing cell's config
LORA_ENVS="${LORA_ENVS} SGLANG_TWO_STREAM_MAX_TOKENS=0"
echo "== [diag] launching eager+lora server (graph off) on ${HEAD_POD}"
launch_server lora off || { echo "ERROR: server failed to even reach READY"; exit 1; }
echo "== [diag] server READY; firing ONE hung lora /generate in background on the pod"

# fire the request detached on the pod so it doesn't block us; it will hang
kh "nohup curl -s http://127.0.0.1:${PORT}/generate -H 'Content-Type: application/json' \
     -d '{\"text\":\"The capital of France is\",\"sampling_params\":{\"max_new_tokens\":24,\"temperature\":0},\"lora_path\":\"${LORA_NAME}\"}' \
     > /tmp/diag_gen.out 2>&1 &" || true

for i in 1 2 3 4 5 6 7; do
  echo "==================== SNAPSHOT $i (t=$(( (i-1)*20 ))s into the hung request) ===================="
  kh '
    echo "--- nvidia-smi (util%, mem) ---"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
    echo "--- top python / compiler procs by CPU ---"
    ps -eo pid,pcpu,pmem,stat,wchan:24,comm --sort=-pcpu 2>/dev/null | grep -E "PID|python|ptxas|cicc|cc1plus|gcc|nvcc|triton" | head -12
    echo "--- live compiler subprocs (ptxas/cicc/cc1plus/nvcc) ---"
    pgrep -a -f "ptxas|cicc|cc1plus|nvcc" 2>/dev/null | head -8 || echo "  (none)"
    echo "--- triton cache files written in last 90s ---"
    find /root/.triton ~/.triton /tmp/triton* -type f -newermt "-90 seconds" 2>/dev/null | head -8
    echo "--- request returned yet? ---"
    test -s /tmp/diag_gen.out && { echo "  GEN OUTPUT:"; head -c 200 /tmp/diag_gen.out; echo; } || echo "  (still hung, no output)"
  ' 2>&1
  [ $i -lt 7 ] && sleep 20
done

echo "== [diag] done sampling; killing server"
kill_all
echo "== [diag] COMPLETE"
