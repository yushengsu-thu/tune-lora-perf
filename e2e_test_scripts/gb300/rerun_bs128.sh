#!/usr/bin/env bash
# One-off: re-run the SUSPECT bench point (sgl-lora cell, base mode, bs128) per the runbook
# xcheck>5% rule. Relaunches the PR sgl-lora server (warm JIT) and benches base bs128 twice.
TAG=$1; PORT=30000; MODEL=/data/Qwen3.5-35B-A3B-FP8; LORAP=/data/qwen35_35b_lora_alpha; H=/tmp/flo_helpers; cd /root/sglang
pkill -9 -f "[s]glang.launch_server" 2>/dev/null; fuser -k $PORT/tcp 2>/dev/null; sleep 5; : >/tmp/srv.log
git fetch https://github.com/yushengsu-thu/sglang trtllm-lora-bf16 >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "[$TAG] HEAD=$(git rev-parse --short HEAD) rerun sgl-lora/base bs128"
OPT="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 SGLANG_OPT_LORA_CUBLAS=1"
LF="--enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton --lora-use-virtual-experts --lora-paths alpha=$LORAP"
setsid env $OPT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True numactl --membind=0,1 python3 -m sglang.launch_server --model-path $MODEL --tp 4 --ep 4 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 128 --mem-fraction-static 0.8 --trust-remote-code --max-prefill-tokens 65536 --chunked-prefill-size 65536 --mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha --moe-runner-backend experimental_sgl_trtllm $LF </dev/null >/tmp/srv.log 2>&1 &
R=0; for i in $(seq 1 225); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { R=1; break; }
  c=$(tr '\r' '\n' </tmp/srv.log 2>/dev/null|grep -acE "Traceback|Error|serve: error|out of memory|CUDA out|Capture cuda graph failed"); [ "${c:-0}" -ge 1 ] && { R=2; break; }; sleep 12; done
[ "$R" = 1 ] || { echo "[$TAG] LAUNCH FAILED"; tr '\r' '\n' </tmp/srv.log|grep -aiE "error|Traceback|out of memory"|tail -5; exit 1; }
echo "[$TAG] READY"
OUT=/tmp/qrerun_$TAG; mkdir -p $OUT
for n in 1 2; do sl0=$(wc -l </tmp/srv.log 2>/dev/null||echo 0)
  python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:$PORT --batch-size 128 --input-len 2048 --output-len 2048 --show-report --result-filename $OUT/base_128_$n.jsonl >$OUT/base_128_$n.log 2>&1
  tail -n +$((sl0+1)) /tmp/srv.log 2>/dev/null|tr '\r' '\n'|grep -aE 'Decode batch' >$OUT/base_128_$n.slog
  echo "[$TAG] RERUN-$n bench/base bs128: $(python3 $H/bench_report.py $OUT/base_128_$n.jsonl $OUT/base_128_$n.slog 2>/dev/null)"
done
pkill -9 -f "[s]glang.launch_server" 2>/dev/null
echo "[$TAG] RERUN DONE"
