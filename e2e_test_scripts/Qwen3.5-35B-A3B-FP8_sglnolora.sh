#!/usr/bin/env bash
# Re-validate: qwen on sgl_flashinfer_trtllm with LoRA OFF must now delegate FP8 to upstream
# (not the sgl-kernel path). Usage: Qwen3.5-35B-A3B-FP8_sglnolora.sh <TAG>  (lorapr-0, TP4)
TAG=$1; PORT=30000; MODEL=/data/Qwen3.5-35B-A3B-FP8; H=/tmp/flo_helpers; cd /root/sglang
pkill -9 -f "[s]glang.launch_server" 2>/dev/null; fuser -k $PORT/tcp 2>/dev/null; sleep 5; : >/tmp/srv.log
git fetch https://github.com/yushengsu-thu/sglang trtllm-lora-bf16 >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "[$TAG] HEAD=$(git rev-parse --short HEAD) sgl backend, EXPERIMENTAL_LORA_OPTI=1, NO --enable-lora"
setsid env SGLANG_EXPERIMENTAL_LORA_OPTI=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True numactl --membind=0,1 python3 -m sglang.launch_server --model-path $MODEL --tp 4 --ep 4 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 128 --mem-fraction-static 0.8 --trust-remote-code --max-prefill-tokens 65536 --chunked-prefill-size 65536 --mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha --moe-runner-backend sgl_flashinfer_trtllm </dev/null >/tmp/srv.log 2>&1 &
R=0; for i in $(seq 1 175); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { R=1; break; }; c=$(tr '\r' '\n' </tmp/srv.log|grep -acE "Traceback|Error|out of memory|Capture cuda graph failed"); [ "${c:-0}" -ge 1 ] && { R=2; break; }; sleep 12; done
[ "$R" = 1 ] || { echo "[$TAG] LAUNCH FAILED"; tr '\r' '\n' </tmp/srv.log|grep -aiE "Error|Traceback|out of memory"|tail -5; echo "[$TAG] DONE"; exit 1; }
echo "[$TAG] READY (no crash)"
echo "[$TAG] gsm8k/base: $(python3 $H/gsm8k_lora.py --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1|grep -aiE 'Accuracy|Truncated'|tr '\n' ' ')"
OUT=/tmp/qsgl_$TAG; mkdir -p $OUT
for bs in 32 64; do sl0=$(wc -l </tmp/srv.log 2>/dev/null||echo 0)
  python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:$PORT --batch-size $bs --input-len 2048 --output-len 2048 --show-report --result-filename $OUT/b$bs.jsonl >$OUT/b$bs.log 2>&1
  tail -n +$((sl0+1)) /tmp/srv.log 2>/dev/null|tr '\r' '\n'|grep -aE 'Decode batch' >$OUT/b$bs.slog
  echo "[$TAG] bench/base bs$bs: $(python3 $H/bench_report.py $OUT/b$bs.jsonl $OUT/b$bs.slog 2>/dev/null)"
done
pkill -9 -f "[s]glang.launch_server" 2>/dev/null
echo "[$TAG] DONE"
