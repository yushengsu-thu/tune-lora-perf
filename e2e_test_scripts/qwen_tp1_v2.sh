#!/usr/bin/env bash
# tp1 re-run on the FIXED branch + a tp1 oss-nolora baseline (% denominator).
# Usage: qwen_tp1_v2.sh <REF=full-lora-opti|main-base> <TAG>   (GPU0, tp1)
REF=$1; TAG=$2; PORT=30001; MODEL=/data/Qwen3.5-35B-A3B-FP8; LORAP=/data/qwen35_35b_lora_alpha; H=/tmp/flo_helpers; cd /root/sglang
pkill -9 -f "[s]glang.launch_server" 2>/dev/null; fuser -k $PORT/tcp 30000/tcp 2>/dev/null; sleep 5; : >/tmp/srv_tp1.log
if [ "$REF" = main-base ]; then URL=https://github.com/sgl-project/sglang; BR=main; BFLAG=""; LF=""; OPT=""; LMODE=0
else URL=https://github.com/jybsuper/sglang; BR=full-lora-opti; BFLAG="--moe-runner-backend sgl_flashinfer_trtllm"; LF="--enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton --lora-use-virtual-experts --lora-paths alpha=$LORAP"; OPT="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 SGLANG_OPT_LORA_CUBLAS=1"; LMODE=1; fi
git fetch $URL $BR >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "[$TAG] HEAD=$(git rev-parse --short HEAD) tp1 REF=$REF backend=${BFLAG:-default} lora=$LMODE"
setsid env CUDA_VISIBLE_DEVICES=0 $OPT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m sglang.launch_server --model-path $MODEL --tp 1 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 64 --mem-fraction-static 0.8 --trust-remote-code --max-prefill-tokens 65536 --chunked-prefill-size 65536 --mamba-scheduler-strategy extra_buffer --attention-backend trtllm_mha $BFLAG $LF </dev/null >/tmp/srv_tp1.log 2>&1 &
R=0; for i in $(seq 1 150); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { R=1; break; }; c=$(tr '\r' '\n' </tmp/srv_tp1.log|grep -acE "Traceback|Error|out of memory|Capture cuda graph failed"); [ "${c:-0}" -ge 1 ] && { R=2; break; }; sleep 10; done
[ "$R" = 1 ] || { echo "[$TAG] LAUNCH FAILED"; tr '\r' '\n' </tmp/srv_tp1.log|grep -aiE "Error|Traceback|out of memory"|tail -5; echo "[$TAG] TP1 DONE"; exit 1; }
echo "[$TAG] READY"
echo "[$TAG] gsm8k/base: $(python3 $H/gsm8k_lora.py --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1|grep -aiE 'Accuracy|Truncated'|tr '\n' ' ')"
[ "$LMODE" = 1 ] && echo "[$TAG] gsm8k/lora: $(python3 $H/gsm8k_lora.py --lora alpha --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1|grep -aiE 'Accuracy|Truncated'|tr '\n' ' ')"
OUT=/tmp/qtp1v2_$TAG; mkdir -p $OUT
for mode in base lora; do [ "$mode" = lora ] && { [ "$LMODE" = 1 ]||continue; LN="--lora-name alpha"; }||LN=""
  for bs in 16 32 64; do sl0=$(wc -l </tmp/srv_tp1.log 2>/dev/null||echo 0)
    python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:$PORT --batch-size $bs --input-len 2048 --output-len 2048 $LN --show-report --result-filename $OUT/${mode}_$bs.jsonl >$OUT/${mode}_$bs.log 2>&1
    tail -n +$((sl0+1)) /tmp/srv_tp1.log 2>/dev/null|tr '\r' '\n'|grep -aE 'Decode batch' >$OUT/${mode}_$bs.slog
    echo "[$TAG] bench/$mode bs$bs: $(python3 $H/bench_report.py $OUT/${mode}_$bs.jsonl $OUT/${mode}_$bs.slog 2>/dev/null)"
  done
done
pkill -9 -f "[s]glang.launch_server" 2>/dev/null
echo "[$TAG] TP1 DONE"
