#!/usr/bin/env bash
# Self-contained QWEN runner (single-pod TP4/EP4, runs all configs sequentially). Usage:
#   qwen_run.sh <REF=full-lora-opti|main> <TAG>
REF=$1; TAG=$2; PORT=30000; MODEL=/data/Qwen3.5-35B-A3B-FP8; LORAP=/data/qwen35_35b_lora_alpha; H=/tmp/flo_helpers; cd /root/sglang
if [ "$REF" = main ]; then URL=https://github.com/sgl-project/sglang; BR=main; else URL=https://github.com/jybsuper/sglang; BR=full-lora-opti; fi
pkill -9 -f "[s]glang.launch_server" 2>/dev/null; pkill -9 -f "[p]ython3 -m sglang" 2>/dev/null; fuser -k $PORT/tcp 2>/dev/null; sleep 5
git fetch $URL $BR >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "[$TAG] HEAD=$(git rev-parse --short HEAD)  REF=$REF"
# config = "name backend lora optenv tests"   (tests: full | gsm8k)
if [ "$REF" = full-lora-opti ]; then
  CONFIGS=("sgl-lora sgl_flashinfer_trtllm 1 PR full" "triton-lora triton 1 NONE full" "nolora sgl_flashinfer_trtllm 0 PR gsm8k")
else
  CONFIGS=("triton-lora triton 1 NONE full" "nolora triton 0 NONE gsm8k")
fi
gen(){ curl -s http://127.0.0.1:$PORT/generate -H "Content-Type: application/json" -d "$1" 2>/dev/null | python3 -c "import sys,json;print(repr(json.load(sys.stdin).get('text','')[:54]))" 2>/dev/null; }
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; name=$1; backend=$2; lora=$3; opt=$4; tests=$5
  echo "[$TAG] ===== CONFIG $name (backend=$backend lora=$lora tests=$tests) ====="
  pkill -9 -f "[s]glang.launch_server" 2>/dev/null; fuser -k $PORT/tcp 2>/dev/null; sleep 5; : >/tmp/srv.log
  OPT=""; [ "$opt" = PR ] && OPT="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP=1 SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 SGLANG_OPT_LORA_CUBLAS=1"
  LF=""; [ "$lora" = 1 ] && LF="--enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton --lora-use-virtual-experts --lora-paths alpha=$LORAP"
  setsid env $OPT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True numactl --membind=0,1 python3 -m sglang.launch_server --model-path $MODEL --tp 4 --ep 4 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 128 --mem-fraction-static 0.8 --trust-remote-code --max-prefill-tokens 65536 --chunked-prefill-size 65536 --mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha --moe-runner-backend $backend $LF </dev/null >/tmp/srv.log 2>&1 &
  R=0; for i in $(seq 1 175); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { R=1; break; }
    c=$(tr '\r' '\n' </tmp/srv.log 2>/dev/null|grep -acE "Traceback|Error|out of memory|CUDA out|Capture cuda graph failed"); [ "${c:-0}" -ge 1 ] && { R=2; break; }; sleep 12; done
  [ "$R" = 1 ] || { echo "[$TAG] $name LAUNCH FAILED"; tr '\r' '\n' </tmp/srv.log|grep -aiE "Error|Traceback|out of memory|assert"|tail -6; continue; }
  echo "[$TAG] $name READY"
  echo "[$TAG] $name base: $(gen '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}')"
  [ "$lora" = 1 ] && echo "[$TAG] $name lora: $(gen '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0},"lora_path":"alpha"}')"
  if [ "$tests" = full ]; then
    CG=0; for p in "What is the capital of Japan?" "Explain gravity briefly." "2+2 equals"; do
      o=$(gen "{\"text\":\"$p\",\"sampling_params\":{\"max_new_tokens\":40,\"temperature\":0}}"); echo "$o"|grep -qE '!!!!|####|@@@@' && CG=1
      [ "$lora" = 1 ] && { ol=$(gen "{\"text\":\"$p\",\"sampling_params\":{\"max_new_tokens\":40,\"temperature\":0},\"lora_path\":\"alpha\"}"); echo "$ol"|grep -qE '!!!!|####|@@@@' && CG=1; }
    done
    echo "[$TAG] $name COHERENCE: $([ $CG = 0 ] && echo COHERENT || echo GARBAGE)"
  fi
  echo "[$TAG] $name gsm8k/base(req-no-lora): $(python3 $H/gsm8k_lora.py --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1|grep -aiE 'Accuracy|Truncated'|tr '\n' ' ')"
  [ "$lora" = 1 ] && echo "[$TAG] $name gsm8k/lora(req-lora): $(python3 $H/gsm8k_lora.py --lora alpha --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1|grep -aiE 'Accuracy|Truncated'|tr '\n' ' ')"
  if [ "$tests" = full ]; then
    OUT=/tmp/qbench_${TAG}_$name; mkdir -p $OUT
    for mode in base lora; do [ "$mode" = lora ] && { [ "$lora" = 1 ]||continue; LN="--lora-name alpha"; }||LN=""
      for bs in 16 32 64 128; do sl0=$(wc -l </tmp/srv.log 2>/dev/null||echo 0)
        python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:$PORT --batch-size $bs --input-len 2048 --output-len 2048 $LN --show-report --result-filename $OUT/${mode}_$bs.jsonl >$OUT/${mode}_$bs.log 2>&1
        tail -n +$((sl0+1)) /tmp/srv.log 2>/dev/null|tr '\r' '\n'|grep -aE 'Decode batch' >$OUT/${mode}_$bs.slog
        echo "[$TAG] $name bench/$mode bs$bs: $(python3 $H/bench_report.py $OUT/${mode}_$bs.jsonl $OUT/${mode}_$bs.slog 2>/dev/null)"
      done
    done
  fi
done
pkill -9 -f "[s]glang.launch_server" 2>/dev/null
echo "[$TAG] QWEN DONE"
