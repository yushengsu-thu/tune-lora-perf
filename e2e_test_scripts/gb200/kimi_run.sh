#!/usr/bin/env bash
# Parameterized KIMI 2-node runner. Usage:
#   kimi_run.sh <ROLE=worker|head> <REF=full-lora-opti|main> <LORA=0|1> <BACKEND> <DISTADDR> <TAG> <TESTS=full|gsm8k_only>
# Kills existing + checks out REF + launches (node-rank from ROLE). HEAD also runs the test matrix.
ROLE=$1; REF=$2; LORA=$3; BACKEND=$4; DISTADDR=$5; TAG=$6; TESTS=${7:-full}
PORT=30000; MODEL=/root/Kimi-K2.5-NVFP4; LORAP=/root/kimi_k25_lora_alpha; H=/tmp/flo_helpers; cd /root/sglang
pkill -9 -f "[s]glang.launch_server" 2>/dev/null; pkill -9 -f "[s]glang::" 2>/dev/null; pkill -9 -f "[p]ython3 -m sglang" 2>/dev/null
fuser -k 20000/tcp ${PORT}/tcp 2>/dev/null; sleep 6; : >/tmp/srv.log
if [ "$REF" = main ]; then URL=https://github.com/yushengsu-thu/sglang; BR=trtllm-lora-bf16; else URL=https://github.com/yushengsu-thu/sglang; BR=trtllm-lora-bf16; fi
git fetch $URL $BR >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "[$TAG] $ROLE HEAD=$(git rev-parse --short HEAD) LORA=$LORA backend=$BACKEND"
OPT=""; [ "$REF" = full-lora-opti ] && OPT="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 SGLANG_OPT_USE_JIT_KERNEL_KIMI_GATE=1 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=1 SGLANG_OPT_FUSED_PERMUTE_QUANT=1 SGLANG_OPT_FUSED_MOE_ACTIVATION_QUANT_FUSE=1"
LF=""; [ "$LORA" = 1 ] && LF="--enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton --lora-use-virtual-experts --lora-paths alpha=$LORAP"
NR=$([ "$ROLE" = worker ] && echo 1 || echo 0)
BF="--moe-runner-backend $BACKEND"; [ "$BACKEND" = default ] && BF=""
setsid env $OPT NCCL_MNNVL_ENABLE=1 NCCL_NVLS_ENABLE=1 NCCL_CUMEM_ENABLE=1 SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True numactl --membind=0,1 python3 -m sglang.launch_server --model-path $MODEL --tp 8 --nnodes 2 --ep-size 8 --dist-init-addr $DISTADDR --dist-timeout 1800 --host 0.0.0.0 --port $PORT --quantization modelopt_fp4 --mem-fraction-static 0.83 --cuda-graph-max-bs 128 --trust-remote-code --max-prefill-tokens 40960 --chunked-prefill-size 40960 $BF $LF --node-rank $NR </dev/null >/tmp/srv.log 2>&1 &
echo "[$TAG] launched node-rank $NR"
[ "$ROLE" = worker ] && exit 0
# ---- HEAD: wait READY ----
R=0; for i in $(seq 1 180); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "[$TAG] READY (iter $i)"; R=1; break; }
  c=$(tr '\r' '\n' </tmp/srv.log 2>/dev/null|grep -acE "Capture cuda graph failed|cudaErrorStreamCapture|Traceback|Error|out of memory"); [ "${c:-0}" -ge 1 ] && { echo "[$TAG] LAUNCH FAILED"; tr '\r' '\n' </tmp/srv.log|grep -aiE "Error|Traceback|Capture|out of memory"|tail -6; R=2; break; }
  sleep 12; done
[ "$R" = 1 ] || { echo "[$TAG] NOT READY"; tail -6 /tmp/srv.log; echo "[$TAG] DONE"; exit 1; }
gen(){ curl -s http://127.0.0.1:$PORT/generate -H "Content-Type: application/json" -d "$1" 2>/dev/null | python3 -c "import sys,json;print(repr(json.load(sys.stdin).get('text','')[:54]))" 2>/dev/null; }
echo "[$TAG] ### SANITY ###"
echo "[$TAG] base: $(gen '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}')"
[ "$LORA" = 1 ] && echo "[$TAG] lora: $(gen '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0},"lora_path":"alpha"}')"
if [ "$TESTS" = full ]; then
  echo "[$TAG] ### COHERENCE ###"
  CG=0; for p in "What is the capital of Japan?" "Explain gravity in one sentence." "2+2 equals"; do
    o=$(gen "{\"text\":\"$p\",\"sampling_params\":{\"max_new_tokens\":40,\"temperature\":0}}")
    echo "[$TAG]   base| $p => $o"; echo "$o" | grep -qE '!!!!|####|@@@@' && CG=1
    if [ "$LORA" = 1 ]; then ol=$(gen "{\"text\":\"$p\",\"sampling_params\":{\"max_new_tokens\":40,\"temperature\":0},\"lora_path\":\"alpha\"}"); echo "[$TAG]   lora| $p => $ol"; echo "$ol" | grep -qE '!!!!|####|@@@@' && CG=1; fi
  done
  echo "[$TAG] COHERENCE: $([ "$CG" = 0 ] && echo COHERENT || echo GARBAGE)"
fi
echo "[$TAG] ### GSM8K base (req-no-lora / lora-disabled) ###"
echo "[$TAG] $(python3 $H/gsm8k_lora.py --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1 | grep -aiE 'Accuracy|Truncated|EOS-empty' | tr '\n' ' ')"
if [ "$LORA" = 1 ]; then
  echo "[$TAG] ### GSM8K lora (req-lora) ###"
  echo "[$TAG] $(python3 $H/gsm8k_lora.py --lora alpha --num-questions 200 --num-shots 5 --parallel 32 --port $PORT --max-new-tokens 512 --model $MODEL 2>&1 | grep -aiE 'Accuracy|Truncated|EOS-empty' | tr '\n' ' ')"
fi
if [ "$TESTS" = full ]; then
  echo "[$TAG] ### BENCH bs16/32/64/128 ###"; OUT=/tmp/kbench_$TAG; mkdir -p $OUT
  for mode in base lora; do
    [ "$mode" = lora ] && { [ "$LORA" = 1 ] || continue; LN="--lora-name alpha"; } || LN=""
    for bs in 16 32 64 128; do sl0=$(wc -l </tmp/srv.log 2>/dev/null||echo 0)
      python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:$PORT --batch-size $bs --input-len 2048 --output-len 2048 $LN --show-report --result-filename $OUT/${mode}_$bs.jsonl >$OUT/${mode}_$bs.log 2>&1
      tail -n +$((sl0+1)) /tmp/srv.log 2>/dev/null|tr '\r' '\n'|grep -aE 'Decode batch' >$OUT/${mode}_$bs.slog
      echo "[$TAG] bench/$mode bs$bs: $(python3 $H/bench_report.py $OUT/${mode}_$bs.jsonl $OUT/${mode}_$bs.slog 2>/dev/null)"
    done
  done
fi
echo "[$TAG] DONE"
