#!/usr/bin/env bash
# Tests-only companion to kimi_run_gb300.sh: assumes a server is ALREADY launched (or warming)
# on PORT and runs the head wait+test matrix against it. Use when the launcher timed out
# waiting on a slow cold autotune but the server is still coming up.
# Usage: kimi_tests_only.sh <LORA=0|1> <TAG> <TESTS=full|gsm8k_only>
LORA=$1; TAG=$2; TESTS=${3:-full}
PORT=30000; MODEL=/root/Kimi-K2.5-NVFP4; H=/tmp/flo_helpers
# ---- HEAD: wait READY (240 x 12s = 48 min, cold sm_103 JIT headroom) ----
R=0; for i in $(seq 1 240); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { echo "[$TAG] READY (iter $i)"; R=1; break; }
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
