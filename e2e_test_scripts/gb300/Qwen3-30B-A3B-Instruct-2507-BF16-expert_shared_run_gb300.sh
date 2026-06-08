#!/usr/bin/env bash
# GB300 (sm_103, gcp-radixark-02) LoRA e2e for Qwen3-30B-A3B-Instruct-2507 **BF16** with
# experts_shared_outer_loras ON, on the experimental_sgl_trtllm path. Sibling of
# Qwen3.5-35B-A3B-FP8_run_gb300.sh; deltas (and why):
#   * MODEL is the bf16 base (NOT FP8); LORAP is the REAL trained adapter
#     /data/lora-diff-Qwen3-30B-A3B-Instruct-2507 (r=32, all-linear) whose vLLM/trainer reference
#     logprobs the dev 4_run_acc already matched to the noise floor.
#   * LoRA flags add --experts-shared-outer-loras and --max-lora-rank 32.
#   * allreduce fusion FORCE-DISABLED (--enforce-disable-flashinfer-allreduce-fusion): the bf16
#     LoRA path crashes in cuda-graph capture with fusion ON (illegal memory access).
#   * NO --mamba-scheduler-strategy: Qwen3-30B-A3B-Instruct-2507 is a standard-attention MoE.
#   * NOCO=1 (default) SKIPS the git checkout — this runner is meant to run on a pod the dev loop
#     already prepared at the task branch (qwen3-30b-a3b-2507-bf16). Set NOCO=0 to fetch+checkout.
#   * gsm8k uses the real adapter, so the req-lora accuracy is a genuine number (NOT the alpha
#     identity adapter's ~0.01) — informational alongside the base ceiling.
# Usage: Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared_run_gb300.sh <TAG>
TAG=${1:-esol}; PORT=30000; MODEL=/data/Qwen3-30B-A3B-Instruct-2507
LORAP=/data/lora-diff-Qwen3-30B-A3B-Instruct-2507; H=/tmp/flo_helpers; cd /root/sglang
BACKEND=${BACKEND:-experimental_sgl_trtllm}
FLASHINFER_PIN=${FLASHINFER_PIN:-0.6.11.post1}
if [ "${NOCO:-1}" != 1 ]; then
  URL=https://github.com/yushengsu-thu/sglang; BR=${BR:-qwen3-30b-a3b-2507-bf16}
  git fetch $URL $BR >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
  [ "${REINSTALL:-0}" = 1 ] && { echo "[$TAG] pip install -e python"; pip install -q -e python >/tmp/pip.log 2>&1; }
fi
echo "[$TAG] HEAD=$(git rev-parse --short HEAD)  backend=$BACKEND flashinfer=$(python3 -c 'import flashinfer; print(flashinfer.__version__)' 2>/dev/null)"
# config = "name lora optenv tests"   (tests: full | gsm8k)
CONFIGS=("sgl-lora 1 PR full" "nolora 0 PR gsm8k")
gen(){ curl -s http://127.0.0.1:$PORT/generate -H "Content-Type: application/json" -d "$1" 2>/dev/null | python3 -c "import sys,json;print(repr(json.load(sys.stdin).get('text','')[:54]))" 2>/dev/null; }
for cfg in "${CONFIGS[@]}"; do
  set -- $cfg; name=$1; lora=$2; opt=$3; tests=$4
  echo "[$TAG] ===== CONFIG $name (lora=$lora tests=$tests) ====="
  pkill -9 -f "[s]glang.launch_server" 2>/dev/null; fuser -k $PORT/tcp 2>/dev/null; sleep 5; : >/tmp/srv.log
  # NEVER add SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP or SGLANG_OPT_LORA_ENABLE_PDL (corrupts base).
  OPT=""; [ "$opt" = PR ] && OPT="SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 SGLANG_OPT_LORA_CUBLAS=1"
  LF=""; [ "$lora" = 1 ] && LF="--enable-lora --max-loras-per-batch 1 --max-lora-rank 32 --lora-backend triton --lora-use-virtual-experts --experts-shared-outer-loras --lora-paths alpha=$LORAP"
  setsid env $OPT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True numactl --membind=0,1 python3 -m sglang.launch_server --model-path $MODEL --tp 4 --ep 4 --host 0.0.0.0 --port $PORT --cuda-graph-max-bs 128 --mem-fraction-static 0.8 --trust-remote-code --max-prefill-tokens 65536 --chunked-prefill-size 4096 --enforce-disable-flashinfer-allreduce-fusion --attention-backend trtllm_mha --moe-runner-backend $BACKEND $LF </dev/null >/tmp/srv.log 2>&1 &
  # 225 x 12s = 45 min — headroom for a cold sm_103 trtllm_lora_temp JIT (warm cache lands ~8 min).
  R=0; for i in $(seq 1 225); do curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null 2>&1 && { R=1; break; }
    c=$(tr '\r' '\n' </tmp/srv.log 2>/dev/null|grep -acE "Traceback|Error|serve: error|out of memory|CUDA out|Capture cuda graph failed"); [ "${c:-0}" -ge 1 ] && { R=2; break; }; sleep 12; done
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
echo "[$TAG] QWEN330-BF16-ESOL DONE"
