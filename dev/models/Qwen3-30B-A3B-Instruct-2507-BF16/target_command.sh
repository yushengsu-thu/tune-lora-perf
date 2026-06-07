#!/usr/bin/env bash
# THE target command for the bf16-in-experimental_sgl_trtllm task: this must launch and serve
# successfully on Qwen/Qwen3-30B-A3B-Instruct-2507 (bf16). Runs IN-POD (cd /root/sglang).
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_EXPERIMENTAL_LORA_OPTI=1 SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 \
SGLANG_OPT_LORA_SHARED_ADD_OVERLAP=1 SGLANG_OPT_LORA_CUBLAS=1 \
numactl --membind=0,1 python3 -m sglang.launch_server \
  --model-path /data/Qwen3-30B-A3B-Instruct-2507 \
  --tp 4 --ep 4 --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.8 --trust-remote-code \
  --cuda-graph-max-bs 128 --max-prefill-tokens 65536 --chunked-prefill-size 4096 \
  --enforce-disable-flashinfer-allreduce-fusion \
  --attention-backend trtllm_mha \
  --moe-runner-backend experimental_sgl_trtllm \
  --lora-use-virtual-experts \
  --enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton \
  --lora-paths alpha=/data/Qwen3-30B-A3B-Instruct-2507-BF16-dummy-lora-r16
