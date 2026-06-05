#!/bin/bash
# Yanbin's launch command: Qwen3.5-35B-A3B-FP8 LoRA server + graph-ON bs64 24-step profile.
# Use together with the kimi skill: have the agent apply the skill on top of this command.

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1 \
numactl --membind=0,1 python3 -m sglang.launch_server \
  --model-path /data/Qwen3.5-35B-A3B-FP8 --tp 4 --ep 4 --host 0.0.0.0 --port 30000 \
  --cuda-graph-max-bs 64 --mem-fraction-static 0.8 --trust-remote-code \
  --max-prefill-tokens 32768 --chunked-prefill-size 4096 \
  --mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha \
  --moe-runner-backend sgl_flashinfer_trtllm \
  --enable-lora --max-loras-per-batch 1 --max-lora-rank 16 --lora-backend triton \
  --lora-use-virtual-experts --lora-paths alpha=/data/qwen35_35b_lora_alpha

python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:30000 \
  --batch-size 64 --input-len 2048 --output-len 48 --lora-name alpha \
  --profile --profile-activities CPU GPU --profile-start-step 8 --profile-steps 24 \
  --profile-prefix q35_graphon_bs64_24step --profile-output-dir /tmp/q35prof_on2 \
  --result-filename /tmp/q35prof_on2/bench.jsonl
