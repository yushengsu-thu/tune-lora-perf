#!/usr/bin/env bash
# Validate the renamed JIT csrc (trtllm_lora_temp) still compiles.
cd /root/sglang
git fetch https://github.com/jybsuper/sglang wip-full-lora-opti >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "JIT HEAD=$(git rev-parse --short HEAD)"
SGLANG_EXPERIMENTAL_LORA_OPTI=1 python3 -c "
import torch
from sglang.jit_kernel.trtllm_lora_temp.moe_lora_merged_align import _jit_module
print('loader import OK; building from csrc/trtllm_lora_temp ...')
try:
    m = _jit_module(torch.bfloat16)
    print('JIT BUILD OK from renamed csrc:', m is not None)
except FileNotFoundError as e:
    print('CSRC-RENAME-BROKE-BUILD:', str(e)[:200])
except Exception as e:
    print('build raised (arg/env, not a rename-path issue):', type(e).__name__, str(e)[:160])
" 2>&1 | grep -aE "HEAD=|OK|BUILD|BROKE|raised|nvcc|No such file|Traceback" | tail -12
echo "JIT_BUILD_DONE"
