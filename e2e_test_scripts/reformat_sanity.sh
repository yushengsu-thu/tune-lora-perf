#!/usr/bin/env bash
cd /root/sglang
git fetch https://github.com/jybsuper/sglang wip-full-lora-opti >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "HEAD=$(git rev-parse --short HEAD)"
echo "=== OFF: gated modules + forward_mla import; experimental pkg must NOT load ==="
SGLANG_EXPERIMENTAL_LORA_OPTI=0 python3 -c "
import sglang.srt.layers.moe.topk
import sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size
import sglang.srt.layers.moe.moe_runner.flashinfer_trtllm
import sglang.srt.lora.layers, sglang.srt.lora.lora_manager, sglang.srt.lora.mem_pool
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla
import sys
leaked = sorted(m for m in sys.modules if 'trtllm_lora_temp' in m)
assert not leaked, f'LEAKED OFF: {leaked}'
print('OFF OK: gated modules + forward_mla import; experimental pkg NOT loaded')
" 2>&1 | tail -5
echo "=== ON: experimental pkg + jit loaders + forward_mla correction modules ==="
SGLANG_EXPERIMENTAL_LORA_OPTI=1 python3 -c "
from sglang.srt.lora.trtllm_lora_temp import install_two_stream_overrides, sgl_backend
from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import kv_b_lora_q_apply, kv_b_lora_q_prepare, kv_b_lora_v_apply, kv_b_lora_v_prepare
from sglang.srt.lora.deepseek_mla_correction import apply_q_correction, apply_v_correction, is_kv_b_lora_active
import sglang.jit_kernel.trtllm_lora_temp.moe_lora_merged_align
import sglang.jit_kernel.trtllm_lora_temp.kimi_k2_moe_fused_gate
import sglang.jit_kernel.trtllm_lora_temp.topk_softmax_pack
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla
print('ON OK: experimental pkg + jit loaders + forward_mla correction modules import')
" 2>&1 | tail -5
echo "REFORMAT_SANITY_DONE"
