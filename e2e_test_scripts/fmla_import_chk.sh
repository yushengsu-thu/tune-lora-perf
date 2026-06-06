#!/usr/bin/env bash
cd /root/sglang
git fetch https://github.com/yushengsu-thu/sglang trtllm-lora-bf16 >/tmp/gf.log 2>&1 && git checkout -f FETCH_HEAD >/tmp/co.log 2>&1
echo "HEAD=$(git rev-parse --short HEAD)"
echo "=== OFF: forward_mla imports; experimental pkg must NOT load ==="
SGLANG_EXPERIMENTAL_LORA_OPTI=0 python3 -c "
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla
import sys
leaked = sorted(m for m in sys.modules if 'trtllm_lora_temp' in m)
assert not leaked, f'LEAKED OFF: {leaked}'
print('OFF OK: forward_mla imports, experimental pkg NOT loaded')
" 2>&1 | tail -5
echo "=== ON: forward_mla + experimental + upstream correction modules resolve ==="
SGLANG_EXPERIMENTAL_LORA_OPTI=1 python3 -c "
import sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla
from sglang.srt.lora.trtllm_lora_temp.deepseek_mla_correction import kv_b_lora_q_apply, kv_b_lora_q_prepare, kv_b_lora_v_apply, kv_b_lora_v_prepare
from sglang.srt.lora.deepseek_mla_correction import apply_q_correction, apply_v_correction, is_kv_b_lora_active
print('ON OK: forward_mla + experimental + upstream correction modules import')
" 2>&1 | tail -5
echo "FMLA_IMPORT_DONE"
