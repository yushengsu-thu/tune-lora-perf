# tune-lora-perf

LoRA performance tuning skills and regression/benchmark workflows for SGLang.

All current development is based on this branch: https://github.com/jybsuper/sglang/tree/nvfp4-lora

## Contents

1. **General skill**: [`skill.md`](skill.md)
2. **Benchmark**: [`sglang-lora-base-perf-benchmark.md`](sglang-lora-base-perf-benchmark.md)
3. **Accuracy regression**:
   - Qwen series — [`sglang-base-variant-regression.md`](sglang-base-variant-regression.md)
   - Kimi — [`kimi-regression/`](kimi-regression)
4. **Profile recipes**:
   - Qwen3.5-35B-A3B-FP8, graph-ON bs64 — [`qwen35_35b_lora_profile_graphon_bs64.md`](qwen35_35b_lora_profile_graphon_bs64.md)
   - Kimi-K2.5-NVFP4 LoRA kernel shapes (bs64, TP8, EP8) — [`kimi_kernel_shapes_bs64_tp8_ep8.md`](kimi_kernel_shapes_bs64_tp8_ep8.md)
