# Qwen3.5-35B-A3B-FP8 LoRA profile — graph-ON, bs64, 24-step

A self-contained profiling recipe for a single LoRA adapter on
**Qwen3.5-35B-A3B-FP8** (attention TP=4, MoE EP=4) on one node. Captures a
CPU+GPU torch-profiler trace of a CUDA-graph-**ON** decode at **batch=64**,
prefix `q35_graphon_bs64_24step`.

Companion to the multi-model profiling workflow in
[`sglang-lora-base-perf-benchmark.md`](./sglang-lora-base-perf-benchmark.md);
the trace it produces is analyzed with
[`kimi-regression/scripts/profile_metrics.py`](./kimi-regression/scripts/profile_metrics.py).

---

## Phase 1 — launch the server (graph ON, LoRA enabled)

```bash
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
```

### Key knobs

| Flag / env | Value | Why it matters for the profile |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | — | Avoids allocator fragmentation churn so the trace shows steady-state kernels, not realloc spikes. |
| `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1` | on | Overlaps the LoRA buffer alloc with the main path (the two-stream overlap under test). |
| `SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1` | on | Enables the fp32-atomics split-K shrink kernel (`_sgemm_lora_a_splitk_kernel`); wins on wide-K shrinks. |
| `numactl --membind=0,1` | NUMA 0,1 | Pins host memory so CPU-side profiler timings aren't polluted by cross-socket traffic. |
| `--tp 4 --ep 4` | 4 GPU | Attention TP=4, MoE EP=4 (single node). |
| `--cuda-graph-max-bs 64` | 64 | Graph captured up to bs64 → the bs64 decode below runs **inside** the CUDA graph (graph ON). |
| `--mem-fraction-static 0.8` | 0.8 | Leaves headroom for the profiler buffers. |
| `--attention-backend trtllm_mha` | — | TRT-LLM MHA attention path. |
| `--moe-runner-backend sgl_flashinfer_trtllm` | — | FlashInfer/TRT-LLM MoE runner (the NVFP4 trtllm path). |
| `--enable-lora --max-loras-per-batch 1 --max-lora-rank 16` | r=16, L=1 | Single adapter, rank 16. |
| `--lora-backend triton` | triton | Triton LoRA kernels (the ones the trace will show). |
| `--lora-use-virtual-experts` | on | Routes MoE LoRA through virtual experts. |
| `--lora-paths alpha=/data/qwen35_35b_lora_alpha` | `alpha` | Adapter name `alpha` — referenced by `--lora-name alpha` below. |

Wait for the server to print `The server is fired up and ready to roll!` before
launching the bench.

---

## Phase 2 — profiled bench (CPU+GPU, 24 captured steps)

```bash
python3 -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:30000 \
  --batch-size 64 --input-len 2048 --output-len 48 --lora-name alpha \
  --profile --profile-activities CPU GPU --profile-start-step 8 --profile-steps 24 \
  --profile-prefix q35_graphon_bs64_24step --profile-output-dir /tmp/q35prof_on2 \
  --result-filename /tmp/q35prof_on2/bench.jsonl
```

### Key knobs

| Flag | Value | Note |
|---|---|---|
| `--model-path None --base-url …:30000` | — | Bench drives the already-running server (no second model load). |
| `--batch-size 64` | 64 | Matches `--cuda-graph-max-bs 64` so decode runs inside the captured graph. |
| `--input-len 2048 --output-len 48` | 2048 / 48 | 2048-token prefill, then 48 decode steps. |
| `--lora-name alpha` | `alpha` | Sends the batch through the `alpha` adapter. |
| `--profile-activities CPU GPU` | both | Captures host + device, so CPU launch overhead and GPU kernels both appear. |
| `--profile-start-step 8` | 8 | Skips the first 8 forwards (warmup / prefill) before recording. |
| `--profile-steps 24` | 24 | Records 24 forwards → pass `--steps 24` to `profile_metrics.py`. |
| `--profile-prefix q35_graphon_bs64_24step` | — | Trace filename prefix. |
| `--profile-output-dir /tmp/q35prof_on2` | — | Where the `*.json[.gz]` trace + `bench.jsonl` land. |

> **Profile window math:** with `--output-len 48`, `start-step 8` + `steps 24`
> captures forwards 8–31 — all decode (the single 2048-tok prefill is step 0),
> so the trace is clean decode-only and needs no prefill-outlier drop.

---

## Analyze the trace

The bench emits **no** `ProfilerStep#` markers, so segment by the largest
inter-kernel gaps and take the median decode-step GPU-busy time:

```bash
python3 kimi-regression/scripts/profile_metrics.py \
  /tmp/q35prof_on2/q35_graphon_bs64_24step*.json* --steps 24 --out /tmp/q35prof_on2/metrics.json
```

Reports `forward_pass_us` (median decode-step GPU-busy) and `per_layer_us`
(pass `--layers N`). Sanity: `forward_pass` should be ≤ the bench ITL
(`1000·bs / decode_tput`), and the LoRA run should be slower than no-LoRA.

> **DECODE-THPT-RULE:** don't trust the bench `output_throughput` line alone —
> cross-check against the **decode throughput printed in the server log**, and
> use this profiler-derived `forward_pass_us` as the independent witness.

---

## Graph-OFF variant

To capture the same decode without the CUDA graph (per-kernel launch overhead
visible), add `--disable-cuda-graph` to the launch and re-run Phase 2 with a
distinct prefix/output dir (e.g. `q35_graphoff_bs64_24step` →
`/tmp/q35prof_off2`).
