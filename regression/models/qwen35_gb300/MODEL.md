# Qwen3.5-35B-A3B-FP8 on GB300 — model knowledge

**GB300 (sm_103, Blackwell Ultra) port of the [`qwen35`](../qwen35/MODEL.md) pack** — read that
file for the full env matrix, robustness items, and adapter caveats; everything there applies.
This file only records the GB300 deltas.

## Target cluster

- kubectl context **`gcp-radixark-02`** (GKE) — 25× `a4x-maxgpu-4g-metal` nodes,
  `nvidia-gb300`, 4 GPU each. (GB200 runs live on the `leira` context instead.)
- Run every step with `kubectl config use-context gcp-radixark-02` (or `--context`).

## Deltas vs the GB200/leira pack

| | `qwen35` (GB200, leira) | `qwen35_gb300` (GB300, gcp-radixark-02) |
|---|---|---|
| pod.yaml | `runtimeClassName: nvidia-legacy`, privileged, `/data` + `/root/.cache` hostPath (persistent weights + JIT cache) | no runtimeClass, no privileged, **no hostPath** — weights re-download to ephemeral `/root` per pod; JIT cache cold per pod |
| MODEL_PATH / LORA_PATH | `/data/...` | `/root/...` |
| pod name | `sglang-qwen35-<ID>` | `sglang-qwen35gb300-<ID>` |
| serving config | identical (PR #27329 launch: tp4/ep4, graph-max-bs 128, experimental_sgl_trtllm variant cell) | identical |

## GB300-specific notes

- The `hf-token-yanbin` secret must exist on the cluster for the private LoRA download
  (copy it from leira: read the token there, `kubectl --context gcp-radixark-02 create secret
  generic hf-token-yanbin --from-literal=token=...`).
- Cold setup is slower than leira: ~40 GB model download + full JIT warmup on every fresh pod
  (no persistent caches). Budget `READY_TIMEOUT_MIN=30` still holds but the FIRST launch pays
  the full deep_gemm JIT.
- sm_103 kernel support depends on the image/flashinfer build — if a cell crashes during JIT
  warmup with an arch error, that's image/commit skew (SKILL.md operational notes), not a
  harness failure.
- **Measured numbers (2026-06-06 validation run, `full-lora-opti@ac51ef5ed`, pinned image
  `97e7cd69…` + flashinfer 0.6.11.post1):** no-LoRA ceiling **3570 / 6206 / 10836 tok/s** at
  bs16/32/64; fast-path LoRA **2771 / 4964 / 8808 = 77.6 / 80.0 / 81.3 %** of the ceiling —
  same shape as GB200 (78.6/81.3/81.8%), absolute throughput ~2% higher. Decode coherent
  (no `!!!!`-collapse), all bench-vs-serverlog sanity ≤3%.
- **The FIRST variant launch cold-compiles the trtllm_lora_temp JIT for sm_103 and can exceed
  30 min** — on the validation run attempt 1 timed out at 30 min and the automatic retry came
  up READY in ~8 min on the warm JIT cache. `READY_TIMEOUT_MIN=45` now gives attempt 1 enough
  headroom (the cache is in ephemeral /root — every fresh pod pays this once).
