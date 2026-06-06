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
| pod.yaml | `runtimeClassName: nvidia-legacy`, privileged, hostPath on `/mnt/nvme-b` (big dedicated raid) | no runtimeClass, no privileged, hostPath on **`/mnt/stateful_partition`** (the node's 95G persistent NVMe partition): `/root/.cache` (JIT — kills the >30-min cold sm_103 build on relaunches) + `/data` (weights — no re-download/relay per pod). Wiped only on node re-image. **Disk budget is tight:** model 40G + adapter 2.4G + JIT cache on a 95G partition — don't park other large artifacts there. |
| MODEL_PATH / LORA_PATH | `/data/...` | `/data/...` (same paths now) |
| pod name | `sglang-qwen35-<ID>` | `sglang-qwen35gb300-<ID>` |
| serving config | identical (PR #27329 launch: tp4/ep4, graph-max-bs 128, experimental_sgl_trtllm variant cell) | identical |
| scheduling | — | needs the `radixark.io/cohort=true:NoSchedule` toleration (free nodes carry it); keep requests small (busy cluster) |

## GB300-specific notes

- The `hf-token-yanbin` secret must exist on the cluster for the private LoRA download
  (copy it from leira: read the token there, `kubectl --context gcp-radixark-02 create secret
  generic hf-token-yanbin --from-literal=token=...`). **Without it**: the base model still
  downloads (public) and setup completes; relay the adapter ONCE per node from a leira pod
  (20MB verified chunks — plain `kubectl cp`/`exec | tar` streams TRUNCATE multi-GB files) into
  `/data/qwen35_35b_lora_alpha/`; it persists on the node afterwards.
- First-ever run on a node pays ~40 GB model download + the full cold JIT; **subsequent pods on
  the same node reuse both** via the `/mnt/stateful_partition` hostPath mounts.
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
