# Kimi-K2.5-NVFP4 on GB300 — model knowledge

**GB300 (sm_103) 2-node MNNVL port of the [`gb200/models/kimi`](../../../gb200/models/kimi/MODEL.md)
pack** — read that file for the env matrix, robustness items, and adapter caveats. This file only
records the GB300 deltas. Cluster basics (context, taints, stateful partition, broadcast):
[`../qwen35/MODEL.md`](../qwen35/MODEL.md).

## Deltas vs gb200/models/kimi

| | gb200 (leira) | gb300 (gcp-radixark-02) |
|---|---|---|
| pods | `mnnvl-kimi-<ID>-{0,1}` | `sglang-gb300-kimi-yushengsu-<ID>-{0,1}` (ID = `$(date +%Y%m%d-%H%M%S)`) |
| MNNVL | native (leira ComputeDomain) | ComputeDomain CRD present on the GKE cluster (DRA); **2-node NCCL MNNVL rendezvous on GKE is UNVERIFIED until the first run** |
| weights | `/root` (pod) — re-downloaded per pod | same, but backed by the node's **2.9T kube-ephemeral-ssd** (~600G download per pod per node; the 95G stateful partition only holds the JIT cache) |
| JIT cache | `/mnt/nvme-b/sglang-dot-cache` hostPath | `/mnt/stateful_partition/sglang-dot-cache` hostPath (persists; broadcastable via `../qwen35/broadcast_jit_cache.sh`) |
| LoRA download | HF secret on leira | **needs `hf-token-yanbin` secret on gcp-radixark-02** (private repo; leira relay is unavailable while that cluster is down) — setup tolerates the failure (`|| true`), but the variant cell can't run without the adapter |
| scheduling | — | needs **2 free cohort nodes** (4 GPU each) + cohort toleration; small requests |
| READY_TIMEOUT_MIN | 40 | 50 (kimi cold autotune + first-ever sm_103 JIT headroom) |

## Status

- Pack created 2026-06-06; **not yet validated on GB300** (pending: HF secret for the kimi LoRA +
  2 free nodes + the MNNVL-on-GKE question above). The qwen35 GB300 pack IS validated.
