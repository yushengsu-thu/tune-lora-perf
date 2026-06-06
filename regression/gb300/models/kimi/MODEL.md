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

## Status — VALIDATED (2026-06-06, pinned `c9f582a27` + image `97e7cd69` + flashinfer 0.6.11)

- **Full e2e PASS on 2-node GB300 MNNVL** (`24wq` head + `qtbb` worker): driver exit 0,
  18/18 traces gzip-OK including the cross-pod 8-rank graph-on pull (TP0-3 head, TP4-7 worker),
  all 6 bench-vs-serverlog sanity checks ≤0.8%, decode coherent with the `alpha-` behavioral
  marker visible.
- **MNNVL-on-GKE works**: ComputeDomain/DRA claim allocation + 2-node NCCL rendezvous confirmed.
- **Throughput (bs16/32/64, tok/s):** ceiling **1245 / 2156 / 3600**; fast-path LoRA
  **1012 / 1907 / 3358 = 81.3 / 88.5 / 93.3 %** — matches the PR's GB200 claim (81/88/93%)
  digit-for-digit; GB300 absolutes slightly higher than the PR's GB200 numbers.
- Cold sm_103 `fp4_gemm` autotune ≈ 13 min (faster than GB200's ~20); the variant launch
  re-tunes (different GEMM shapes on the LoRA path). One transient rank death after capture
  (no traceback, NOT cgroup-OOM) was auto-recovered by the launch retry — attempt 2 READY in
  182 s on the warm cache. Same class as the GB200-era "transient rank death" note.
