# GB300 port of the e2e test scripts (gcp-radixark-02)

GB300 (sm_103, GKE `a4x-maxgpu-4g-metal` / `nvidia-gb300`, 18 nodes × 4 GPU, one NVL72
fabric) port of the GB200 scripts in `../gb200/` (historical — that cluster is gone) —
see `../README.md` for the shared files and `../../E2E_FULL_TEST_RUNBOOK.md` for the
methodology. **Helpers (`gsm8k_lora.py`, `bench_report.py`, `prompts_check.py`) are
hardware-agnostic — deploy the ones from `../` unchanged.** Validated end-to-end on
2026-06-06; numbers + the kimi NVFP4+LoRA bug report in [`results/RESULTS.md`](results/RESULTS.md).

Every step: `kubectl config use-context gcp-radixark-02` (or `--context gcp-radixark-02`).

## What changed vs GB200 (and why)

| delta | reason |
|---|---|
| pod specs: no `runtimeClassName`, `radixark.io/cohort=true:NoSchedule` toleration, hostPath on `/mnt/stateful_partition` (95G persistent NVMe) | GKE cluster conventions — copied from `../../regression/gb300/models/Qwen3.5-35B-A3B-FP8/pod.yaml` and verified against working pods on the cluster |
| image **pinned by digest** `97e7cd69…` + `flashinfer==0.6.11.post1` re-pin (setup + per-run guard in scripts; `FLASHINFER_PIN` env to override) | floating `dev-cu13` / the branch pyproject's flashinfer 0.6.12 changed `get_sf_out_offset_*` signatures → `trtllm_lora_temp` JIT compile error on sm_103. Re-pin when the PR rebases. |
| READY waits: qwen 35→45 min, tp1 25→45 min, kimi 36→48 min | first sm_103 cold JIT exceeded 30 min on the 2026-06-06 regression validation; warm-cache relaunch ≈8 min. Cache persists per node via the `dot-cache` hostPath — broadcast it with `../../regression/gb300/models/Qwen3.5-35B-A3B-FP8/broadcast_jit_cache.sh` |
| `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` **removed** from `Qwen3.5-35B-A3B-FP8_run_gb300.sh`'s PR opt set | the runbook guardrail says NEVER set it (corrupts base / decode garbage) — the GB200 script still carried it; bug fixed in this port |
| PR moe backend parameterized: `PRBACKEND` (default `experimental_sgl_trtllm`) | verified on `full-lora-opti@ac51ef5ed` (2026-06-06): the old `sgl_flashinfer_trtllm` name is gone from the choices — the rebased branch uses the post-PR-#27329-merge naming. Override `PRBACKEND` for older refs. |
| kimi: ComputeDomain `numNodes: 2` + `allocationMode: Single`, memory limit 1800→850Gi, weights on ephemeral `/root` | CD shape copied from a working 2-node deployment on this cluster; node allocatable is ~907Gi memory / 2.5TiB ephemeral; the 95G stateful partition can't hold Kimi's ~600G weights (→ re-download per pod creation; only the JIT cache persists) |

Optional per-run env for all scripts: `REINSTALL=1` re-runs `pip install -e python` after the
checkout (the GB200 flow never reinstalled; use it when the ref's deps changed).

## Files

| file | what |
|---|---|
| `Qwen3.5-35B-A3B-FP8-pod.yaml` | single-node TP4/EP4 qwen pod (4 GPU). `ID=$(date +%Y%m%d-%H%M%S) envsubst < Qwen3.5-35B-A3B-FP8-pod.yaml \| kubectl apply -f -` |
| `Qwen3.5-35B-A3B-FP8-tp1-pods.yaml` | tp1 pr/oss pods (1 GPU each), `kubectl apply -f` directly |
| `Kimi-K2.5-NVFP4-gb300.yaml` | headless svc + ComputeDomain + head/worker kimi pods (2×4 GPU MNNVL) |
| `Qwen3.5-35B-A3B-FP8_run_gb300.sh` | `bash Qwen3.5-35B-A3B-FP8_run_gb300.sh <full-lora-opti\|main> <TAG>` |
| `Qwen3.5-35B-A3B-FP8_base_gb300.sh` | `bash Qwen3.5-35B-A3B-FP8_base_gb300.sh <TAG>` — oss no-LoRA ceiling (%-denominator) |
| `Qwen3.5-35B-A3B-FP8_tp1_gb300.sh` | `bash Qwen3.5-35B-A3B-FP8_tp1_gb300.sh <full-lora-opti\|main-base> <TAG>` |
| `Kimi-K2.5-NVFP4_run_gb300.sh` | `bash Kimi-K2.5-NVFP4_run_gb300.sh <worker\|head> <full-lora-opti\|main> <LORA> <BACKEND> <DISTADDR\|-> <TAG> <full\|gsm8k_only>` — `-` = the Kimi-K2.5-NVFP4-gb300.yaml head FQDN. **Worker FIRST, then head.** |
| `jit_cache.sh` | `bash jit_cache.sh <restore\|save> <model> <pod> [git-url branch]` — laptop JIT-cache warm-start/save (wrapper over `../../dev/jit_store.sh`); skips the cold sm_103 JIT on a fresh pod |

## Run order (qwen, one 4-GPU pod)

```bash
kubectl --context gcp-radixark-02 apply -f <(ID=$(date +%Y%m%d-%H%M%S) envsubst < Qwen3.5-35B-A3B-FP8-pod.yaml)
# wait for /root/.setup-done, then deploy helpers:
for f in gsm8k_lora.py bench_report.py prompts_check.py; do
  kubectl --context gcp-radixark-02 exec -i <pod> -- bash -c "mkdir -p /tmp/flo_helpers; cat > /tmp/flo_helpers/$f" < ../$f
done
kubectl --context gcp-radixark-02 exec -i <pod> -- bash -c 'cat > /tmp/Qwen3.5-35B-A3B-FP8_run_gb300.sh' < Qwen3.5-35B-A3B-FP8_run_gb300.sh
# OPTIONAL JIT warm-start — skips the >30-min cold sm_103 JIT if a matching cache was saved before.
# Checks out the ref on the pod, then restores dev/models/<model>/jit-cache/<fp>.tgz; the run
# script's own checkout is then a no-op and its launch lands warm:
bash jit_cache.sh restore qwen <pod> https://github.com/yushengsu-thu/sglang trtllm-lora-bf16
# fire detached (runbook §5): setsid bash /tmp/Qwen3.5-35B-A3B-FP8_run_gb300.sh full-lora-opti g300pr > /tmp/run.log 2>&1
# then the same pod sequentially: Qwen3.5-35B-A3B-FP8_run_gb300.sh main / Qwen3.5-35B-A3B-FP8_base_gb300.sh
# AFTER a successful run, save the freshly-compiled cache to the laptop store (fp-keyed):
bash jit_cache.sh save qwen <pod>
```

> **JIT cache (compile parts) — `jit_cache.sh`:** a thin wrapper over the shared store
> `../../dev/jit_store.sh` (one fp-keyed cache per model under `dev/models/<model>/jit-cache/`,
> reused by the dev loop, the regression driver, and these e2e scripts). `restore` warms a fresh pod
> (skipping the cold compile of `moe_fused_gate` / `moe_lora` / `custom_all_reduce` / `deep_gemm` /
> `trtllm_lora_temp`); `save` captures it after a run. The cache also persists per node via the
> `dot-cache` hostPath and spreads node→node via
> `../../regression/gb300/models/Qwen3.5-35B-A3B-FP8/broadcast_jit_cache.sh`; the laptop store is the
> durable copy that survives every node going cold / re-imaged.

## Kimi run order

```bash
kubectl --context gcp-radixark-02 apply -f Kimi-K2.5-NVFP4-gb300.yaml      # svc + CD + 2 pods
# ~600G download per pod — watch /root/setup.log on both. Then per cell:
#   0. (optional) warm-start BOTH pods:  bash jit_cache.sh restore kimi <head-pod>; bash jit_cache.sh restore kimi <worker-pod>
#   1. worker pod:  bash /tmp/Kimi-K2.5-NVFP4_run_gb300.sh worker <REF> <LORA> <BACKEND> - <TAG> full
#   2. head pod:    bash /tmp/Kimi-K2.5-NVFP4_run_gb300.sh head   <REF> <LORA> <BACKEND> - <TAG> full
#   3. (after) save the JIT kernels (fp4 autotune is NOT cacheable):  bash jit_cache.sh save kimi <head-pod>
```

## Guardrails (unchanged from `../README.md`)

- Never set `SGLANG_OPT_LORA_DOWN_FINALIZE_OVERLAP` / `SGLANG_OPT_LORA_ENABLE_PDL`.
- Keep flashinfer autotune ON; one server per pod / exclusive port; fire detached and verify
  via `/tmp/srv.log`.
- gsm8k signal: the **base** number (~0.77–0.81 qwen / ~0.95 kimi); `req-lora` ~0.01–0.04
  is expected (alpha = behavioral identity adapter).
- `hf-token-yanbin` secret must exist on the cluster (private LoRA adapters); without it the
  qwen adapter can be relayed per `../../regression/gb300/models/Qwen3.5-35B-A3B-FP8/MODEL.md`, but the
  kimi adapter download will fail.
- GB300 reference numbers (regression 2026-06-06): qwen no-LoRA ceiling 3570/6206/10836 tok/s
  @bs16/32/64; fast-path LoRA 77.6/80.0/81.3% of ceiling (GB200: 78.6/81.3/81.8%).
