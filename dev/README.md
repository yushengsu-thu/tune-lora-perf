# dev/ — GB300 dev loop (launch → upload code → bench → profile → publish)

Minimal scripts to test **local sglang dev code** (LoRA vs no-LoRA) on the **GB300 cluster
(`gcp-radixark-02`)**. The scripts are **model-general**: every model is a directory under
[`models/`](models/) named `${MODEL_NAME}-${PRECISION}` (currently `Qwen3.5-35B-A3B-FP8` and
`Kimi-K2.5-NVFP4`), and **all model-specific parameters live in that directory's `model.env`**.
One script = one step; every step **verifies itself** and exits non-zero on failure, so the
chain (`run_all.sh`) stops at the first problem. Pod specs default to the validated
[`../regression/gb300`](../regression/gb300) packs — knowledge/caveats live in those
`MODEL.md` files.

## Codebase & image

| | |
|---|---|
| **Docker image** | `lmsysorg/sglang:nightly-dev-cu13-20260603-83bc7766` (pinned dated nightly; ships flashinfer-jit-cache `0.6.11.post1+cu130`, torch `2.11.0+cu130`, py3.12, sglang `@83bc77661`). Set in the pod packs the dev loop reuses ([`../regression/gb300/models/<model>/pod.yaml`](../regression/gb300)). |
| **sglang codebase** | your **local checkout** (`SGLANG_SRC`, default `/Users/yushengsu/Downloads/tml/sglang`); `2_upload_code.sh` uploads the **current branch's committed HEAD**. The LoRA-perf work tracks **`trtllm-lora-bf16`** on [`github.com/yushengsu-thu/sglang`](https://github.com/yushengsu-thu/sglang) (override what to upload with `SGLANG_BRANCH=<ref>`). |
| **flashinfer** | re-pinned to `0.6.11.post1` after `pip install -e` (matches the image's baked jit-cache; the branch pyproject's 0.6.12 breaks the sm_103 `trtllm_lora_temp` JIT). Override with `FLASHINFER_PIN`. |

```
dev/
├── common.sh            # model-agnostic config + proven launch/wait/kill helpers
├── jit_store.sh         # ★ shared laptop JIT-cache store (sourced by common.sh; CLI for regression/e2e)
├── jit_fp.cmd           # the compile-input fingerprint command (single source of truth)
├── models/              # ★ one dir per model = ${MODEL_NAME}-${PRECISION}
│   ├── Qwen3.5-35B-A3B-FP8/
│   │   ├── model.env                    # ALL qwen params (tp/ep, paths, flags, envs, recipes)
│   │   └── jit-cache/<fp>.tgz           # saved compile cache per code fingerprint (+ .meta, INDEX)
│   └── Kimi-K2.5-NVFP4/{model.env,jit-cache/}
├── 1_launch_node.sh     # launch pod(s), wait for weights+install, verify GPUs
├── 2_upload_code.sh     # push the CURRENT local sglang branch + pip install -e + RESTORE warm cache
├── 3_run_benchmark.sh   # LoRA vs no-LoRA bench → input/extend, decode, e2e table
├── 4_run_acc.sh         # LoRA vs no-LoRA accuracy → per-token logprob diff vs ACC_TOL
├── 5_run_profile.sh     # LoRA vs no-LoRA torch profiles (cuda-graph ON+OFF) → <DATE>-<TIME>/{lora,no-lora}/graph_{on,off}/
├── 6_upload_results.sh  # push the run dir to github.com/<you>/lora_perf_lora_profile
├── 7_broadcast_jit_cache.sh  # (optional) copy this node's JIT cache to ALL GB300 nodes (node→node)
├── 8_save_jit_cache.sh  # SAVE the node's compiled JIT cache to dev/models/<m>/jit-cache/ (node→laptop)
├── run_all.sh           # the whole chain: run_all.sh <model|all>  (now ends with step 8)
├── .state/<model>.env   # pod ID + RUN_DIR handoff between steps (written by 1, read by 2-8)
└── results/<model>/<DATE>-<TIME>/   # everything a run produces, locally
```

## Prerequisites (once)

- `kubectl` context **`gcp-radixark-02`** works (`kubectl --context gcp-radixark-02 get nodes`).
  All scripts pin this context per-command — they never touch your current context, and never
  use leira (gone).
- `gh auth status` OK (step 6 creates/pushes the results repo).
- Local sglang checkout at `/Users/yushengsu/Downloads/tml/sglang` (override: `SGLANG_SRC=…`);
  the **current branch's committed HEAD** is what gets uploaded.
- The `hf-token-yanbin` secret exists on the cluster (private LoRA adapters — see
  `../regression/gb300/models/Qwen3.5-35B-A3B-FP8/MODEL.md`).

## Run everything

```bash
bash dev/run_all.sh Qwen3.5-35B-A3B-FP8   # any dir name under dev/models/
bash dev/run_all.sh qwen                  # …or any unique case-insensitive prefix of one
bash dev/run_all.sh all                   # every model under dev/models/, in turn
# State files and results dirs are keyed by the FULL model dir name (after prefix resolution):
#   dev/.state/Qwen3.5-35B-A3B-FP8.env , dev/results/Kimi-K2.5-NVFP4/<DATE>-<TIME>/ , …
```

## Add a new model

Create `dev/models/${MODEL_NAME}-${PRECISION}/model.env` (copy an existing one) and set every
variable in it — that file is the **single place** for model parameters:

| group | variables |
|---|---|
| topology | `NNODES`, `TP`, `EP`, `GPUS_PER_NODE`, `DIST_PORT` (required if `NNODES>1`) |
| pods | `POD_PREFIX`; optional `PACK`/`POD_YAML` (default: `../regression/gb300/models/<model>/pod.yaml`) |
| paths | `MODEL_PATH`, `LORA_PATH` (in-pod) |
| server | `SERVER_COMMON` (flags both cells share), `ENV_COMMON`, `LORA_EXTRA` (lora-cell flags), `LORA_ENVS` (lora-cell `SGLANG_*` envs) |
| recipes | `BENCH_BS`/`BENCH_IN`/`BENCH_OUT`, `ACC_TOL`, `PROF_RECIPE` (`bs start steps outlen`), `TRACE_RANKS` |
| timeouts | `READY_TIMEOUT_MIN` (server JIT/autotune), `POD_READY_TIMEOUT` (k8s) |

`common.sh` sources the file and fails fast if any required variable is missing. The dir name is
the model name everywhere (CLI arg, state file, results dir, published runs); a unique prefix of
it works as a CLI shorthand automatically. A multi-node model also needs a pod yaml whose pods
follow `${POD_PREFIX}-${ID}-<n>` + a `-head` service (see the kimi regression pack).

Total ≈ 2–5 h per model (first-ever run on a node also pays the weight download and the cold
sm_103 JIT/autotune — Qwen ~40GB+45min JIT once per node; Kimi ~600GB to ephemeral disk and
fp4 autotune ~10-25 min on EVERY launch). Steps 3/4/5 each launch their own servers (2 cells
each = 6 launches total) — that's the price of standalone, individually-rerunnable steps.

## The steps (inputs / outputs)

### 1. `1_launch_node.sh <model>` — launch the node

| | |
|---|---|
| input | model name; optional `ID=<dns-safe-id>` (default `date +%Y%m%d-%H%M%S`) |
| does | **free-node pre-check** (the pod requests a full node's 4 GPUs, so the K8s scheduler auto-places it on an empty GB300 node; the pre-check fails fast instead of hanging Pending when none is free) → `kubectl apply` the pack's `POD_YAML` (e.g. qwen: 1 pod `sglang-gb300-qwen35-yushengsu-<ID>`; kimi: 2 pods + ComputeDomain/MNNVL, anti-affinity forces two different nodes), wait Ready, wait `/root/.setup-done` (weights + base install) |
| output | running pod(s); `dev/.state/<model>.env` with the pod `ID` |
| verify | pod Ready + setup-done on every pod + ≥4 GPUs visible per pod |

### 2. `2_upload_code.sh <model>` — upload the dev code

| | |
|---|---|
| input | state from step 1; `$SGLANG_SRC` current branch (committed HEAD; dirty tree ⇒ warning) |
| does | thin git bundle (only commits the pod lacks) → `kubectl cp` → checkout on every pod → `pip install -e python` → re-pin flashinfer `0.6.11.post1` (image-matching JIT cache) → **laptop JIT-cache RESTORE**: fingerprint the compile-relevant inputs (flashinfer/torch versions + every `*.cu/cuh/cpp/h` / `jit`/`kernel` source); if `dev/models/<model>/jit-cache/<fp>.tgz` exists for that fingerprint, **extract it into every pod's `/root/.cache`** so the launch skips the >30-min cold sm_103 JIT. No saved cache for this code (compile inputs changed / first time) → the next launch JIT-compiles, then **step 8** saves a new fp-keyed tarball |
| output | every pod's `/root/sglang` at your local HEAD; warm `/root/.cache` when a matching cache was saved |
| verify | in-pod `git rev-parse HEAD` == local HEAD + `import sglang` succeeds, per pod (the restore is best-effort: a miss just means the launch compiles) |

### 3. `3_run_benchmark.sh <model>` — LoRA vs no-LoRA benchmark

| | |
|---|---|
| input | state; server cells: **no-lora** (stock backend) vs **lora** (`experimental_sgl_trtllm` + the model's required `SGLANG_*` env set, requests routed to the `alpha` adapter) |
| does | per cell: launch graph-ON server (1 retry; patient cold-JIT wait) → `bench_one_batch_server` with the pack's `BENCH_BS`/`BENCH_IN`/`BENCH_OUT` (currently bs 16/32/64, in=out=2048) → server-log slice → coherence probe → kill |
| output | `results/<model>/<DATE>-<TIME>/bench/{no-lora,lora}/bs<bs>.{jsonl,log,serverlog}` + `bench/summary.md` — table of **input/extend tok/s, decode tok/s, ITL ms, e2e s** + lora/no-lora decode ratio |
| verify | every `bs<bs>.jsonl` parses for both cells + per-cell post-load coherence (no `!!!!` decode collapse) |

### 4. `4_run_acc.sh <model>` — LoRA vs no-LoRA accuracy

| | |
|---|---|
| input | state; `${LORA_PATH}/compare_sample_train_data.pt` (token sequences **+ the vLLM/trainer reference logprobs**, shipped inside the adapter repo). Overrides: `ACC_DATA=<in-pod path>`, or `ACC_HF_FILE=<file>` (+`ACC_HF_REPO`, default `yushengsu/datasets`) to download a reference `.pt` from the private HF dataset (pod `HF_TOKEN` authenticates) |
| does | per cell: launch graph-ON server → teacher-forced **prefill-only** logprob capture (`/generate` with `max_new_tokens=0` + `return_logprob`) → coherence probe → kill; then locally: **(a)** lora-vs-no-lora per-token diff, **(b)** KL (=½·mean((a−b)²), `lora-dev-script`'s `kl_v2`) of sglang-lora against the `.pt`'s `training_logprobs` (trainer) and `sampling_logprobs` (original **vLLM** sampler), next to the inherent KL(vLLM, trainer) noise floor |
| output | `results/<model>/<DATE>-<TIME>/acc/{no-lora,lora}/logprobs.json` + `acc/summary.md` — lora-vs-no-lora table (n, mean/max abs, p50/p95, half-MSE, verdict vs `ACC_TOL`) + 3-row KL table: KL(vLLM,trainer) / KL(sglang,trainer) / KL(sglang,vLLM) — sglang matches the vLLM-era accuracy when KL(sglang,trainer) ≈ the floor |
| verify | both logprob sets captured + equal length + per-cell coherence. `max > ACC_TOL` is a **warning** by default (`ACC_STRICT=1` makes it fail); the KL-vs-reference rows are informational |

> acc is prefill-only — it **cannot** see decode-accumulating garbage (proven: a corrupted-decode
> server scored clean acc while generating `!!!!`). The per-cell coherence probe is the decode gate.

> ⚠ **Reference-applicability rule:** the `.pt` references in `hf.co/datasets/yushengsu/datasets`
> only apply when the served **adapter carries the `experts_shared_outer_loras` tag**
> ([sgl-project/sglang#21466](https://github.com/sgl-project/sglang/pull/21466)). A **general**
> adapter (e.g. `jybsuper/qwen35_35b_lora_alpha`) must use its **own bundled**
> `compare_sample_train_data.pt` (the default) — against a mismatched reference the KL table is
> meaningless (measured 2026-06-06: KL≈0.42–0.52 vs floor 0.0006, **even for the no-lora cell**;
> the lora-vs-no-lora table stays valid either way). The script detects the tag in
> `adapter_config.json` and refuses a mismatched `ACC_HF_FILE` (`ACC_FORCE=1` overrides).

### 5. `5_run_profile.sh <model>` — LoRA vs no-LoRA profile

| | |
|---|---|
| input | state; graph-ON `PROF_RECIPE`+`TRACE_RANKS` (qwen `bs64 start8 steps24`, kimi `bs16 start4 steps12` — clean-decode windows) and graph-OFF `PROF_OFF`+`TRACE_RANKS_OFF` (default light `bs16 start4 steps12`, TP0 only) |
| does | per cell, **both cuda-graph ON and OFF**: launch server (OFF adds `--disable-cuda-graph`) → `bench_one_batch_server --profile` (CPU+GPU) → pull that mode's traces (gzip-verified; multi-node ranks map to pods via `GPUS_PER_NODE`). graph-ON = real timing (all `TRACE_RANKS`); graph-OFF = kernel structure (`TRACE_RANKS_OFF`, usually TP0, ~10x bigger). 4 launches total (2 cells × {on,off}) |
| output | `results/<model>/<DATE>-<TIME>/{no-lora,lora}/graph_{on,off}/bs<bs>-TP-<r>.trace.json.gz` (+ `bench.log` per mode) — the pair feeds `analyze_llm_torch_profile.py --mapping-input graph_off --formal-input graph_on`; the `<DATE>-<TIME>` dir holds both cells' profiles, shared with steps 3/4 |
| verify | every expected trace exists locally and passes `gzip -t` |

### 6. `6_upload_results.sh <model>` — publish

| | |
|---|---|
| input | the run dir from step 3/4/5 state |
| does | check `github.com/<you>/lora_perf_lora_profile` exists (else `gh repo create` private) → commit the run dir to `runs/<model>/<DATE>-<TIME>/` → push (files >95MB skipped + listed) |
| output | `https://github.com/<you>/lora_perf_lora_profile/tree/main/runs/<model>/<DATE>-<TIME>` |
| verify | pushed path readable via the GitHub API |

### 7. `7_broadcast_jit_cache.sh <model>` — (optional) warm every node

The JIT/compile cache (deep_gemm, flashinfer, triton, trtllm_lora_temp) already **persists
per node**: the pod mounts `/root/.cache` on the node's
`/mnt/stateful_partition/sglang-dot-cache` (hostPath), so a relaunch on the *same* node skips
the >30-min cold sm_103 JIT. This step copies that dir from the current run's node to **every
other GB300 GPU node** — **in-cluster direct transfer**: temporary non-GPU sync pods, the
source serves the tarball over the pod network and all targets pull **in parallel** with
size+gzip verification (~1.3GB → 16 nodes in minutes; mechanics in
[`../regression/gb300/models/Qwen3.5-35B-A3B-FP8/broadcast_jit_cache.sh`](../regression/gb300/models/Qwen3.5-35B-A3B-FP8/broadcast_jit_cache.sh)),
so **any future pod lands warm no matter which node it gets**. The cache dir is node-level and
model-agnostic — one broadcast covers everything compiled on the source node. Run it after a
successful run, while the pods still exist (the source node is looked up from the head pod).
Not part of `run_all` (multi-GB transfer; run it when the cache actually changed — step 2
tells you: its `jit_stamp` check prints **RECOMPILE expected** when the uploaded code changed
a compile input, and after the next successful launch the refreshed cache+stamp is what this
step broadcasts). Kimi's fp4 autotune is process-local and can't be cached — only its JIT
kernels benefit.

### 8. `8_save_jit_cache.sh <model>` — save the compiled cache to the laptop (node → laptop)

Downloads the node's freshly-compiled cache into the **per-model laptop store**
`dev/models/<model>/jit-cache/<fp>.tgz`, **keyed by the fingerprint of the code that built it**.
It captures only the **compile output** — `deep_gemm` autotune, `tvm-ffi` (the sgl_kernel JIT:
`moe_fused_gate` / `moe_lora` / `custom_all_reduce` / `topk_softmax` / …), `flashinfer`, `sglang`,
`torch` (and `triton` / `trtllm_lora_temp` when present) — and **excludes** the `huggingface`
(~658M) + `pip` (~1.1G) download caches: ~13MB gzip, which streams fine over `kubectl exec` (the
multi-GB in-cluster broadcast in step 7 is only needed for the FULL cache). Run it after a
successful run, while the pods still exist; it **skips** if that fingerprint is already saved
(`FORCE=1` overwrites). On the next `2_upload_code.sh`, if the code's compile inputs are unchanged,
that step **restores** this tarball so the launch is warm.

> **Steps 7 vs 8 (both warm future launches, different reach):** step 7 broadcasts a still-warm
> node's cache to the OTHER GB300 nodes **in-cluster** (node→node — breadth *now*). Step 8 saves it
> to the **laptop** (node→laptop — durability): it survives every node going cold / re-imaged / the
> pods being deleted, and re-warms a *fresh* pod on the next push. Use both. The cache extracts to
> the SAME in-pod path (`/root/.cache`) on every node, so it is byte-relocatable; flashinfer/
> deep_gemm/tvm-ffi key kernels by content hash, so a restore is never *incorrect* — at worst the
> launch recompiles the kernels that actually changed. Kimi's fp4 autotune is process-local and
> can't be cached — only its JIT kernels benefit.

The store is **shared** with the regression driver and the e2e GB300 scripts via `dev/jit_store.sh`
(`jit_store.sh save|restore|fits <model> <pod> [--context CTX]`) — one fp-keyed store per model
warms all three flows. The `<fp>.tgz` blobs are gitignored (force-add to share across machines);
the `<fp>.meta` + `INDEX` sidecars are tracked.

### `run_all.sh <model|all>` — everything

Runs 1→2→3→4→5→6→8; each step's own verification gates the next; first failure aborts. Step 8
saves the freshly-compiled cache at the end so the next push lands warm.
(7 is manual — broadcast when you want to warm the *other* nodes too.)

## Notes

- **Steps are resumable**: each reads `dev/.state/<model>.env`, so you can rerun any single step
  (e.g. iterate `2 → 3` on the same pods after a code change). Step 1 resets the state.
- **Cleanup** when done:
  `ID=<id> sh -c 'sed "s/\${ID}/$ID/g" ../regression/gb300/models/<model>/pod.yaml | kubectl --context gcp-radixark-02 delete -f - --ignore-not-found'`
  (`<model>` = the dev/models dir name; if the pack overrides `POD_YAML`, delete that yaml
  instead; `<id>` is in `dev/.state/<model>.env`).
- Decode throughput is the headline number; `bs<bs>.serverlog` keeps the scheduler's own
  `gen throughput` lines as ground truth if a bench number looks suspicious (>5% mismatch ⇒
  rerun — see `../regression/SKILL.md` item 4).
- Kimi NVFP4+LoRA had a 4-bug crash chain at the PR merge commit
  (`../e2e_test_scripts/gb300/results/RESULTS.md`) — if the lora cell dies in JIT warmup on your
  branch, that's the first place to look.
