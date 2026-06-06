# dev/ — GB300 dev loop (launch → upload code → bench → profile → publish)

Minimal scripts to test **local sglang dev code** (LoRA vs no-LoRA) on the **GB300 cluster
(`gcp-radixark-02`)** for **Qwen3.5-35B-A3B-FP8** and **Kimi-K2.5-NVFP4**.
One script = one step; every step **verifies itself** and exits non-zero on failure, so the
chain (`run_all.sh`) stops at the first problem. Pod specs and server flags are reused from the
validated [`../regression/gb300`](../regression/gb300) packs — knowledge/caveats live in those
`MODEL.md` files.

```
dev/
├── common.sh            # shared config (qwen|kimi) + proven launch/wait/kill helpers
├── 1_launch_node.sh     # launch pod(s), wait for weights+install, verify GPUs
├── 2_upload_code.sh     # push the CURRENT local sglang branch to the pods + pip install -e
├── 3_run_benchmark.sh   # LoRA vs no-LoRA bench → input/extend, decode, e2e table
├── 4_run_acc.sh         # LoRA vs no-LoRA accuracy → per-token logprob diff vs ACC_TOL
├── 5_run_profile.sh     # LoRA vs no-LoRA torch profiles → <DATE>-<TIME>/{lora,no-lora}/
├── 6_upload_results.sh  # push the run dir to github.com/<you>/lora_perf_lora_profile
├── run_all.sh           # the whole chain: run_all.sh <qwen|kimi|all>
├── .state/<model>.env   # pod ID + RUN_DIR handoff between steps (written by 1, read by 2-6)
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
bash dev/run_all.sh qwen     # or: kimi, or: all
# `qwen`/`kimi` are CLI shorthands for Qwen3.5-35B-A3B-FP8 / Kimi-K2.5-NVFP4 (full names also
# accepted); state files and results dirs are keyed by the FULL model name:
#   dev/.state/Qwen3.5-35B-A3B-FP8.env , dev/results/Kimi-K2.5-NVFP4/<DATE>-<TIME>/ , …
```

Total ≈ 2–5 h per model (first-ever run on a node also pays the weight download and the cold
sm_103 JIT/autotune — Qwen ~40GB+45min JIT once per node; Kimi ~600GB to ephemeral disk and
fp4 autotune ~10-25 min on EVERY launch). Steps 3/4/5 each launch their own servers (2 cells
each = 6 launches total) — that's the price of standalone, individually-rerunnable steps.

## The steps (inputs / outputs)

### 1. `1_launch_node.sh <qwen|kimi>` — launch the node

| | |
|---|---|
| input | model name; optional `ID=<dns-safe-id>` (default `date +%Y%m%d-%H%M%S`) |
| does | **free-node pre-check** (the pod requests a full node's 4 GPUs, so the K8s scheduler auto-places it on an empty GB300 node; the pre-check fails fast instead of hanging Pending when none is free) → `kubectl apply` the regression pod yaml (qwen: 1 pod `sglang-gb300-qwen3vl-yushengsu-<ID>`; kimi: 2 pods + ComputeDomain/MNNVL, anti-affinity forces two different nodes), wait Ready, wait `/root/.setup-done` (weights + base install) |
| output | running pod(s); `dev/.state/<model>.env` with the pod `ID` |
| verify | pod Ready + setup-done on every pod + ≥4 GPUs visible per pod |

### 2. `2_upload_code.sh <qwen|kimi>` — upload the dev code

| | |
|---|---|
| input | state from step 1; `$SGLANG_SRC` current branch (committed HEAD; dirty tree ⇒ warning) |
| does | thin git bundle (only commits the pod lacks) → `kubectl cp` → checkout on every pod → `pip install -e python` → re-pin flashinfer `0.6.11.post1` (image-matching JIT cache) |
| output | every pod's `/root/sglang` at your local HEAD |
| verify | in-pod `git rev-parse HEAD` == local HEAD + `import sglang` succeeds, per pod |

### 3. `3_run_benchmark.sh <qwen|kimi>` — LoRA vs no-LoRA benchmark

| | |
|---|---|
| input | state; server cells: **no-lora** (stock backend) vs **lora** (`experimental_sgl_trtllm` + the model's required `SGLANG_*` env set, requests routed to the `alpha` adapter) |
| does | per cell: launch graph-ON server (1 retry; patient cold-JIT wait) → `bench_one_batch_server` bs 16/32/64, in=out=2048 → server-log slice → coherence probe → kill |
| output | `results/<model>/<DATE>-<TIME>/bench/{no-lora,lora}/bs<bs>.{jsonl,log,serverlog}` + `bench/summary.md` — table of **input/extend tok/s, decode tok/s, ITL ms, e2e s** + lora/no-lora decode ratio |
| verify | every `bs<bs>.jsonl` parses for both cells + per-cell post-load coherence (no `!!!!` decode collapse) |

### 4. `4_run_acc.sh <qwen|kimi>` — LoRA vs no-LoRA accuracy

| | |
|---|---|
| input | state; `${LORA_PATH}/compare_sample_train_data.pt` (token sequences **+ the vLLM/trainer reference logprobs**, shipped inside the adapter repo). Overrides: `ACC_DATA=<in-pod path>`, or `ACC_HF_FILE=<file>` (+`ACC_HF_REPO`, default `yushengsu/datasets`) to download a reference `.pt` from the private HF dataset (pod `HF_TOKEN` authenticates) |
| does | per cell: launch graph-ON server → teacher-forced **prefill-only** logprob capture (`/generate` with `max_new_tokens=0` + `return_logprob`) → coherence probe → kill; then locally: **(a)** lora-vs-no-lora per-token diff, **(b)** KL (=½·mean((a−b)²), `lora-dev-script`'s `kl_v2`) of sglang-lora against the `.pt`'s `training_logprobs` (trainer) and `sampling_logprobs` (original **vLLM** sampler), next to the inherent KL(vLLM, trainer) noise floor |
| output | `results/<model>/<DATE>-<TIME>/acc/{no-lora,lora}/logprobs.json` + `acc/summary.md` — lora-vs-no-lora table (n, mean/max abs, p50/p95, half-MSE, verdict vs `ACC_TOL`) + 3-row KL table: KL(vLLM,trainer) / KL(sglang,trainer) / KL(sglang,vLLM) — sglang matches the vLLM-era accuracy when KL(sglang,trainer) ≈ the floor |
| verify | both logprob sets captured + equal length + per-cell coherence. `max > ACC_TOL` is a **warning** by default (`ACC_STRICT=1` makes it fail); the KL-vs-reference rows are informational |

> acc is prefill-only — it **cannot** see decode-accumulating garbage (proven: a corrupted-decode
> server scored clean acc while generating `!!!!`). The per-cell coherence probe is the decode gate.

### 5. `5_run_profile.sh <qwen|kimi>` — LoRA vs no-LoRA profile

| | |
|---|---|
| input | state; profile recipe per model (qwen `bs64 start8 steps24`, kimi `bs16 start4 steps12` — clean-decode windows) |
| does | per cell: launch graph-ON server → `bench_one_batch_server --profile` (CPU+GPU) → pull every per-rank trace (gzip-verified, kimi ranks 4-7 pulled from the worker pod) |
| output | `results/<model>/<DATE>-<TIME>/{no-lora,lora}/bs<bs>-TP-<r>.trace.json.gz` (+ `bench.log`) — the `<DATE>-<TIME>` dir holds both cells' profiles, shared with steps 3/4 |
| verify | every expected trace exists locally and passes `gzip -t` |

### 6. `6_upload_results.sh <qwen|kimi>` — publish

| | |
|---|---|
| input | the run dir from step 3/4/5 state |
| does | check `github.com/<you>/lora_perf_lora_profile` exists (else `gh repo create` private) → commit the run dir to `runs/<model>/<DATE>-<TIME>/` → push (files >95MB skipped + listed) |
| output | `https://github.com/<you>/lora_perf_lora_profile/tree/main/runs/<model>/<DATE>-<TIME>` |
| verify | pushed path readable via the GitHub API |

### `run_all.sh <qwen|kimi|all>` — everything

Runs 1→2→3→4→5→6; each step's own verification gates the next; first failure aborts.

## Notes

- **Steps are resumable**: each reads `dev/.state/<model>.env`, so you can rerun any single step
  (e.g. iterate `2 → 3` on the same pods after a code change). Step 1 resets the state.
- **Cleanup** when done:
  `ID=<id> sh -c 'sed "s/\${ID}/$ID/g" ../regression/gb300/models/<pack>/pod.yaml | kubectl --context gcp-radixark-02 delete -f - --ignore-not-found'`
  (pack = `Qwen3.5-35B-A3B-FP8` | `Kimi-K2.5-NVFP4`; `<id>` is in `dev/.state/<model>.env`).
- Decode throughput is the headline number; `bs<bs>.serverlog` keeps the scheduler's own
  `gen throughput` lines as ground truth if a bench number looks suspicious (>5% mismatch ⇒
  rerun — see `../regression/SKILL.md` item 4).
- Kimi NVFP4+LoRA had a 4-bug crash chain at the PR merge commit
  (`../e2e_test_scripts/gb300/results/RESULTS.md`) — if the lora cell dies in JIT warmup on your
  branch, that's the first place to look.
