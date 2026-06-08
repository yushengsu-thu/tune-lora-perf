# Journal — Qwen3-30B-A3B-Instruct-2507 (BF16, experts_shared_outer_loras) on GB300 via `tune-lora-perf/dev`

> Running log of this dev task: every directive + every action/command executed, with outcomes.
> Newest entries appended at the bottom of the chronological log.

## [to-do optimization]

Invasive / not-yet-attempted follow-ups. Strike through (`~~...~~`) when attempted, with a pointer
to the log entry.

- **If the KL-vs-reference shows shared-outer is the wrong serving mode for this adapter**: confirm
  the adapter's true expert-LoRA layout (per-expert vs shared-outer / expert_dim=1) from
  adapter_model.safetensors shapes, and decide whether `--experts-shared-outer-loras` should be ON
  for it (the flag force-overrides auto-detect).
- **Two-stream behaviour with shared-outer**: the gate_up/down LoRA deltas under shared-outer take a
  different triton kernel branch (num_experts_for_weight==0 sentinel); confirm the O1-bf16 side-stream
  overlap still nets a speedup in this layout (phase-2 work landed for the non-shared-outer case).

---

## Goal (directive, 2026-06-08)

> 在 /Users/yushengsu/Downloads/tml/sglang 的既有 branch qwen3-30b-a3b-2507-bf16 上，新增並驗證一個
> 「experts_shared_outer_loras 開啟」的 BF16 變體 model pack，用真實 adapter + vLLM/trainer reference
> 做 logprob 比較。
>
> 【建立 model pack】新增 dev/models/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared/（model.env /
> pod.yaml / target_command.sh），差異：LORA_EXTRA 加 --experts-shared-outer-loras；--max-lora-rank 32；
> 換掉 dummy，服務真實 adapter yushengsu/lora-diff-Qwen3-30B-A3B-Instruct-2507（含 adapter +
> compare_sample_train_data.pt），啟動前下到每個 pod 的 /data；target_command.sh 固化啟動命令；acc 不設
> ACC_HF_FILE（用 bundled .pt）。其餘維持 trtllm_mha / experimental_sgl_trtllm / virtual-experts /
> allreduce-fusion off / bf16 / TP4EP4 / 四個 opt env。ACC_TOL 依實測調整。
>
> 【acc】沿用 dev/4_run_acc.sh（已內建 sglang↔vLLM/trainer KL 比較），不另寫腳本。
>
> 【測試】用 dev 在現有 pod（別重建、別重抓 base；python 改動走 warm JIT cache）：1) lora 啟動成功；
> 2) coherence；3) 4_run_acc 印 KL 表，判讀 KL≈floor→shared-outer 正確 / KL≫floor→記錄結論；
> 4) 3_run_benchmark bs 16/32/64 比 lora vs no-lora decode tok/s（含 ITL/ratio）；5) gb300 LoRA e2e。
>
> 【sglang 改動】預期不需要（flag 已被 experimental_sgl_trtllm+bf16 路徑支援；透傳 + dtype-agnostic
> 佈局 + 無守衛）。撞到 bug 可自行評估修，記 journal；硬性限制：FP8/FP4 byte-for-byte 不變、不動權重
> 佈局、每階段 commit。
>
> 【發佈】全部驗證通過 + 確認後才 push / 更新 PR。

## Fixed environment

| item | value |
|---|---|
| sglang branch | `qwen3-30b-a3b-2507-bf16` @ `526e0ae22` (the bf16-in-experimental work + O1-bf16 two-stream) |
| codebase | /Users/yushengsu/Downloads/tml/sglang |
| docker image | `lmsysorg/sglang:nightly-dev-cu13-20260603-83bc7766` |
| cluster / pod | gcp-radixark-02, REUSING `sglang-gb300-qwen330-yushengsu-bf16test-20260607` (node 6zvh) |
| base model | Qwen/Qwen3-30B-A3B-Instruct-2507 (bf16, ~60G, already on /data) |
| adapter | yushengsu/lora-diff-Qwen3-30B-A3B-Instruct-2507 (r=32, α=32, all-linear) + bundled compare_sample_train_data.pt |
| config | TP4/EP4, experimental_sgl_trtllm, trtllm_mha, virtual-experts, **--experts-shared-outer-loras**, allreduce-fusion off |

---

## Chronological log

### 1. Pack + tooling created (2026-06-08)
- Added `ensure_hf_lora` to `dev/common.sh` (parallel to `ensure_dummy_lora`): a model.env with
  `LORA_PATH=hf` + `LORA_HF_REPO` resolves to a `/data/<repo-basename>` dir, patches LORA_EXTRA's
  `--lora-paths` token, sets `LORA_FROM_HF=1`; `ensure_hf_lora` then `hf download`s the repo onto each
  pod (pod HF_TOKEN auth, idempotent on adapter_model.safetensors). Wired into `1_launch_node.sh`
  after `ensure_dummy_lora`.
- New pack `dev/models/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared/`: model.env (rank 32,
  `--experts-shared-outer-loras`, `LORA_PATH=hf` + LORA_HF_REPO), pod.yaml (copy of base BF16 pack,
  same POD_PREFIX → reuses the running pod), target_command.sh (fixed launch cmd), this journal.
- Wired the new model's state to the running BF16 pod (`ID=bf16test-20260607`, node 6zvh) and ran
  `ensure_hf_lora` → adapter_model.safetensors (1.98G) + adapter_config.json + the 22K
  compare_sample_train_data.pt landed at /data/lora-diff-Qwen3-30B-A3B-Instruct-2507. Pod code
  already at 526e0ae22 (this task touched only dev/ laptop scripts; no sglang change, no re-upload).
- Committed stage 1 (tune-lora-perf `a9f6b2b`): ensure_hf_lora + the new pack.

### 2. Verify — launch + coherence + acc (4_run_acc, 2026-06-08) — PASS
`./4_run_acc.sh Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared`
(results dir dev/results/.../20260608-130204/acc/)
- **Launch**: both cells READY, NO FP8 assert / NO illegal memory access — `--experts-shared-outer-loras`
  serves cleanly on the bf16 experimental_sgl_trtllm path (as predicted: flag is plumbed + dtype-agnostic).
- **Coherence**: lora + base both coherent ("Paris. The capital of the United States is Washington, D.C. ...").
- **KL vs vLLM/trainer reference (THE gate)** — bundled compare_sample_train_data.pt, 1820 tokens:

  | pair | KL (0.5·mean((a−b)²)) |
  |---|---|
  | orig_sampler (vLLM) vs trainer — noise floor | 0.004243 |
  | **sglang-lora vs trainer** | **0.005636 ≈ floor** |
  | sglang-lora vs orig_sampler (vLLM) | 0.005670 ≈ floor |

  → **`--experts-shared-outer-loras` is the CORRECT serving mode for this adapter** — sglang reproduces
  the vLLM/trainer reference logprobs to within the inherent noise floor. The adapter genuinely IS a
  shared-outer (expert_dim=1) adapter; serving it with the flag matches the reference.
- lora-vs-no-lora |diff| table (mean 0.996, EXCEEDS the 0.30 placeholder tol) is EXPECTED: this is a
  real trained adapter that legitimately diverges from base (not the near-identity alpha mock); the
  KL-vs-reference table above is the real correctness gate. (ACC_TOL is informational here.)

### 3. Bench — lora vs no-lora (3_run_benchmark, 2026-06-08) — PASS
`./3_run_benchmark.sh Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared` (bs 16/32/64, in=out=2048):

| cell | bs16 | bs32 | bs64 |
|---|---|---|---|
| no-lora decode tok/s | 3782.0 | 6757.8 | 11437.3 |
| lora decode tok/s | 2314.5 | 4207.6 | 7265.5 |
| lora ITL ms | 6.91 | 7.61 | 8.81 |
| lora / no-lora | 61.2% | 62.3% | 63.5% |

- lora cell coherent ("Paris. The capital of the United States is Washington, D.C. ...").
- This is the real r=32 shared-outer adapter; the two-stream O1-bf16 overlap (SGLANG_EXPERIMENTAL_LORA_OPTI=1)
  is active. (For reference the earlier non-shared-outer r=16 dummy two-stream run was
  68.9/70.4/73.2% of ceiling — different adapter/rank/layout, not a like-for-like comparison;
  the goal's ask was lora-vs-no-lora, captured above.)
- results dir: dev/results/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared/20260608-130204/bench/

### 4. gb300 LoRA e2e (2026-06-08) — PASS
New runner `e2e_test_scripts/gb300/Qwen3-30B-A3B-Instruct-2507-BF16-expert_shared_run_gb300.sh`
(sibling of the qwen FP8 runner; bf16 base + real adapter + --experts-shared-outer-loras + rank 32,
allreduce-fusion off, no mamba-strategy, NOCO=1 so it runs on the dev-prepared pod). Deployed the
model-agnostic helpers (gsm8k_lora.py / bench_report.py / prompts_check.py) + runner to the pod and
fired detached. HEAD=526e0ae22, backend=experimental_sgl_trtllm, flashinfer=0.6.11.post1.

- **sgl-lora cell**: READY, COHERENT (base + lora).
  - **gsm8k 5-shot**: base **0.950** (1/200 truncated), **lora 0.940** (2/200 truncated) — the real
    adapter served with shared-outer ON preserves accuracy (lora ≈ base), consistent with the
    acc KL ≈ floor result.
  - bench (in=out=2048, server-log xcheck all OK):

    | bs | base tput | base ITL | lora tput | lora ITL |
    |---|---|---|---|---|
    | 16  | 2441.9  | 6.55 | 2317.0  | 6.91 |
    | 32  | 4396.8  | 7.28 | 4200.2  | 7.62 |
    | 64  | 7598.6  | 8.42 | 7268.6  | 8.81 |
    | 128 | 12741.5 | 10.05 | 12258.7 | 10.44 |
- **nolora cell**: READY, base coherent, gsm8k base **0.950** (matches the sgl-lora cell's base ceiling).
- The single "Killed" line in run.log is the inter-cell server teardown (kill_all) — harmless; the
  nolora cell launched READY immediately after.

## Verification summary (all PASS)
1. Launch (target_command): clean, no FP8 assert / no illegal memory access.
2. Coherence: lora + base coherent.
3. acc KL vs vLLM/trainer: 0.0056 ≈ floor 0.0042 → **--experts-shared-outer-loras is the correct
   serving mode** for this adapter (sglang reproduces the reference logprobs).
4. bench (lora vs no-lora) bs 16/32/64: lora 61.2/62.3/63.5% of no-lora ceiling.
5. gb300 e2e: gsm8k base 0.950 / lora 0.940; bench bs16-128 xcheck OK.

No sglang code change was needed — the flag was already supported on the bf16 experimental path.
