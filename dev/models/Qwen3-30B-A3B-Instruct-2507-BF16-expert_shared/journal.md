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
