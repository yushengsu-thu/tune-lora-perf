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
- ~~**Two-stream behaviour with shared-outer**: confirm the O1-bf16 side-stream overlap still nets a
  speedup in this layout.~~ → **DONE** (§6/§7: +20-22% on shared-outer, +20-25% on normal r32).

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

### 5. allreduce-fusion + extra flags investigation (2026-06-08)
User asked to add to BOTH cells: `--reasoning-parser qwen3 --tool-call-parser qwen3_coder
--mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion`, then re-profile;
and (follow-up) "if it crashes on allreduce fusion, FIX it, don't disable it."

Findings (launch sanity via dev launch_server, fusion ON):
- **`--mamba-scheduler-strategy extra_buffer` is the crash**, NOT allreduce fusion. Traceback:
  `schedule_batch.py:2102 _mamba_radix_cache_v2_req_prepare_for_extend →
   req.mamba_ping_pong_track_buffer[...] → TypeError: 'NoneType' object is not subscriptable`.
  Qwen3-30B-A3B-Instruct-2507 is a STANDARD-attention MoE (no mamba/linear layers), so the mamba
  radix-cache path has no track buffer. This flag does not apply to this model (that's why the base
  BF16 pack dropped it) — not a bug to fix.
- **`--enable-flashinfer-allreduce-fusion` works end-to-end on the current branch (526e0ae22).** With
  fusion ON + the two parser flags (no mamba flag): cuda-graph capture completed 100%, server READY,
  and REAL base + lora inference both coherent ("Paris. The capital of the United States is
  Washington, D.C."). NO illegal memory access, no traceback. server_args confirms
  enable_flashinfer_allreduce_fusion=True, experts_shared_outer_loras=True.
  → The documented "allreduce fusion ON → illegal memory access in cuda-graph capture" crash does
  NOT reproduce now (it predated the bf16 two-stream work / current branch state). Nothing to fix
  in the fusion path; the earlier `--enforce-disable-flashinfer-allreduce-fusion` was based on a
  stale observation. The two parser flags are API-layer (harmless to perf).
- **Bench fusion ON vs OFF (decode tok/s, bs 16/32/64, both cells; fusion ON also stresses bs64 +
  cuda-graph and stayed clean)** — model.env flipped to `--enable-flashinfer-allreduce-fusion` +
  the parser flags:

  | cell | bs | fusion OFF | fusion ON | Δ |
  |---|---|---|---|---|
  | no-lora | 16 | 3782.0 | 3932.5 | +4.0% |
  | no-lora | 32 | 6757.8 | 6991.4 | +3.5% |
  | no-lora | 64 | 11437.3 | 11757.9 | +2.8% |
  | lora | 16 | 2314.5 | 2365.6 | +2.2% |
  | lora | 32 | 4207.6 | 4278.6 | +1.7% |
  | lora | 64 | 7265.5 | 7424.1 | +2.2% |

  lora ITL 6.91/7.61/8.81 → 6.76/7.48/8.62 ms; lora coherent. → **fusion ON is a clean win** (both
  cells faster, no correctness/stability cost). model.env now defaults to fusion ON + parsers.
- **torch profiler traces (5_run_profile, fusion ON)**: lora + no-lora, bs64, per-rank CPU+GPU
  traces captured (cuda-graph ON = real timing) — another fusion-ON load pass, lora coherent, no
  crash. Traces in dev/results/.../20260608-130204/{no-lora,lora}/bs64-TP-{0..3}.trace.json.gz
  (no-lora ~5M/rank, lora ~71M/rank), all gzip-valid. The pod's GKE API-server connection truncates
  single >64MB streams (both `kubectl exec cat` and `kubectl cp` capped ~64MB), so one 72MB lora
  rank failed the legacy cat-retry pull → added a size-verified 8MB-chunk fallback to
  common.sh `pull_trace` (split on pod → per-chunk exact-size check + retry → reassemble); all 4
  lora ranks then validated. Open the traces in perfetto/chrome tracing to inspect the kernel
  timeline (the fused allreduce+rmsnorm replaces the standalone all-reduce + norm kernels).
- **Decision (user)**: keep fusion ON + the two parser flags as the pack default; mamba flag stays
  out. model.env / target_command.sh / the gb300 e2e runner all synced to this config.

### 6. two-stream ON vs OFF on shared-outer (2026-06-08)
Isolated the O1-bf16 two-stream contribution: same config (fusion ON, shared-outer, rank 32) with
the two-stream override DISABLED by clearing LORA_ENVS (no `SGLANG_EXPERIMENTAL_LORA_OPTI` →
`install_two_stream_overrides` not called → single-stream MoE LoRA dispatch). No side stream means
the `SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC` "decode garbage" hazard doesn't apply; decode stayed
coherent. lora decode tok/s, in=out=2048:

| bs | two-stream OFF | two-stream ON | gain |
|---|---|---|---|
| 16 | 1932.3 | 2365.6 | +22.4% |
| 32 | 3577.2 | 4278.6 | +19.6% |
| 64 | 6185.9 | 7424.1 | +20.0% |

→ Two-stream is a **+20-22%** decode win on this shared-outer real-adapter path — much larger than
the +6-8% measured on the earlier non-shared-outer r=16 dummy (the real r=32 + shared-outer LoRA
delta is heavier, so overlapping the gate_up shrink/expand on the side stream hides more work).
Both cells coherent.

### 7. Full factorial: {normal vs shared-outer} × {single vs two stream} × {fusion ON vs OFF} (2026-06-08)
All at rank 32 on the same pod, so the only knob between A and B is the `--experts-shared-outer-loras`
flag (caveat: A = a generated per-expert dummy targeting q/k/v/o/gate/up/down; B = the real all-linear
shared-outer adapter — module coverage differs, so the A-vs-B absolute gap mixes "module set" with
"shared-outer layout". The single-vs-two and ON-vs-OFF columns within a row are clean.) no-lora reuses
the default-backend baseline per fusion state (ON 3932.5/6991.4/11757.9, OFF 3782.0/6757.8/11437.3).

lora decode tok/s (bs16/32/64) — normal also measured at rank 16 (existing per-expert dummy):

| | single, fus ON | two, fus ON | single, fus OFF | two, fus OFF |
|---|---|---|---|---|
| A. normal r16 | 2160.9/4005.9/7067.5 | 2701.8/4887.1/8445.4 | 2110.5/3920.4/6897.2 | 2642.2/4762.5/8306.3 |
| A. normal r32 | 2073.8/3897.6/6939.0 | 2594.9/4723.5/8316.4 | 2046.2/3791.9/6784.4 | 2483.9/4558.2/8095.8 |
| B. shared-outer r32 | 1932.3/3577.2/6185.9 | 2365.6/4278.6/7424.1 | 1894.9/3487.9/6070.4 | 2314.5/4207.6/7265.5 |

lora / no-lora ratio (bs16/32/64):

| | single, fus ON | two, fus ON | single, fus OFF | two, fus OFF |
|---|---|---|---|---|
| A. normal r16 | 54.9/57.3/60.1% | 68.7/69.9/71.8% | 55.8/58.0/60.3% | 69.9/70.5/72.6% |
| A. normal r32 | 52.7/55.7/59.0% | 66.0/67.6/70.7% | 54.1/56.1/59.3% | 65.7/67.5/70.8% |
| B. shared-outer r32 | 49.1/51.2/52.6% | 60.2/61.2/63.1% | 50.1/51.6/53.1% | 61.2/62.3/63.5% |

Takeaways:
- **rank**: r16 > r32 > shared-outer (lower rank = lighter LoRA; r16↔r32 gap is small, ~1-3pt).
- **two-stream is the dominant win**: r16 ~+25%, r32 ~+20%, shared-outer ~+20% over single-stream
  (both fusion states).
- **fusion ON vs OFF barely moves the lora cells** (<1pt ratio at fixed stream; fusion's benefit is
  mostly on the no-lora fused-backend path).
- normal is faster than shared-outer at equal rank (e.g. two+ON bs64: A 8316 vs B 7424) — partly the
  shared-outer triton branch, partly B's wider (all-linear) module coverage.
- coherence: B (real adapter) identical coherent text across all four cells; A (random dummy) text
  varies single-vs-two (numerical noise amplified by random weights) — not a correctness issue.
- artifact: a generated normal per-expert dummy at rank 32 now lives at
  /data/Qwen3-30B-A3B-Instruct-2507-dummy-lora-r32 on the pod (for this comparison).
