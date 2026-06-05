#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8 ONLY — acc + bench + prompts + profile, base vs variant, SINGLE node (tp4/ep4).
# HARDENED port of kimi-regression/scripts/run_kimi.sh; launch flags from tune-lora-perf/run_script.sh
# (Yanbin's Qwen3.5 LoRA launch + graph-ON bs64 24-step profile).
#
# Per cell (base, then variant): launch graph-ON -> acc (logprobs) -> bench (bs16/32/64) ->
# prompt-check -> profile graph-ON bs64 (24-step, per run_script.sh) -> relaunch graph-OFF ->
# profile bs16 (12-step). Downloads incrementally.
#
# Robustness carried over from kimi-regression (see its SKILL.md "Hard-won robustness"):
#   * kill_all kills LOCAL orphaned kubectl-exec launchers + VERIFIES GPU=0 before any launch.
#   * patient wait_ready (FP8 JIT warmup + cuda-graph capture), DIED only when ALL sglang procs gone.
#   * launch retries once (transient rank death happens).
#   * bench: --result-filename + tee (NEVER grep|tail), per-bs server-log slice + serverlog_sanity.py
#     cross-check (bench output_throughput is occasionally a phantom — server log is ground truth).
#   * server log '>>' append-only across launches (preserves the scheduler's gen-throughput lines).
#   * qwen3.5-specific: SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 is REQUIRED on the LoRA cell
#     (mamba + cuda-graph WAR -> 'Thinking!!!!' decode garbage without it). The prompt-check catches
#     exactly that failure mode, which the prefill-only acc CANNOT see.
set -uo pipefail   # NOT -e: failures handled explicitly (launch retry); -e would abort on a transient.

# ===== identity / pod =====
ID="${ID:?export ID=<dns-safe-identifier> (names the pod: sglang-qwen35-<ID>)}"
POD=sglang-qwen35-${ID}
MODEL_PATH=/data/Qwen3.5-35B-A3B-FP8     # persistent per-node big disk (see qwen35-pod.yaml /data mount)
LORA_PATH=/data/qwen35_35b_lora_alpha
LORA_NAME=alpha
MAX_LORA_RANK=16
PORT=30000
ACC_DATA="${LORA_PATH}/compare_sample_train_data.pt"   # ships inside the LoRA adapter repo

# ===== cells: base (control) vs variant (candidate) — EDIT per A/B =====
# Each cell = REF (injected branch, SKILL.md §3) + LORA on/off + EXTRA server flags + ENVS (env prefix).
# DEFAULT: no-LoRA base  vs  trtllm-LoRA + the qwen3.5 opt stack (run_script.sh's launch).
# For an ACC REGRESSION check, make the two cells NUMERICALLY EQUIVALENT (e.g. trtllm-LoRA vs
# triton-LoRA, or env-on vs env-off) — base-vs-LoRA is an *intended* diff, not a regression.
BASE_REF=__bench_base;       BASE_LORA=off; BASE_EXTRA="";  BASE_ENVS=""
VARIANT_REF=__bench_variant; VARIANT_LORA=on
VARIANT_EXTRA="--moe-runner-backend sgl_flashinfer_trtllm --lora-use-virtual-experts"
# qwen3.5 LoRA opt stack (from run_script.sh / the kimi-regression env matrix):
#   MAIN_ALLOC=1  REQUIRED (mamba graph-on coherence — without it decode = 'Thinking!!!!')
#   SHRINK_SPLIT_K=1  opt-in fp32 split-K shrink (PR #26962), perf-only
# (the NVFP4 per-token/fusion envs are no-ops on FP8 — do not add them)
VARIANT_ENVS="SGLANG_OPT_LORA_OVERLAP_MAIN_ALLOC=1 SGLANG_ENABLE_LORA_SHRINK_SPLIT_K=1"
cell_ref(){   [ "$1" = base ] && echo "$BASE_REF"   || echo "$VARIANT_REF"; }
cell_lora(){  [ "$1" = base ] && echo "$BASE_LORA"  || echo "$VARIANT_LORA"; }
cell_extra(){ [ "$1" = base ] && echo "$BASE_EXTRA" || echo "$VARIANT_EXTRA"; }
cell_envs(){  [ "$1" = base ] && echo "$BASE_ENVS"  || echo "$VARIANT_ENVS"; }

# ===== workload =====
IN=2048; OUT=2048; PROF_OUT_ON=48; PROF_OUT_OFF=64; BENCH_BS="16 32 64"
# graph-ON profile = bs64, start-step 8, 24 steps (run_script.sh's recipe: forwards 8-31 are all decode,
# clean window). graph-OFF profile = bs16, start-step 4, 12 steps (kernel structure; rank-0 suffices).
OUTROOT=/tmp/qwen35_reg
RUN_ROOT="${RUN_ROOT:-$HOME/Downloads/sglang_qwen35_reg_${ID}_$(date +%Y%m%d_%H%M%S)}"
LOCAL_OUT="${RUN_ROOT}/qwen35"; mkdir -p "$LOCAL_OUT"
# Default to this script's own directory (works from the repo checkout OR ~/.claude/skills).
SKILL_SCRIPTS="${SKILL_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Server args common to BOTH cells — from run_script.sh (Yanbin's launch command).
COMMON="--model-path ${MODEL_PATH} --tp 4 --ep 4 --host 0.0.0.0 --port ${PORT} \
--cuda-graph-max-bs 64 --mem-fraction-static 0.8 --trust-remote-code \
--max-prefill-tokens 32768 --chunked-prefill-size 4096 \
--mamba-scheduler-strategy extra_buffer --enable-flashinfer-allreduce-fusion --attention-backend trtllm_mha"
ENV_COMMON="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"   # both cells (fair)

kh(){ kubectl exec "${POD}" -- bash -lc "$1"; }

# ---- BULLETPROOF cleanup (kimi-regression robustness #1, single-pod version) ----
kill_all(){
  pkill -9 -f "kubectl exec.*launch_server" 2>/dev/null || true   # LOCAL orphaned launch clients
  sleep 2
  for i in $(seq 1 30); do
    kubectl exec "$POD" -- bash -lc 'for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done; pkill -9 -f "[s]glang" 2>/dev/null; pkill -9 -f "[b]ench_one_batch" 2>/dev/null; true' >/dev/null 2>&1
    g=$(kh 'nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null|wc -l'|tr -d ' ')
    [ "${g:-1}" = 0 ] && { echo "  GPU clean (iter $i)"; break; }
    sleep 4
  done
  sleep 6
}

checkout(){
  kh "cd /root/sglang; git checkout -q --detach $1; pip install -e python >/tmp/pip.log 2>&1; git --no-pager log -1 --oneline"
}

# Pre-warm the HF dynamic-module cache so 4 ranks don't race copying trust_remote_code *.py.
prewarm(){ kubectl exec "$POD" -- bash -lc 'python3 -c "from transformers import AutoConfig,AutoTokenizer; m=\"/data/Qwen3.5-35B-A3B-FP8\"; AutoConfig.from_pretrained(m,trust_remote_code=True); AutoTokenizer.from_pretrained(m,trust_remote_code=True)" 2>/dev/null'; echo "prewarmed ${POD}"; }

# Record num_hidden_layers into meta.env (summary.py uses it for the per-layer metric).
# Qwen3.5's config.json nests it under text_config (multimodal-style layout) — check both.
record_layers(){
  local n; n=$(kh "python3 -c \"import json; c=json.load(open('${MODEL_PATH}/config.json')); print(c.get('num_hidden_layers') or c.get('text_config',{}).get('num_hidden_layers') or '')\"" 2>/dev/null | tr -d '[:space:]')
  [ -n "${n:-}" ] && echo "layers=${n}" >> "${RUN_ROOT}/qwen35/meta.env" && echo "  layers=${n}"
}

# ---- observable, patient wait_ready (FP8 JIT warmup + graph capture; cache shared across configs) ----
wait_ready(){
  for i in $(seq 1 180); do
    kh "curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1" && { echo "  READY (~$((i*10+12))s)"; return 0; }
    n=$(kh 'pgrep -cf "[s]glang" 2>/dev/null'|tr -d ' ')
    [ "${n:-0}" = 0 ] && { echo "  DIED (0 sglang procs)"; kh 'tr "\r" "\n" </tmp/server.log|grep -aviE "shards: +[0-9]+%|profile/s"|tail -15'; return 1; }
    [ $((i % 9)) = 0 ] && echo "  ...i=$i procs=$n | $(kh 'tr "\r" "\n" </tmp/server.log 2>/dev/null|grep -aiE "autotune|Tuning|capturing|warmup|ready"|tail -1')"
    sleep 10
  done
  echo "  TIMEOUT"; return 1
}

start_server(){ kubectl exec "$POD" -- bash -lc "cd /root/sglang && ${ENV_COMMON} $1 exec numactl --membind=0,1 python3 -m sglang.launch_server ${COMMON} $2 >> /tmp/server.log 2>&1" >/dev/null 2>&1 & }
# NOTE: '>>' (append, never truncate) — the server log is NEVER overwritten across launches/cells, so
# the scheduler's per-batch 'gen throughput' (ground-truth decode rate) is preserved. bench() slices
# THIS bench's lines via wc-l-before / tail-after into <cell>/bench/bs<bs>.serverlog.
# Server runs in the exec FOREGROUND; the LOCAL kubectl exec is backgrounded (trailing '&' is local).
# An in-pod '& echo $!' / setsid hangs the exec (sglang workers keep the stream open).

launch(){  # $1=cell  $2=on|off  → retries once on failure
  local lora extra envs lora_flags="" graph_flags="" flags attempt
  lora=$(cell_lora "$1"); extra=$(cell_extra "$1"); envs=$(cell_envs "$1")
  [ "$lora" = on ] && lora_flags="--enable-lora --max-loras-per-batch 1 --max-lora-rank ${MAX_LORA_RANK} --lora-backend triton --lora-paths ${LORA_NAME}=${LORA_PATH}"
  [ "$2" = off ] && graph_flags="--disable-cuda-graph"
  flags="${graph_flags} ${extra} ${lora_flags}"
  for attempt in 1 2; do
    kill_all
    start_server "$envs" "$flags"
    sleep 12
    kubectl exec "$POD" -- bash -lc 'pgrep -f "[s]glang.launch_server" >/dev/null 2>&1' || { echo "  server not up — restart"; start_server "$envs" "$flags"; }
    wait_ready && return 0
    echo "  launch attempt ${attempt} failed ($1 graph-$2) — $([ "$attempt" = 1 ] && echo 'retry clean' || echo 'give up')"
  done
  return 1
}

acc(){   local name=""; [ "$(cell_lora "$1")" = on ] && name="${LORA_NAME}"; local d="${OUTROOT}/$1/acc"
  kh "mkdir -p ${d}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${name}' --out ${d}/logprobs.json 2>&1 | tee ${d}/acc.log"; }
bench(){ local la="";   [ "$(cell_lora "$1")" = on ] && la="--lora-name ${LORA_NAME}"; local d="${OUTROOT}/$1/bench"
  # ALWAYS capture the server-log slice per bs + sanity-check bench-vs-server decode throughput.
  # The sanity verdict is ALSO written to bs<bs>.sanity (build_readme.py reads it — in kimi-regression
  # it only went to the driver's stdout and the published README's sanity column was always 'OK').
  kh "mkdir -p ${d}; cd /root/sglang; for bs in ${BENCH_BS}; do sl0=\$(wc -l </tmp/server.log 2>/dev/null||echo 0); python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} --batch-size \${bs} --input-len ${IN} --output-len ${OUT} ${la} --show-report --result-filename ${d}/bs\${bs}.jsonl 2>&1 | tee ${d}/bs\${bs}.log; tail -n +\$((sl0+1)) /tmp/server.log | tr '\r' '\n' | grep -aE 'Prefill batch|Decode batch' > ${d}/bs\${bs}.serverlog || true; python3 /root/serverlog_sanity.py ${d}/bs\${bs}.jsonl ${d}/bs\${bs}.serverlog 2>&1 | tee ${d}/bs\${bs}.sanity; done"; }
# Always-on prompt check: a clear table of the RAW output of every endpoint (base + LoRA, correctly
# routed) for this cell. Runs AFTER bench (server has seen sustained load) — a coherent prefix that
# collapsed to '!!!!' / 'Thinking!!!!' here is the qwen3.5 MAIN_ALLOC cuda-graph WAR (or a cousin),
# which the prefill-only acc CANNOT see. Output -> <cell>/prompts/prompts.md.
prompts(){ local lora ln=""; lora=$(cell_lora "$1"); local d="${OUTROOT}/$1/prompts"; [ "$lora" = on ] && ln="${LORA_NAME}"
  kh "mkdir -p ${d}; cd /root/sglang; python3 /root/prompts_probe.py --port ${PORT} --model ${MODEL_PATH} --lora '${ln}' --cell '$1' 2>&1 | tee ${d}/prompts.md"; }
prof(){  # $1=cell $2=on|off $3=bs $4=start-step $5=steps $6=output-len
  local la=""; [ "$(cell_lora "$1")" = on ] && la="--lora-name ${LORA_NAME}"; local d="${OUTROOT}/$1/profile_graph_$2/bs$3"
  kh "rm -rf ${d}; mkdir -p ${d}; cd /root/sglang; python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} --batch-size $3 --input-len ${IN} --output-len $6 ${la} --profile --profile-activities CPU GPU --profile-start-step $4 --profile-steps $5 --profile-prefix qwen35_$1_graph_$2_bs$3 --profile-output-dir ${d} --result-filename ${d}/bench.jsonl 2>&1 | tee ${d}/bench.log; find ${d} -name '*.trace.json.gz' -printf '  %p %s\n'|sort"; }
dl(){ mkdir -p "${LOCAL_OUT}"; kubectl exec "${POD}" -- bash -lc "cd ${OUTROOT} && tar -czf - $1" 2>/dev/null | tar -xzf - -C "${LOCAL_OUT}"; }
# Flattened, ASYMMETRIC trace pull (kimi-regression robustness #9, single-pod version — all 4 TP ranks
# live on this one pod, so no cross-pod split):
#   graph-ON  (bs64) -> ALL 4 TP ranks (the real-timing trace you actually read — get it complete).
#   graph-OFF (bs16) -> ONLY TP0 (~10x bigger per rank; kernel STRUCTURE only, 1 rank suffices).
# Layout: ${LOCAL_OUT}/<cell>/traces/graph_{on,off}/bs<bs>-TP-<r>.trace.json.gz (+ server_args.json on graph-on)
pull_traces(){  # $1=cell  $2=on|off  $3=bs
  # bash 3.2 (macOS): all RHS in a single `local` line is evaluated before any LHS is in scope, so
  # referencing $cell/$g in the same line breaks under `set -u`. Positionals first, derived second.
  local cell=$1 g=$2 bs=$3 ranks r s w
  local src="${OUTROOT}/${cell}/profile_graph_${g}/bs${bs}" dst="${LOCAL_OUT}/${cell}/traces/graph_${g}"
  mkdir -p "$dst"
  [ "$g" = on ] && ranks="0 1 2 3" || ranks="0"
  # The bench can return BEFORE the profiler finishes flushing the trace files (and a kubectl-exec
  # network blip can kill the local prof exec mid-run while the in-pod bench continues) — so WAIT
  # for the trace to appear and stop growing before pulling, up to ~6 min.
  # Filename carries BOTH ranks on an EP run: ...-TP-<r>-EP-<r>.trace.json.gz (plain -TP-<r>. without EP).
  for r in $ranks; do
    for w in $(seq 1 24); do
      s=$(kubectl exec "$POD" -- bash -lc "find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) -printf '%s\n' 2>/dev/null | head -1")
      [ "${s:-0}" -ge 10000 ] && break
      sleep 15
    done
    # Streaming ~100MB through `kubectl exec | cat` can TRUNCATE on a network blip — verify the
    # gzip integrity after each pull and retry up to 3 times; if whole-file pulls keep truncating,
    # fall back to a server-side `split -b 20m` + per-chunk size-verified pull + local reassembly
    # (20MB chunks survive the blips that kill 90MB streams).
    for w in 1 2 3; do
      kubectl exec "$POD" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" > "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null
      gzip -t "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null && break
      echo "  graph_${g} TP${r} pull corrupt/truncated (attempt ${w}) — retrying"
    done
    if ! gzip -t "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null; then
      echo "  graph_${g} TP${r}: whole-file pull keeps truncating — falling back to 20MB split chunks"
      kubectl exec "$POD" -- bash -lc "f=\$(find ${src} \( -name '*-TP-${r}-EP-*.trace.json.gz' -o -name '*-TP-${r}.trace.json.gz' \) 2>/dev/null | head -1); rm -rf /tmp/.pull_split; mkdir -p /tmp/.pull_split; [ -n \"\$f\" ] && split -b 20m \"\$f\" /tmp/.pull_split/part_; ls /tmp/.pull_split/ 2>/dev/null" > /tmp/.pull_parts.$$ 2>/dev/null
      : > "${dst}/bs${bs}-TP-${r}.trace.json.gz"
      while read -r part; do
        [ -n "$part" ] || continue
        for w in 1 2 3 4 5; do
          kubectl exec "$POD" -- bash -lc "cat /tmp/.pull_split/${part}" > /tmp/.pull_chunk.$$ 2>/dev/null
          want=$(kubectl exec "$POD" -- bash -lc "stat -c%s /tmp/.pull_split/${part}" 2>/dev/null | tr -d '[:space:]')
          got=$(stat -f%z /tmp/.pull_chunk.$$ 2>/dev/null || stat -c%s /tmp/.pull_chunk.$$ 2>/dev/null || echo 0)
          [ -n "$want" ] && [ "$want" = "$got" ] && break
          echo "    ${part}: ${got}/${want:-?}B (attempt ${w}) — retrying"
        done
        cat /tmp/.pull_chunk.$$ >> "${dst}/bs${bs}-TP-${r}.trace.json.gz"
      done < /tmp/.pull_parts.$$
      rm -f /tmp/.pull_parts.$$ /tmp/.pull_chunk.$$
      kubectl exec "$POD" -- bash -lc "rm -rf /tmp/.pull_split" 2>/dev/null
    fi
    s=$(stat -f%z "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null || stat -c%s "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null || echo 0)
    if [ "${s:-0}" -lt 10000 ] || ! gzip -t "${dst}/bs${bs}-TP-${r}.trace.json.gz" 2>/dev/null; then
      echo "  MISSING graph_${g} TP${r} (${s}B)"; rm -f "${dst}/bs${bs}-TP-${r}.trace.json.gz"
    else
      echo "  graph_${g}/bs${bs}_TP${r}  $((s/1024/1024))M"
    fi
  done
  if [ "$g" = on ]; then
    kubectl exec "$POD" -- bash -lc "f=\$(find ${src} -name 'server_args.json' 2>/dev/null | head -1); [ -n \"\$f\" ] && cat \"\$f\"" > "${dst}/server_args.json" 2>/dev/null
    [ -s "${dst}/server_args.json" ] || rm -f "${dst}/server_args.json"
  fi
  return 0   # informational pulls only — MUST NOT fail run_cell's `launch && {...} ||` chain
}

# acc_capture.py (per-token logprobs over compare_sample_train_data.pt via /generate return_logprob)
cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests, json
ap=argparse.ArgumentParser(); ap.add_argument("--port",required=True); ap.add_argument("--data",required=True); ap.add_argument("--lora",default=""); ap.add_argument("--out",required=True); a=ap.parse_args()
data=torch.load(a.data,weights_only=False); toks=data["tokens"]
if torch.is_tensor(toks): toks=toks.tolist()
seqs=toks if (toks and isinstance(toks[0],list)) else [toks]; lp=[]
for s in seqs:
    p={"input_ids":s,"sampling_params":{"max_new_tokens":0,"temperature":0.0},"return_logprob":True,"logprob_start_len":0}
    if a.lora: p["lora_path"]=a.lora
    r=requests.post(f"http://127.0.0.1:{a.port}/generate",json=p,timeout=1800); r.raise_for_status()
    lp+=[x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]   # [1:] skips BOS (no logprob)
json.dump(lp,open(a.out,"w")); print("wrote",len(lp),"logprobs ->",a.out)
PY
kubectl cp /tmp/acc_capture.py ${POD}:/root/acc_capture.py >/dev/null

# Prompt-check uses the standalone scripts/prompts_check.py (single source; also runnable ad-hoc).
kubectl cp "${SKILL_SCRIPTS}/prompts_check.py" ${POD}:/root/prompts_probe.py >/dev/null 2>&1 \
  || echo "WARN: ${SKILL_SCRIPTS}/prompts_check.py not found — set SKILL_SCRIPTS to the skill's scripts dir"

# serverlog_sanity.py: cross-checks each bench's output_throughput against the server log's own
# decode "gen throughput" (catches anomalous bench numbers — ground truth is the scheduler's log).
kubectl cp "${SKILL_SCRIPTS}/serverlog_sanity.py" ${POD}:/root/serverlog_sanity.py >/dev/null 2>&1 \
  || echo "WARN: ${SKILL_SCRIPTS}/serverlog_sanity.py not found"

run_cell(){  # $1=cell (base|variant)  — runs ALL FOUR tests (acc, bench, prompts, profile)
  echo "================= CELL $1 ================="
  checkout "$(cell_ref "$1")"
  launch "$1" on || { echo "[$1] graph-ON launch FAILED after retry — skipping cell"; return 1; }
  acc   "$1"; dl "$1/acc";                    echo "[$(date +%H:%M:%S)] qwen35 $1 ACC done"   | tee -a "${RUN_ROOT}/progress.log"
  bench "$1"; dl "$1/bench";                  echo "[$(date +%H:%M:%S)] qwen35 $1 BENCH done" | tee -a "${RUN_ROOT}/progress.log"
  prompts "$1"; dl "$1/prompts";              echo "[$(date +%H:%M:%S)] qwen35 $1 PROMPTS done" | tee -a "${RUN_ROOT}/progress.log"
  prof "$1" on 64 8 24 ${PROF_OUT_ON}; pull_traces "$1" on 64     # graph-on: bs64 24-step (run_script.sh), all 4 TP ranks
  if launch "$1" off; then
    prof "$1" off 16 4 12 ${PROF_OUT_OFF}; pull_traces "$1" off 16
  else
    echo "[$1] graph-OFF launch FAILED — graph-off profile skipped"
  fi
  echo "[$(date +%H:%M:%S)] qwen35 $1 PROFILE done"  | tee -a "${RUN_ROOT}/progress.log"
}

prewarm
record_layers
run_cell base
run_cell variant
kill_all
ntr=$(find "${LOCAL_OUT}" -name '*.trace.json.gz' | wc -l | tr -d ' ')
if [ "${ntr}" -gt 0 ] && find "${LOCAL_OUT}" -name '*.trace.json.gz' -exec gzip -t {} + 2>/dev/null; then
  echo "traces integrity OK (${ntr} files)"
else
  echo "WARN: ${ntr} trace files locally — check pull_traces output for MISSING ranks"
fi
echo "[$(date +%H:%M:%S)] qwen35 DONE (all local) -> ${LOCAL_OUT}" | tee -a "${RUN_ROOT}/progress.log"

# Optional publish to a private GitHub results repo + Release (opt-in: set RESULTS_REPO=<owner>/<repo>).
# Small artifacts (acc/bench/prompts/README) -> a new commit at runs/<RUN_TAG>/; big traces -> a
# Release tagged <RUN_TAG>. Append-only. See SKILL.md §5.5.
if [ -n "${RESULTS_REPO:-}" ]; then
  echo "[$(date +%H:%M:%S)] qwen35 PUBLISH -> $RESULTS_REPO" | tee -a "${RUN_ROOT}/progress.log"
  RUN_ROOT="$RUN_ROOT" RESULTS_REPO="$RESULTS_REPO" \
    bash "${SKILL_SCRIPTS}/publish.sh" 2>&1 | tee -a "${RUN_ROOT}/publish.log" || \
    echo "[$(date +%H:%M:%S)] qwen35 PUBLISH FAILED (run still local at ${LOCAL_OUT})" | tee -a "${RUN_ROOT}/progress.log"
fi
