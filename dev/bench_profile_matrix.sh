#!/usr/bin/env bash
# Per-opt single×two matrix: {OFF-delta vs ON-delta} × {single-stream, two-stream}.
#   bench (graph-ON, real timing) for all 4 cells; profile (graph-OFF, kernel structure) for the
#   2 ON cells. single = SGLANG_TWO_STREAM_MAX_TOKENS=0 (two-stream gate off); two = default
#   (256, two-stream active for decode). OFF/ON are env-delta strings appended to LORA_ENVS
#   (explicit, so they override any model.env default) — pass a single flag ("FLAG=0" / "FLAG=1")
#   or a bundle ("A=0 B=0" / "A=1 B=1").
#   Usage: bash dev/bench_profile_matrix.sh <model> <out-subdir> "<off-delta>" "<on-delta>"
. "$(dirname "$0")/common.sh" "${1:-}"
SUB="${2:?need out subdir}"; OFF_DELTA="${3:?need off-delta}"; ON_DELTA="${4:?need on-delta}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/${SUB}"
LA="--lora-name ${LORA_NAME}"; BASE="$LORA_ENVS"
echo "== [matrix] $MODEL  off='${OFF_DELTA}'  on='${ON_DELTA}'  -> ${OUT}"

stream_env(){ [ "$1" = single ] && echo "SGLANG_TWO_STREAM_MAX_TOKENS=0" || echo ""; }
delta(){ [ "$1" = on ] && echo "$ON_DELTA" || echo "$OFF_DELTA"; }

# ---- bench: 4 cells, graph-ON ----
for VAR in off on; do for ST in single two; do
  LORA_ENVS="$BASE $(delta $VAR) $(stream_env $ST)"
  echo "---- bench $VAR $ST  ($LORA_ENVS) ----"
  launch_server lora on || { echo "ERROR launch bench $VAR/$ST"; exit 1; }
  coherence_check lora || { echo "ERROR coherence $VAR/$ST"; exit 1; }
  D="/tmp/mx/bench/${VAR}_${ST}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; for bs in ${BENCH_BS}; do
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size \${bs} --input-len ${BENCH_IN} --output-len ${BENCH_OUT} ${LA} \
          --show-report --result-filename ${D}/bs\${bs}.jsonl 2>&1 | tail -3
      done" || echo "  WARN: bench $VAR/$ST exec returned nonzero (benign bench-client shutdown) — jsonl validated at summary"
  pull_dir "$D" "${OUT}/bench/${VAR}_${ST}"
done; done

# ---- profile: graph-OFF kernel structure, ON-delta, both streams ----
for ST in single two; do
  LORA_ENVS="$BASE $ON_DELTA $(stream_env $ST)"
  echo "---- profile on $ST (graph-off) ----"
  launch_server lora off || { echo "ERROR launch prof $ST"; exit 1; }
  coherence_check lora || true
  D="/tmp/mx/prof/${ST}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size 16 --input-len ${BENCH_IN} --output-len 48 ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step 6 --profile-steps 12 \
        --profile-prefix ${MODEL}_${ST}_on --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tail -3" || echo "  WARN: prof $ST exec returned nonzero (benign) — pulling trace next"
  mkdir -p "${OUT}/profile/${ST}_on"
  pull_trace 0 "$D" "${OUT}/profile/${ST}_on/bs16-TP-0.trace.json.gz" && echo "  pulled ${ST}_on" || echo "  WARN pull ${ST}_on"
done
kill_all

# ---- matrix summary ----
python3 - "${OUT}/bench" "$BENCH_BS" "$OFF_DELTA" "$ON_DELTA" > "${OUT}/summary.md" <<'PY'
import json,sys,pathlib
root,bss,off,on=pathlib.Path(sys.argv[1]),sys.argv[2].split(),sys.argv[3],sys.argv[4]
def last(p):
    try: return [json.loads(l) for l in (root/p).read_text().splitlines() if l.strip()][-1]
    except: return None
g=lambda v:("%.1f"%v) if isinstance(v,(int,float)) else "—"
print(f"# single×two matrix\n\n- OFF = `{off}`\n- ON  = `{on}`\n")
print("| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |")
print("|---|---|---|---|---|---|")
cell={}
for v in("off","on"):
  for st in("single","two"):
    for bs in bss:
      r=last(f"{v}_{st}/bs{bs}.jsonl")
      if r: cell[(v,st,bs)]=r; print(f"| {v} | {st} | {bs} | {g(r.get('input_throughput'))} | {g(r.get('output_throughput'))} | {g(r.get('latency'))} |")
def rat(a,b): return ("%.1f%%"%(100*a/b)) if (a and b) else "—"
b0=bss[0]
print(f"\n## ON/OFF ratio @bs{b0} (prefill & decode >100% = faster)\n")
print("| stream | prefill | decode | e2e |\n|---|---|---|---|")
for st in("single","two"):
  o=cell.get(("off",st,b0)); n=cell.get(("on",st,b0))
  if o and n:
    print(f"| {st} | {rat(n['input_throughput'],o['input_throughput'])} | {rat(n['output_throughput'],o['output_throughput'])} | {rat(n['latency'],o['latency'])} |")
PY
echo "== [matrix] done -> ${OUT}/summary.md"; cat "${OUT}/summary.md"
