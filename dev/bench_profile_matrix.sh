#!/usr/bin/env bash
# Per-opt single×two matrix: {FLAG off/on} × {single-stream, two-stream}.
#   bench (graph-ON, real timing) for all 4 cells; profile (graph-OFF, kernel structure) for the
#   2 flag-on cells. single = SGLANG_TWO_STREAM_MAX_TOKENS=0 (two-stream gate off); two = default
#   (256, two-stream active for decode). FLAG is the opt's gate, held off(=0)/on(=1).
#   Usage: bash dev/bench_profile_matrix.sh <model> <FLAG_NAME> <out-subdir>
#   e.g.   bash dev/bench_profile_matrix.sh Qwen3-...-expert_shared SGLANG_OPT_LORA_FUSED_MERGED_ALIGN opt1-matrix
. "$(dirname "$0")/common.sh" "${1:-}"
FLAG="${2:?need FLAG name}"; SUB="${3:?need out subdir}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/${SUB}"
LA="--lora-name ${LORA_NAME}"; BASE="$LORA_ENVS"
echo "== [matrix] $MODEL  FLAG=$FLAG  -> ${OUT}"

stream_env(){ [ "$1" = single ] && echo "SGLANG_TWO_STREAM_MAX_TOKENS=0" || echo ""; }

# ---- bench: 4 cells, graph-ON ----
for FV in 0 1; do for ST in single two; do
  LORA_ENVS="$BASE $FLAG=$FV $(stream_env $ST)"
  echo "---- bench flag=$FV stream=$ST  ($LORA_ENVS) ----"
  launch_server lora on || { echo "ERROR launch bench $FV/$ST"; exit 1; }
  coherence_check lora || { echo "ERROR coherence $FV/$ST"; exit 1; }
  D="/tmp/mx/bench/${FV}_${ST}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; for bs in ${BENCH_BS}; do
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size \${bs} --input-len ${BENCH_IN} --output-len ${BENCH_OUT} ${LA} \
          --show-report --result-filename ${D}/bs\${bs}.jsonl 2>&1 | tail -3
      done" || { echo "ERROR bench $FV/$ST"; exit 1; }
  pull_dir "$D" "${OUT}/bench/${FV}_${ST}"
done; done

# ---- profile: graph-OFF kernel structure, flag-on, both streams ----
for ST in single two; do
  LORA_ENVS="$BASE $FLAG=1 $(stream_env $ST)"
  echo "---- profile flag=1 stream=$ST (graph-off) ----"
  launch_server lora off || { echo "ERROR launch prof $ST"; exit 1; }
  coherence_check lora || true
  D="/tmp/mx/prof/${ST}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
      python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
        --batch-size 16 --input-len ${BENCH_IN} --output-len 48 ${LA} \
        --profile --profile-activities CPU GPU --profile-start-step 6 --profile-steps 12 \
        --profile-prefix ${MODEL}_${ST}_on --profile-output-dir ${D} \
        --result-filename ${D}/bench.jsonl 2>&1 | tail -3" || { echo "ERROR prof $ST"; exit 1; }
  mkdir -p "${OUT}/profile/${ST}_on"
  pull_trace 0 "$D" "${OUT}/profile/${ST}_on/bs16-TP-0.trace.json.gz" && echo "  pulled ${ST}_on" || echo "  WARN pull ${ST}_on"
done
kill_all

# ---- matrix summary ----
python3 - "${OUT}/bench" "$BENCH_BS" "$FLAG" > "${OUT}/summary.md" <<'PY'
import json,sys,pathlib
root,bss,flag=pathlib.Path(sys.argv[1]),sys.argv[2].split(),sys.argv[3]
def last(p):
    try: return [json.loads(l) for l in (root/p).read_text().splitlines() if l.strip()][-1]
    except: return None
g=lambda v:("%.1f"%v) if isinstance(v,(int,float)) else "—"
print(f"# single×two matrix — {flag} off/on × stream\n")
print("| flag | stream | bs | prefill tok/s | decode tok/s | e2e s |")
print("|---|---|---|---|---|---|")
cell={}
for fv in("0","1"):
  for st in("single","two"):
    for bs in bss:
      r=last(f"{fv}_{st}/bs{bs}.jsonl")
      if r: cell[(fv,st,bs)]=r; print(f"| {'on' if fv=='1' else 'off'} | {st} | {bs} | {g(r.get('input_throughput'))} | {g(r.get('output_throughput'))} | {g(r.get('latency'))} |")
print("\n## decode tok/s matrix (rows=flag, cols=stream)\n")
print("| flag \\\\ stream | single | two |\n|---|---|---|")
for fv in("0","1"):
  row=[]
  for st in("single","two"):
    r=cell.get((fv,st,bss[0]))  # bs = first (16)
    row.append(g(r.get('output_throughput')) if r else "—")
  print(f"| {'on' if fv=='1' else 'off'} (bs{bss[0]}) | {row[0]} | {row[1]} |")
def rat(a,b): return ("%.1f%%"%(100*a/b)) if (a and b) else "—"
print("\n## opt effect (on/off) per stream, decode bs"+bss[0])
o0=cell.get(("0","single",bss[0])); o1=cell.get(("1","single",bss[0]))
t0=cell.get(("0","two",bss[0]));    t1=cell.get(("1","two",bss[0]))
if o0 and o1: print(f"- single-stream: {rat(o1['output_throughput'],o0['output_throughput'])}")
if t0 and t1: print(f"- two-stream:    {rat(t1['output_throughput'],t0['output_throughput'])}")
PY
echo "== [matrix] done -> ${OUT}/summary.md"; cat "${OUT}/summary.md"
