#!/usr/bin/env bash
# 3-way bench: base (no-LoRA ceiling) vs LoRA-before (all session opts OFF) vs LoRA-after
# (opt1+opt2+opt3 + common opt-in, = current model.env defaults). Two-stream (production) for the
# LoRA cells. Reports prefill / decode / e2e per bs + ratios (after/base = % of ceiling,
# after/before = cumulative opt speedup).
#   Usage: bash dev/bench_3way.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
OUT="${DEV_DIR}/results/${MODEL}/3way"
LA="--lora-name ${LORA_NAME}"; BASE_LORA="$LORA_ENVS"
# "before" = turn OFF every optimization this session touched (back to pre-opt LoRA behavior)
OFF_DELTA="SGLANG_OPT_LORA_FUSED_MERGED_ALIGN=0 SGLANG_OPT_LORA_FUSED_TOPK_PACK=0 \
SGLANG_OPT_LORA_LEAN_INFO=0 SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN=0 SGLANG_OPT_FUSED_MOE_ACTIVATION_VEC=0"
echo "== [3way] $MODEL bs=${BENCH_BS} in=${BENCH_IN} out=${BENCH_OUT} -> ${OUT}"

run_bench(){ # $1=cell-name $2=lora|no-lora $3=lora-name-arg
  local D="/tmp/3way/$1"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; for bs in ${BENCH_BS}; do
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size \${bs} --input-len ${BENCH_IN} --output-len ${BENCH_OUT} $3 \
          --show-report --result-filename ${D}/bs\${bs}.jsonl 2>&1 | tail -2
      done" || echo "  WARN: $1 bench nonzero (jsonl validated at summary)"
  pull_dir "$D" "${OUT}/$1"
}

# 1) base (no-LoRA) ceiling
echo "---- base (no-lora) ----"
launch_server no-lora || { echo ERROR base launch; exit 1; }
coherence_check no-lora || true
run_bench base no-lora ""

# 2) LoRA-before (all session opts off)
echo "---- lora-before (opts OFF) ----"
LORA_ENVS="$BASE_LORA $OFF_DELTA"
launch_server lora || { echo ERROR before launch; exit 1; }
coherence_check lora || { echo ERROR before coherence; exit 1; }
run_bench before lora "$LA"

# 3) LoRA-after (current defaults: opt1+opt2+opt3+common all on)
echo "---- lora-after (opts ON) ----"
LORA_ENVS="$BASE_LORA"
launch_server lora || { echo ERROR after launch; exit 1; }
coherence_check lora || { echo ERROR after coherence; exit 1; }
run_bench after lora "$LA"
kill_all

python3 - "$OUT" "$BENCH_BS" > "${OUT}/summary.md" <<'PY'
import json,sys,pathlib
root,bss=pathlib.Path(sys.argv[1]),sys.argv[2].split()
def last(p):
    try: return [json.loads(l) for l in (root/p).read_text().splitlines() if l.strip()][-1]
    except: return None
g=lambda v:("%.0f"%v) if isinstance(v,(int,float)) else "—"
cells=("base","before","after")
print("# 3-way — base (no-LoRA) vs LoRA before/after (opt1+opt2+opt3 + common)\n")
print("| cell | bs | prefill tok/s | decode tok/s | e2e s |")
print("|---|---|---|---|---|")
d={}
for c in cells:
  for bs in bss:
    r=last(f"{c}/bs{bs}.jsonl")
    if r: d[(c,bs)]=r; print(f"| {c} | {bs} | {g(r.get('input_throughput'))} | {g(r.get('output_throughput'))} | {r.get('latency')} |")
rat=lambda a,b:("%.1f%%"%(100*a/b)) if (a and b) else "—"
print("\n## LoRA-after vs LoRA-before (cumulative opt speedup)\n| bs | prefill | decode | e2e(lower=better) |\n|---|---|---|---|")
for bs in bss:
  a,b=d.get(("after",bs)),d.get(("before",bs))
  if a and b: print(f"| {bs} | {rat(a['input_throughput'],b['input_throughput'])} | {rat(a['output_throughput'],b['output_throughput'])} | {rat(a['latency'],b['latency'])} |")
print("\n## LoRA-after as % of base (no-LoRA ceiling)\n| bs | prefill | decode | e2e |\n|---|---|---|---|")
for bs in bss:
  a,bse=d.get(("after",bs)),d.get(("base",bs))
  if a and bse: print(f"| {bs} | {rat(a['input_throughput'],bse['input_throughput'])} | {rat(a['output_throughput'],bse['output_throughput'])} | {rat(a['latency'],bse['latency'])} |")
PY
echo "== [3way] done =="; cat "${OUT}/summary.md"
