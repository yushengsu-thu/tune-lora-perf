#!/usr/bin/env bash
# opt2 A/B bench: LoRA cell with SGLANG_OPT_LORA_FUSED_TOPK_PACK off vs on, same warm pod.
# off = separate _pack_topk_for_flashinfer_routed (cast/shift/or elementwise chain after gating);
# on = the flashinfer routed-pack fused INTO the gating kernel (StandardTopKOutputPacked), so the
# dispatch skips the separate pack. opt1's fused-align stays default-on for both, isolating opt2.
#   Usage: bash dev/bench_opt2_ab.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

OUT="${DEV_DIR}/results/${MODEL}/opt2-ab"
echo "== [opt2-ab] $MODEL  bs=${BENCH_BS} in=${BENCH_IN} out=${BENCH_OUT}  -> ${OUT}"
LA="--lora-name ${LORA_NAME}"
BASE_LORA_ENVS="$LORA_ENVS"

for VAR in off on; do
  # SGLANG_OPT_LORA_FUSED_TOPK_PACK defaults True, so "off" MUST set =0 explicitly.
  if [ "$VAR" = on ]; then LORA_ENVS="$BASE_LORA_ENVS SGLANG_OPT_LORA_FUSED_TOPK_PACK=1"
  else LORA_ENVS="$BASE_LORA_ENVS SGLANG_OPT_LORA_FUSED_TOPK_PACK=0"; fi
  echo "---- variant: $VAR   LORA_ENVS=$LORA_ENVS ----"
  launch_server lora || { echo "ERROR: launch failed ($VAR)"; exit 1; }
  coherence_check lora || { echo "ERROR: decode garbage ($VAR)"; exit 1; }
  D="/tmp/opt1_ab/${VAR}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; for bs in ${BENCH_BS}; do
        sl0=\$(wc -l </tmp/server.log 2>/dev/null || echo 0)
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size \${bs} --input-len ${BENCH_IN} --output-len ${BENCH_OUT} ${LA} \
          --show-report --result-filename ${D}/bs\${bs}.jsonl 2>&1 | tee ${D}/bs\${bs}.log
        tail -n +\$((sl0+1)) /tmp/server.log | tr '\r' '\n' | grep -aE 'Prefill batch|Decode batch' > ${D}/bs\${bs}.serverlog || true
      done" || { echo "ERROR: bench failed ($VAR)"; exit 1; }
  pull_dir "$D" "${OUT}/${VAR}"
done
kill_all

python3 - "$OUT" "$BENCH_BS" <<'PY'
import json, sys, pathlib
root, bss = pathlib.Path(sys.argv[1]), sys.argv[2].split()
def last(p):
    try: return [json.loads(l) for l in p.read_text().splitlines() if l.strip()][-1]
    except Exception: return None
rows = {}
for var in ("off","on"):
    for bs in bss:
        r = last(root/var/f"bs{bs}.jsonl")
        if r: rows[(var,int(bs))] = r
f = lambda v: ("%.1f"%v) if isinstance(v,(int,float)) else "—"
L = ["# opt1 A/B — SGLANG_OPT_LORA_FUSED_MERGED_ALIGN off vs on (LoRA cell, shared_outer)","",
     "| variant | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |","|---|---|---|---|---|---|"]
for var in ("off","on"):
    for bs in (int(b) for b in bss):
        r = rows.get((var,bs))
        if not r: L.append(f"| {var} | {bs} | — | — | — | — |"); continue
        L.append(f"| {var} | {bs} | {f(r.get('input_throughput'))} | {f(r.get('output_throughput'))} "
                 f"| {f(r.get('median_itl'))} | {f(r.get('latency'))} |")
L += ["","on/off ratio (prefill & decode: >100%=opt1 faster; e2e: <100%=opt1 faster)","",
      "| bs | prefill | decode | e2e |","|---|---|---|---|"]
def ratio(a,b): return ("%.1f%%"%(100.0*a/b)) if (a and b) else "—"
for bs in (int(b) for b in bss):
    o,n = rows.get(("off",bs)), rows.get(("on",bs))
    if not (o and n): L.append(f"| {bs} | — | — | — |"); continue
    L.append(f"| {bs} | {ratio(n.get('input_throughput'),o.get('input_throughput'))} "
             f"| {ratio(n.get('output_throughput'),o.get('output_throughput'))} "
             f"| {ratio(n.get('latency'),o.get('latency'))} |")
out = root/"summary.md"; out.write_text("\n".join(L)+"\n")
print("\n".join(L)); print(f"\nwrote {out}")
PY
echo "== [opt1-ab] done -> ${OUT}/summary.md"
