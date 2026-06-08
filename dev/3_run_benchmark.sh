#!/usr/bin/env bash
# 3. Benchmark LoRA vs no-LoRA (bench_one_batch_server; bs/in/out from the model pack).
#    Input : model name (dir under dev/models/ or unique prefix); pods from step 1 running
#            the code from step 2.
#    Output: $RUN_DIR/bench/{no-lora,lora}/bs<bs>.{jsonl,log,serverlog} +
#            $RUN_DIR/bench/summary.md: per-cell input/extend tok/s, decode tok/s, ITL, e2e latency,
#            plus a lora/no-lora ratio table for all three (extend, decode, e2e).
#    Verify: every bs<bs>.jsonl parses for both cells + a post-load coherence check per cell
#            (catches '!!!!' decode collapse the numbers can't show).
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
OUTROOT=/tmp/dev_run
echo "== [3/bench] $MODEL  bs=${BENCH_BS}  in=${BENCH_IN} out=${BENCH_OUT}  -> ${RUN_DIR}/bench"

for CELL in no-lora lora; do
  echo "---- cell: $CELL ----"
  launch_server "$CELL" || { echo "ERROR: $CELL server failed to launch"; exit 1; }
  LA=""; [ "$CELL" = lora ] && LA="--lora-name ${LORA_NAME}"
  D="${OUTROOT}/${CELL}/bench"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; for bs in ${BENCH_BS}; do
        sl0=\$(wc -l </tmp/server.log 2>/dev/null || echo 0)
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size \${bs} --input-len ${BENCH_IN} --output-len ${BENCH_OUT} ${LA} \
          --show-report --result-filename ${D}/bs\${bs}.jsonl 2>&1 | tee ${D}/bs\${bs}.log
        tail -n +\$((sl0+1)) /tmp/server.log | tr '\r' '\n' | grep -aE 'Prefill batch|Decode batch' > ${D}/bs\${bs}.serverlog || true
      done" || { echo "ERROR: bench failed ($CELL)"; exit 1; }
  coherence_check "$CELL" || exit 1
  pull_dir "$D" "${RUN_DIR}/bench/${CELL}"
done
kill_all

# ---- verify + summarize locally (input/extend, decode, e2e) ----
python3 - "$RUN_DIR/bench" "$BENCH_BS" <<'PY' || exit 1
import json, sys, pathlib
root, bss = pathlib.Path(sys.argv[1]), sys.argv[2].split()
rows, missing = [], []
for cell in ("no-lora", "lora"):
    for bs in bss:
        p = root / cell / f"bs{bs}.jsonl"
        try:
            r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()][-1]
        except Exception:
            missing.append(str(p)); continue
        rows.append((cell, int(bs), r.get("input_throughput"), r.get("output_throughput"),
                     r.get("median_itl"), r.get("latency")))
if missing:
    print("ERROR: missing/unparsable bench results:", *missing, sep="\n  "); sys.exit(1)
f = lambda v, fmt="%.1f": (fmt % v) if isinstance(v, (int, float)) else "—"
L = ["# bench summary — LoRA vs no-LoRA", "",
     "| cell | bs | input/extend tok/s | decode tok/s | ITL ms | e2e s |", "|---|---|---|---|---|---|"]
dec, ext, e2e = {}, {}, {}
for cell, bs, it, ot, itl, lat in rows:
    dec[(cell, bs)] = ot
    ext[(cell, bs)] = it
    e2e[(cell, bs)] = lat
    itl = itl if itl else (1000.0 * bs / ot if ot else None)
    L.append(f"| {cell} | {bs} | {f(it)} | {f(ot)} | {f(itl,'%.2f')} | {f(lat,'%.2f')} |")
ratio = lambda a, b: (("%.1f%%" % (100.0 * a / b)) if (a and b) else "—")
L += ["", "lora / no-lora ratio  (extend & decode tok/s: higher=faster; e2e latency: higher=slower)",
      "", "| bs | extend | decode | e2e |", "|---|---|---|---|"]
for bs in (int(b) for b in bss):
    L.append(f"| {bs} | {ratio(ext.get(('lora', bs)), ext.get(('no-lora', bs)))} "
             f"| {ratio(dec.get(('lora', bs)), dec.get(('no-lora', bs)))} "
             f"| {ratio(e2e.get(('lora', bs)), e2e.get(('no-lora', bs)))} |")
out = root / "summary.md"; out.write_text("\n".join(L) + "\n")
print("\n".join(L)); print(f"\nwrote {out}")
PY

echo "== [3/bench] PASS — ${RUN_DIR}/bench/summary.md"
