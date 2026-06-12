#!/usr/bin/env bash
# opt8 step0 PROBE — piecewise CUDA graph for the LoRA prefill path (zero-code A/B).
#
# WHY (run 20260612-051818): prefill is HOST-BOUND (graph-off idle 49%, 13.1k launches; in
# production the host skew shows up as allreduce spin: 158.9 ms of a 219.7 ms 4096-tok chunk
# vs no-lora's 12.0 ms). server_args._handle_piecewise_cuda_graph condition 7 force-disables
# piecewise whenever LoRA is enabled — the no-lora cell is eligible, every LoRA config is
# locked to eager prefill. `--enforce-piecewise-cuda-graph` skips the auto-disable, so the
# probe needs NO code change.
#
# off = stock lora cell (eager prefill, today's default)
# on  = + --enforce-piecewise-cuda-graph
#
# Outcomes:
#   * on-launch crashes        -> probe NEGATIVE-but-informative: the crash log IS the work list
#                                 (saved to <out>/on/launch_tail.log).
#   * boots but no capture log -> piecewise still disabled somewhere else; check launch_tail.log.
#   * boots + captures + coherent -> read the triplet; prefill is the headline (ceiling ~3.6x).
#
#   Usage: bash dev/probe_opt8_piecewise.sh <model>
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

OUT="${DEV_DIR}/results/${MODEL}/opt8-probe"
mkdir -p "$OUT"
echo "== [opt8-probe] $MODEL  bs=${BENCH_BS} in=${BENCH_IN} out=${BENCH_OUT}  -> ${OUT}"
LA="--lora-name ${LORA_NAME}"
BASE_LORA_EXTRA="$LORA_EXTRA"

for VAR in off on; do
  if [ "$VAR" = on ]; then LORA_EXTRA="$BASE_LORA_EXTRA --enforce-piecewise-cuda-graph"
  else LORA_EXTRA="$BASE_LORA_EXTRA"; fi
  echo "---- variant: $VAR ----"
  mkdir -p "${OUT}/${VAR}"
  sl_launch0=$(kh "wc -l </tmp/server.log 2>/dev/null || echo 0" | tr -dc 0-9)
  if ! launch_server lora; then
    kh "tail -n 120 /tmp/server.log" > "${OUT}/${VAR}/launch_tail.log" 2>/dev/null || true
    echo "ERROR: launch failed ($VAR) — tail saved to ${OUT}/${VAR}/launch_tail.log"
    [ "$VAR" = on ] && { echo "[opt8-probe] ON-variant launch crash = the work list; see launch_tail.log"; exit 2; }
    exit 1
  fi
  # piecewise evidence: capture lines logged during THIS launch
  kh "tail -n +$((sl_launch0+1)) /tmp/server.log | grep -aiE 'piecewise' | head -20" \
     > "${OUT}/${VAR}/piecewise_evidence.log" 2>/dev/null || true
  echo "  piecewise log lines ($VAR):"; sed 's/^/    /' "${OUT}/${VAR}/piecewise_evidence.log" | head -8
  coherence_check lora || { echo "ERROR: decode garbage ($VAR)"; kh "tail -n 120 /tmp/server.log" > "${OUT}/${VAR}/launch_tail.log" 2>/dev/null; exit 3; }
  D="/tmp/opt8_probe/${VAR}"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; for bs in ${BENCH_BS}; do
        sl0=\$(wc -l </tmp/server.log 2>/dev/null || echo 0)
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size \${bs} --input-len ${BENCH_IN} --output-len ${BENCH_OUT} ${LA} \
          --show-report --result-filename ${D}/bs\${bs}.jsonl 2>&1 | tee ${D}/bs\${bs}.log
        tail -n +\$((sl0+1)) /tmp/server.log | tr '\r' '\n' | grep -aE 'gen throughput|Prefill batch|Decode batch' > ${D}/bs\${bs}.serverlog || true
      done" || { echo "ERROR: bench failed ($VAR)"; exit 1; }
  pull_dir "$D" "${OUT}/${VAR}"
  for bs in ${BENCH_BS}; do
    echo "  cell ${VAR} bs${bs}:"
    python3 "${DEV_DIR}/serverlog_sanity.py" "${OUT}/${VAR}/bs${bs}.jsonl" "${OUT}/${VAR}/bs${bs}.serverlog" || true
  done
done
kill_all
LORA_EXTRA="$BASE_LORA_EXTRA"

python3 - "$OUT" "$BENCH_BS" <<'PY'
import json, sys, pathlib
root, bss = pathlib.Path(sys.argv[1]), sys.argv[2].split()
def last(p):
    try: return [json.loads(l) for l in p.read_text().splitlines() if l.strip()][-1]
    except Exception: return None
rows = {(v,int(bs)): last(root/v/f"bs{bs}.jsonl") for v in ("off","on") for bs in bss}
f = lambda v: ("%.1f"%v) if isinstance(v,(int,float)) else "—"
L = ["# opt8 step0 probe — --enforce-piecewise-cuda-graph off vs on (LoRA cell)","",
     "| variant | bs | prefill tok/s | decode tok/s | ITL ms | e2e s |","|---|---|---|---|---|---|"]
for v in ("off","on"):
    for bs in (int(b) for b in bss):
        r = rows.get((v,bs)) or {}
        itl = r.get("median_itl") or (1000.0*bs/r["output_throughput"] if r.get("output_throughput") else None)
        L.append(f"| {v} | {bs} | {f(r.get('input_throughput'))} | {f(r.get('output_throughput'))} | {f(itl)} | {f(r.get('latency'))} |")
L += ["", "on/off ratio (prefill & decode: higher=faster; e2e: lower=faster)","",
      "| bs | prefill | decode | e2e |","|---|---|---|---|"]
ratio = lambda a,b,k: ("%.1f%%"%(100.0*a[k]/b[k])) if (a and b and a.get(k) and b.get(k)) else "—"
for bs in (int(b) for b in bss):
    on, off = rows.get(("on",bs)), rows.get(("off",bs))
    L.append(f"| {bs} | {ratio(on,off,'input_throughput')} | {ratio(on,off,'output_throughput')} | {ratio(on,off,'latency')} |")
out = root/"summary.md"; out.write_text("\n".join(L)+"\n")
print("\n".join(L)); print(f"\nwrote {out}")
PY
echo "== [opt8-probe] DONE — ${OUT}/summary.md (+ per-variant piecewise_evidence.log)"
