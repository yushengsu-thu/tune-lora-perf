#!/usr/bin/env python3
"""Sanity-check a bench result against the SERVER LOG's own decode throughput.

WHY (learned the hard way): bench_one_batch_server's reported `output_throughput` is occasionally
ANOMALOUS. A kimi V5 (down-overlap) run once reported 3078 tok/s; a verified rerun showed it was
really ~2440 (a +26% phantom that made a useless overlap look like a big win). The scheduler logs
the ground-truth per-batch decode rate ("gen throughput (token/s): X"). If the bench number
disagrees with the server's decode median by more than ~5%, the bench number is SUSPECT — rerun
before trusting/reporting it. ALWAYS capture the server log alongside the bench.

Usage: serverlog_sanity.py <bench.jsonl> <server_log_slice>
  where <server_log_slice> = the Prefill/Decode batch lines logged DURING this bench
  (snapshot wc -l of /tmp/server.log before the bench, tail -n +N after).
"""
import json, re, sys, statistics as st

try:
    # dev harness writes multi-line jsonl; the LAST line is the run of record.
    lines = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()]
    d = json.loads(lines[-1])
    txt = open(sys.argv[2], errors="ignore").read()
except Exception as e:
    print(f"  [SANITY] skipped ({e})")
    sys.exit(0)

bench_dec = float(d.get("output_throughput") or 0)
bench_pre = float(d.get("input_throughput") or 0)
dec = [float(x) for x in re.findall(r"gen throughput \(token/s\): ([0-9.]+)", txt)]
smed = st.median(dec) if dec else 0.0
smax = max(dec) if dec else 0.0
diff = 100 * (bench_dec / smed - 1) if smed else 0.0
if not smed:
    flag = "(no server decode lines captured — check the slice / server.log path)"
elif abs(diff) <= 5:
    flag = "OK"
else:
    flag = "*** WARN: bench decode != server decode -> bench number SUSPECT, RERUN before trusting ***"

print(
    f"  [SANITY] bench_decode={bench_dec:.1f}  server_decode_median={smed:.1f} "
    f"(max {smax:.1f}, n={len(dec)})  diff={diff:+.1f}%  {flag}"
)
print(f"           bench_prefill(input)={bench_pre:.1f}   (full server prefill/decode lines kept in the *.serverlog slice)")
