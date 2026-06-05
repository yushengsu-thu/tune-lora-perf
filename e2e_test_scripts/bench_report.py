#!/usr/bin/env python3
# bench_report.py <bench.jsonl> <server.serverlog> -> one clean line:
#   e2e + bench decode tput + ITL  +  server-log decode median  +  bench-vs-server xcheck (OK / SUSPECT>5%)
# Separate file (single-quoted invocation) to avoid the nested-quote f-string garble of an inline echo.
import sys, json, re, statistics as st
try:
    last = [l for l in open(sys.argv[1]).read().splitlines() if l.strip()][-1]
    d = json.loads(last)
except Exception:
    print("(no bench jsonl)"); sys.exit(0)
bs = d.get("batch_size", 0); ot = d.get("output_throughput", 0) or 0; lat = d.get("latency", 0) or 0
itl = 1000.0 * bs / ot if ot else 0.0
sd = None
if len(sys.argv) > 2:
    try:
        v = [float(x) for x in re.findall(r"gen throughput \(token/s\): ([0-9.]+)", open(sys.argv[2]).read())]
        sd = st.median(v) if v else None
    except Exception:
        sd = None
xc = (100.0 * (ot / sd - 1) if (ot and sd) else None)
flag = "n/a" if xc is None else ("OK" if abs(xc) <= 5 else "SUSPECT>5%-RERUN")
print("e2e=%.2fs tput=%.1f itl=%.2fms | server_decode=%s xcheck=%s %s" % (
    lat, ot, itl,
    ("%.1f" % sd if sd else "-"),
    ("%+.1f%%" % xc if xc is not None else "-"), flag))
