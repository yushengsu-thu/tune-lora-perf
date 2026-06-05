#!/usr/bin/env python3
"""Decode forward-pass + per-layer GPU time from a torch-profiler trace (graph-OFF).

WHY: bench `output_throughput` is NOT trustworthy alone (a kimi V5 run reported a +26% phantom).
This is an INDEPENDENT, profiler-derived decode-speed witness. Lower forward-pass => faster.

Method: bench `--profile` runs a KNOWN `--profile-steps` count (default 12) and emits NO `ProfilerStep#`
markers, so we segment by taking the (steps-1) LARGEST inter-kernel gaps as the step boundaries (NOT every
small gap — that over-splits). Then drop PREFILL steps (GPU-busy > 3x median; the 32768-tok EXTEND dwarfs
16-tok decode). forward_pass_us = median decode-step GPU-busy; per_layer_us = forward_pass / --layers.
Sanity: forward_pass should be <= the bench ITL (= 1000*bs/decode_tput) and lora > no-lora. If the numbers
look wrong, DON'T report them — fall back to the `llm-pipeline-analysis` skill (rigorous anchor-kernel timing).

Usage: profile_metrics.py <trace.json[.gz]> --steps S [--layers N] [--out metrics.json]
"""
import argparse, gzip, json, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("trace"); ap.add_argument("--steps", type=int, default=12, help="profiler --profile-steps used")
ap.add_argument("--layers", type=int, default=0); ap.add_argument("--out", default=None)
a = ap.parse_args()
op = gzip.open if a.trace.endswith(".gz") else open
try:
    tr = json.load(op(a.trace, "rt", errors="ignore"))
except Exception as e:
    print(json.dumps({"trace": a.trace, "error": str(e)})); raise SystemExit
ev = tr.get("traceEvents", tr if isinstance(tr, list) else [])
kern = sorted([e for e in ev if e.get("ph") == "X" and "dur" in e and str(e.get("cat", "")).lower() == "kernel"],
              key=lambda k: k["ts"])
res = {"trace": a.trace.split("/")[-1], "n_kernels": len(kern), "layers": a.layers, "steps_arg": a.steps}
if len(kern) >= a.steps * 4 and a.steps >= 2:
    ends = [k["ts"] + k["dur"] for k in kern]; starts = [k["ts"] for k in kern]
    gaps = sorted(((starts[i + 1] - ends[i], i) for i in range(len(kern) - 1)), reverse=True)
    bounds = sorted(i for _, i in gaps[:a.steps - 1])                 # the (steps-1) largest gaps = step boundaries
    segs, s = [], 0
    for b in bounds:
        segs.append((s, b)); s = b + 1
    segs.append((s, len(kern) - 1))
    busy = [sum(kern[j]["dur"] for j in range(a0, a1 + 1)) for a0, a1 in segs]
    medb = st.median(busy)
    decode = [b for b in busy if b <= 3 * medb]                       # drop prefill EXTEND outliers
    fp = st.median(decode) if decode else 0.0
    res.update(method="top-(steps-1)-gap segmentation (no ProfilerStep markers)",
               n_decode_steps=len(decode), n_prefill_dropped=len(busy) - len(decode),
               forward_pass_us=round(fp, 1), forward_pass_ms=round(fp / 1000.0, 3),
               per_layer_us=(round(fp / a.layers, 2) if a.layers else None))
else:
    res.update(method="too few kernels -> use llm-pipeline-analysis", forward_pass_us=None, per_layer_us=None)
print(json.dumps(res, indent=2))
if a.out:
    open(a.out, "w").write(json.dumps(res))
