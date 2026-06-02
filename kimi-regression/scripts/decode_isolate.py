#!/usr/bin/env python3
"""Decode-isolated profile analysis for Kimi traces — the fix for the prefill-contamination trap.

A bench_one_batch_server profile window is ~75-80% PREFILL (2 big EXTEND steps of 32768 tokens dwarf
the 16-token decode steps), and the 2-stream LoRA overlap is DECODE-ONLY. So aggregate kernel sums
are prefill-dominated and MISLEADING for decode throughput. This isolates the `step[DECODE]` annotation
regions and reports decode-only kernel time, wall (interval-union), overlap factor (sum/wall), and the
top decode kernels — optionally diffed against a base trace dir.

Usage:
    python3 decode_isolate.py --input  <variant profile_graph_on/bs16 dir>
    python3 decode_isolate.py --input  <variant dir>  --base <base dir>     # show the LoRA-added decode kernels
Picks the *-TP-0.trace.json.gz in each dir. Reads decode vs prefill via the step[DECODE]/step[EXTEND]
user_annotations the sglang profiler emits, so it works for any model with those markers."""
import argparse, gzip, json, glob, bisect
from collections import defaultdict

def load(d):
    g = sorted(glob.glob(f"{d}/**/*TP-0.trace.json.gz", recursive=True)) or sorted(glob.glob(f"{d}/*TP-0.trace.json.gz"))
    if not g: raise SystemExit(f"no *-TP-0.trace.json.gz under {d}")
    with gzip.open(g[0]) as f: return json.load(f), g[0]

def merge(iv):
    iv = sorted([x for x in iv if x[1] > x[0]]); out = []
    for s, e in iv:
        if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return out

def analyze(d):
    t, path = load(d); evs = t["traceEvents"]
    ann = lambda kw: merge([(e["ts"], e["ts"]+e.get("dur", 0)) for e in evs
                            if e.get("cat") in ("user_annotation", "gpu_user_annotation") and kw in e.get("name", "")])
    dec, ext = ann("DECODE"), ann("EXTEND")
    ds, es = [m[0] for m in dec], [m[0] for m in ext]
    in_any = lambda ts, m, st: (lambda i: i >= 0 and ts <= m[i][1])(bisect.bisect_right(st, ts)-1)
    kern = [e for e in evs if e.get("cat") == "kernel" and "dur" in e]
    by, iv, dec_sum, ext_sum = defaultdict(float), [], 0.0, 0.0
    for e in kern:
        ts, dur = e["ts"], e["dur"]
        if ext and in_any(ts, ext, es): ext_sum += dur; continue
        if dec and not in_any(ts, dec, ds): continue
        dec_sum += dur; by[e["name"][:70]] += dur; iv.append((ts, ts+dur))
    iv.sort(); union = 0.0; ce = -1e30
    for s, e2 in iv:
        if e2 <= ce: continue
        union += e2 - max(s, ce); ce = max(ce, e2)
    return dict(path=path, dec=dec_sum, ext=ext_sum, union=union, by=by)

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--base", default=None)
a = ap.parse_args()
v = analyze(a.input)
ovl = v["dec"]/v["union"] if v["union"] else 0
print(f"trace: {v['path'].split('/')[-1][:60]}")
print(f"  DECODE kernel sum={v['dec']/1000:.1f}ms  wall(union)={v['union']/1000:.1f}ms  overlap-factor={ovl:.2f}  | PREFILL sum={v['ext']/1000:.0f}ms")
print(f"  (prefill is {100*v['ext']/(v['ext']+v['dec']+1e-9):.0f}% of profiled GPU time — excluded from the decode metrics above)")
print("  top DECODE kernels:")
for nm, d in sorted(v["by"].items(), key=lambda kv: -kv[1])[:14]:
    print(f"    {d/1000:7.1f}ms  {nm}")
if a.base:
    b = analyze(a.base)
    bo = b["dec"]/b["union"] if b["union"] else 0
    print(f"\nBASE decode: sum={b['dec']/1000:.1f}ms wall={b['union']/1000:.1f}ms overlap={bo:.2f}")
    print(f">> variant/base decode wall = {v['union']/b['union'] if b['union'] else 0:.2f}x  (≈ inverse of the decode throughput ratio)")
    print("  decode kernels ADDED/grown in variant (the LoRA cost):")
    allk = set(v["by"]) | set(b["by"])
    diff = sorted(((nm, v["by"].get(nm, 0) - b["by"].get(nm, 0)) for nm in allk), key=lambda x: -x[1])
    for nm, dd in diff[:12]:
        if dd <= 0: break
        print(f"    +{dd/1000:6.1f}ms  {nm}")
