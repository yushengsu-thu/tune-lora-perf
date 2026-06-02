#!/usr/bin/env python3
"""Kimi-K2.5-NVFP4 base-vs-variant summary: acc-diff + perf-delta (incl. prefill/decode split).
Pure stdlib; run locally on the RUN_ROOT the driver used. Usage:
    python3 summary.py <RUN_ROOT>     # reads <RUN_ROOT>/kimi/{base,variant}/{acc,bench}
Env: ACC_TOL (default 0.30 — Kimi atomic-add noise floor!), PERF_TOL (default 0.05)."""
import json, os, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_ROOT", ".")).expanduser()
kimi = root / "kimi"
IN = 2048
# Kimi's MoE/LoRA uses atomic_add -> run-to-run logprob nondeterminism ~0.26-0.30. So the default
# ACC_TOL is the NOISE FLOOR, not 0.01. A logprob diff <= this is noise, not a regression. To be
# rigorous, run the SAME config twice to measure the actual floor, then judge the variant against it.
ACC_TOL = float(os.environ.get("ACC_TOL", "0.30"))
PERF_TOL = float(os.environ.get("PERF_TOL", "0.05"))

def env(p):
    d = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1); d[k] = v
    return d

def load_lp(p):
    try: return [float(x) for x in json.loads(p.read_text())] if p.exists() else None
    except Exception: return None

def last_row(p):
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows[-1] if rows else None

def pctl(v, q):
    s = sorted(v); return s[min(len(s)-1, max(0, int(round(q*(len(s)-1)))))]

M = env(kimi / "meta.env")
L = ["# Kimi-K2.5-NVFP4 — base vs variant regression summary", "",
     f"Run `{root.name}`.  base = {M.get('base_src','?')}  |  variant = {M.get('variant_src','?')}",
     f"acc tol (max |Δlogprob|) = {ACC_TOL} (Kimi atomic-add noise floor); perf tol = {PERF_TOL:.0%}.", ""]

L += ["## Provenance", "", "| cell | src | commit |", "|---|---|---|",
      f"| base | {M.get('base_src','?')} | `{M.get('base_commit','?')[:12]}` |",
      f"| variant | {M.get('variant_src','?')} | `{M.get('variant_commit','?')[:12]}` |", ""]

# ---- Accuracy: per-token logprob |diff| (variant vs base) ----
L += ["## Accuracy (per-token logprob |diff|, variant vs base)", "",
      "| n | mean abs | max abs | p50 | p95 | half-MSE | verdict |",
      "|---:|---:|---:|---:|---:|---:|:--|"]
a = load_lp(kimi/"base"/"acc"/"logprobs.json"); b = load_lp(kimi/"variant"/"acc"/"logprobs.json")
if not a or not b:
    L.append(f"| {('base+' if not a else '')+('variant' if not b else '')} MISSING | | | | | | n/a |")
else:
    n = min(len(a), len(b)); diff = [abs(a[i]-b[i]) for i in range(n)]
    mean = sum(diff)/n; mx = max(diff); hmse = 0.5*sum((a[i]-b[i])**2 for i in range(n))/n
    verdict = "PASS (≤ noise)" if mx <= ACC_TOL else "**REGRESS / intended-diff**"
    note = "" if len(a)==len(b) else f" ⚠{len(a)}v{len(b)}"
    L.append(f"| {n}{note} | {mean:.5f} | {mx:.5f} | {pctl(diff,0.5):.5f} | {pctl(diff,0.95):.5f} | {hmse:.5f} | {verdict} |")
L += ["", "> If base/variant are *numerically equivalent* configs (e.g. 2-stream vs no-2-stream), max ≤ tol = no regression.",
      "> If they differ on purpose (base no-LoRA vs variant LoRA), the diff is the intended effect — read the numbers.", ""]

# ---- Performance: latency / throughput + prefill/decode split ----
L += ["## Performance (variant vs base)", "",
      "| bs | base lat | var lat | base tok/s | var tok/s | var % base | base pre/dec(s) | var pre/dec(s) | verdict |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|:--|"]
def split(row, bs):  # derive prefill/decode time from latency + input/output throughput
    lat = row.get("latency"); it = row.get("input_throughput"); ot = row.get("output_throughput")
    if not (lat and ot): return (None, None)
    dec = (bs*IN)/ot; pre = lat - dec
    return (pre, dec)
for bs in (16, 32, 64):
    bb = last_row(kimi/"base"/"bench"/f"bs{bs}.jsonl"); vv = last_row(kimi/"variant"/"bench"/f"bs{bs}.jsonl")
    if not bb or not vv:
        L.append(f"| {bs} | {'MISSING' if not bb else 'ok'} | {'MISSING' if not vv else 'ok'} | | | | | | n/a |"); continue
    bl, vl = bb.get("latency"), vv.get("latency"); bt, vt = bb.get("output_throughput"), vv.get("output_throughput")
    ratio = (vt/bt*100) if (bt and vt) else None
    bp, bd = split(bb, bs); vp, vd = split(vv, bs)
    vd_s = "n/a" if ratio is None else ("PASS" if ratio >= (1-PERF_TOL)*100 else "**REGRESS / intended**")
    pre_b = f"{bp:.1f}/{bd:.1f}" if bp is not None else "-"; pre_v = f"{vp:.1f}/{vd:.1f}" if vp is not None else "-"
    L.append(f"| {bs} | {bl:.2f} | {vl:.2f} | {bt:.0f} | {vt:.0f} | {ratio:.1f}% | {pre_b} | {pre_v} | {vd_s} |")
L += ["", "> tok/s is decode (output) throughput. pre/dec split: prefill = latency − (bs·2048)/decode_tput.",
      "> The headline % is decode-dominated (out=2048); for the *profile* analysis use decode_isolate.py", ""]

out = root / "summary.md"; out.write_text("\n".join(L) + "\n")
print(out); print("\n".join(L))
