#!/usr/bin/env python3
"""Generic base-vs-variant summary: acc-diff + perf-delta (incl. prefill/decode split).
Pure stdlib; run locally on the RUN_ROOT the driver used. Usage:
    python3 summary.py <RUN_ROOT>     # auto-discovers <RUN_ROOT>/<model>/ via its meta.env
Model parameters (acc_tol / layers / display name) come from <model>/meta.env — written by
run_regression.sh from models/<m>/model.env. Env overrides: ACC_TOL, PERF_TOL, LAYERS.
Works on legacy kimi-regression / qwen35_35b-regression run folders too (same meta.env layout)."""
import json, os, subprocess, sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get("RUN_ROOT", ".")).expanduser()

def env(p):
    d = {}
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1); d[k.strip()] = v.strip()
    return d

# ---- auto-discover the model dir: any RUN_ROOT subdir holding a meta.env ----
cands = sorted(d for d in (root.iterdir() if root.is_dir() else []) if d.is_dir() and (d / "meta.env").exists())
if not cands:
    sys.exit(f"summary.py: no <model>/meta.env found under {root}")
mdl = cands[0]
M = env(mdl / "meta.env")
DISPLAY = M.get("model_display", mdl.name)
IN = 2048; OUT = 2048
# acc tol = the model's run-to-run logprob NOISE FLOOR (e.g. kimi atomic-add ~0.26-0.30 -> 0.30;
# qwen3.5 unmeasured -> 0.05 placeholder). A diff <= the floor is noise, not a regression. To be
# rigorous, run the SAME config twice to measure the actual floor, then judge against it.
ACC_TOL = float(os.environ.get("ACC_TOL", M.get("acc_tol", "0.05")))
PERF_TOL = float(os.environ.get("PERF_TOL", "0.05"))
# Layer count: from model.env (static, e.g. kimi 61) or a record_layers hook (dynamic, qwen35).
LAYERS = int(os.environ.get("LAYERS", os.environ.get("KIMI_LAYERS", os.environ.get("QWEN35_LAYERS", M.get("layers", "0") or "0"))))
# graph-off profile steps (for the auto profile_metrics run) — "bs start steps outlen" in meta.env
PROF_OFF_STEPS = (M.get("prof_off", "16 4 12 64").split() + ["12"])[2]

def load_lp(p):
    try: return [float(x) for x in json.loads(p.read_text())] if p.exists() else None
    except Exception: return None

def last_row(p):
    if not p.exists(): return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return rows[-1] if rows else None

def pctl(v, q):
    s = sorted(v); return s[min(len(s)-1, max(0, int(round(q*(len(s)-1)))))]

L = [f"# {DISPLAY} — base vs variant regression summary", "",
     f"Run `{root.name}`.  base = {M.get('base_src','?')}  |  variant = {M.get('variant_src','?')}",
     f"acc tol (max |Δlogprob|) = {ACC_TOL} (the model's noise floor — see models/{M.get('model', mdl.name)}/MODEL.md); perf tol = {PERF_TOL:.0%}.", ""]

L += ["## Provenance", "", "| cell | src | commit |", "|---|---|---|",
      f"| base | {M.get('base_src','?')} | `{M.get('base_commit','?')[:12]}` |",
      f"| variant | {M.get('variant_src','?')} | `{M.get('variant_commit','?')[:12]}` |", ""]

# ---- Accuracy: per-token logprob |diff| (variant vs base) ----
L += ["## Accuracy (per-token logprob |diff|, variant vs base)", "",
      "| n | mean abs | max abs | p50 | p95 | half-MSE | verdict |",
      "|---:|---:|---:|---:|---:|---:|:--|"]
a = load_lp(mdl/"base"/"acc"/"logprobs.json"); b = load_lp(mdl/"variant"/"acc"/"logprobs.json")
if not a or not b:
    L.append(f"| {('base+' if not a else '')+('variant' if not b else '')} MISSING | | | | | | n/a |")
else:
    n = min(len(a), len(b)); diff = [abs(a[i]-b[i]) for i in range(n)]
    mean = sum(diff)/n; mx = max(diff); hmse = 0.5*sum((a[i]-b[i])**2 for i in range(n))/n
    verdict = "PASS (≤ tol)" if mx <= ACC_TOL else "**REGRESS / intended-diff**"
    note = "" if len(a)==len(b) else f" ⚠{len(a)}v{len(b)}"
    L.append(f"| {n}{note} | {mean:.5f} | {mx:.5f} | {pctl(diff,0.5):.5f} | {pctl(diff,0.95):.5f} | {hmse:.5f} | {verdict} |")
L += ["", "> If base/variant are *numerically equivalent* configs (e.g. env-on vs env-off), max ≤ tol = no regression.",
      "> If they differ on purpose (base no-LoRA vs variant LoRA), the diff is the intended effect — read the numbers.",
      "> NOTE: acc is teacher-forced PREFILL-only — it cannot see decode-accumulating garbage.",
      "> The per-cell prompt-check table is the decode gate; paste it next to this.", ""]

# ---- Performance: latency / throughput + prefill/decode split ----
L += ["## Performance (variant vs base)", "",
      "| bs | base lat | var lat | base tok/s | var tok/s | var % base | base pre/dec(s) | var pre/dec(s) | verdict |",
      "|---:|---:|---:|---:|---:|---:|---:|---:|:--|"]
def split(row, bs):  # derive prefill/decode time from latency + output throughput (decode tokens = bs*OUT)
    lat = row.get("latency"); ot = row.get("output_throughput")
    if not (lat and ot): return (None, None)
    dec = (bs*OUT)/ot; pre = lat - dec
    return (pre, dec)
for bs in (16, 32, 64):
    bb = last_row(mdl/"base"/"bench"/f"bs{bs}.jsonl"); vv = last_row(mdl/"variant"/"bench"/f"bs{bs}.jsonl")
    if not bb or not vv:
        L.append(f"| {bs} | {'MISSING' if not bb else 'ok'} | {'MISSING' if not vv else 'ok'} | | | | | | n/a |"); continue
    bl, vl = bb.get("latency"), vv.get("latency"); bt, vt = bb.get("output_throughput"), vv.get("output_throughput")
    ratio = (vt/bt*100) if (bt and vt) else None
    bp, bd = split(bb, bs); vp, vd = split(vv, bs)
    vd_s = "n/a" if ratio is None else ("PASS" if ratio >= (1-PERF_TOL)*100 else "**REGRESS / intended**")
    pre_b = f"{bp:.1f}/{bd:.1f}" if bp is not None else "-"; pre_v = f"{vp:.1f}/{vd:.1f}" if vp is not None else "-"
    L.append(f"| {bs} | {bl:.2f} | {vl:.2f} | {bt:.0f} | {vt:.0f} | {ratio:.1f}% | {pre_b} | {pre_v} | {vd_s} |")
L += ["", "> tok/s is decode (output) throughput. pre/dec split: prefill = latency − (bs·2048)/decode_tput.",
      "> The headline % IS the decode (output_throughput) ratio — the profiler is kernel-structure only.", ""]

# ---- Speed: 5 independent measurements + server-log cross-check ----
import re as _re, statistics as _st
def _server_decode_median(cfg, bs):
    p = mdl/cfg/"bench"/f"bs{bs}.serverlog"
    if not p.exists(): return None
    d = [float(x) for x in _re.findall(r"gen throughput \(token/s\): ([0-9.]+)", p.read_text(errors="ignore"))]
    return _st.median(d) if d else None
L += ["## Speed — 5 independent measurements (do NOT trust bench `output_throughput` alone)", "",
      "A change is only 'faster' if the bench gain survives the **server-log cross-check** AND the "
      "**profiler forward-pass/per-layer** time agrees (a kimi bench once read a +26% phantom).", "",
      "**(3) server decode tok/s + (4) bench ITL + (5) bench e2e** — per bs, base & variant:", "",
      "| bs | cfg | (5) e2e lat(s) | (4) ITL(ms) | bench decode tok/s | (3) server decode tok/s | bench/server | sanity |",
      "|---:|:--|---:|---:|---:|---:|---:|:--|"]
for bs in (16, 32, 64):
    for cfg in ("base", "variant"):
        row = last_row(mdl/cfg/"bench"/f"bs{bs}.jsonl")
        if not row: L.append(f"| {bs} | {cfg} | MISSING | | | | | n/a |"); continue
        e2e = row.get("latency"); ot = row.get("output_throughput")
        itl = row.get("median_itl") or (1000.0*bs/ot if ot else None)   # ms/decode-step = bs/decode_tput
        sdec = _server_decode_median(cfg, bs); ratio = (100*(ot/sdec-1) if (ot and sdec) else None)
        san = "no-serverlog" if sdec is None else ("OK" if abs(ratio) <= 5 else "**SUSPECT >5% → RERUN**")
        L.append(f"| {bs} | {cfg} | {(f'{e2e:.2f}' if e2e else '—')} | {(f'{itl:.2f}' if itl else '—')} | "
                 f"{(f'{ot:.0f}' if ot else '—')} | {(f'{sdec:.0f}' if sdec else '—')} | "
                 f"{(f'{ratio:+.1f}%' if ratio is not None else '—')} | {san} |")
L += ["", "**(1) per-layer time + (2) forward-pass time** — profiler-derived (graph-off decode), INDEPENDENT of bench (lower = faster):", "",
      "| cfg | (2) forward-pass (ms) | (1) per-layer (µs) | source |", "|:--|---:|---:|:--|"]
_pm = Path(__file__).resolve().parent / "profile_metrics.py"
for cfg in ("base", "variant"):
    pmj = mdl/cfg/"profile_metrics.json"; d = {}
    # Self-contained: if (1)+(2) weren't pre-computed, auto-run profile_metrics.py on the graph-off
    # trace (--steps from meta.env prof_off; --layers from meta.env / model.env).
    if not pmj.exists() and _pm.exists():
        trs = sorted((mdl/cfg/"traces"/"graph_off").glob("*.trace.json.gz"))
        if trs:
            try:
                subprocess.run([sys.executable, str(_pm), str(trs[0]), "--steps", str(PROF_OFF_STEPS),
                                "--layers", str(LAYERS), "--out", str(pmj)],
                               check=False, capture_output=True, timeout=180)
            except Exception: pass
    if pmj.exists():
        try: d = json.loads(pmj.read_text())
        except Exception: pass
    fp, pl, m = d.get("forward_pass_ms"), d.get("per_layer_us"), d.get("method", "")
    src = (m[:34] if m else "—") if d else "no graph-off trace found"
    L.append(f"| {cfg} | {(f'{fp:.3f}' if fp else '—')} | {(f'{pl:.2f}' if pl else '—')} | {src} |")
L += ["", "> Cross-check rule: server decode tok/s is the scheduler's ground truth — >5% gap vs bench decode ⇒ bench SUSPECT, rerun. "
      "Forward-pass/per-layer (profiler) is a second independent witness. Report **all five** + both cross-checks; never conclude from bench e2e alone.", ""]

out = root / "summary.md"; out.write_text("\n".join(L) + "\n")
print(out); print("\n".join(L))
