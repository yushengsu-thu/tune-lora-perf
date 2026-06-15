#!/usr/bin/env python3
"""GPU-busy witness — from a GPU-only torch trace, confirm the GPU is actually busy.

WHY: the host-bound question ("is the GPU working, or idling while the CPU dispatches?")
deserves a direct, independent reading. A GPU-only profile (`--profile-activities GPU`)
records just the GPU op timeline (no CPU side), so the GPU-active fraction over the
captured span is a clean busy/idle witness — a cross-check on the CPU+GPU trace +
`sanity_check_opt.py`'s wall-vs-GPU-busy idle ratio.

  busy% = sum(GPU op durations) / (last op end - first op start)
  - >=85%  -> GPU-BOUND (busy); kernel/compute opts can pay off.
  - <60%   -> HOST-BOUND (GPU idles in the gaps between launches); only host-side
              work (fewer launches / graphing) moves e2e.
  - >100%  -> multi-stream overlap (two-stream side stream): definitely busy.

A GPU-only trace has no CPU `step[...]` markers, so this is a global fraction over the
whole captured window (warmup + profiled steps) — coarse but honest about idle gaps.
For the prefill host-bound view, feed a graph-OFF gpu-only trace (prefill is eager);
graph-ON gpu-only reflects the decode window the recipe profiles.

Usage: gpu_busy_witness.py <gpu_only_trace.json[.gz]>
"""
import gzip
import json
import sys

if len(sys.argv) < 2:
    print("usage: gpu_busy_witness.py <gpu_only_trace.json[.gz]>")
    raise SystemExit(2)

path = sys.argv[1]
opener = gzip.open if path.endswith(".gz") else open
try:
    obj = json.load(opener(path, "rt", errors="ignore"))
    events = obj.get("traceEvents", obj if isinstance(obj, list) else [])
except Exception as e:  # noqa: BLE001 - a witness never hard-fails the profile step
    print(f"  [GPU-BUSY] skipped ({e})")
    raise SystemExit(0)

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
gpu = [
    e
    for e in events
    if e.get("ph") == "X" and "dur" in e and str(e.get("cat", "")).lower() in GPU_CATS
]
if not gpu:
    print(
        "  [GPU-BUSY] no GPU ops found — is this a --profile-activities GPU trace? (skipped)"
    )
    raise SystemExit(0)

busy_us = sum(e["dur"] for e in gpu)
start = min(e["ts"] for e in gpu)
end = max(e["ts"] + e["dur"] for e in gpu)
span_us = end - start
frac = (100.0 * busy_us / span_us) if span_us else 0.0

if frac >= 100:
    verdict = "GPU-BOUND (busy; multi-stream overlap)"
elif frac >= 85:
    verdict = "GPU-BOUND (busy) — kernel/compute opts can pay off"
elif frac >= 60:
    verdict = "borderline — partial GPU idle"
else:
    verdict = "HOST-BOUND (GPU idle between launches) — only host-side work moves e2e"

print(
    f"  [GPU-BUSY] GPU-active {busy_us/1e3:.1f} ms / span {span_us/1e3:.1f} ms "
    f"= {frac:.0f}%  ({len(gpu)} GPU ops)  -> {verdict}"
)
