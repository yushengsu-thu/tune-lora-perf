"""Pre-optimization sanity check — run BEFORE committing days to a kernel optimization.

Encodes the three checks that would have saved / did save days in the opt1–opt7 arc:
  1. HOST-BOUND CHECK: per phase (prefill/decode), wall vs GPU-busy from a torch trace.
     If the GPU is idle >40% of the wall, GPU-side optimizations will NOT move e2e
     (opt6 −34% act kernel and opt7 −62% pipeline were both e2e-neutral for this reason);
     only host-side work (fewer launches / graphing) will.
  2. E2E CEILING: given X µs/layer of GPU time you plan to remove, the maximum possible
     e2e gain under the measured idle ratio. If it's under the ±2% noise floor, stop.
  3. KERNEL TRIAGE: actual µs/call vs theoretical HBM time (bytes / peak BW). Ratio >3×
     means the kernel is CONFIG-bound (bad grid/occupancy — fix the launch, ~30 lines),
     not bandwidth-bound (needs fusion/surgery). This turned opt7-P3 from a multi-day
     cpasync-mainloop fork into a 30-line gather kernel (permute 180 µs -> 12.7 µs).

CAVEAT: feed it graph-OFF (eager) traces and read the PREFILL row — production decode runs
under CUDA graph (one replay launch/step), so the DECODE row of an eager trace does NOT
reflect production decode (which is not host-bound; that's why opt1/opt2 paid off there).

Usage (trace = the harness's graph-OFF profile output, e.g. opt*/profile/*/bs16-TP-0.trace.json.gz):
  python3 dev/sanity_check_opt.py <trace.json.gz>
  python3 dev/sanity_check_opt.py <trace.json.gz> --remove-us-per-layer 168 --layers 384
  python3 dev/sanity_check_opt.py <trace.json.gz> --kernel permuteKernel --bytes 70e6
"""

import argparse
import collections
import gzip
import json

HBM_BW = 8e12  # GB300 effective HBM bandwidth (B/s); adjust per platform
NOISE_FLOOR = 0.02  # measured run-to-run noise (opt4 identical-cell column)


def load(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)["traceEvents"]


def phase_windows(evs):
    steps = sorted(
        (e for e in evs if isinstance(e.get("name"), str) and e["name"].startswith("step[")),
        key=lambda e: e["ts"],
    )
    return [(e["ts"], e["ts"] + e["dur"], "PREFILL" if "EXTEND" in e["name"] else "DECODE") for e in steps]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--remove-us-per-layer", type=float, help="GPU us/layer the opt would remove")
    ap.add_argument("--layers", type=int, default=384, help="layer-forwards per prefill (chunks x layers)")
    ap.add_argument("--kernel", help="substring of a kernel name to triage")
    ap.add_argument("--bytes", type=float, help="theoretical bytes moved per kernel call")
    args = ap.parse_args()

    evs = load(args.trace)
    spans = phase_windows(evs)
    kern = [e for e in evs if e.get("cat") == "kernel"]

    # ---- 1. host-bound check ----
    wall = collections.Counter()
    busy = collections.Counter()
    launches = collections.Counter()
    for s0, s1, kind in spans:
        wall[kind] += s1 - s0
    for e in kern:
        w = None
        for s0, s1, kind in spans:
            if e["ts"] >= s0:
                w = kind
            else:
                break
        if w:
            busy[w] += e["dur"]
            launches[w] += 1
    print("== 1. host-bound check (profiler-inflated wall; ratios are what matter) ==")
    hostbound = {}
    for k in ("PREFILL", "DECODE"):
        if wall[k] == 0:
            continue
        idle = 1 - busy[k] / wall[k]
        hostbound[k] = idle > 0.4
        verdict = "HOST-BOUND — GPU-side opts will NOT move e2e" if hostbound[k] else "GPU-bound — kernel opts pay off"
        print(f"  {k}: wall {wall[k]/1e3:.0f} ms, GPU-busy {busy[k]/1e3:.0f} ms, "
              f"idle {idle:.0%}, {launches[k]} launches -> {verdict}")

    # ---- 2. e2e ceiling for a planned GPU-time removal ----
    if args.remove_us_per_layer:
        total_removed_ms = args.remove_us_per_layer * args.layers / 1e3
        for k in ("PREFILL",):
            if wall[k] == 0:
                continue
            # under host-bound, removed GPU time is absorbed into idle: ceiling ~= 0;
            # under GPU-bound, ceiling = removed / busy fraction of wall
            ceiling = 0.0 if hostbound.get(k) else total_removed_ms / (wall[k] / 1e3)
            print(f"\n== 2. e2e ceiling: removing {args.remove_us_per_layer} us/layer x {args.layers} "
                  f"= {total_removed_ms:.0f} ms GPU ==")
            print(f"  {k}: max e2e gain ~ {ceiling:.1%} "
                  f"({'BELOW' if ceiling < NOISE_FLOOR else 'above'} the ±{NOISE_FLOOR:.0%} noise floor"
                  f"{' — DO NOT proceed on e2e grounds' if ceiling < NOISE_FLOOR else ''})")

    # ---- 3. kernel triage: config-bound vs bandwidth-bound ----
    if args.kernel:
        hits = [e for e in kern if args.kernel in e["name"]]
        if not hits:
            print(f"\n== 3. kernel triage: no kernel matching '{args.kernel}' ==")
            return
        tot = sum(e["dur"] for e in hits)
        per = tot / len(hits)
        print(f"\n== 3. kernel triage: '{args.kernel}' {len(hits)}x, {per:.1f} us/call ==")
        if args.bytes:
            theo = args.bytes / HBM_BW * 1e6
            ratio = per / theo
            verdict = ("CONFIG-bound (bad grid/occupancy) — fix the launch config (~30 lines), "
                       "NOT fusion surgery" if ratio > 3 else "bandwidth-bound — fusion/removal is the right lever")
            print(f"  theoretical HBM time {theo:.1f} us -> actual/theoretical = {ratio:.1f}x -> {verdict}")
        g = hits[0].get("args", {}).get("grid")
        occ = hits[0].get("args", {}).get("est. achieved occupancy %")
        if g:
            print(f"  sample launch: grid={g} occupancy={occ}%")


if __name__ == "__main__":
    main()
