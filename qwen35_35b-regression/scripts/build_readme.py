#!/usr/bin/env python3
"""Generate a per-run README.md from a qwen35_35b-regression RUN_ROOT.

Discovers cells automatically (any subdir under `qwen35/` with bench/acc/prompts/traces),
pulls throughput from bench jsonls, sanity status from bench/bs*.sanity, and coherence
from prompts.md.

Usage:
    python3 build_readme.py <RUN_ROOT> <RUN_TAG> <RESULTS_REPO>
"""
import sys, json
from pathlib import Path


def fail(msg):
    print(f"build_readme: {msg}", file=sys.stderr)
    sys.exit(1)


if len(sys.argv) != 4:
    fail("usage: build_readme.py <RUN_ROOT> <RUN_TAG> <RESULTS_REPO>")

RUN_ROOT = Path(sys.argv[1])
RUN_TAG = sys.argv[2]
RESULTS_REPO = sys.argv[3]
QW = RUN_ROOT / "qwen35"
if not QW.is_dir():
    fail(f"no qwen35/ folder under {RUN_ROOT}")

# meta.env (base_commit, variant_commit, layers, etc.)
meta = {}
mf = QW / "meta.env"
if mf.exists():
    for line in mf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

# Discover cells: any subdir of qwen35/ that has at least one of bench/, acc/, prompts/, traces/.
cells = sorted(
    d.name
    for d in QW.iterdir()
    if d.is_dir()
    and any((d / sub).exists() for sub in ("bench", "acc", "prompts", "traces"))
)
if not cells:
    fail("no cells found (need <cell>/bench or <cell>/acc)")

def _cell_sort_key(c):
    if c == "base":
        return (0, c)
    if c == "variant":
        return (1, c)
    return (2, c)
cells.sort(key=_cell_sort_key)


def tput(cell, bs):
    p = QW / cell / "bench" / f"bs{bs}.jsonl"
    if not p.exists():
        return None
    lines = [x for x in p.read_text().splitlines() if x.strip()]
    if not lines:
        return None
    try:
        return round(json.loads(lines[-1]).get("output_throughput", 0), 1)
    except json.JSONDecodeError:
        return None


def sanity_status(cell):
    # run_qwen35.sh writes the serverlog_sanity verdict to bench/bs<N>.sanity (a file, not just the
    # driver's stdout — so this column reflects the real verdict, unlike the kimi-regression README).
    suspects = []
    sanity_files = sorted((QW / cell / "bench").glob("bs*.sanity"))
    for f in sanity_files:
        if "SUSPECT" in f.read_text(errors="replace"):
            suspects.append(f.stem)
    if not sanity_files:
        return "no-sanity-files"
    return "OK" if not suspects else f"SUSPECT ({', '.join(suspects)})"


def prompts_status(cell):
    p = QW / cell / "prompts" / "prompts.md"
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    if "GARBAGE" in text or "!!!!!!!" in text:
        return "GARBAGE"
    return "coherent"


def trace_counts(cell):
    on_dir = QW / cell / "traces" / "graph_on"
    off_dir = QW / cell / "traces" / "graph_off"
    on = len(list(on_dir.rglob("*.trace.json.gz"))) if on_dir.exists() else 0
    off = len(list(off_dir.rglob("*.trace.json.gz"))) if off_dir.exists() else 0
    return on, off


def has_subdir(cell, sub):
    return (QW / cell / sub).is_dir()


# ----- assemble -----
lines = []
lines.append(f"# `{RUN_TAG}`")
lines.append("")
lines.append("Qwen3.5-35B-A3B-FP8 regression run produced by the `qwen35_35b-regression` skill")
lines.append("(single node, tp4/ep4; launch flags from run_script.sh).")
lines.append(f"Cells: `{', '.join(cells)}`.")
lines.append("")

# ---- Cell configuration ----
lines.append("## Cells in this run")
lines.append("")
for c in cells:
    parts = []
    cell_md = QW / c / "cell.md"
    if cell_md.exists():
        parts.append(cell_md.read_text(errors="replace").strip())
    elif c == "base":
        parts.append(f"**no-LoRA control** on `{meta.get('base_commit','?')[:10]}` (src `{meta.get('base_src','?')}`)")
    elif c.startswith("base_"):
        parts.append(f"**no-LoRA** on `{meta.get('base_commit','?')[:10]}`, override: `{c[len('base_'):]}` (drop `cell.md` to describe the exact envs)")
    elif c == "variant":
        parts.append(f"**LoRA candidate** on `{meta.get('variant_commit','?')[:10]}` (src `{meta.get('variant_src','?')}`)")
    elif c.startswith("variant_"):
        parts.append(f"**LoRA candidate** on `{meta.get('variant_commit','?')[:10]}`, override: `{c[len('variant_'):]}` (drop `cell.md` to describe the exact envs)")
    else:
        parts.append(f"**custom cell** (assumed variant) on `{meta.get('variant_commit','?')[:10]}` — see `cells/{c}/`, drop a `cell.md` to describe it")

    on, off = trace_counts(c)
    extra = []
    if has_subdir(c, "acc"):
        extra.append("acc")
    if has_subdir(c, "bench"):
        extra.append("bench")
    if has_subdir(c, "prompts"):
        extra.append("prompts")
    if on or off:
        extra.append(f"traces (graph-on={on}, graph-off={off})")
    parts.append(f"contents: {', '.join(extra)}")
    lines.append(f"- `{c}` — " + "; ".join(parts))
lines.append("")

# ---- Perf table ----
lines.append("## Performance (output_throughput, tok/s) — graph-on, in=out=2048")
lines.append("")
lines.append("| bs | " + " | ".join(f"`{c}`" for c in cells) + " |")
lines.append("|---|" + "|".join(["---"] * len(cells)) + "|")
for bs in (16, 32, 64):
    row = [str(bs)]
    for c in cells:
        v = tput(c, bs)
        row.append(f"{v}" if v else "—")
    lines.append("| " + " | ".join(row) + " |")
lines.append("")

# % of base
if "base" in cells and len(cells) > 1:
    others = [c for c in cells if c != "base"]
    lines.append("### % of base (no-LoRA) tput")
    lines.append("")
    lines.append("| bs | " + " | ".join(f"`{c}`" for c in others) + " |")
    lines.append("|---|" + "|".join(["---"] * len(others)) + "|")
    for bs in (16, 32, 64):
        row = [str(bs)]
        b = tput("base", bs)
        for c in others:
            v = tput(c, bs)
            row.append(f"{100*v/b:.1f}%" if v and b else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

# ---- Correctness ----
lines.append("## Correctness")
lines.append("")
lines.append("| cell | bench sanity (bench vs server-log decode, <5%) | prompts (decode gate) |")
lines.append("|---|---|---|")
for c in cells:
    san = sanity_status(c) if has_subdir(c, "bench") else "—"
    pr = prompts_status(c) or "—"
    lines.append(f"| `{c}` | {san} | {pr} |")
lines.append("")
lines.append("Acc (per-token logprob diff vs base) is in the run's `summary.md` (built by `summary.py`).")
lines.append("")

# ---- Trace download ----
lines.append("## Traces — separate GitHub Release")
lines.append("")
lines.append(f"All `.trace.json.gz` files for this run are attached to the release **`{RUN_TAG}`** in `{RESULTS_REPO}`. Each cell has its own tarball (`<cell>_traces.tar.gz`) containing `traces/graph_on/` (all 4 TP ranks, **bs64**, 24-step) and `traces/graph_off/` (rank-0 only, **bs16**, 12-step).")
lines.append("")
lines.append("Download one cell's traces:")
lines.append("")
lines.append("```bash")
lines.append(f"gh release download {RUN_TAG} --repo {RESULTS_REPO} --pattern '<cell>_traces.tar.gz'")
lines.append("tar -xzf <cell>_traces.tar.gz")
lines.append("```")
lines.append("")
lines.append("Open `.trace.json.gz` in `chrome://tracing` or [perfetto.dev/viewer](https://ui.perfetto.dev/).")
lines.append("")

# ---- File layout ----
lines.append("## Files in this folder (`cells/<name>/`)")
lines.append("")
lines.append("- `acc/logprobs.json` — per-token logprobs from teacher-forced prefill over the adapter's `compare_sample_train_data.pt`.")
lines.append("- `bench/bs<N>.jsonl` — `bench_one_batch_server` output for batch size N, in=out=2048 (last line has `output_throughput`, `latency`, …).")
lines.append("- `bench/bs<N>.log` — full bench stdout (`--show-report` table inside).")
lines.append("- `bench/bs<N>.serverlog` — sgl scheduler's own `Prefill/Decode batch ... gen throughput` lines (ground truth).")
lines.append("- `bench/bs<N>.sanity` — `serverlog_sanity.py` verdict for that bench (OK / SUSPECT).")
lines.append("- `prompts/prompts.md` — 8 prompts × 3 endpoints, base vs LoRA outputs side-by-side. Decode garbage (`!!!!` / `Thinking!!!!`) shows up here.")
lines.append("")

print("\n".join(lines))
