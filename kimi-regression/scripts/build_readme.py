#!/usr/bin/env python3
"""Generate a per-run README.md from a kimi-regression RUN_ROOT.

Discovers cells automatically (any subdir under `kimi/` with `bench/` or `acc/`),
pulls throughput from bench jsonls, sanity status from bench logs, acc MAE from
`acc/acc_vs_*.txt`, and coherence from prompts.md.

Usage:
    python3 build_readme.py <RUN_ROOT> <RUN_TAG> <RESULTS_REPO>
"""
import sys, json, re
from pathlib import Path


def fail(msg):
    print(f"build_readme: {msg}", file=sys.stderr)
    sys.exit(1)


if len(sys.argv) != 4:
    fail("usage: build_readme.py <RUN_ROOT> <RUN_TAG> <RESULTS_REPO>")

RUN_ROOT = Path(sys.argv[1])
RUN_TAG = sys.argv[2]
RESULTS_REPO = sys.argv[3]
KIMI = RUN_ROOT / "kimi"
if not KIMI.is_dir():
    fail(f"no kimi/ folder under {RUN_ROOT}")

# meta.env (base_commit, variant_commit, etc.)
meta = {}
mf = KIMI / "meta.env"
if mf.exists():
    for line in mf.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            meta[k.strip()] = v.strip()

# Discover cells: any subdir of kimi/ that has at least one of bench/, acc/, prompts/, traces/.
# (profile-only cells like variant_2stream_off have only traces/ and still belong in the README.)
cells = sorted(
    d.name
    for d in KIMI.iterdir()
    if d.is_dir()
    and any((d / sub).exists() for sub in ("bench", "acc", "prompts", "traces"))
)
if not cells:
    fail("no cells found (need <cell>/bench or <cell>/acc)")

# Always put base first if present, then variant, then others alphabetical
def _cell_sort_key(c):
    if c == "base":
        return (0, c)
    if c == "variant":
        return (1, c)
    return (2, c)
cells.sort(key=_cell_sort_key)


def tput(cell, bs):
    p = KIMI / cell / "bench" / f"bs{bs}.jsonl"
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
    suspects = []
    for log in sorted((KIMI / cell / "bench").glob("bs*.log")):
        if "SUSPECT" in log.read_text(errors="replace"):
            suspects.append(log.stem)
    return "OK" if not suspects else f"SUSPECT ({', '.join(suspects)})"


def acc_vs_files(cell):
    """Return list of (label, mae, pearson) tuples for any acc_vs_*.txt in this cell."""
    out = []
    for p in sorted((KIMI / cell / "acc").glob("acc_vs_*.txt")):
        label = p.stem.replace("acc_vs_", "")
        text = p.read_text(errors="replace")
        mae = re.search(r"mean\|variant-base\|\s*=\s*([\d.]+)", text)
        pear = re.search(r"pearson corr\s*=\s*([\d.]+)", text)
        if mae and pear:
            out.append((label, float(mae.group(1)), float(pear.group(1))))
    return out


def prompts_status(cell):
    p = KIMI / cell / "prompts" / "prompts.md"
    if not p.exists():
        return None
    text = p.read_text(errors="replace")
    if "GARBAGE" in text or "!!!!!!!" in text:
        return "GARBAGE"
    # If prompts ran and didn't garbage, mark coherent
    return "coherent"


def trace_counts(cell):
    # rglob, not glob — traces from the skill's pull_traces() flatten directly under graph_{on,off}/,
    # but ad-hoc tar-streamed pulls (e.g. profile-only custom cells) may preserve a timestamp subdir.
    on_dir = KIMI / cell / "traces" / "graph_on"
    off_dir = KIMI / cell / "traces" / "graph_off"
    on = len(list(on_dir.rglob("*.trace.json.gz"))) if on_dir.exists() else 0
    off = len(list(off_dir.rglob("*.trace.json.gz"))) if off_dir.exists() else 0
    return on, off


def has_subdir(cell, sub):
    return (KIMI / cell / sub).is_dir()


# ----- assemble -----
lines = []
lines.append(f"# `{RUN_TAG}`")
lines.append("")
lines.append(f"Kimi-K2.5-NVFP4 regression run produced by the `kimi-regression` skill.")
lines.append(f"Cells: `{', '.join(cells)}`.")
lines.append("")

# ---- Cell configuration ----
lines.append("## Cells in this run")
lines.append("")
for c in cells:
    parts = []
    cell_md = KIMI / c / "cell.md"
    if cell_md.exists():
        # Explicit cell description always wins.
        parts.append(cell_md.read_text(errors="replace").strip())
    elif c == "base":
        commit = meta.get("base_commit", "?")[:10]
        src = meta.get("base_src", "?")
        parts.append(f"**no-LoRA control** on `{commit}` (src `{src}`)")
    elif c.startswith("base_"):
        # e.g. base_ep8 = no-LoRA variant of base with a launch tweak; commit = base_commit.
        commit = meta.get("base_commit", "?")[:10]
        suffix = c[len("base_"):]
        parts.append(f"**no-LoRA** on `{commit}`, override: `{suffix}` (drop `cell.md` to describe the exact envs)")
    elif c == "variant":
        commit = meta.get("variant_commit", "?")[:10]
        src = meta.get("variant_src", "?")
        parts.append(f"**LoRA candidate** on `{commit}` (src `{src}`)")
    elif c.startswith("variant_"):
        commit = meta.get("variant_commit", "?")[:10]
        suffix = c[len("variant_"):]
        parts.append(f"**LoRA candidate** on `{commit}`, override: `{suffix}` (drop `cell.md` to describe the exact envs)")
    else:
        commit = meta.get("variant_commit", "?")[:10]
        parts.append(f"**custom cell** (assumed variant) on `{commit}` — see `cells/{c}/`, drop a `cell.md` to describe it")

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
lines.append("| cell | bench sanity (bench vs server-log decode, <5%) | acc diffs | prompts |")
lines.append("|---|---|---|---|")
for c in cells:
    san = sanity_status(c) if has_subdir(c, "bench") else "—"
    accs = acc_vs_files(c)
    accs_s = "; ".join(f"vs {lbl}: MAE {m:.3f}, pearson {p:.3f}" for lbl, m, p in accs) if accs else "—"
    pr = prompts_status(c) or "—"
    lines.append(f"| `{c}` | {san} | {accs_s} | {pr} |")
lines.append("")

# ---- Trace download ----
lines.append("## Traces — separate GitHub Release")
lines.append("")
lines.append(f"All `.trace.json.gz` files for this run are attached to the release **`{RUN_TAG}`** in `{RESULTS_REPO}`. Each cell has its own tarball (`<cell>_traces.tar.gz`) containing `traces/graph_on/` (all TP ranks across both pods, bs64 unless noted) and `traces/graph_off/` (rank-0 only, bs64).")
lines.append("")
lines.append("Download one cell's traces:")
lines.append("")
lines.append("```bash")
lines.append(f"gh release download {RUN_TAG} --repo {RESULTS_REPO} --pattern '<cell>_traces.tar.gz'")
lines.append("tar -xzf <cell>_traces.tar.gz")
lines.append("```")
lines.append("")
lines.append("Download every cell's traces (the whole release):")
lines.append("")
lines.append("```bash")
lines.append(f"gh release download {RUN_TAG} --repo {RESULTS_REPO}")
lines.append("for t in *_traces.tar.gz; do tar -xzf \"$t\"; done")
lines.append("```")
lines.append("")
lines.append("Open `.trace.json.gz` in `chrome://tracing` or [perfetto.dev/viewer](https://ui.perfetto.dev/).")
lines.append("")

# ---- File layout ----
lines.append("## Files in this folder (`cells/<name>/`)")
lines.append("")
lines.append("- `acc/logprobs.json` — per-token logprobs from teacher-forced prefill over the adapter's `compare_sample_train_data.pt`.")
lines.append("- `acc/acc_vs_<ref>.txt` — per-token diff vs a reference cell (e.g. cutlass-LoRA gold). Reports `mean|variant-base|` (MAE), `max|variant-base|`, and `pearson corr`.")
lines.append("- `bench/bs<N>.jsonl` — `bench_one_batch_server` output for batch size N, in=out=2048 (the last line has `output_throughput`, `latency`, `last_ttft`, etc.).")
lines.append("- `bench/bs<N>.log` — full bench stdout (`--show-report` table inside).")
lines.append("- `bench/bs<N>.serverlog` — sgl scheduler's own `Prefill batch ... gen throughput` and `Decode batch ... gen throughput` lines, used by the sanity check.")
lines.append("- `prompts/prompts.md` — 8 prompts × 3 endpoints, base vs LoRA outputs side-by-side. Decode garbage (`!!!!`-collapse) shows up here.")
lines.append("")
lines.append(f"_See the [`kimi-regression` skill SKILL.md](https://github.com/{RESULTS_REPO}/blob/main/SKILL.md) for the full meaning of each artifact._")

Path(sys.argv[2])  # noop, ensure tag is a string
print("\n".join(lines))
