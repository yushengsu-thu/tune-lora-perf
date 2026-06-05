#!/usr/bin/env bash
# Publish a finished regression run to a private GitHub repo + release (model-agnostic:
# the model dir + tag prefix are discovered from the run's <model>/meta.env).
#
# What goes where (split by file size):
#   - Tiny artifacts (acc/, bench/, prompts/, README.md) -> a NEW commit at
#     <RESULTS_REPO>/runs/<RUN_TAG>/cells/<cell>/  (regular git, append-only history).
#   - Big traces (traces/graph_{on,off}/*.trace.json.gz) -> a NEW GitHub Release tagged
#     <RUN_TAG>, one tarball per cell (`<cell>_traces.tar.gz`), 2 GB asset limit (well under).
#
# Each new run is APPEND-ONLY: a new folder + a new release. Previous runs/releases stay.
#
# Inputs (env):
#   RUN_ROOT       (required) — local run folder, e.g. ~/Downloads/sglang_<model>_reg_<id>_<ts>
#   RESULTS_REPO   (required) — <owner>/<repo>, can be private; you must have push + release auth
#   RUN_TAG        (optional) — defaults to: <tag_prefix>-<variant-shorthash>-<timestamp>
#                               (tag_prefix from meta.env, e.g. kimi-reg / qwen35-reg)
#   RESULTS_LOCAL  (optional) — local clone path, defaults to ~/.cache/sglang-results-<repo>
#   PUBLISH_DRY    (optional) — set to 1 to skip git push + gh release create (everything else runs)
#
# Pre-reqs:
#   - `gh` CLI authenticated (`gh auth status`); push rights to RESULTS_REPO; org allows releases.
set -euo pipefail

: "${RUN_ROOT:?must set RUN_ROOT (the local regression run folder)}"
: "${RESULTS_REPO:?must set RESULTS_REPO (e.g. <owner>/<results-repo>)}"

# ---- discover the model dir: the RUN_ROOT subdir holding meta.env ----
MODEL_OUT=""
for d in "$RUN_ROOT"/*/; do
  [ -f "${d}meta.env" ] && { MODEL_OUT="${d%/}"; break; }
done
[ -n "$MODEL_OUT" ] || { echo "publish: no <model>/meta.env under $RUN_ROOT" >&2; exit 1; }
META="$MODEL_OUT/meta.env"
MODEL_NAME=$(basename "$MODEL_OUT")

# Read meta.env: variant commit shorthash + tag prefix drive the default tag
variant_commit=$(grep -E '^variant_commit=' "$META" | head -1 | cut -d= -f2 || true)
tag_prefix=$(grep -E '^tag_prefix=' "$META" | head -1 | cut -d= -f2 || true)
[ -z "$tag_prefix" ] && tag_prefix="${MODEL_NAME}-reg"

TS=$(basename "$RUN_ROOT" | sed -nE 's/.*_([0-9]{8}_[0-9]{6})$/\1/p')
[ -z "$TS" ] && TS="$(date +%Y%m%d_%H%M%S)"
VSHORT="${variant_commit:0:10}"
[ -z "$VSHORT" ] && VSHORT="unknown"
DEFAULT_TAG="${tag_prefix}-${VSHORT}-${TS}"
RUN_TAG="${RUN_TAG:-$DEFAULT_TAG}"

# Tag must be a valid git ref + DNS-safe asset prefix
case "$RUN_TAG" in
  *[!a-zA-Z0-9._-]*) echo "publish: RUN_TAG has invalid chars: $RUN_TAG" >&2; exit 1;;
esac

repo_safe=$(echo "$RESULTS_REPO" | tr '/' '_')
RESULTS_LOCAL="${RESULTS_LOCAL:-$HOME/.cache/sglang-results-${repo_safe}}"
SKILL_SCRIPTS="$(cd "$(dirname "$0")" && pwd)"

echo "publish: MODEL=$MODEL_NAME"
echo "publish: RUN_TAG=$RUN_TAG"
echo "publish: RESULTS_REPO=$RESULTS_REPO"
echo "publish: RESULTS_LOCAL=$RESULTS_LOCAL"

# ---- 1. Clone (or fast-forward) the results repo ----
if [ ! -d "$RESULTS_LOCAL/.git" ]; then
  echo "publish: cloning $RESULTS_REPO -> $RESULTS_LOCAL"
  mkdir -p "$(dirname "$RESULTS_LOCAL")"
  gh repo clone "$RESULTS_REPO" "$RESULTS_LOCAL"
else
  echo "publish: fast-forwarding existing clone"
  ( cd "$RESULTS_LOCAL" && git fetch --quiet origin && git checkout --quiet "$(git symbolic-ref --short refs/remotes/origin/HEAD | cut -d/ -f2)" && git pull --quiet --ff-only )
fi

# Refuse to clobber an existing run folder
RUN_DIR="$RESULTS_LOCAL/runs/$RUN_TAG"
if [ -e "$RUN_DIR" ]; then
  echo "publish: $RUN_DIR already exists in $RESULTS_REPO — set a different RUN_TAG to avoid clobbering" >&2
  exit 1
fi

# ---- 2. Build README for this run ----
mkdir -p "$RUN_DIR"
python3 "$SKILL_SCRIPTS/build_readme.py" "$RUN_ROOT" "$RUN_TAG" "$RESULTS_REPO" > "$RUN_DIR/README.md"
echo "publish: README -> $RUN_DIR/README.md ($(wc -l <"$RUN_DIR/README.md") lines)"

# ---- 3. Copy small artifacts (everything except traces) into runs/<tag>/cells/ ----
mkdir -p "$RUN_DIR/cells"
# rsync excludes traces dir entirely (those go to the release)
rsync -a --exclude='traces/' --exclude='meta.env' --exclude='*.out' --exclude='progress.log' \
      "$MODEL_OUT/" "$RUN_DIR/cells/"
# Keep meta.env at the run-folder level (useful for re-running build_readme.py later)
cp "$META" "$RUN_DIR/meta.env"

small_bytes=$(du -sk "$RUN_DIR" | awk '{print $1}')
echo "publish: small-files staged for commit: $((small_bytes)) KB at $RUN_DIR"

# ---- 4. Tar traces per cell -> staging dir for the release ----
TRACES_STAGE=$(mktemp -d -t reg-traces-XXXXXX)
trap 'rm -rf "$TRACES_STAGE"' EXIT
trace_assets=()
for cell_path in "$MODEL_OUT"/*/; do
  cell=$(basename "$cell_path")
  if [ -d "$cell_path/traces" ] && [ -n "$(find "$cell_path/traces" -name '*.trace.json.gz' -print -quit 2>/dev/null)" ]; then
    out="$TRACES_STAGE/${cell}_traces.tar.gz"
    ( cd "$cell_path" && tar -czf "$out" traces )
    sz=$(du -h "$out" | awk '{print $1}')
    echo "publish: tarred $cell traces -> ${cell}_traces.tar.gz ($sz)"
    trace_assets+=("$out")
  fi
done

# ---- 5. Commit + push the small files ----
if [ "${PUBLISH_DRY:-0}" = "1" ]; then
  echo "publish: PUBLISH_DRY=1 — skipping git push and gh release create"
  echo "publish: (would commit) $RUN_DIR -> $RESULTS_REPO main"
  echo "publish: (would upload) ${trace_assets[*]:-<none>} -> release $RUN_TAG"
  exit 0
fi

(
  cd "$RESULTS_LOCAL"
  git add "runs/$RUN_TAG"
  git -c "user.name=$(git config user.name || echo regression-bot)" \
      -c "user.email=$(git config user.email || echo regression@local)" \
      commit -m "$RUN_TAG: $MODEL_NAME regression run

Cells: $(ls -1 "$RUN_DIR/cells" | tr '\n' ' ')
Traces in release: $RUN_TAG"
  git push origin HEAD
)
COMMIT_URL="https://github.com/$RESULTS_REPO/tree/main/runs/$RUN_TAG"
echo "publish: committed -> $COMMIT_URL"

# ---- 6. Create the GitHub Release with the trace tarballs ----
# The release notes ARE the README (so reading the release page = reading the per-run README).
gh release create "$RUN_TAG" \
    --repo "$RESULTS_REPO" \
    --title "$RUN_TAG" \
    --notes-file "$RUN_DIR/README.md" \
    "${trace_assets[@]}"
RELEASE_URL="https://github.com/$RESULTS_REPO/releases/tag/$RUN_TAG"
echo "publish: release -> $RELEASE_URL"

# Surface a Slack-pasteable line as the last thing
echo ""
echo "================ PUBLISHED ================"
echo "  Summary: $COMMIT_URL"
echo "  Traces:  $RELEASE_URL"
