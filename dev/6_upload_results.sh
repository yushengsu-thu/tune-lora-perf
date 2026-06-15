#!/usr/bin/env bash
# 6. Upload the run results to the GitHub repo `lora_perf_lora_profile`
#    (created automatically, private, if it doesn't exist).
#    Input : model name (dir under dev/models/ or unique prefix); $RUN_DIR from step 3/4 state.
#    Split:  SMALL artifacts (bench summary/jsonl/serverlog, gpu_busy_witness output, bench.log,
#            README) are committed under runs/<model>/<DATE-TIME>/. LARGE torch traces
#            (`*.trace.json.gz`: graph-on/off, gpu-only, perfetto copies) are uploaded as assets to
#            a per-run GitHub RELEASE (tag <model>-<DATE-TIME>) — kept OUT of git history so clones
#            stay small (2 GiB/file limit; our traces are ~tens of MB). The repo README links to the
#            release + the `gh release download` command.
#    profile upload is MANDATORY: if traces exist they MUST land in the release (the step ERRORS
#    otherwise — do NOT skip the profile on a flaky uplink; the pod pull already retries).
#    Verify: the pushed repo path is visible via the API AND the release has all the trace assets.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
[ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ] || { echo "ERROR: no RUN_DIR — run step 3/4/5 first"; exit 1; }

OWNER=$(gh api user --jq .login)
REPO="${RESULTS_REPO:-${OWNER}/lora_perf_lora_profile}"
# RESULTS_RUN_NAME overrides the default <MODEL>/<timestamp> path with a flat run name under runs/.
RUN_TAG="${RESULTS_RUN_NAME:-${MODEL}/$(basename "$RUN_DIR")}"
REL_TAG="${RESULTS_REL_TAG:-${MODEL}-$(basename "$RUN_DIR")}"   # release tag: flat, slash-free, per-run
REL_URL="https://github.com/${REPO}/releases/tag/${REL_TAG}"
echo "== [6/upload] $RUN_DIR  ->  repo ${REPO}/runs/${RUN_TAG}  + traces in release ${REL_TAG}"

# repo: check / create
if ! gh repo view "$REPO" >/dev/null 2>&1; then
  echo "-- repo $REPO not found — creating (private)"
  gh repo create "$REPO" --private -d "LoRA vs no-LoRA perf benchmarks + torch profiles (GB300 dev runs)" || exit 1
fi

WORK=$(mktemp -d -t lora-perf-up-XXXXXX); trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 "https://github.com/${REPO}" "$WORK" 2>/dev/null \
  || { git -C "$WORK" init -q; git -C "$WORK" remote add origin "https://github.com/${REPO}"; }

DST="${WORK}/runs/${RUN_TAG}"; mkdir -p "$DST"
# Split traces (-> release) from small artifacts (-> repo). Any OTHER >95MB file is skipped+listed.
TRACES=(); SKIPPED=""
while IFS= read -r f; do
  rel="${f#"$RUN_DIR"/}"
  case "$rel" in
    *.trace.json.gz) TRACES+=("$f"); continue ;;     # profile traces -> GitHub Release, not repo
  esac
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [ "$sz" -gt 99614720 ]; then SKIPPED="${SKIPPED}\n- ${rel} ($((sz/1024/1024))MB, non-trace >95MB)"; continue; fi
  mkdir -p "$DST/$(dirname "$rel")"; cp "$f" "$DST/$rel"
done < <(find "$RUN_DIR" -type f)

{ echo "# ${RUN_TAG}"
  echo "- model: ${MODEL}  pods ID: ${ID}"
  echo "- uploaded: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  echo
  echo "## Profile traces — GitHub Release (kept out of the repo to save space)"
  echo "- release: <${REL_URL}>"
  echo "- download all traces for this run:"
  echo '  ```'
  echo "  gh release download ${REL_TAG} -R ${REPO} -D ./traces"
  echo '  ```'
  if [ ${#TRACES[@]} -gt 0 ]; then
    echo "- ${#TRACES[@]} trace asset(s) (named \`cell__graph_mode__file\`):"
    for t in "${TRACES[@]}"; do echo "  - \`${t#"$RUN_DIR"/}\`"; done
  fi
  [ -n "$SKIPPED" ] && printf '\n## Skipped (non-trace >95MB)%b\n' "$SKIPPED"
} > "$DST/README.md"

git -C "$WORK" add -A
git -C "$WORK" -c user.name="${GIT_NAME:-$OWNER}" -c user.email="${GIT_EMAIL:-${OWNER}@users.noreply.github.com}" \
  commit -q -m "run ${RUN_TAG} (traces -> release ${REL_TAG})" || echo "nothing to commit"
git -C "$WORK" push -q origin HEAD:main || git -C "$WORK" push -q origin HEAD:master || exit 1

# ---- profile traces -> GitHub Release (MANDATORY when traces exist) ----
if [ ${#TRACES[@]} -gt 0 ]; then
  STAGE="${WORK}/relassets"; mkdir -p "$STAGE"; ASSETS=()
  for t in "${TRACES[@]}"; do
    rel="${t#"$RUN_DIR"/}"; flat="${rel//\//__}"     # cell/mode path -> flat unique asset name
    cp "$t" "$STAGE/$flat"; ASSETS+=("$STAGE/$flat")
  done
  if gh release view "$REL_TAG" -R "$REPO" >/dev/null 2>&1; then
    gh release upload "$REL_TAG" -R "$REPO" --clobber "${ASSETS[@]}" \
      || { echo "ERROR: release asset upload failed ($REL_TAG)"; exit 1; }
  else
    gh release create "$REL_TAG" -R "$REPO" -t "${RUN_TAG} — profile traces" \
      -n "torch profiler traces for runs/${RUN_TAG} (graph-on / graph-off / gpu-only / perfetto). Asset name = cell__graph_mode__file." \
      "${ASSETS[@]}" || { echo "ERROR: release create failed ($REL_TAG)"; exit 1; }
  fi
  n=$(gh release view "$REL_TAG" -R "$REPO" --json assets --jq '.assets | length' 2>/dev/null || echo 0)
  [ "${n:-0}" -ge "${#TRACES[@]}" ] || { echo "ERROR: release ${REL_TAG} has ${n} assets, expected >= ${#TRACES[@]}"; exit 1; }
  echo "  ${#TRACES[@]} trace(s) -> ${REL_URL}"
else
  echo "  WARN: no *.trace.json.gz under $RUN_DIR — nothing to release (profile missing?)"
fi

# ---- verify repo path via API ----
gh api "repos/${REPO}/contents/runs/${RUN_TAG}" --jq '.[].name' >/dev/null \
  || { echo "ERROR: pushed path not visible via API"; exit 1; }
echo "== [6/upload] PASS — repo: https://github.com/${REPO}/tree/main/runs/${RUN_TAG}  | traces: ${REL_URL}"
