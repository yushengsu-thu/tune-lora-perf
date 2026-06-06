#!/usr/bin/env bash
# 6. Upload the run results to the GitHub repo `lora_perf_lora_profile`
#    (created automatically, private, if it doesn't exist).
#    Input : model name (dir under dev/models/ or unique prefix); $RUN_DIR from step 3/4 state.
#    Output: a commit at <repo>/runs/<model>/<DATE>-<TIME>/ with bench + profiles.
#    Verify: the pushed path is visible via the GitHub API.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state
[ -n "${RUN_DIR:-}" ] && [ -d "$RUN_DIR" ] || { echo "ERROR: no RUN_DIR — run step 3/4/5 first"; exit 1; }

OWNER=$(gh api user --jq .login)
REPO="${RESULTS_REPO:-${OWNER}/lora_perf_lora_profile}"
RUN_TAG="${MODEL}/$(basename "$RUN_DIR")"
echo "== [6/upload] $RUN_DIR  ->  ${REPO}/runs/${RUN_TAG}"

# repo: check / create
if ! gh repo view "$REPO" >/dev/null 2>&1; then
  echo "-- repo $REPO not found — creating (private)"
  gh repo create "$REPO" --private -d "LoRA vs no-LoRA perf benchmarks + torch profiles (GB300 dev runs)" || exit 1
fi

WORK=$(mktemp -d -t lora-perf-up-XXXXXX); trap 'rm -rf "$WORK"' EXIT
git clone -q --depth 1 "https://github.com/${REPO}" "$WORK" 2>/dev/null \
  || { git -C "$WORK" init -q; git -C "$WORK" remote add origin "https://github.com/${REPO}"; }

DST="${WORK}/runs/${RUN_TAG}"; mkdir -p "$DST"
# GitHub hard-limit guard: files >95MB are skipped (listed in the run README)
SKIPPED=""
while IFS= read -r f; do
  rel="${f#"$RUN_DIR"/}"
  sz=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f")
  if [ "$sz" -gt 99614720 ]; then SKIPPED="${SKIPPED}\n- ${rel} ($((sz/1024/1024))MB)"; continue; fi
  mkdir -p "$DST/$(dirname "$rel")"; cp "$f" "$DST/$rel"
done < <(find "$RUN_DIR" -type f)
{ echo "# ${RUN_TAG}"
  echo "- model: ${MODEL}  pods ID: ${ID}"
  echo "- uploaded: $(date '+%Y-%m-%d %H:%M:%S %Z')"
  [ -n "$SKIPPED" ] && printf '\n## Skipped (>95MB GitHub limit)%b\n' "$SKIPPED"
} > "$DST/README.md"

git -C "$WORK" add -A
git -C "$WORK" -c user.name="${GIT_NAME:-$OWNER}" -c user.email="${GIT_EMAIL:-${OWNER}@users.noreply.github.com}" \
  commit -q -m "run ${RUN_TAG}" || { echo "nothing to commit"; }
git -C "$WORK" push -q origin HEAD:main || git -C "$WORK" push -q origin HEAD:master || exit 1

# ---- verify via API ----
gh api "repos/${REPO}/contents/runs/${RUN_TAG}" --jq '.[].name' >/dev/null \
  || { echo "ERROR: pushed path not visible via API"; exit 1; }
echo "== [6/upload] PASS — https://github.com/${REPO}/tree/main/runs/${RUN_TAG}"
