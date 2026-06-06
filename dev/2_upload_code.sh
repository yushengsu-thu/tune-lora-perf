#!/usr/bin/env bash
# 2. Upload the dev code from $SGLANG_SRC (default /Users/yushengsu/Downloads/tml/sglang)
#    to every pod and install it.
#    Input : model name (dir under dev/models/ or unique prefix); state from step 1.
#            Optional: SGLANG_SRC=<path> ; SGLANG_BRANCH=<branch|ref> picks WHAT to upload
#            (without touching your checkout — default: the checkout's current HEAD).
#    Guard : uploading the wrong branch wastes hours, so the branch is always CONFIRMED:
#            SGLANG_BRANCH=... counts as explicit; otherwise an interactive [y/N] prompt asks;
#            non-interactive runs (run_all in background/CI) FAIL unless SGLANG_BRANCH or
#            SGLANG_CONFIRM=1 is given.
#    Output: every pod's /root/sglang checked out at the chosen commit, `pip install -e python`
#            done, flashinfer re-pinned to the image-matching ${FLASHINFER_PIN}.
#    Verify: in-pod HEAD == chosen commit on every pod + `import sglang` works.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

# ---- what to upload: SGLANG_BRANCH if given, else the checkout's current HEAD ----
UPLOAD_REF="${SGLANG_BRANCH:-HEAD}"
git -C "$SGLANG_SRC" rev-parse --verify --quiet "${UPLOAD_REF}^{commit}" >/dev/null \
  || { echo "ERROR: SGLANG_BRANCH='$UPLOAD_REF' is not a branch/ref in $SGLANG_SRC"; exit 1; }
LOCAL_HEAD=$(git -C "$SGLANG_SRC" rev-parse "${UPLOAD_REF}^{commit}")
if [ "$UPLOAD_REF" = HEAD ]; then BRANCH=$(git -C "$SGLANG_SRC" branch --show-current)
else BRANCH="$UPLOAD_REF"; fi
echo "== [2/upload] $MODEL  src=$SGLANG_SRC  branch=${BRANCH:-detached}  HEAD=${LOCAL_HEAD:0:12}"
[ "$UPLOAD_REF" = HEAD ] && [ -n "$(git -C "$SGLANG_SRC" status --porcelain)" ] && \
  echo "WARN: working tree is DIRTY — only the COMMITTED state of HEAD is uploaded"

# ---- confirm the branch before spending pod-hours on it ----
if [ -z "${SGLANG_BRANCH:-}" ] && [ "${SGLANG_CONFIRM:-0}" != 1 ]; then
  if [ -t 0 ]; then
    printf "Upload branch '%s' @ %s and run the e2e on it? [y/N] " "${BRANCH:-detached}" "${LOCAL_HEAD:0:12}"
    read -r ANS
    case "$ANS" in y|Y|yes|YES) ;; *) echo "aborted — pick one with SGLANG_BRANCH=<branch>"; exit 1 ;; esac
  else
    echo "ERROR: non-interactive run and no branch chosen — refusing to guess."
    echo "       SGLANG_BRANCH=<branch>  uploads that branch (checkout untouched), or"
    echo "       SGLANG_CONFIRM=1        accepts the current checkout (${BRANCH:-detached} @ ${LOCAL_HEAD:0:12})"
    exit 1
  fi
fi

# Thin bundle: only the commits the pod is missing (boundary = the pod's current HEAD, which the
# local repo must contain — it does if local has fetched upstream main, since pods clone the fork).
# If the pod's repo ALREADY contains the chosen commit (e.g. an ancestor of its HEAD), a thin
# bundle would be empty — skip the bundle and just checkout.
POD_HEAD=$(kh 'cd /root/sglang && git rev-parse HEAD' | tr -d '[:space:]')
echo "-- pod HEAD: ${POD_HEAD:0:12}"
NEED_BUNDLE=1
if kh "cd /root/sglang && git cat-file -e ${LOCAL_HEAD}" >/dev/null 2>&1; then
  NEED_BUNDLE=0; echo "-- pods already contain ${LOCAL_HEAD:0:12} — checkout only, no bundle"
elif git -C "$SGLANG_SRC" cat-file -e "$POD_HEAD" 2>/dev/null; then
  git -C "$SGLANG_SRC" bundle create /tmp/dev_sglang.bundle "$UPLOAD_REF" --not "$POD_HEAD" 2>/dev/null \
    || git -C "$SGLANG_SRC" bundle create /tmp/dev_sglang.bundle "$UPLOAD_REF" --not "${POD_HEAD}^"
else
  echo "WARN: pod HEAD unknown locally — bundling vs merge-base with origin/main"
  MB=$(git -C "$SGLANG_SRC" merge-base "$UPLOAD_REF" origin/main)
  git -C "$SGLANG_SRC" bundle create /tmp/dev_sglang.bundle "$UPLOAD_REF" --not "${MB}^"
fi

for P in "${PODS[@]}"; do
  echo "-- $P: push + checkout + install"
  if [ "$NEED_BUNDLE" = 1 ]; then
    $KC cp /tmp/dev_sglang.bundle "$P:/root/dev.bundle"
    kp "$P" "cd /root/sglang && git fetch -f /root/dev.bundle '${UPLOAD_REF}':refs/heads/__dev" \
      || { echo "ERROR: bundle fetch failed on $P"; exit 1; }
  fi
  kp "$P" "cd /root/sglang \
    && git checkout -qf --detach ${LOCAL_HEAD} \
    && { [ -f \"\$HOME/.cargo/env\" ] && . \"\$HOME/.cargo/env\"; true; } \
    && pip install -e python >/tmp/pip.log 2>&1 \
    && pip install -q \"flashinfer_python[cu13]==${FLASHINFER_PIN}\" \"flashinfer_cubin==${FLASHINFER_PIN}\" >/dev/null 2>&1 \
    && python3 -c 'import sglang, flashinfer; print(\"  import OK — sglang @\", flashinfer.__version__)'" \
    || { echo "ERROR: install failed on $P"; kp "$P" 'tail -20 /tmp/pip.log'; exit 1; }
  GOT=$(kp "$P" 'cd /root/sglang && git rev-parse HEAD' | tr -d '[:space:]')
  [ "$GOT" = "$LOCAL_HEAD" ] || { echo "ERROR: $P HEAD=$GOT != local $LOCAL_HEAD"; exit 1; }
  echo "  $P @ ${GOT:0:12} OK"
done

echo "== [2/upload] PASS — all pods at ${LOCAL_HEAD:0:12} (${BRANCH:-detached})"
