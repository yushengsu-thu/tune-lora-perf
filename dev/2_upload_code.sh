#!/usr/bin/env bash
# 2. Upload the dev code (CURRENT branch of $SGLANG_SRC, default /Users/yushengsu/Downloads/tml/sglang)
#    to every pod and install it.
#    Input : model name (qwen|kimi); state from step 1. Optional SGLANG_SRC override.
#    Output: every pod's /root/sglang checked out at the local HEAD commit, `pip install -e python`
#            done, flashinfer re-pinned to the image-matching ${FLASHINFER_PIN}.
#    Verify: in-pod HEAD == local HEAD on every pod + `import sglang` works.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state

BRANCH=$(git -C "$SGLANG_SRC" branch --show-current)
LOCAL_HEAD=$(git -C "$SGLANG_SRC" rev-parse HEAD)
echo "== [2/upload] $MODEL  src=$SGLANG_SRC  branch=${BRANCH:-detached}  HEAD=${LOCAL_HEAD:0:12}"
[ -n "$(git -C "$SGLANG_SRC" status --porcelain)" ] && \
  echo "WARN: working tree is DIRTY — only the COMMITTED state of HEAD is uploaded"

# Thin bundle: only the commits the pod is missing (boundary = the pod's current HEAD, which the
# local repo must contain — it does if local has fetched upstream main, since pods clone the fork).
POD_HEAD=$(kh 'cd /root/sglang && git rev-parse HEAD' | tr -d '[:space:]')
echo "-- pod HEAD: ${POD_HEAD:0:12}"
if git -C "$SGLANG_SRC" cat-file -e "$POD_HEAD" 2>/dev/null; then
  git -C "$SGLANG_SRC" bundle create /tmp/dev_sglang.bundle HEAD --not "$POD_HEAD" 2>/dev/null \
    || git -C "$SGLANG_SRC" bundle create /tmp/dev_sglang.bundle HEAD --not "${POD_HEAD}^"
else
  echo "WARN: pod HEAD unknown locally — bundling vs merge-base with origin/main"
  MB=$(git -C "$SGLANG_SRC" merge-base HEAD origin/main)
  git -C "$SGLANG_SRC" bundle create /tmp/dev_sglang.bundle HEAD --not "${MB}^"
fi

for P in "${PODS[@]}"; do
  echo "-- $P: push + checkout + install"
  $KC cp /tmp/dev_sglang.bundle "$P:/root/dev.bundle"
  kp "$P" "cd /root/sglang \
    && git fetch -f /root/dev.bundle HEAD:refs/heads/__dev \
    && git checkout -qf --detach __dev \
    && pip install -e python >/tmp/pip.log 2>&1 \
    && pip install -q \"flashinfer_python[cu13]==${FLASHINFER_PIN}\" \"flashinfer_cubin==${FLASHINFER_PIN}\" >/dev/null 2>&1 \
    && python3 -c 'import sglang, flashinfer; print(\"  import OK — sglang @\", flashinfer.__version__)'" \
    || { echo "ERROR: install failed on $P"; kp "$P" 'tail -20 /tmp/pip.log'; exit 1; }
  GOT=$(kp "$P" 'cd /root/sglang && git rev-parse HEAD' | tr -d '[:space:]')
  [ "$GOT" = "$LOCAL_HEAD" ] || { echo "ERROR: $P HEAD=$GOT != local $LOCAL_HEAD"; exit 1; }
  echo "  $P @ ${GOT:0:12} OK"
done

echo "== [2/upload] PASS — all pods at ${LOCAL_HEAD:0:12} (${BRANCH:-detached})"
