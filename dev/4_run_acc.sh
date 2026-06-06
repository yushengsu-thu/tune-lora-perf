#!/usr/bin/env bash
# 4. Accuracy, two comparisons in one run:
#    (a) LoRA vs no-LoRA teacher-forced per-token logprob diff (regression's acc test).
#    (b) sglang vs the vLLM/trainer reference (lora-dev-script/check_sglang_lora_correctness.py):
#        the .pt ships training_logprobs (trainer) + sampling_logprobs (original vLLM sampler);
#        report KL(=0.5·mean((a−b)²)) of sglang-lora against both, next to the inherent
#        KL(vLLM, trainer) noise floor. Reference .pt files live in the (private) HF dataset
#        hf.co/datasets/yushengsu/datasets — select one with ACC_HF_FILE=<path-in-repo>.
#    ⚠ REFERENCE-APPLICABILITY RULE: the yushengsu/datasets references ONLY apply when the
#        served adapter carries the `experts_shared_outer_loras` tag (the semantics of
#        github.com/sgl-project/sglang/pull/21466). A GENERAL adapter (e.g.
#        jybsuper/qwen35_35b_lora_alpha) must use its OWN bundled compare_sample_train_data.pt
#        (the default ACC_DATA) — against the wrong reference the KL table is meaningless
#        (measured 2026-06-06: KL≈0.42–0.52 vs floor 0.0006, even for the no-lora cell).
#        The script detects the tag in ${LORA_PATH}/adapter_config.json and refuses a
#        mismatched ACC_HF_FILE (ACC_FORCE=1 overrides).
#    Feeds compare_sample_train_data.pt to /generate with max_new_tokens=0 + return_logprob
#    (prefill-only, no sampling) for BOTH cells.
#    NOTE: prefill-only — it CANNOT see decode garbage; the coherence gate (run per cell here
#    too) is the decode check.
#    Input : model name (dir under dev/models/ or unique prefix); pods from step 1
#            running the code from step 2.
#            Optional: ACC_DATA=<in-pod .pt path> | ACC_HF_FILE=<file in ACC_HF_REPO
#            (default yushengsu/datasets)> to use a reference .pt from HF.
#    Output: $RUN_DIR/acc/{no-lora,lora}/logprobs.json + $RUN_DIR/acc/summary.md
#            (lora-vs-no-lora table + the 3-row KL-vs-reference table).
#    Verify: both logprob sets captured + same length. max>ACC_TOL is a WARNING by default
#            (alpha is a near-identity adapter so diff ≈ LoRA-path numerical noise; a custom
#            adapter legitimately diverges) — set ACC_STRICT=1 to fail on it. The KL-vs-
#            reference rows are informational (no hard gate — tolerances differ per adapter).
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
OUTROOT=/tmp/dev_run
ACC_DATA="${ACC_DATA:-${LORA_PATH}/compare_sample_train_data.pt}"
echo "== [4/acc] $MODEL  data=${ACC_DATA}  tol=${ACC_TOL}  -> ${RUN_DIR}/acc"

# ---- reference-applicability gate (see the header RULE) ----
# yushengsu/datasets references <-> experts_shared_outer_loras-tagged adapters ONLY.
ESOL=$(kh "grep -qF experts_shared_outer_loras ${LORA_PATH}/adapter_config.json 2>/dev/null && echo 1 || echo 0" | tr -d '[:space:]')
echo "-- adapter experts_shared_outer_loras tag: ${ESOL:-0}"
if [ -n "${ACC_HF_FILE:-}" ] && [ "${ESOL:-0}" != 1 ] && [ "${ACC_FORCE:-0}" != 1 ]; then
  echo "ERROR: ACC_HF_FILE is set but the served adapter (${LORA_PATH}) has NO"
  echo "       experts_shared_outer_loras tag — the yushengsu/datasets references only apply"
  echo "       to adapters with that tag (sgl-project/sglang#21466). For a general adapter use"
  echo "       its bundled compare_sample_train_data.pt (the default). ACC_FORCE=1 overrides."
  exit 1
fi
if [ "${ESOL:-0}" = 1 ] && [ -z "${ACC_HF_FILE:-}" ]; then
  echo "NOTE: adapter IS experts_shared_outer_loras-tagged — its bundled .pt may not match;"
  echo "      pick a reference from hf.co/datasets/yushengsu/datasets via ACC_HF_FILE=<file>."
fi

# Optional: pull a reference .pt from the (private) HF dataset repo instead of the adapter's copy.
#   ACC_HF_FILE=<path-in-repo> [ACC_HF_REPO=yushengsu/datasets] bash 4_run_acc.sh qwen
# The pod's HF_TOKEN (hf-token-yanbin secret) authenticates the download.
if [ -n "${ACC_HF_FILE:-}" ]; then
  ACC_HF_REPO="${ACC_HF_REPO:-yushengsu/datasets}"
  echo "-- downloading reference data ${ACC_HF_REPO}:${ACC_HF_FILE} onto the pod"
  kh "hf download '${ACC_HF_REPO}' '${ACC_HF_FILE}' --repo-type dataset --local-dir /root/acc_data >/dev/null && ls -l /root/acc_data/${ACC_HF_FILE}" \
    || { echo "ERROR: HF download failed (private repo — pod HF_TOKEN must have access)"; exit 1; }
  ACC_DATA="/root/acc_data/${ACC_HF_FILE}"
fi

# capture helper (regression's acc_capture.py + the vLLM/trainer reference logprobs that ship
# INSIDE the .pt: training_logprobs = trainer reference, sampling_logprobs = the original
# (vLLM) sampler — same data lora-dev-script/check_sglang_lora_correctness.py compares against)
cat > /tmp/acc_capture.py <<'PY'
import argparse, torch, requests, json
ap=argparse.ArgumentParser(); ap.add_argument("--port",required=True); ap.add_argument("--data",required=True); ap.add_argument("--lora",default=""); ap.add_argument("--out",required=True); a=ap.parse_args()
data=torch.load(a.data,weights_only=False); toks=data["tokens"]
if torch.is_tensor(toks): toks=toks.tolist()
seqs=toks if (toks and isinstance(toks[0],list)) else [toks]; lp=[]
for s in seqs:
    p={"input_ids":s,"sampling_params":{"max_new_tokens":0,"temperature":0.0},"return_logprob":True,"logprob_start_len":0}
    if a.lora: p["lora_path"]=a.lora
    r=requests.post(f"http://127.0.0.1:{a.port}/generate",json=p,timeout=1800); r.raise_for_status()
    lp+=[x[0] for x in r.json()["meta_info"]["input_token_logprobs"]][1:]   # [1:] skips BOS (no logprob)
def ref(k):
    v=data.get(k); v=v.tolist() if torch.is_tensor(v) else v
    return [float(x) for x in v] if v is not None else None
json.dump({"sglang":lp,"training_logprobs":ref("training_logprobs"),"sampling_logprobs":ref("sampling_logprobs")},open(a.out,"w"))
print("wrote",len(lp),"logprobs ->",a.out)
PY
$KC cp /tmp/acc_capture.py "${HEAD_POD}:/root/acc_capture.py"

for CELL in no-lora lora; do
  echo "---- cell: $CELL ----"
  launch_server "$CELL" || { echo "ERROR: $CELL server failed to launch"; exit 1; }
  LN=""; [ "$CELL" = lora ] && LN="$LORA_NAME"
  D="${OUTROOT}/${CELL}/acc"
  kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; python3 /root/acc_capture.py --port ${PORT} --data '${ACC_DATA}' --lora '${LN}' --out ${D}/logprobs.json 2>&1 | tee ${D}/acc.log" \
    || { echo "ERROR: acc capture failed ($CELL)"; exit 1; }
  coherence_check "$CELL" || exit 1
  pull_dir "$D" "${RUN_DIR}/acc/${CELL}"
done
kill_all

# ---- verify + diff locally (lora vs no-lora + the vLLM/trainer reference comparison) ----
STRICT="${ACC_STRICT:-0}" python3 - "$RUN_DIR/acc" "$ACC_TOL" <<'PY' || exit 1
import json, os, sys, pathlib
root, tol = pathlib.Path(sys.argv[1]), float(sys.argv[2])
def load(c):
    p = root / c / "logprobs.json"
    try: d = json.loads(p.read_text())
    except Exception: print(f"ERROR: missing/unparsable {p}"); sys.exit(1)
    if isinstance(d, list): d = {"sglang": d}          # old flat format
    return d
A, B = load("no-lora"), load("lora")
a, b = A.get("sglang") or [], B.get("sglang") or []
if not a or not b: print("ERROR: empty logprobs"); sys.exit(1)
if len(a) != len(b): print(f"ERROR: length mismatch no-lora={len(a)} lora={len(b)}"); sys.exit(1)
d = sorted(abs(x - y) for x, y in zip(a, b))
n = len(d); mean = sum(d)/n; mx = d[-1]
p = lambda q: d[min(n-1, int(q*n))]
def kl_v2(x, y):                                       # lora-dev-script's metric: 0.5*mean((x-y)^2)
    m = min(len(x), len(y))
    return 0.5*sum((x[i]-y[i])**2 for i in range(m))/m
hmse = kl_v2(a, b)
ok = mx <= tol
verdict = "PASS (≤ tol)" if ok else "EXCEEDS tol — LoRA-path numerical regression OR an intentionally divergent adapter"
L = ["# acc summary — teacher-forced prefill logprobs", "",
     "## sglang: lora vs no-lora (per-token |diff|)", "",
     "| n | mean abs | max abs | p50 | p95 | half-MSE | tol | verdict |",
     "|---:|---:|---:|---:|---:|---:|---:|:--|",
     f"| {n} | {mean:.5f} | {mx:.5f} | {p(.5):.5f} | {p(.95):.5f} | {hmse:.5f} | {tol} | {verdict} |"]
# vLLM/trainer reference comparison (lora-dev-script/check_sglang_lora_correctness.py):
# training_logprobs = trainer reference, sampling_logprobs = original (vLLM) sampler.
tr, sp = B.get("training_logprobs"), B.get("sampling_logprobs")
if tr and sp:
    floor = kl_v2(tr, sp)
    k_tr, k_sp = kl_v2(b, tr), kl_v2(b, sp)
    note = "≈ floor: sglang matches the vLLM-era accuracy" if k_tr <= max(2*floor, 1e-6) else "**above the vLLM noise floor — inspect**"
    if len(tr) != len(b): L += ["", f"⚠ length mismatch ref={len(tr)} vs sglang={len(b)} — KL over the common prefix"]
    L += ["", "## vs vLLM/trainer reference (KL = 0.5·mean((a−b)²), from the .pt)", "",
          "| pair | KL | meaning |", "|---|---:|---|",
          f"| orig_sampler (vLLM) vs trainer | {floor:.6f} | inherent noise floor |",
          f"| sglang-lora vs trainer | {k_tr:.6f} | {note} |",
          f"| sglang-lora vs orig_sampler (vLLM) | {k_sp:.6f} | direct sglang↔vLLM gap |"]
else:
    L += ["", "> No `training_logprobs`/`sampling_logprobs` in the .pt — for the vLLM-reference",
          "> comparison point ACC_HF_FILE at a file from hf.co/datasets/yushengsu/datasets."]
L += ["", "> Prefill-only: decode health is gated separately (coherence check per cell)."]
out = root / "summary.md"; out.write_text("\n".join(L) + "\n")
print("\n".join(L)); print(f"\nwrote {out}")
if not ok and os.environ.get("STRICT") == "1": sys.exit(1)
PY

echo "== [4/acc] PASS — ${RUN_DIR}/acc/summary.md"
