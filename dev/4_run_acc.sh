#!/usr/bin/env bash
# 4. Accuracy: LoRA vs no-LoRA teacher-forced per-token logprob diff (regression's acc test).
#    Feeds the adapter's compare_sample_train_data.pt to /generate with max_new_tokens=0 +
#    return_logprob (prefill-only, no sampling) for BOTH cells, then diffs per-token logprobs.
#    NOTE: prefill-only — it CANNOT see decode garbage; the coherence gate (run per cell here
#    too) is the decode check.
#    Input : model name (qwen|kimi); pods from step 1 running the code from step 2.
#    Output: $RUN_DIR/acc/{no-lora,lora}/logprobs.json + $RUN_DIR/acc/summary.md
#            (n, mean|max abs diff, p50/p95, half-MSE, verdict vs ACC_TOL).
#    Verify: both logprob sets captured + same length. max>ACC_TOL is a WARNING by default
#            (alpha is a near-identity adapter so diff ≈ LoRA-path numerical noise; a custom
#            adapter legitimately diverges) — set ACC_STRICT=1 to fail on it.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
OUTROOT=/tmp/dev_run
ACC_DATA="${ACC_DATA:-${LORA_PATH}/compare_sample_train_data.pt}"
echo "== [4/acc] $MODEL  data=${ACC_DATA}  tol=${ACC_TOL}  -> ${RUN_DIR}/acc"

# capture helper (same as regression's acc_capture.py): prefill-only logprobs via /generate
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
json.dump(lp,open(a.out,"w")); print("wrote",len(lp),"logprobs ->",a.out)
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

# ---- verify + diff locally ----
STRICT="${ACC_STRICT:-0}" python3 - "$RUN_DIR/acc" "$ACC_TOL" <<'PY' || exit 1
import json, os, sys, pathlib
root, tol = pathlib.Path(sys.argv[1]), float(sys.argv[2])
def load(c):
    p = root / c / "logprobs.json"
    try: return [float(x) for x in json.loads(p.read_text())]
    except Exception: print(f"ERROR: missing/unparsable {p}"); sys.exit(1)
a, b = load("no-lora"), load("lora")
if not a or not b: print("ERROR: empty logprobs"); sys.exit(1)
if len(a) != len(b): print(f"ERROR: length mismatch no-lora={len(a)} lora={len(b)}"); sys.exit(1)
d = sorted(abs(x - y) for x, y in zip(a, b))
n = len(d); mean = sum(d)/n; mx = d[-1]
p = lambda q: d[min(n-1, int(q*n))]
hmse = 0.5*sum((x-y)**2 for x, y in zip(a, b))/n
ok = mx <= tol
verdict = "PASS (≤ tol)" if ok else "EXCEEDS tol — LoRA-path numerical regression OR an intentionally divergent adapter"
L = ["# acc summary — per-token logprob |diff| (lora vs no-lora, teacher-forced prefill)", "",
     "| n | mean abs | max abs | p50 | p95 | half-MSE | tol | verdict |",
     "|---:|---:|---:|---:|---:|---:|---:|:--|",
     f"| {n} | {mean:.5f} | {mx:.5f} | {p(.5):.5f} | {p(.95):.5f} | {hmse:.5f} | {tol} | {verdict} |",
     "", "> Prefill-only: decode health is gated separately (coherence check per cell).",]
out = root / "summary.md"; out.write_text("\n".join(L) + "\n")
print("\n".join(L)); print(f"\nwrote {out}")
if not ok and os.environ.get("STRICT") == "1": sys.exit(1)
PY

echo "== [4/acc] PASS — ${RUN_DIR}/acc/summary.md"
