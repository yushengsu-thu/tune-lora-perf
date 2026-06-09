#!/usr/bin/env bash
# 5. Profile LoRA vs no-LoRA (torch profiler, CPU+GPU) under BOTH cuda-graph ON and OFF.
#      graph-ON  -> PROF_RECIPE + TRACE_RANKS     : REAL timing (all ranks you actually read).
#      graph-OFF -> PROF_OFF    + TRACE_RANKS_OFF : kernel STRUCTURE for source attribution
#                                                   (TP0 suffices; per-rank trace ~10x bigger).
#    Mirrors regression/scripts/run_regression.sh's two-trace profile, so the pair feeds
#    analyze_llm_torch_profile.py --mapping-input graph_off --formal-input graph_on directly.
#    Input : model name (dir under dev/models/ or unique prefix); pods from step 1
#            running the code from step 2.
#    Output: $RUN_DIR/  (= dev/results/<model>/<DATE>-<TIME>/, shared with step 3)
#              ├── no-lora/graph_{on,off}/  bs<bs>-TP-<r>.trace.json.gz  (+ bench.log per mode)
#              └── lora/   graph_{on,off}/  bs<bs>-TP-<r>.trace.json.gz
#    Verify: every expected per-rank trace exists locally and passes `gzip -t`.
. "$(dirname "$0")/common.sh" "${1:-}"
load_state; ensure_run_dir
OUTROOT=/tmp/dev_run
# graph-OFF recipe/ranks are optional in model.env; default to a light TP0-only pass (graph-off is
# slow + traces are ~10x bigger, and structure attribution needs a small bs / few decode steps —
# same recipe regression uses). Override PROF_OFF/TRACE_RANKS_OFF per model if needed.
PROF_OFF="${PROF_OFF:-16 4 12 64}"        # bs start-step steps output-len (kernel structure only)
TRACE_RANKS_OFF="${TRACE_RANKS_OFF:-0}"   # TP0 suffices for structure

# recipe + ranks for a graph mode -> sets P_BS P_START P_STEPS P_OUTLEN RANKS
prof_mode(){  # $1=on|off
  if [ "$1" = on ]; then read -r P_BS P_START P_STEPS P_OUTLEN <<< "$PROF_RECIPE"; RANKS="$TRACE_RANKS"
  else                   read -r P_BS P_START P_STEPS P_OUTLEN <<< "$PROF_OFF";    RANKS="$TRACE_RANKS_OFF"; fi
}

FAIL=0
for CELL in no-lora lora; do
  echo "==== cell: $CELL ===="
  for G in on off; do
    prof_mode "$G"
    echo "---- $CELL graph-$G  bs=${P_BS} start=${P_START} steps=${P_STEPS} outlen=${P_OUTLEN}  ranks='${RANKS}' ----"
    # graph-OFF is supplementary (kernel structure): a failed graph-OFF launch skips ONLY that pass,
    # like regression's run_cell — it must not lose the graph-ON traces or block the other cell.
    if ! launch_server "$CELL" "$G"; then
      if [ "$G" = on ]; then echo "  [$CELL] graph-ON launch FAILED — skipping cell"; FAIL=1; continue 2
      else echo "  [$CELL] graph-OFF launch FAILED — graph-off profile skipped"; continue; fi
    fi
    LA=""; [ "$CELL" = lora ] && LA="--lora-name ${LORA_NAME}"
    D="${OUTROOT}/${CELL}/graph_${G}/profile"
    kh "rm -rf ${D}; mkdir -p ${D}; cd /root/sglang; \
        python -m sglang.bench_one_batch_server --model-path None --base-url http://127.0.0.1:${PORT} \
          --batch-size ${P_BS} --input-len ${BENCH_IN} --output-len ${P_OUTLEN} ${LA} \
          --profile --profile-activities CPU GPU --profile-start-step ${P_START} --profile-steps ${P_STEPS} \
          --profile-prefix ${MODEL}_${CELL}_graph_${G}_bs${P_BS} --profile-output-dir ${D} \
          --result-filename ${D}/bench.jsonl 2>&1 | tee ${D}/bench.log" \
      || { echo "ERROR: profile bench failed ($CELL graph-$G)"; [ "$G" = on ] && exit 1; echo "  graph-off best-effort — skipping"; continue; }
    coherence_check "$CELL" || { [ "$G" = on ] && exit 1; echo "  [$CELL] graph-off coherence flagged — continuing"; }
    OUT="${RUN_DIR}/${CELL}/graph_${G}"; mkdir -p "$OUT"
    for R in $RANKS; do
      if pull_trace "$R" "$D" "${OUT}/bs${P_BS}-TP-${R}.trace.json.gz"; then
        S=$(stat -f%z "${OUT}/bs${P_BS}-TP-${R}.trace.json.gz" 2>/dev/null || echo 0)
        echo "  ${CELL}/graph_${G}/bs${P_BS}-TP-${R}  $((S/1024/1024))M OK"
      else
        # graph-ON is the trace you actually read (hard fail); graph-OFF is best-effort (warn).
        echo "  MISSING ${CELL} graph-${G} TP${R}"; [ "$G" = on ] && FAIL=1
      fi
    done
    kh "cat ${D}/bench.log" > "${OUT}/bench.log" 2>/dev/null || true
  done
done
kill_all

# ---- verify: every expected trace present + gzip-valid ----
for CELL in no-lora lora; do
  for G in on off; do
    prof_mode "$G"
    for R in $RANKS; do
      F="${RUN_DIR}/${CELL}/graph_${G}/bs${P_BS}-TP-${R}.trace.json.gz"
      # graph-ON must be present+valid; graph-OFF is best-effort (skipped/truncated -> warn only).
      [ -s "$F" ] && gzip -t "$F" 2>/dev/null || { echo "$([ "$G" = on ] && echo ERROR || echo WARN): bad/missing $F"; [ "$G" = on ] && FAIL=1; }
    done
  done
done
[ "$FAIL" = 0 ] || exit 1
echo "== [5/profile] PASS — traces in ${RUN_DIR}/<cell>/graph_{on,off}/"
