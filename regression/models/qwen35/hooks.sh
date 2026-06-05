# Qwen3.5-specific LOGIC for run_regression.sh (sourced after model.env; functions only —
# they run with the driver's variables in scope: PODS, kh/kp, LOCAL_OUT, MODEL_PATH, ...).
# Only define the hooks this model needs; the driver calls them ONLY if defined.

# hook_post_setup — called once after prewarm, before the first cell.
#
# Qwen3.5's layer count is NOT hardcoded (model.env LAYERS="") — read num_hidden_layers from the
# model's config.json and record it into meta.env so summary.py can compute the per-layer metric.
# NOTE: qwen3.5's config.json nests it under text_config (multimodal-style layout) — check both.
hook_post_setup(){
  local n
  n=$(kh "python3 -c \"import json; c=json.load(open('${MODEL_PATH}/config.json')); print(c.get('num_hidden_layers') or c.get('text_config',{}).get('num_hidden_layers') or '')\"" 2>/dev/null | tr -d '[:space:]')
  if [ -n "${n:-}" ] && ! grep -q '^layers=' "${LOCAL_OUT}/meta.env" 2>/dev/null; then
    echo "layers=${n}" >> "${LOCAL_OUT}/meta.env"; echo "  [hook] layers=${n}"
  fi
}
