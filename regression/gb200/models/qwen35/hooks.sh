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

# hook_post_checkout — called after every checkout's `pip install -e python`.
#
# The branch's pyproject pins flashinfer_python==0.6.12 ("aligned with jit-cache in Dockerfile"),
# but PR #27329's trtllm_lora_temp JIT kernels only COMPILE against the 0.6.11 trtllm headers
# (0.6.12 changed get_sf_out_offset_* signatures — the PR's own CI Extra is red on this), and the
# pinned pod image ships flashinfer-jit-cache 0.6.11.post1+cu130. So after each editable install
# re-pin the python side back to the image-matching version (FLASHINFER_PIN in model.env).
# Remove this hook (and the image digest pin in pod.yaml) when the PR rebases onto 0.6.12.
hook_post_checkout(){
  local P
  [ -n "${FLASHINFER_PIN:-}" ] || return 0
  for P in "${PODS[@]}"; do
    kp "$P" "pip install -q \"flashinfer_python[cu13]==${FLASHINFER_PIN}\" \"flashinfer_cubin==${FLASHINFER_PIN}\" 2>&1 | tail -1; python3 -c 'import flashinfer; print(\"  [hook] flashinfer\", flashinfer.__version__)'"
  done
}
