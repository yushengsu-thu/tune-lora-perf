#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8 regression on GB300 (gcp-radixark-02) — thin wrapper over the generic driver.
# All parameters: models/qwen35_gb300/model.env (+ hooks.sh); deltas vs GB200: models/qwen35_gb300/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/run_regression.sh" qwen35_gb300 "$@"
