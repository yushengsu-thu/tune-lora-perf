#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8 regression on GB300 (gcp-radixark-02) — thin wrapper over the generic driver.
# Parameters: models/qwen35/model.env (+ hooks.sh); GB300 deltas: models/qwen35/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/run_regression.sh" gb300/models/qwen35 "$@"
