#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8 regression on GB200 (leira) — thin wrapper over the generic driver.
# Parameters: models/qwen35/model.env (+ hooks.sh); the why: models/qwen35/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/run_regression.sh" gb200/models/qwen35 "$@"
