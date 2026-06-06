#!/usr/bin/env bash
# Qwen3.5-35B-A3B-FP8 regression entry point — thin wrapper over the generic driver.
# All qwen35 parameters: models/qwen35/model.env (+ hooks.sh); the why: models/qwen35/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/run_regression.sh" qwen35 "$@"
