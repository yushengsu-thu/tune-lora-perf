#!/usr/bin/env bash
# Kimi-K2.5-NVFP4 regression on GB200 (leira) — thin wrapper over the generic driver.
# Parameters: models/kimi/model.env (+ hooks.sh); the why: models/kimi/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/run_regression.sh" gb200/models/kimi "$@"
