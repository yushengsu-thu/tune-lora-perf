#!/usr/bin/env bash
# Kimi-K2.5-NVFP4 regression entry point — thin wrapper over the generic driver.
# All kimi parameters: models/kimi/model.env (+ hooks.sh); the why: models/kimi/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/scripts/run_regression.sh" kimi "$@"
