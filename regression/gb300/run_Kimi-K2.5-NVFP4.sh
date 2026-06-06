#!/usr/bin/env bash
# Kimi-K2.5-NVFP4 regression on GB300 (gcp-radixark-02, 2-node MNNVL) — thin wrapper.
# Parameters: models/Kimi-K2.5-NVFP4/model.env (+ hooks.sh); GB300 deltas: models/Kimi-K2.5-NVFP4/MODEL.md.
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../scripts/run_regression.sh" gb300/models/Kimi-K2.5-NVFP4 "$@"
