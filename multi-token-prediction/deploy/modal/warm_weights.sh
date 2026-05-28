#!/usr/bin/env bash
# Pre-download Gemma target + drafter into the hf-cache Modal volume.
# Idempotent: HF snapshot_download is content-addressed.
#
# Usage:
#   bash deploy/modal/warm_weights.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"

echo "==> warming hf-cache volume (target + drafter)"
"$MODAL_BIN" run deploy/modal/modal_app.py::warm_weights
