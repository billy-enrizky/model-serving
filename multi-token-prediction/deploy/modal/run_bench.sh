#!/usr/bin/env bash
# Run bench/load_runner against a deployed Modal endpoint.
# Bench executes on a Modal GPU matching the server (so NVML reads the
# correct sm_/HBM specs) and writes results into the mtp-bench-results
# Modal volume, then pulls them locally.
#
# Usage:
#   bash deploy/modal/run_bench.sh <label> [requests] [concurrency] [max_tokens]
#
# Env overrides (set when running parallel agents per GPU to avoid races):
#   MTP_GPU       GPU string for the bench container (must match server's GPU).
#                 Default: H100. Examples: A100-80GB, B200, A10.
#   BENCH_URL     Server URL to hit. Default: contents of .state/url.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"
STATE_DIR="$REPO_ROOT/deploy/modal/.state"

URL="${BENCH_URL:-$(cat "$STATE_DIR/url")}"
GPU="${MTP_GPU:-H100}"

LABEL="${1:?usage: run_bench.sh <label> [requests] [concurrency] [max_tokens]}"
REQUESTS="${2:-16}"
CONCURRENCY="${3:-1}"
MAX_TOKENS="${4:-128}"

echo "==> bench label=$LABEL gpu=$GPU requests=$REQUESTS concurrency=$CONCURRENCY max_tokens=$MAX_TOKENS"
echo "==> against $URL"

MTP_GPU="$GPU" "$MODAL_BIN" run deploy/modal/modal_app.py::bench_run \
  --base-url "$URL" \
  --label "$LABEL" \
  --requests "$REQUESTS" \
  --concurrency "$CONCURRENCY" \
  --max-tokens "$MAX_TOKENS"

echo
echo "==> pulling results from mtp-bench-results volume"
mkdir -p "$REPO_ROOT/metrics/runs"
LATEST_REMOTE="$("$MODAL_BIN" volume ls mtp-bench-results 2>/dev/null \
  | grep -E "^[0-9]{8}T[0-9]{6}_${LABEL}$" | tail -n1)"
if [[ -z "$LATEST_REMOTE" ]]; then
  echo "No remote dir matching label '$LABEL' found in volume." >&2
  exit 1
fi
DEST="$REPO_ROOT/metrics/runs/$LATEST_REMOTE"
rm -rf "$DEST"
mkdir -p "$DEST"
TMP_PARENT="$(mktemp -d)"
"$MODAL_BIN" volume get --force mtp-bench-results "$LATEST_REMOTE/" "$TMP_PARENT/"
mv "$TMP_PARENT/$LATEST_REMOTE"/* "$DEST/"
rm -rf "$TMP_PARENT"
echo "==> local: $DEST"
