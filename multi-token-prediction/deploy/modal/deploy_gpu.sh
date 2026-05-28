#!/usr/bin/env bash
# Deploy mtp-gemma-server targeting a specific GPU + MTP setting.
#
# Usage:
#   bash deploy/modal/deploy_gpu.sh <gpu> <num_assistant_tokens>
#
# Examples:
#   bash deploy/modal/deploy_gpu.sh H100      4
#   bash deploy/modal/deploy_gpu.sh A100-80GB 0
#   bash deploy/modal/deploy_gpu.sh B200      4
#
# Writes the deployed URL to deploy/modal/.state/url_<gpu_lower>.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${1:?usage: deploy_gpu.sh <gpu> <num_assistant_tokens>}"
NUM="${2:?usage: deploy_gpu.sh <gpu> <num_assistant_tokens>}"

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"
STATE_DIR="$REPO_ROOT/deploy/modal/.state"
mkdir -p "$STATE_DIR"

GPU_LOWER="$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"
LOG_FILE="$STATE_DIR/deploy_${GPU_LOWER}.log"
URL_FILE="$STATE_DIR/url_${GPU_LOWER}"

echo "==> deploy gpu=$GPU NUM_ASSISTANT_TOKENS=$NUM"
MTP_GPU="$GPU" MTP_NUM_ASSISTANT="$NUM" \
  COLUMNS=200 \
  "$MODAL_BIN" deploy deploy/modal/modal_app.py 2>&1 | tee "$LOG_FILE"

# Modal wraps URLs at terminal width. Strip whitespace and join continuation
# lines that begin with a leading-space to recover the full URL.
URL="$(tr -d '\r' < "$LOG_FILE" \
  | awk '/^[[:space:]]*│[[:space:]]+https:\/\// { sub(/^[[:space:]]*│[[:space:]]+/,""); printf "%s",$0; next }
         /^[[:space:]]*│[[:space:]]+l\.run/    { sub(/^[[:space:]]*│[[:space:]]+/,""); print $0; next }
         { print "" }' \
  | grep -Eo 'https://[a-z0-9.-]+\.modal\.run' | head -n1 || true)"
if [[ -z "$URL" ]]; then
  # Fallback: strip every whitespace + box-drawing in entire log, re-grep
  URL="$(tr -d ' \n\r│├└─' < "$LOG_FILE" | grep -Eo 'https://[a-z0-9.-]+\.modal\.run' | head -n1 || true)"
fi
if [[ -z "$URL" ]]; then
  echo "Could not extract URL from $LOG_FILE" >&2
  exit 1
fi
printf '%s' "$URL" > "$URL_FILE"
# Canonical .state/url is intentionally NOT updated here. When multiple GPUs
# are deployed in parallel (e.g. via dispatching-parallel-agents), the per-GPU
# URL file is the source of truth. Use `MTP_GPU=<gpu> bash deploy/modal/refresh_canonical_url.sh`
# (or run_ab.sh, which passes BENCH_URL explicitly) to avoid races.

echo
echo "Deployed: $URL"
echo "URL written to $URL_FILE"
