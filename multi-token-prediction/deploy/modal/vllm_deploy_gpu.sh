#!/usr/bin/env bash
# Deploy vLLM Modal app for a given GPU + mode.
# Usage: MTP_GPU=H100 VLLM_MODE=mtp bash vllm_deploy_gpu.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/deploy/modal"

: "${MTP_GPU:?Set MTP_GPU (e.g. H100)}"
: "${VLLM_MODE:?Set VLLM_MODE (mtp|baseline)}"
: "${VLLM_VERSION:=latest}"

export MTP_GPU VLLM_MODE VLLM_VERSION

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"
STATE_DIR="$REPO_ROOT/deploy/modal/.state"
mkdir -p "$STATE_DIR"

GPU_LOWER="$(echo "$MTP_GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"
LOG="$STATE_DIR/deploy_vllm_${GPU_LOWER}_${VLLM_MODE}.log"

echo "[deploy] vLLM ${VLLM_VERSION} on ${MTP_GPU} mode=${VLLM_MODE}"
COLUMNS=200 "$MODAL_BIN" deploy vllm_modal_app.py 2>&1 | tee "$LOG"

# Modal wraps URLs at terminal width. Recover the full URL like deploy_gpu.sh does.
URL="$(tr -d '\r' < "$LOG" \
  | awk '/^[[:space:]]*│[[:space:]]+https:\/\// { sub(/^[[:space:]]*│[[:space:]]+/,""); printf "%s",$0; next }
         /^[[:space:]]*│[[:space:]]+l\.run/    { sub(/^[[:space:]]*│[[:space:]]+/,""); print $0; next }
         { print "" }' \
  | grep -Eo 'https://[a-z0-9.-]+\.modal\.run' | head -n1 || true)"
if [[ -z "$URL" ]]; then
  URL="$(tr -d ' \n\r│├└─' < "$LOG" | grep -Eo 'https://[a-z0-9.-]+\.modal\.run' | head -n1 || true)"
fi
if [[ -z "$URL" ]]; then
  echo "[deploy] failed to capture URL from $LOG" >&2
  exit 1
fi
URL_FILE="$STATE_DIR/url_vllm_${GPU_LOWER}_${VLLM_MODE}"
printf '%s' "$URL" > "$URL_FILE"
echo "[deploy] URL: $URL"
echo "[deploy] URL written to $URL_FILE"
