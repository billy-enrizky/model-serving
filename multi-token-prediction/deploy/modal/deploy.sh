#!/usr/bin/env bash
# Deploy the MTP Gemma server to Modal (H100, scale-to-zero).
# Captures the deployed web URL into deploy/modal/.state/url.
#
# Usage:
#   bash deploy/modal/deploy.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"
STATE_DIR="$REPO_ROOT/deploy/modal/.state"
mkdir -p "$STATE_DIR"

LOG_FILE="$STATE_DIR/deploy.log"

echo "==> modal deploy deploy/modal/modal_app.py"
COLUMNS=200 "$MODAL_BIN" deploy deploy/modal/modal_app.py 2>&1 | tee "$LOG_FILE"

# Modal CLI wraps URLs at terminal width. Strip whitespace + box-drawing
# characters from the entire log before regrepping.
URL="$(tr -d ' \n\r│├└─' < "$LOG_FILE" | grep -Eo 'https://[a-z0-9.-]+\.modal\.run' | head -n1 || true)"
if [[ -z "$URL" ]]; then
  echo "Could not extract URL from deploy output. Check $LOG_FILE." >&2
  exit 1
fi
printf '%s' "$URL" > "$STATE_DIR/url"
echo
echo "Deployed: $URL"
echo "URL written to $STATE_DIR/url"
