#!/usr/bin/env bash
# Hit /healthz and /v1/chat/completions on the deployed Modal endpoint.
# Reads URL from deploy/modal/.state/url and API key from deploy/modal/.state/api_key.
#
# Usage:
#   bash deploy/modal/smoke_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$REPO_ROOT/deploy/modal/.state"

URL="$(cat "$STATE_DIR/url")"
KEY="$(cat "$STATE_DIR/api_key")"

echo "==> URL: $URL"

pretty() {
  python3 -c 'import json,sys; print(json.dumps(json.loads(sys.stdin.read()), indent=2))' \
    || { echo "(non-JSON response)"; }
}

echo "==> GET /healthz"
curl -sS --max-time 600 "$URL/healthz" | pretty

echo
echo "==> GET /v1/models"
curl -sS --max-time 600 -H "X-API-Key: $KEY" "$URL/v1/models" | pretty

echo
echo "==> POST /v1/chat/completions (non-stream, 32 tokens)"
curl -sS --max-time 600 \
  -H "X-API-Key: $KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "model": "gemma-4-E2B-it",
        "messages": [{"role": "user", "content": "In one sentence, why is the sky blue?"}],
        "max_tokens": 32,
        "temperature": 0.0
      }' \
  "$URL/v1/chat/completions" | pretty
