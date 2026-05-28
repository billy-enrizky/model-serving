#!/usr/bin/env bash
# Idempotent Modal account bootstrap.
#
# Sets the Modal auth token, then ensures the HF_TOKEN secret, MODEL_API_KEY
# secret, and hf-cache volume exist. Re-running is safe.
#
# Required env (read from .env at repo root, or pass on command line):
#   MODAL_TOKEN_ID         Modal API token id (ak-...)
#   MODAL_TOKEN_SECRET     Modal API token secret (as-...)
#   HF_TOKEN               HuggingFace token (gated Gemma access)
#   MODEL_API_KEY          Server bearer key (auto-generated if unset)
#
# Usage:
#   bash deploy/modal/setup_modal.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present (export every var inside).
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"
if [[ ! -x "$MODAL_BIN" ]]; then
  echo "modal not installed in .venv. Run: uv pip install modal" >&2
  exit 1
fi

: "${MODAL_TOKEN_ID:?MODAL_TOKEN_ID required}"
: "${MODAL_TOKEN_SECRET:?MODAL_TOKEN_SECRET required}"
: "${HF_TOKEN:?HF_TOKEN required}"

if [[ -z "${MODEL_API_KEY:-}" ]]; then
  MODEL_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
  echo "Generated MODEL_API_KEY=$MODEL_API_KEY"
fi

echo "==> modal token set"
"$MODAL_BIN" token set \
  --token-id "$MODAL_TOKEN_ID" \
  --token-secret "$MODAL_TOKEN_SECRET" >/dev/null

echo "==> ensuring secret hf-token"
"$MODAL_BIN" secret create hf-token \
  "HF_TOKEN=$HF_TOKEN" --force >/dev/null 2>&1 \
  || "$MODAL_BIN" secret create hf-token "HF_TOKEN=$HF_TOKEN" >/dev/null 2>&1 \
  || echo "  (already exists, leaving as is)"

echo "==> ensuring secret mtp-api-key"
"$MODAL_BIN" secret create mtp-api-key \
  "MODEL_API_KEY=$MODEL_API_KEY" --force >/dev/null 2>&1 \
  || "$MODAL_BIN" secret create mtp-api-key "MODEL_API_KEY=$MODEL_API_KEY" >/dev/null 2>&1 \
  || echo "  (already exists, leaving as is)"

echo "==> ensuring volume hf-cache"
"$MODAL_BIN" volume create hf-cache >/dev/null 2>&1 || echo "  (already exists)"

# Persist generated key so the smoke test can read it.
mkdir -p "$REPO_ROOT/deploy/modal/.state"
printf '%s' "$MODEL_API_KEY" > "$REPO_ROOT/deploy/modal/.state/api_key"
chmod 600 "$REPO_ROOT/deploy/modal/.state/api_key"

echo "Modal bootstrap complete."
echo "API key written to deploy/modal/.state/api_key"
