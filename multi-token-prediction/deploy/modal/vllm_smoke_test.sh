#!/usr/bin/env bash
# Hit /v1/models and /v1/chat/completions on a deployed vLLM endpoint.
# Usage: MTP_GPU=H100 VLLM_MODE=mtp bash vllm_smoke_test.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/deploy/modal"

: "${MTP_GPU:?Set MTP_GPU}"
: "${VLLM_MODE:?Set VLLM_MODE (mtp|baseline)}"

GPU_LOWER="$(echo "$MTP_GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"
URL="$(cat "$REPO_ROOT/deploy/modal/.state/url_vllm_${GPU_LOWER}_${VLLM_MODE}")"
KEY="${MODEL_API_KEY:-$(cat "$REPO_ROOT/deploy/modal/.state/api_key" 2>/dev/null || true)}"
[ -z "${KEY:-}" ] && { echo "Set MODEL_API_KEY or write .state/api_key" >&2; exit 1; }

echo "[smoke] /v1/models"
curl -fsSL --max-time 1200 -H "Authorization: Bearer $KEY" "$URL/v1/models" | head -c 500; echo

echo "[smoke] /v1/chat/completions (3 retries for cold start)"
for i in 1 2 3; do
  if curl -fsSL --max-time 1200 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
    -X POST "$URL/v1/chat/completions" -d '{
      "model":"gemma-4-E2B-it",
      "messages":[{"role":"user","content":"What is 2+2?"}],
      "max_tokens":32, "temperature":0.0
    }' | head -c 500; then
    echo; break
  fi
  echo "[smoke] retry $i"; sleep 5
done
