#!/usr/bin/env bash
# Full A/B for vLLM on a given GPU: deploy MTP, bench, deploy baseline, bench.
# Bench is executed inside a Modal GPU container (so NVML reads the right
# specs) hitting the deployed vLLM HTTPS endpoint via bearer auth.
# Usage: bash vllm_run_ab.sh <GPU>
# Example: bash vllm_run_ab.sh H100
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT/deploy/modal"

MTP_GPU="${1:?Pass GPU as arg, e.g. H100}"
export MTP_GPU
GPU_LOWER="$(echo "$MTP_GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"

KEY="${MODEL_API_KEY:-$(cat "$REPO_ROOT/deploy/modal/.state/api_key" 2>/dev/null || true)}"
[ -z "${KEY:-}" ] && { echo "Set MODEL_API_KEY or write .state/api_key" >&2; exit 1; }
export MODEL_API_KEY="$KEY"

REQ=16
CON=1
MAX=128
PROMPT_SET="${PROMPT_SET:-generic}"
LABEL_SUFFIX=""
[ "$PROMPT_SET" != "generic" ] && LABEL_SUFFIX="_${PROMPT_SET}"

run_mode() {
  local MODE="$1"
  local LABEL="vllm_${MODE}_${GPU_LOWER}${LABEL_SUFFIX}_c1"
  echo "[ab] ===== mode=${MODE} label=${LABEL} ====="

  # Stop prior app to flush warm pool (Modal warm-container quirk).
  "$MODAL_BIN" app stop -y "vllm-gemma-${GPU_LOWER}-${MODE}" 2>/dev/null || true

  VLLM_MODE="$MODE" bash vllm_deploy_gpu.sh
  local URL
  URL="$(cat "$REPO_ROOT/deploy/modal/.state/url_vllm_${GPU_LOWER}_${MODE}")"

  # Warm: 5 retries (vLLM cold start can be 60-90s).
  for i in 1 2 3 4 5 6 7 8; do
    echo "[warm] try $i $URL"
    if curl -fsSL --max-time 1200 -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
      -X POST "$URL/v1/chat/completions" -d '{
        "model":"gemma-4-E2B-it",
        "messages":[{"role":"user","content":"warm"}],
        "max_tokens":4, "temperature":0.0
      }' >/dev/null; then break; fi
    sleep 20
  done

  # Run bench inside Modal GPU container so NVML reports correct GPU specs.
  cd "$REPO_ROOT"
  MTP_GPU="$MTP_GPU" "$MODAL_BIN" run deploy/modal/modal_app.py::bench_run \
    --base-url "$URL" \
    --label "$LABEL" \
    --requests "$REQ" \
    --concurrency "$CON" \
    --max-tokens "$MAX" \
    --prompt-set "$PROMPT_SET" \
    --auth-mode bearer

  # Pull results from mtp-bench-results volume.
  mkdir -p "$REPO_ROOT/metrics/runs"
  local LATEST_REMOTE
  LATEST_REMOTE="$("$MODAL_BIN" volume ls mtp-bench-results 2>/dev/null \
    | grep -E "^[0-9]{8}T[0-9]{6}_${LABEL}$" | tail -n1)"
  if [[ -z "$LATEST_REMOTE" ]]; then
    echo "[ab] No remote dir matching label '$LABEL' found in volume." >&2
    return 1
  fi
  local DEST="$REPO_ROOT/metrics/runs/$LATEST_REMOTE"
  rm -rf "$DEST"
  mkdir -p "$DEST"
  local TMP_PARENT
  TMP_PARENT="$(mktemp -d)"
  "$MODAL_BIN" volume get --force mtp-bench-results "$LATEST_REMOTE/" "$TMP_PARENT/"
  mv "$TMP_PARENT/$LATEST_REMOTE"/* "$DEST/"
  rm -rf "$TMP_PARENT"
  echo "[ab] local: $DEST"

  # Capture vLLM Prometheus metrics for spec-decode acceptance.
  curl -fsSL --max-time 30 -H "Authorization: Bearer $KEY" "$URL/metrics" \
    > "$DEST/vllm_metrics.prom" 2>/dev/null || echo "[ab] /metrics scrape failed (non-fatal)"

  cd "$REPO_ROOT/deploy/modal"
}

VLLM_MODES="${VLLM_MODES:-mtp baseline}"
for MODE in $VLLM_MODES; do
  run_mode "$MODE"
done
echo "[ab] vLLM A/B on $MTP_GPU done (modes=$VLLM_MODES)."
