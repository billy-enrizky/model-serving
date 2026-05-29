#!/usr/bin/env bash
# Full A/B sweep on a single GPU: deploy N=4, warm, bench MTP, deploy N=0,
# warm, bench baseline, restore N=4. Reads URL from per-GPU state file so
# parallel agents on different GPUs do not race on the canonical .state/url.
#
# Usage:
#   bash deploy/modal/run_ab.sh <gpu> [requests] [concurrency] [max_tokens]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${1:?usage: run_ab.sh <gpu> [requests] [concurrency] [max_tokens]}"
REQ="${2:-16}"
CON="${3:-1}"
MAX="${4:-128}"

GPU_LOWER="$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"
KEY="$(cat "$REPO_ROOT/deploy/modal/.state/api_key")"

PROMPT_SET="${PROMPT_SET:-generic}"
LABEL_SUFFIX=""
[ "$PROMPT_SET" != "generic" ] && LABEL_SUFFIX="_${PROMPT_SET}"
MODES="${MODES:-mtp baseline}"
export PROMPT_SET

warm() {
  local url="$1"
  echo "==> warming $url"
  # -L follows Modal's 303 cold-start redirect. Without it, slow cold starts
  # return an empty 303 body and look like a failure.
  for i in 1 2 3 4 5; do
    body="$(curl -sSL --max-time 1200 \
      -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
      -d '{"model":"gemma-4-E2B-it","messages":[{"role":"user","content":"warm"}],"max_tokens":4,"temperature":0.0}' \
      "$url/v1/chat/completions" || true)"
    if [[ "$body" == \{* ]]; then
      echo "  warm ok (attempt $i): $(echo "$body" | head -c 120)"
      return 0
    fi
    echo "  attempt $i empty/failed, retrying after 15s"
    sleep 15
  done
  echo "  warm failed after 5 attempts" >&2
  return 1
}

run_one() {
  local LABEL="$1"
  local N="$2"
  bash deploy/modal/deploy_gpu.sh "$GPU" "$N"
  local URL
  URL="$(cat "$REPO_ROOT/deploy/modal/.state/url_${GPU_LOWER}")"
  warm "$URL"
  BENCH_URL="$URL" MTP_GPU="$GPU" PROMPT_SET="$PROMPT_SET" \
    bash deploy/modal/run_bench.sh "$LABEL" "$REQ" "$CON" "$MAX"
}

for MODE in $MODES; do
  case "$MODE" in
    mtp)      run_one "mtp_n4_${GPU_LOWER}${LABEL_SUFFIX}_c${CON}"      4 ;;
    baseline) run_one "baseline_n0_${GPU_LOWER}${LABEL_SUFFIX}_c${CON}" 0 ;;
    *) echo "Unknown MODE='$MODE' (expect mtp|baseline)" >&2; exit 1 ;;
  esac
done

# Restore N=4 deploy so the app stays in MTP-on state for next session,
# but only if we touched the app (mtp or baseline).
if [[ " $MODES " == *" baseline "* ]] || [[ " $MODES " == *" mtp "* ]]; then
  bash deploy/modal/deploy_gpu.sh "$GPU" 4
fi

echo
echo "==> A/B done for $GPU (modes=$MODES, prompt_set=$PROMPT_SET)."
for MODE in $MODES; do
  case "$MODE" in
    mtp)      echo "    metrics/runs/<ts>_mtp_n4_${GPU_LOWER}${LABEL_SUFFIX}_c${CON}" ;;
    baseline) echo "    metrics/runs/<ts>_baseline_n0_${GPU_LOWER}${LABEL_SUFFIX}_c${CON}" ;;
  esac
done
