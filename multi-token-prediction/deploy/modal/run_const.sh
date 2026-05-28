#!/usr/bin/env bash
# Run transformers MTP with schedule=constant N=4 on a given GPU.
# Usage: bash run_const.sh <GPU>
# Example: bash run_const.sh H100
#
# Writes a separate Modal app (mtp-gemma-server[-<gpu>]-const) so the
# heuristic warm pool is untouched. Bench label: tx_mtp_const_<gpu>_c1.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

GPU="${1:?usage: run_const.sh <GPU>}"
GPU_LOWER="$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"
LABEL="tx_mtp_const_${GPU_LOWER}_c1"

MODAL_BIN="$REPO_ROOT/.venv/bin/modal"
KEY="${MODEL_API_KEY:-$(cat "$REPO_ROOT/deploy/modal/.state/api_key" 2>/dev/null || true)}"
[ -z "${KEY:-}" ] && { echo "Set MODEL_API_KEY or write .state/api_key" >&2; exit 1; }
export MODEL_API_KEY="$KEY"

# Stop both possible app pools so deploy starts cold and env is fresh.
APP_HEURISTIC="mtp-gemma-server"
[ "$GPU" != "H100" ] && APP_HEURISTIC="${APP_HEURISTIC}-${GPU_LOWER}"
APP_CONST="${APP_HEURISTIC}-const"
"$MODAL_BIN" app stop -y "$APP_HEURISTIC" 2>/dev/null || true
"$MODAL_BIN" app stop -y "$APP_CONST" 2>/dev/null || true

# Deploy with schedule=constant N=4. modal_app.py adds -const to app name when
# MTP_SCHEDULE=constant so the heuristic warm pool is not reused.
MTP_SCHEDULE=constant bash deploy/modal/deploy_gpu.sh "$GPU" 4
URL="$(cat "$REPO_ROOT/deploy/modal/.state/url_${GPU_LOWER}")"

# Warm: 5 retries.
for i in 1 2 3 4 5; do
  if curl -fsSL --max-time 1200 -H "X-API-Key: $KEY" -H "Content-Type: application/json" \
    -X POST "$URL/v1/chat/completions" -d '{
      "model":"gemma-4-E2B-it",
      "messages":[{"role":"user","content":"warm"}],
      "max_tokens":4, "temperature":0.0
    }' >/dev/null; then echo "[warm] ok attempt $i"; break; fi
  echo "[warm] retry $i"
  sleep 15
done

# Run bench inside Modal GPU container (NVML probing).
BENCH_URL="$URL" MTP_GPU="$GPU" \
  bash deploy/modal/run_bench.sh "$LABEL" 16 1 128

# Verify constant schedule actually ran (proposed_tokens should be steady).
LATEST="$(ls -1dt "$REPO_ROOT"/metrics/runs/*_"$LABEL" 2>/dev/null | head -n1)"
if [[ -n "$LATEST" ]]; then
  python3 -c "
import json
d = json.load(open('$LATEST/result.json'))
proposed = [r['proposed_tokens'] for r in d['requests']]
print(f'proposed per request: {proposed}')
if proposed:
    print(f'mean: {sum(proposed)/len(proposed):.1f}')
print(f'throughput: {d[\"aggregate\"].get(\"system_throughput_tokens_per_sec\",0):.2f} tok/s')
print(f'acceptance: {d[\"aggregate\"].get(\"speculative_decoding\",{}).get(\"overall_acceptance_rate\",0)*100:.2f}%')
"
fi
