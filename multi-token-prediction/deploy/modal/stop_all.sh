#!/usr/bin/env bash
# Stop every currently-deployed Modal app so the n=3 sweep starts under the
# free-workspace 8-Web-Function cap. Modal's `app stop` takes an app id; we
# parse deployed ids from `app list` (col 2 = id, col 4 = state).
#
# Usage: bash deploy/modal/stop_all.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODAL_BIN="$REPO_ROOT/.venv/bin/modal"

# bash 3.2 (macOS default) has no mapfile; collect ids with a while-read loop.
IDS="$(
  "$MODAL_BIN" app list 2>/dev/null \
    | sed 's/│/|/g' \
    | grep -E 'ap-' \
    | awk -F'|' '{gsub(/^ +| +$/,"",$2); gsub(/^ +| +$/,"",$4); if ($4=="deployed") print $2}'
)"

if [[ -z "$IDS" ]]; then
  echo "[stop_all] no deployed apps."
  exit 0
fi

N="$(echo "$IDS" | wc -l | tr -d ' ')"
echo "[stop_all] stopping $N deployed app(s)"
echo "$IDS" | while read -r id; do
  [ -z "$id" ] && continue
  "$MODAL_BIN" app stop -y "$id" 2>/dev/null && echo "[stop_all] stopped $id" \
    || echo "[stop_all] could not stop $id (skip)"
done
echo "[stop_all] done."
