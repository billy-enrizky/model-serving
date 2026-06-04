#!/usr/bin/env bash
# n=3 re-bench sweep on Modal. Repeats every cell RUNS times (default 3) as
# independent COLD invocations, so each run captures real cold-start + run-to-run
# variance (the suspect single-sample cells the README distrusts).
#
# Every MTP run is pinned to schedule=constant, gamma=4:
#   - transformers: MTP_SCHEDULE=constant is forced here AND inside run_const.sh.
#     We never call deploy_gpu.sh directly (its default is heuristic).
#   - vLLM: num_speculative_tokens=4 is baked into vllm_modal_app.py (constant
#     by construction; no schedule concept).
#
# Each run gets RUN_TAG=_r<N> so the 3 invocations land in distinct
# metrics/runs dirs (..._c1_r1, ..._c1_r2, ..._c1_r3) instead of overwriting.
#
# Each driver stops its Modal app before redeploy, so runs are cold and the
# free-workspace 8-app cap is respected (one cell deployed at a time).
#
# Usage:
#   bash deploy/modal/run_n3.sh                 # full matrix, RUNS=3
#   RUNS=3 bash deploy/modal/run_n3.sh          # explicit
#   GPUS="H100" REGIMES="structured" ENGINES="vllm" MODES="mtp" \
#     bash deploy/modal/run_n3.sh              # single-cell smoke test
#
# Env:
#   RUNS      repetitions per cell (default 3)
#   GPUS      space-separated GPUs (default "A10 A100-80GB B200 H100")
#   REGIMES   space-separated prompt sets (default "generic code structured")
#   ENGINES   "transformers vllm" (default both)
#   MODES     "mtp baseline" (default both)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

RUNS="${RUNS:-3}"
GPUS="${GPUS:-A10 A100-80GB B200 H100}"
REGIMES="${REGIMES:-generic code structured}"
ENGINES="${ENGINES:-transformers vllm}"
MODES="${MODES:-mtp baseline}"

# Hard constraint: every MTP run is constant gamma=4. Force it here so a stray
# deploy_gpu.sh default (heuristic) can never leak into the sweep.
export MTP_SCHEDULE=constant

# run_cell <run> <regime> <gpu> <engine>: deploy + bench one (engine,gpu,regime)
# cell for run-index <run>. Drivers force RUN_TAG/PROMPT_SET; each stops its own
# (and sibling) app first, so at most ~2 apps are live at once.
run_cell() {
  local RUN="$1" REGIME="$2" GPU="$3" ENGINE="$4"
  export RUN_TAG="_r${RUN}" PROMPT_SET="$REGIME"
  echo "===================================================================="
  echo "==> run=$RUN regime=$REGIME gpu=$GPU engine=$ENGINE tag=$RUN_TAG"
  echo "===================================================================="
  case "$ENGINE" in
    transformers)
      if [[ " $MODES " == *" mtp "* ]]; then
        bash deploy/modal/run_const.sh "$GPU" \
          || echo "!! FAIL tx mtp $GPU $REGIME $RUN_TAG" >&2
      fi
      if [[ " $MODES " == *" baseline "* ]]; then
        MODES=baseline bash deploy/modal/run_ab.sh "$GPU" \
          || echo "!! FAIL tx baseline $GPU $REGIME $RUN_TAG" >&2
      fi
      ;;
    vllm)
      VLLM_MODES="$MODES" bash deploy/modal/vllm_run_ab.sh "$GPU" \
        || echo "!! FAIL vllm $GPU $REGIME $RUN_TAG" >&2
      ;;
    *) echo "Unknown ENGINE='$ENGINE' (expect transformers|vllm)" >&2; exit 1 ;;
  esac
}

# cell_complete <run> <regime> <gpu> <engine>: true if every result dir this
# cell should produce already exists (used by the retry pass to skip good cells).
cell_complete() {
  local RUN="$1" REGIME="$2" GPU="$3" ENGINE="$4"
  local glow; glow="$(echo "$GPU" | tr '[:upper:]' '[:lower:]' | tr -d -- '-!+')"
  local sfx=""; [ "$REGIME" != "generic" ] && sfx="_${REGIME}"
  local labels=()
  case "$ENGINE" in
    transformers)
      [[ " $MODES " == *" mtp "* ]]      && labels+=("transformers_mtp_const_${glow}${sfx}_c1_r${RUN}")
      [[ " $MODES " == *" baseline "* ]] && labels+=("baseline_n0_${glow}${sfx}_c1_r${RUN}")
      ;;
    vllm)
      [[ " $MODES " == *" mtp "* ]]      && labels+=("vllm_mtp_${glow}${sfx}_c1_r${RUN}")
      [[ " $MODES " == *" baseline "* ]] && labels+=("vllm_baseline_${glow}${sfx}_c1_r${RUN}")
      ;;
  esac
  local lbl
  for lbl in "${labels[@]}"; do
    ls -1d "$REPO_ROOT"/metrics/runs/*_"$lbl" >/dev/null 2>&1 || return 1
  done
  return 0
}

START_TS="$(date '+%Y-%m-%d %H:%M:%S')"
echo "==> n=3 sweep start $START_TS"
echo "    RUNS=$RUNS GPUS=[$GPUS] REGIMES=[$REGIMES] ENGINES=[$ENGINES] MODES=[$MODES]"
echo "    MTP_SCHEDULE=constant (gamma=4)"
echo

# Clear the workspace to 0 deployed apps so the sweep starts under the
# free-workspace 8-Web-Function cap (stale apps filled it in the smoke run).
echo "==> stopping all deployed apps before sweep"
bash deploy/modal/stop_all.sh || true
echo

# Main sweep.
for RUN in $(seq 1 "$RUNS"); do
  for REGIME in $REGIMES; do
    for GPU in $GPUS; do
      for ENGINE in $ENGINES; do
        run_cell "$RUN" "$REGIME" "$GPU" "$ENGINE"
      done
    done
  done
done

# Retry pass: one extra attempt for any cell missing its result dir(s) (e.g. a
# transient network drop or a cap hit). Re-stop apps first to reclaim slots.
echo
echo "==> retry pass: re-running cells with missing result dirs"
bash deploy/modal/stop_all.sh || true
RETRIED=0
for RUN in $(seq 1 "$RUNS"); do
  for REGIME in $REGIMES; do
    for GPU in $GPUS; do
      for ENGINE in $ENGINES; do
        if ! cell_complete "$RUN" "$REGIME" "$GPU" "$ENGINE"; then
          echo "==> RETRY run=$RUN $REGIME $GPU $ENGINE (missing dir)"
          run_cell "$RUN" "$REGIME" "$GPU" "$ENGINE"
          RETRIED=$((RETRIED+1))
        fi
      done
    done
  done
done
echo "==> retry pass done ($RETRIED cell(s) retried)."

echo
echo "==> n=3 sweep done (start $START_TS, end $(date '+%Y-%m-%d %H:%M:%S'))."
echo "    Aggregate with: python3 bench/summarize_n3.py"
