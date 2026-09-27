#!/usr/bin/env bash
# Run one MA replicate; resources are supplied by the caller / compute pool.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GROUP="G1-MA"
if (( $# < 1 || $# > 3 )); then
  echo "Usage: bash $0 123|456|789 [TASK.json] [--plan]" >&2
  exit 2
fi
SEED="$1"
[[ "$SEED" =~ ^(123|456|789)$ ]] || { echo "Invalid seed: $SEED" >&2; exit 2; }
shift
TASK="config/screening/conditional_g2.example.json"
if (( $# )) && [[ "$1" != --plan ]]; then TASK="$1"; shift; fi
EXTRA=()
if (( $# )); then
  [[ $# == 1 && "$1" == --plan ]] || { echo "Only --plan is supported after TASK.json" >&2; exit 2; }
  EXTRA+=(--plan)
fi
[[ -f "$TASK" ]] || { echo "Task file missing: $TASK" >&2; exit 2; }
mkdir -p run/submission_logs
LOG="$ROOT/run/submission_logs/${GROUP}_seed${SEED}_$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
exec > >(tee -a "$LOG") 2>&1
echo "Job: $GROUP seed $SEED; task: $TASK"
echo "Combined stdout/stderr log: $LOG"
echo "Optional hard gates: minimum_dc=${COMMON_MINIMUM_DC:-disabled}, maximum_vis=${COMMON_MAXIMUM_VIS:-disabled}"
if bash "$ROOT/submit/generation/run_conditional_G1G2_MA.sh" "$GROUP" "$SEED" "$TASK" "${EXTRA[@]}"; then
  echo "Completed: $GROUP seed $SEED"
else
  status=$?
  echo "FAILED: $GROUP seed $SEED; exit=$status; log=$LOG" >&2
  exit "$status"
fi
