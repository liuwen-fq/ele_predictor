#!/usr/bin/env bash
# Submit one factual MA target-ablation arm on the existing fixed Q dataset.
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_mechanism_auxiliary_ablation.sh G2 sigma
#   CUDA_VISIBLE_DEVICES=1 bash submit/generation/run_mechanism_auxiliary_ablation.sh G2 r
#   CUDA_VISIBLE_DEVICES=2 bash submit/generation/run_mechanism_auxiliary_ablation.sh G2 joint
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_mechanism_auxiliary_ablation.sh G0 sigma
#
# Each arm preserves the source Q formulas, σ/r labels, candidate vocabulary,
# decoder, training budget, and inference condition p(formula | σ, r).  Only
# the training-time MA target changes.
set -euo pipefail

SCOPE="${1:?usage: $0 G0|G2 sigma|r|joint}"
TARGET="${2:?usage: $0 G0|G2 sigma|r|joint}"
case "$SCOPE" in G0|G2) ;; *)
  echo "Unknown scope: $SCOPE (expected G0 or G2)" >&2; exit 2;; esac
case "$TARGET" in sigma|r|joint) ;; *)
  echo "Unknown MA target: $TARGET (expected sigma, r, or joint)" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ "$SCOPE" == "G0" && "$TARGET" == "joint" ]]; then
  # G0-MA is the canonical factual two-label (I_sigma, I_r) joint arm.
  exec bash "$ROOT/submit/generation/run_mechanism_auxiliary.sh" G0-MA
fi
exec bash "$ROOT/submit/generation/run_mechanism_auxiliary.sh" \
  "${SCOPE}-MA-${TARGET}"
