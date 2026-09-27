#!/usr/bin/env bash
# Run one member of a same-seed G2-Q / MA-target-ablation replicate.
#
# Usage:
#   bash submit/generation/run_G2_MA_replicate.sh G2-Q 456
#   bash submit/generation/run_G2_MA_replicate.sh G2-MA 456
#   bash submit/generation/run_G2_MA_replicate.sh G2-MA-sigma 456
#   bash submit/generation/run_G2_MA_replicate.sh G2-MA-r 456
#   bash submit/generation/run_G2_MA_replicate.sh G2-MA-joint 456
#
# Always run the Q member first.  MA and shuffle then reuse its decoder,
# fixed 30,000-row Q data, vocabulary, and seed-specific output directory.
set -euo pipefail

GROUP="${1:?usage: $0 G2-Q|G2-MA|G2-MA-shuffle|G2-MA-sigma|G2-MA-r|G2-MA-joint SEED}"
REPLICA_SEED="${2:?usage: $0 G2-Q|G2-MA|G2-MA-shuffle|G2-MA-sigma|G2-MA-r|G2-MA-joint SEED}"
case "$GROUP" in
  G2-Q|G2-MA|G2-MA-shuffle|G2-MA-sigma|G2-MA-r|G2-MA-joint) ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac
[[ "$REPLICA_SEED" =~ ^[0-9]+$ ]] || {
  echo "SEED must be a non-negative integer: $REPLICA_SEED" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_mechanism_auxiliary_comparison/G2-comp/G2_seed_${REPLICA_SEED}}"
MAIN="$ROOT/submit/generation/run_mechanism_auxiliary.sh"

if [[ "$GROUP" == "G2-Q" ]]; then
  exec env SEED="$REPLICA_SEED" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
    bash "$MAIN" G2-Q
fi

Q_RUN="$EXPERIMENT_ROOT/G2-Q"
if [[ ! -f "$Q_RUN/train_decoder/optimal.pt" \
      || ! -f "$Q_RUN/generated/generated_formulas.json" ]]; then
  echo "The same-seed G2-Q replicate must finish first: $Q_RUN" >&2
  exit 1
fi

exec env SEED="$REPLICA_SEED" EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  SOURCE_GROUP_DIR="$Q_RUN" SOURCE_DECODER_DIR="$Q_RUN/train_decoder" \
  VOCAB_DIR="$Q_RUN/candidate_pool" \
  bash "$MAIN" "$GROUP"
