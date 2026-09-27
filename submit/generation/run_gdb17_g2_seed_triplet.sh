#!/usr/bin/env bash
# Run one complete GDB17 G2 replicate in the only valid dependency order.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_gdb17_g2_seed_triplet.sh 123
#
# The underlying runner reuses one shared GDB17 catalogue, one fixed G2
# vocabulary, and one 30,000-formula oracle-labelled synthetic set.  Within
# this seed, G2-Q trains the decoder once; G2-MA and G2-MA-shuffle reuse it.
set -euo pipefail

SEED="${1:?usage: $0 123|456|789}"
[[ "$SEED" =~ ^(123|456|789)$ ]] || {
  echo "SEED must be one of 123, 456, 789: $SEED" >&2
  exit 2
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="$ROOT/submit/generation/run_gdb17_mechanism_auxiliary_replicate.sh"

for GROUP in G2-Q G2-MA G2-MA-shuffle; do
  echo "[$(date -Iseconds)] GDB17 G2 seed $SEED: $GROUP"
  bash "$RUNNER" "$GROUP" "$SEED"
done

echo "Completed GDB17 G2 triplet for seed $SEED."
