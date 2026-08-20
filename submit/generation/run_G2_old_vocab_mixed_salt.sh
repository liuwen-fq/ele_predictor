#!/usr/bin/env bash
# Controlled ablation: old G2 vocabulary, current mixed-salt synthetic data.
# Only solvents.json and salts.json are copied.  Embeddings, oracle labels,
# diffusion/decoder checkpoints, and generated formulas are built afresh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REFERENCE_G2="${REFERENCE_G2:-run/bck_generation_joint_conditions/G2}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_ablation_old_g2_vocab_mixed_salt}"
TARGET_POOL="$EXPERIMENT_ROOT/G2/candidate_pool"

for source in "$REFERENCE_G2/candidate_pool/solvents.json" \
              "$REFERENCE_G2/candidate_pool/salts.json"; do
  [[ -f "$source" ]] || { echo "Missing old-G2 vocabulary file: $source" >&2; exit 1; }
done
if [[ -e "$TARGET_POOL/solvents.json" || -e "$TARGET_POOL/salts.json" ]]; then
  echo "A fixed vocabulary already exists at $TARGET_POOL." >&2
  echo "Choose another EXPERIMENT_ROOT to avoid overwriting this ablation." >&2
  exit 1
fi

mkdir -p "$TARGET_POOL"
cp "$REFERENCE_G2/candidate_pool/solvents.json" "$TARGET_POOL/solvents.json"
cp "$REFERENCE_G2/candidate_pool/salts.json" "$TARGET_POOL/salts.json"

# The standard G2 runner detects the fixed vocabulary and skips new guided
# selection.  build_synthetic_generator_data.py supplies the current 1--2
# salt / Dirichlet construction; all downstream stages are retrained.
exec env EXPERIMENT_ROOT="$EXPERIMENT_ROOT" ALLOW_FIXED_VOCAB=1 \
  "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_generation_experiment.sh" G2
