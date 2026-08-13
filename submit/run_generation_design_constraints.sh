#!/usr/bin/env bash
# Build generation-facing solvent structure/property constraints from existing
# train/valid/test attribution outputs.  No model training or preprocessing.
set -euo pipefail

cd "$(dirname "$0")/.."
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$PWD:$PWD/scripts/test_results${PYTHONPATH:+:$PYTHONPATH}"

ROOT="run/property_fragment_attribution_v2"
for file in \
  "$ROOT/prediction_inputs/ft_train_predictions.json" \
  "$ROOT/prediction_inputs/ft_valid_predictions.json" \
  "$ROOT/cross_split/cross_split_motif_stability.csv"; do
    [[ -f "$file" ]] || { echo "Missing prerequisite: $file" >&2; exit 1; }
done

python3 -u scripts/test_results/build_generation_design_constraints.py \
  --train-input "$ROOT/prediction_inputs/ft_train_predictions.json" \
  --valid-input "$ROOT/prediction_inputs/ft_valid_predictions.json" \
  --cross-split-csv "$ROOT/cross_split/cross_split_motif_stability.csv" \
  --output-dir "$ROOT/generation_design_constraints"
