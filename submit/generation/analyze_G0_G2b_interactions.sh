#!/usr/bin/env bash
# Uniform-current-oracle interaction attribution for the released G0 and G2-b.
# This does not train or sample a generator; it only rescans existing formulas.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_bamboo_composition_comparison}"
ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
SEED="${SEED:-123}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
BATCH_SIZE="${BATCH_SIZE:-64}"

for condition in condition_k20_r0p5 condition_k30_r0p3; do
  python3 -u scripts/generation/analyze_generated_interactions.py \
    --experiment-root "$EXPERIMENT_ROOT/$condition" \
    --groups G0 G2-b \
    --oracle-config "$ORACLE_CONFIG" \
    --oracle-checkpoint "$ORACLE_CKPT" \
    --batch-size "$BATCH_SIZE" --bootstrap "$BOOTSTRAP" --seed "$SEED"
done
