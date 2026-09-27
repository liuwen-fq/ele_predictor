#!/usr/bin/env bash
# Sample one independent seed from the *released* Bamboo-Mixer generator.
# The diffusion/decoder/checkpoints remain those distributed by Bamboo; only
# the native generator's fixed batch-seed sequence receives an offset.
# Decoded formulas are then rescored by the frozen current oracle so their
# metrics can be compared fairly with G0-Q/G0-MA/G2-MA.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_released_bamboo_G0_replicate.sh 123
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_released_bamboo_G0_replicate.sh 456
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/run_released_bamboo_G0_replicate.sh 789
set -euo pipefail

SEED="${1:?usage: $0 SEED}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a non-negative integer" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
BAMBOO_ROOT="${BAMBOO_ROOT:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer}"
ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_released_bamboo_replicates/G0-comp/G0_seed_${SEED}}"
RUN="$EXPERIMENT_ROOT/G0"
OUTPUT="$RUN/generated/generated_formulas.json"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
for path in "$BAMBOO_ROOT/ckpts/generator/diffusion.pt" \
            "$BAMBOO_ROOT/ckpts/generator/decoder.pt" \
            "$BAMBOO_ROOT/ckpts/formula/optimal.pt" \
            "$BAMBOO_ROOT/ckpts/mono/optimal.pt" \
            "$BAMBOO_ROOT/config/test_results/generate_config.yaml" \
            "$BAMBOO_ROOT/run/solv_embs.pt" "$BAMBOO_ROOT/run/salt_embs.pt" \
            "$ORACLE_CONFIG" "$ORACLE_CKPT/pretrain.pt" "$ORACLE_CKPT/optimal.pt"; do
  need_file "$path"
done

if [[ -f "$OUTPUT" ]]; then
  echo "Existing released-Bamboo replicate retained: $OUTPUT" >&2
  exit 0
fi
mkdir -p "$RUN/candidate_pool" "$RUN/generated"
cp emb_dict/solvents.json "$RUN/candidate_pool/solvents.json"
cp emb_dict/salts.json "$RUN/candidate_pool/salts.json"

python3 -u scripts/generation/run_released_bamboo_condition.py \
  --bamboo-root "$BAMBOO_ROOT" --output "$OUTPUT" \
  --target-conductivity "$TARGET_CONDUCTIVITY" --anion-ratio "$ANION_RATIO" \
  --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
  --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED" \
  --current-oracle-config "$ORACLE_CONFIG" \
  --current-oracle-checkpoint "$ORACLE_CKPT"

python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$EXPERIMENT_ROOT" --groups G0 \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100 \
  --joint-sigma-scale 2 --joint-anion-ratio-scale 0.1

printf 'Finished released Bamboo G0 replicate seed=%s: %s\n' "$SEED" "$OUTPUT"
