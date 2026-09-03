#!/usr/bin/env bash
# Resample the already trained G2-b and G2 generators at a new joint condition.
# This deliberately does not rebuild embeddings, synthetic data, diffusion, or
# decoder checkpoints.
set -euo pipefail

TARGET_CONDUCTIVITY="${1:?usage: $0 TARGET_CONDUCTIVITY ANION_RATIO}"
ANION_RATIO="${2:?usage: $0 TARGET_CONDUCTIVITY ANION_RATIO}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SOURCE_ROOT:-run/generation_bamboo_composition_comparison}"
TAG="k${TARGET_CONDUCTIVITY//./p}_r${ANION_RATIO//./p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/condition_${TAG}}"
ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
SEED="${SEED:-123}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
need_file "$ORACLE_CONFIG"
need_file "$ORACLE_CKPT/optimal.pt"
need_file "$ORACLE_CKPT/pretrain.pt"

for GROUP in G2-b G2; do
  SOURCE="$SOURCE_ROOT/$GROUP"
  DEST="$OUTPUT_ROOT/$GROUP"
  for path in "$SOURCE/configs/diffusion.yaml" "$SOURCE/train_diffusion/optimal.pt" \
              "$SOURCE/configs/decoder.yaml" "$SOURCE/train_decoder/optimal.pt" \
              "$SOURCE/candidate_pool/solvents.json" "$SOURCE/candidate_pool/salts.json"; do
    need_file "$path"
  done
  mkdir -p "$DEST"
  if [[ ! -e "$DEST/candidate_pool" ]]; then
    ln -s "$(cd "$SOURCE/candidate_pool" && pwd)" "$DEST/candidate_pool"
  fi
  if [[ -e "$DEST/generated/generated_formulas.json" ]]; then
    echo "Condition output already exists: $DEST/generated/generated_formulas.json" >&2
    echo "Use OUTPUT_ROOT=... to create a separate replicate rather than overwriting it." >&2
    exit 1
  fi
  mkdir -p "$DEST/generated"
  python3 -u scripts/generation/run_conditioned_generation.py \
    --diffusion-config "$SOURCE/configs/diffusion.yaml" \
    --diffusion-checkpoint "$SOURCE/train_diffusion/optimal.pt" \
    --decoder-config "$SOURCE/configs/decoder.yaml" \
    --decoder-checkpoint "$SOURCE/train_decoder/optimal.pt" \
    --solvents "$SOURCE/candidate_pool/solvents.json" \
    --salts "$SOURCE/candidate_pool/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output "$DEST/generated/generated_formulas.json" \
    --target-conductivity "$TARGET_CONDUCTIVITY" --anion-ratio "$ANION_RATIO" \
    --anion-ratio-min 0.0 --anion-ratio-max 1.0 \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED" \
    --selection-policy joint_target --conductivity-tolerance 2.0 \
    --anion-ratio-tolerance 0.10 --joint-anion-weight 1.0
done

python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$OUTPUT_ROOT" --groups G2-b G2 \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100

printf 'Finished resampling condition κ=%s, anion ratio=%s. Report: %s/generation_experiment_report_zh.md\n' \
  "$TARGET_CONDUCTIVITY" "$ANION_RATIO" "$OUTPUT_ROOT"
