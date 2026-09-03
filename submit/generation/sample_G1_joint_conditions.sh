#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SOURCE_ROOT:-run/generation_bamboo_composition_comparison}"
G1_ROOT="$SOURCE_ROOT/G1"
ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
SEED="${SEED:-123}"

need_file() {
  [[ -f "$1" ]] || {
    echo "Required file is missing: $1" >&2
    exit 1
  }
}

for path in \
  "$G1_ROOT/configs/diffusion.yaml" \
  "$G1_ROOT/train_diffusion/optimal.pt" \
  "$G1_ROOT/configs/decoder.yaml" \
  "$G1_ROOT/train_decoder/optimal.pt" \
  "$G1_ROOT/candidate_pool/solvents.json" \
  "$G1_ROOT/candidate_pool/salts.json" \
  "$ORACLE_CONFIG" \
  "$ORACLE_CKPT/optimal.pt" \
  "$ORACLE_CKPT/pretrain.pt"
do
  need_file "$path"
done

sample_condition() {
  local conductivity="$1"
  local ratio="$2"
  local tag="k${conductivity//./p}_r${ratio//./p}"
  local dest="$SOURCE_ROOT/condition_${tag}/G1"

  mkdir -p "$dest/generated"

  if [[ ! -e "$dest/candidate_pool" ]]; then
    ln -s "$(cd "$G1_ROOT/candidate_pool" && pwd)" "$dest/candidate_pool"
  fi

  if [[ -f "$dest/generated/generated_formulas.json" ]]; then
    echo "Skip existing result: $dest/generated/generated_formulas.json"
    return
  fi

  python3 -u scripts/generation/run_conditioned_generation.py \
    --diffusion-config "$G1_ROOT/configs/diffusion.yaml" \
    --diffusion-checkpoint "$G1_ROOT/train_diffusion/optimal.pt" \
    --decoder-config "$G1_ROOT/configs/decoder.yaml" \
    --decoder-checkpoint "$G1_ROOT/train_decoder/optimal.pt" \
    --solvents "$G1_ROOT/candidate_pool/solvents.json" \
    --salts "$G1_ROOT/candidate_pool/salts.json" \
    --oracle-config "$ORACLE_CONFIG" \
    --oracle-checkpoint "$ORACLE_CKPT" \
    --output "$dest/generated/generated_formulas.json" \
    --target-conductivity "$conductivity" \
    --anion-ratio "$ratio" \
    --anion-ratio-min 0.0 \
    --anion-ratio-max 1.0 \
    --temperature-c "$TEMPERATURE_C" \
    --concentration "$SALT_MOLAR_RATIO" \
    --num-batches "$GEN_BATCHES" \
    --batch-size "$GEN_BATCH_SIZE" \
    --seed "$SEED" \
    --selection-policy joint_target \
    --conductivity-tolerance 2.0 \
    --anion-ratio-tolerance 0.10 \
    --joint-anion-weight 1.0
}

sample_condition 20 0.5
sample_condition 30 0.3
