#!/usr/bin/env bash
# Train one member of the paired G2-b/G2 composition-prior ablation.
#
# G2-b: fixed released Bamboo 62-solvent vocabulary.
# G2:   current explanation/property-guided 62-solvent vocabulary.
#
# Both use the released Bamboo 1--6 solvent-count empirical prior, fixed two
# salts, no oracle-label joint rebalancing, and this repository's current
# frozen dual-task oracle.  Therefore vocabulary source is the only intended
# experimental variable.
set -euo pipefail

GROUP="${1:?usage: $0 G2-b|G2}"
case "$GROUP" in G2-b|G2) ;; *) echo "Unknown group: $GROUP" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_bamboo_composition_comparison}"
RUN="$EXPERIMENT_ROOT/$GROUP"
POOL="$RUN/candidate_pool"
POOL_SIZE="${POOL_SIZE:-62}"
SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-500}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
SEED="${SEED:-123}"
CANDIDATE_POOL_SEED="${CANDIDATE_POOL_SEED:-$SEED}"
TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
MONO_CONFIG="${MONO_CONFIG:-$ORACLE_CONFIG}"
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
BAMBOO_SALTS="${BAMBOO_SALTS:-$ROOT/emb_dict/salts.json}"
SOLVENT_LIKE_POOL="${SOLVENT_LIKE_POOL:-$ROOT/emb_dict/solvent_like_candidates.json}"
SOLVENT_LIKE_AUDIT="${SOLVENT_LIKE_AUDIT:-$ROOT/emb_dict/solvent_like_candidates_audit.json}"
PROPERTY_CONSTRAINTS="${PROPERTY_CONSTRAINTS:-run/property_fragment_attribution_v2/generation_design_constraints/generation_property_constraints.csv}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
for path in "$ORACLE_CONFIG" "$ORACLE_CKPT/optimal.pt" "$ORACLE_CKPT/pretrain.pt" \
            "$MONO_CKPT" "$BAMBOO_SOLVENTS" "$BAMBOO_SALTS"; do
  need_file "$path"
done

mkdir -p "$RUN"
TIMING="$RUN/timings.csv"
[[ -f "$TIMING" ]] || printf 'stage,start_utc,end_utc,elapsed_seconds,detail\n' > "$TIMING"

record_skip() {
  printf '%s,%s,%s,%s,%s\n' "$1" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0 "$2" >> "$TIMING"
}
run_stage() {
  local stage="$1" start end start_utc end_utc; shift
  start="$(date +%s)"; "$@"; end="$(date +%s)"
  start_utc=$(date -u -d "@$start" +%FT%TZ)
  end_utc=$(date -u -d "@$end" +%FT%TZ)
  printf '%s,%s,%s,%s,%s\n' "$stage" "$start_utc" "$end_utc" \
    "$((end - start))" executed >> "$TIMING"
}
train_or_resume() {
  local stage="$1" work="$2" config="$3"; shift 3
  local checkpoints=()
  shopt -s nullglob; checkpoints=("$work"/ckpt/ckpt_epoch_*.pt); shopt -u nullglob
  if [[ -f "$work/FINISHED" ]]; then
    record_skip "$stage" finished
  elif (( ${#checkpoints[@]} > 0 )); then
    run_stage "$stage" python3 -u scripts/train/train.py --task formula --conf "$config" --restart
  elif [[ -f "$work/optimal.pt" ]]; then
    echo "Incomplete $stage: optimal.pt exists without FINISHED/epoch checkpoint: $work" >&2
    exit 1
  else
    run_stage "$stage" python3 -u scripts/train/train.py --task formula --conf "$config"
  fi
}

if [[ "$GROUP" == "G2-b" ]]; then
  # Fixed published vocabulary.  Copy only once; embeddings and every model
  # artifact below are rebuilt in this experiment root.
  if [[ ! -e "$POOL/solvents.json" && ! -e "$POOL/salts.json" ]]; then
    mkdir -p "$POOL"
    cp "$BAMBOO_SOLVENTS" "$POOL/solvents.json"
    cp "$BAMBOO_SALTS" "$POOL/salts.json"
    record_skip candidate_pool "fixed_released_Bamboo_vocabulary"
  elif [[ -f "$POOL/solvents.json" && -f "$POOL/salts.json" ]]; then
    record_skip candidate_pool "existing_fixed_released_Bamboo_vocabulary"
  else
    echo "Incomplete G2-b fixed vocabulary in $POOL; use a fresh EXPERIMENT_ROOT." >&2
    exit 1
  fi
else
  if [[ ! -f "$POOL/solvents.json" ]]; then
    need_file "$SOLVENT_LIKE_POOL"; need_file "$SOLVENT_LIKE_AUDIT"; need_file "$PROPERTY_CONSTRAINTS"
    run_stage candidate_pool python3 -u scripts/generation/build_candidate_pool.py \
      --mode guided --output-dir "$POOL" --baseline-solvents "$BAMBOO_SOLVENTS" \
      --salts "$BAMBOO_SALTS" --candidate-source "$SOLVENT_LIKE_POOL" \
      --property-constraints "$PROPERTY_CONSTRAINTS" --model-config "$MONO_CONFIG" \
      --mono-checkpoint "$MONO_CKPT" --pool-size "$POOL_SIZE" --seed "$CANDIDATE_POOL_SEED" \
      --guided-selection-strategy "${GUIDED_SELECTION_STRATEGY:-coverage_v1}" \
      --guided-property-weight "${GUIDED_PROPERTY_WEIGHT:-1.0}" \
      --guided-fragment-weight "${GUIDED_FRAGMENT_WEIGHT:-1.0}" \
      --guided-family-coverage-weight "${GUIDED_FAMILY_COVERAGE_WEIGHT:-0.7}" \
      --guided-scaffold-coverage-weight "${GUIDED_SCAFFOLD_COVERAGE_WEIGHT:-0.25}" \
      --guided-redundancy-weight "${GUIDED_REDUNDANCY_WEIGHT:-1.0}" \
      --anion-guided-weight "${GUIDED_ANION_WEIGHT:-0.35}"
  else
    record_skip candidate_pool existing_output
  fi
fi

if [[ ! -f "$POOL/solv_embs.pt" ]]; then
  run_stage vocabulary_embedding python3 -u scripts/generation/build_generator_vocab_embeddings.py \
    --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
    --model-config "$MONO_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$POOL"
else record_skip vocabulary_embedding existing_output; fi

if [[ ! -f "$RUN/generator_data/processed_data.pkl" ]]; then
  run_stage synthetic_formula_oracle python3 -u scripts/generation/build_synthetic_generator_data.py \
    --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output-dir "$RUN/generator_data" --num-formulas "$SYNTHETIC_FORMULAS" \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --oversample-factor 1.0 --label-policy sanitize_keep_all \
    --solvent-count-policy bamboo_empirical_1_to_6 \
    --salt-mixture-mode bamboo_fixed_two_salt --salt-mixture-bins 5 \
    --joint-balance-strategy none --min-nonempty-anion-bins 1 \
    --anion-ratio-min 0.0 --anion-ratio-max 1.0 --seed "$SEED"
else record_skip synthetic_formula_oracle existing_output; fi

if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
  run_stage write_train_configs python3 -u scripts/generation/write_generator_train_configs.py \
    --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED"
else record_skip write_train_configs existing_output; fi

train_or_resume train_diffusion "$RUN/train_diffusion" "$RUN/configs/diffusion.yaml"
train_or_resume train_decoder "$RUN/train_decoder" "$RUN/configs/decoder.yaml"

if [[ ! -f "$RUN/generated/generated_formulas.json" ]]; then
  run_stage sample_and_score python3 -u scripts/generation/run_conditioned_generation.py \
    --diffusion-config "$RUN/configs/diffusion.yaml" --diffusion-checkpoint "$RUN/train_diffusion/optimal.pt" \
    --decoder-config "$RUN/configs/decoder.yaml" --decoder-checkpoint "$RUN/train_decoder/optimal.pt" \
    --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output "$RUN/generated/generated_formulas.json" --target-conductivity "$TARGET_CONDUCTIVITY" \
    --anion-ratio "$ANION_RATIO" --anion-ratio-min 0.0 --anion-ratio-max 1.0 \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED" \
    --selection-policy joint_target --conductivity-tolerance 2.0 \
    --anion-ratio-tolerance 0.10 --joint-anion-weight 1.0
else record_skip sample_and_score existing_output; fi

run_stage analyze_generation python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$EXPERIMENT_ROOT" --groups G2-b G2 \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100

printf 'Finished %s. Raw results: %s\n' "$GROUP" "$RUN/generated/generated_formulas.json"
