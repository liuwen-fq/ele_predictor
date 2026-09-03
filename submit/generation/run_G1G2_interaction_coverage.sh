#!/usr/bin/env bash
# Paired interaction-coverage retraining for G1 (random vocabulary) and G2
# (explanation/property-guided vocabulary).
#
# Each pair uses exactly one vocabulary and one oracle-scored oversampled
# candidate cache:
#   * -Q: seeded random, quota-matched synthetic-data control;
#   * -I: within each (sigma, anion-ratio, solvent-count) cell, select
#         low/middle/high explanation-interaction coverage.
# The oracle is frozen and never retrained by this pipeline.
set -euo pipefail

GROUP="${1:?usage: $0 G1-Q|G1-I|G2-Q|G2-I}"
case "$GROUP" in
  G1-Q) BASE_GROUP="G1"; VOCAB_MODE="random"; MECHANISM_SELECTION="matched_random" ;;
  G1-I) BASE_GROUP="G1"; VOCAB_MODE="random"; MECHANISM_SELECTION="interaction_sigma_tertiles" ;;
  G2-Q) BASE_GROUP="G2"; VOCAB_MODE="guided"; MECHANISM_SELECTION="matched_random" ;;
  G2-I) BASE_GROUP="G2"; VOCAB_MODE="guided"; MECHANISM_SELECTION="interaction_sigma_tertiles" ;;
  *) echo "Unknown group: $GROUP (expected G1-Q|G1-I|G2-Q|G2-I)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Use a fresh root for a new pair.  Q and I of the same BASE_GROUP must share
# this root, seed, vocabulary and scored-candidate cache.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_interaction_coverage_comparison}"
RUN="$EXPERIMENT_ROOT/$GROUP"
VOCAB_ROOT="$EXPERIMENT_ROOT/vocabularies/$BASE_GROUP"
POOL="$VOCAB_ROOT/candidate_pool"
SHARED_CANDIDATE_CACHE="${SHARED_CANDIDATE_CACHE:-$EXPERIMENT_ROOT/shared_${BASE_GROUP}_oracle_scored_candidates}"

POOL_SIZE="${POOL_SIZE:-62}"
SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-2.0}"
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
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
BAMBOO_SALTS="${BAMBOO_SALTS:-$ROOT/emb_dict/salts.json}"
SOLVENT_LIKE_POOL="${SOLVENT_LIKE_POOL:-$ROOT/emb_dict/solvent_like_candidates.json}"
SOLVENT_LIKE_AUDIT="${SOLVENT_LIKE_AUDIT:-$ROOT/emb_dict/solvent_like_candidates_audit.json}"
PROPERTY_CONSTRAINTS="${PROPERTY_CONSTRAINTS:-run/property_fragment_attribution_v2/generation_design_constraints/generation_property_constraints.csv}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
for path in "$ORACLE_CONFIG" "$ORACLE_CKPT/optimal.pt" "$ORACLE_CKPT/pretrain.pt" \
            "$MONO_CKPT" "$BAMBOO_SOLVENTS" "$BAMBOO_SALTS" "$SOLVENT_LIKE_POOL"; do
  need_file "$path"
done
if [[ "$VOCAB_MODE" == "guided" ]]; then
  need_file "$SOLVENT_LIKE_AUDIT"
  need_file "$PROPERTY_CONSTRAINTS"
fi

mkdir -p "$RUN" "$VOCAB_ROOT"
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
  printf '%s,%s,%s,%s,%s\n' "$stage" "$start_utc" "$end_utc" "$((end - start))" executed >> "$TIMING"
}
train_or_resume() {
  local stage="$1" work="$2" config="$3"
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

# The lock makes concurrent Q/I submissions safe: both runs get precisely the
# same G1 or G2 vocabulary and vocabulary embeddings.
prepare_vocab() {
  local lock="$VOCAB_ROOT/.build.lock"
  exec 9>"$lock"
  flock 9
  if [[ ! -f "$POOL/solvents.json" ]]; then
    mkdir -p "$POOL"
    if [[ "$VOCAB_MODE" == "random" ]]; then
      python3 -u scripts/generation/build_candidate_pool.py \
        --mode random --output-dir "$POOL" --baseline-solvents "$BAMBOO_SOLVENTS" \
        --salts "$BAMBOO_SALTS" --candidate-source "$SOLVENT_LIKE_POOL" \
        --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" \
        --pool-size "$POOL_SIZE" --seed "$CANDIDATE_POOL_SEED"
    else
      python3 -u scripts/generation/build_candidate_pool.py \
        --mode guided --output-dir "$POOL" --baseline-solvents "$BAMBOO_SOLVENTS" \
        --salts "$BAMBOO_SALTS" --candidate-source "$SOLVENT_LIKE_POOL" \
        --property-constraints "$PROPERTY_CONSTRAINTS" --model-config "$ORACLE_CONFIG" \
        --mono-checkpoint "$MONO_CKPT" --pool-size "$POOL_SIZE" --seed "$CANDIDATE_POOL_SEED" \
        --guided-selection-strategy "${GUIDED_SELECTION_STRATEGY:-coverage_v1}" \
        --guided-property-weight "${GUIDED_PROPERTY_WEIGHT:-1.0}" \
        --guided-fragment-weight "${GUIDED_FRAGMENT_WEIGHT:-1.0}" \
        --guided-family-coverage-weight "${GUIDED_FAMILY_COVERAGE_WEIGHT:-0.7}" \
        --guided-scaffold-coverage-weight "${GUIDED_SCAFFOLD_COVERAGE_WEIGHT:-0.25}" \
        --guided-redundancy-weight "${GUIDED_REDUNDANCY_WEIGHT:-1.0}" \
        --anion-guided-weight "${GUIDED_ANION_WEIGHT:-0.35}"
    fi
  fi
  if [[ ! -f "$POOL/solv_embs.pt" ]]; then
    python3 -u scripts/generation/build_generator_vocab_embeddings.py \
      --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
      --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$POOL"
  fi
  flock -u 9
}

if [[ ! -f "$POOL/solvents.json" || ! -f "$POOL/solv_embs.pt" ]]; then
  run_stage vocabulary_and_embedding prepare_vocab
else
  record_skip vocabulary_and_embedding existing_shared_vocabulary
fi

if [[ ! -e "$RUN/candidate_pool" && ! -L "$RUN/candidate_pool" ]]; then
  ln -s "$(cd "$POOL" && pwd)" "$RUN/candidate_pool"
fi

if [[ ! -f "$RUN/generator_data/processed_data.pkl" ]]; then
  mkdir -p "$SHARED_CANDIDATE_CACHE"
  command -v flock >/dev/null || { echo "flock is required for paired Q/I cache safety" >&2; exit 1; }
  run_stage synthetic_formula_oracle flock "$SHARED_CANDIDATE_CACHE/.build.lock" \
    python3 -u scripts/generation/build_synthetic_generator_data.py \
    --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output-dir "$RUN/generator_data" --num-formulas "$SYNTHETIC_FORMULAS" \
    --candidate-cache-dir "$SHARED_CANDIDATE_CACHE" \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --oversample-factor "$OVERSAMPLE_FACTOR" --label-policy sanitize_keep_all \
    --solvent-count-policy bamboo_empirical_1_to_6 \
    --salt-mixture-mode bamboo_fixed_two_salt --salt-mixture-bins 5 \
    --joint-balance-strategy none --min-nonempty-anion-bins 1 \
    --anion-ratio-min 0.0 --anion-ratio-max 1.0 \
    --mechanism-selection "$MECHANISM_SELECTION" --seed "$SEED"
else
  record_skip synthetic_formula_oracle existing_output
fi

if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
  run_stage write_train_configs python3 -u scripts/generation/write_generator_train_configs.py \
    --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED"
else
  record_skip write_train_configs existing_output
fi

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
else
  record_skip sample_and_score existing_output
fi

PAIR_GROUPS=("${BASE_GROUP}-Q" "${BASE_GROUP}-I")
run_stage analyze_generation python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$EXPERIMENT_ROOT" --groups "${PAIR_GROUPS[@]}" \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100

printf 'Finished %s. Raw results: %s\n' "$GROUP" "$RUN/generated/generated_formulas.json"
