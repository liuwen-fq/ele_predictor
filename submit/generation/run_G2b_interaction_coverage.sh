#!/usr/bin/env bash
# Matched G2-b interaction-coverage experiment.
#
# G2-b-Q: seeded random selection from an oversampled candidate set.
# G2-b-I: identical candidate construction and final (sigma, ratio,
#          solvent-count) quotas, but interaction-coverage selection within
#          each cell.
#
# The generator still receives only [sigma, anion_ratio].  The frozen oracle
# is used for labels and explanation proxies only; it is never retrained here.
set -euo pipefail

GROUP="${1:?usage: $0 G2-b-Q|G2-b-I}"
case "$GROUP" in
  G2-b-Q) MECHANISM_SELECTION="matched_random" ;;
  G2-b-I) MECHANISM_SELECTION="interaction_sigma_tertiles" ;;
  *) echo "Unknown group: $GROUP (expected G2-b-Q or G2-b-I)" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Q and I must use the same values for every parameter below.  Keep separate
# experiment roots only when intentionally starting a new paired comparison.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_bamboo_interaction_comparison}"
RUN="$EXPERIMENT_ROOT/$GROUP"
POOL="$RUN/candidate_pool"
SHARED_CANDIDATE_CACHE="${SHARED_CANDIDATE_CACHE:-$EXPERIMENT_ROOT/shared_oracle_scored_candidates}"
SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-2.0}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-500}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
SEED="${SEED:-123}"
TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
BAMBOO_SALTS="${BAMBOO_SALTS:-$ROOT/emb_dict/salts.json}"

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

if [[ ! -e "$POOL/solvents.json" && ! -e "$POOL/salts.json" ]]; then
  mkdir -p "$POOL"
  cp "$BAMBOO_SOLVENTS" "$POOL/solvents.json"
  cp "$BAMBOO_SALTS" "$POOL/salts.json"
  record_skip candidate_pool fixed_released_Bamboo_vocabulary
elif [[ -f "$POOL/solvents.json" && -f "$POOL/salts.json" ]]; then
  record_skip candidate_pool existing_fixed_released_Bamboo_vocabulary
else
  echo "Incomplete fixed vocabulary in $POOL; use a fresh EXPERIMENT_ROOT." >&2
  exit 1
fi

if [[ ! -f "$POOL/solv_embs.pt" ]]; then
  run_stage vocabulary_embedding python3 -u scripts/generation/build_generator_vocab_embeddings.py \
    --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
    --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$POOL"
else
  record_skip vocabulary_embedding existing_output
fi

if [[ ! -f "$RUN/generator_data/processed_data.pkl" ]]; then
  # Serialise the first cache build.  If Q and I are submitted together, the
  # second process waits here and then reuses the completed exact candidate
  # pool rather than producing a separately scored one.
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

run_stage analyze_generation python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$EXPERIMENT_ROOT" --groups G2-b-Q G2-b-I \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100

printf 'Finished %s. Raw results: %s\n' "$GROUP" "$RUN/generated/generated_formulas.json"
