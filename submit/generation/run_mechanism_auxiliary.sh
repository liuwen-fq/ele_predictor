#!/usr/bin/env bash
# Controlled interaction-auxiliary diffusion experiment.
#
# All MA arms retain the ordinary deployable generation interface:
#     p(formula | sigma, anion_ratio)
# They never sample or require an interaction vector at inference.  The only
# intervention is a training-time Huber loss that predicts a formula's
# residualized explanation interaction pair from the denoiser's reconstructed
# clean latent embedding.
#
# G2-MA and G2-MA-shuffle reuse the exact same completed G2-Q synthetic
# formulas, BOWs, sigma/r labels, vocabulary, decoder, training budget, and
# generation settings.  The shuffled arm changes only the within-cell pairing
# of the two auxiliary residual labels.  G0-MA is the corresponding
# fixed-Bamboo-vocabulary, current-oracle controlled arm; it is not the
# released Bamboo checkpoint.
set -euo pipefail

GROUP="${1:?usage: $0 G2-Q|G2-MA|G2-MA-shuffle|G2-MA-sigma|G2-MA-r|G2-MA-joint|G0-MA|G0-MA-shuffle|G0-MA-sigma|G0-MA-r|G0-MA-joint}"
# Read this early because the standalone G2-Q replicate uses it in its
# default output directory.  The assignment below therefore must not replace
# a seed supplied by the submission environment.
SEED="${SEED:-123}"
case "$GROUP" in
  G2-Q)
    # A fresh, seed-specific Q replicate.  It reads the completed Q synthetic
    # data and vocabulary but retrains *both* diffusion and decoder, so MA
    # arms can share a same-seed decoder rather than the historical seed-123
    # checkpoint.
    PREP_MODE=""
    RETRAIN_DECODER=1
    BASELINE_GROUP="G2-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_replicates/G2_seed_${SEED}"
    DEFAULT_SOURCE_GROUP="run/generation_interaction_coverage_comparison/G2-Q"
    DEFAULT_SOURCE_DECODER=""
    MA_TARGETS=""
    ;;
  G2-MA)
    PREP_MODE="true"
    RETRAIN_DECODER=0
    BASELINE_GROUP="G2-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_comparison/G2-comp"
    DEFAULT_SOURCE_GROUP="run/generation_interaction_coverage_comparison/G2-Q"
    DEFAULT_SOURCE_DECODER="run/generation_interaction_coverage_comparison/G2-Q/train_decoder"
    MA_TARGETS="sigma_r"
    ;;
  G2-MA-shuffle)
    PREP_MODE="shuffle_within_cell"
    RETRAIN_DECODER=0
    BASELINE_GROUP="G2-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_comparison/G2-comp"
    DEFAULT_SOURCE_GROUP="run/generation_interaction_coverage_comparison/G2-Q"
    DEFAULT_SOURCE_DECODER="run/generation_interaction_coverage_comparison/G2-Q/train_decoder"
    MA_TARGETS="sigma_r"
    ;;
  G2-MA-sigma|G2-MA-r|G2-MA-joint)
    PREP_MODE="true"
    RETRAIN_DECODER=0
    BASELINE_GROUP="G2-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_comparison/G2-comp"
    DEFAULT_SOURCE_GROUP="run/generation_interaction_coverage_comparison/G2-Q"
    DEFAULT_SOURCE_DECODER="run/generation_interaction_coverage_comparison/G2-Q/train_decoder"
    case "$GROUP" in
      G2-MA-sigma) MA_TARGETS="sigma" ;;
      G2-MA-r) MA_TARGETS="r" ;;
      G2-MA-joint) MA_TARGETS="sigma_r" ;;
    esac
    ;;
  G0-MA)
    PREP_MODE="true"
    RETRAIN_DECODER=0
    BASELINE_GROUP="G0-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_comparison/G0-comp"
    # The condition-specific directory retains the Q synthetic formulas and
    # candidate vocabulary, whereas the parent comparison root retains the
    # matching decoder checkpoint.
    DEFAULT_SOURCE_GROUP="run/generation_bamboo_composition_comparison/condition_k20_r0p5/G2-b-Q"
    DEFAULT_SOURCE_DECODER="run/generation_bamboo_composition_comparison/G2-b-Q/train_decoder"
    MA_TARGETS="sigma_r"
    ;;
  G0-MA-shuffle)
    PREP_MODE="shuffle_within_cell"
    RETRAIN_DECODER=0
    BASELINE_GROUP="G0-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_comparison/G0-comp"
    DEFAULT_SOURCE_GROUP="run/generation_bamboo_composition_comparison/condition_k20_r0p5/G2-b-Q"
    DEFAULT_SOURCE_DECODER="run/generation_bamboo_composition_comparison/G2-b-Q/train_decoder"
    MA_TARGETS="sigma_r"
    ;;
  G0-MA-sigma|G0-MA-r|G0-MA-joint)
    PREP_MODE="true"
    RETRAIN_DECODER=0
    BASELINE_GROUP="G0-Q"
    DEFAULT_EXPERIMENT_ROOT="run/generation_mechanism_auxiliary_comparison/G0-comp"
    DEFAULT_SOURCE_GROUP="run/generation_bamboo_composition_comparison/condition_k20_r0p5/G2-b-Q"
    DEFAULT_SOURCE_DECODER="run/generation_bamboo_composition_comparison/G2-b-Q/train_decoder"
    case "$GROUP" in
      G0-MA-sigma) MA_TARGETS="sigma" ;;
      G0-MA-r) MA_TARGETS="r" ;;
      G0-MA-joint) MA_TARGETS="sigma_r" ;;
    esac
    ;;
  *)
    echo "Unknown group: $GROUP" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$DEFAULT_EXPERIMENT_ROOT}"
RUN="$EXPERIMENT_ROOT/$GROUP"
SOURCE_GROUP_DIR="${SOURCE_GROUP_DIR:-$DEFAULT_SOURCE_GROUP}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$SOURCE_GROUP_DIR/generator_data}"
SOURCE_DECODER_DIR="${SOURCE_DECODER_DIR:-$DEFAULT_SOURCE_DECODER}"
VOCAB_DIR="${VOCAB_DIR:-$SOURCE_GROUP_DIR/candidate_pool}"

TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-500}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
ANION_RATIO_TOLERANCE="${ANION_RATIO_TOLERANCE:-0.10}"
MECHANISM_AUX_WEIGHT="${MECHANISM_AUX_WEIGHT:-0.05}"
MECHANISM_AUX_HIDDEN_DIM="${MECHANISM_AUX_HIDDEN_DIM:-192}"

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
need_dir() { [[ -d "$1" ]] || { echo "Required directory is missing: $1" >&2; exit 1; }; }
for path in "$SOURCE_DATA_DIR/processed_data.pkl" "$SOURCE_DATA_DIR/synthetic_data.json" \
            "$SOURCE_DATA_DIR/dataset_config.yaml" "$SOURCE_DATA_DIR/synthetic_metadata.json" \
            "$VOCAB_DIR/solvents.json" "$VOCAB_DIR/salts.json" \
            "$ORACLE_CONFIG" "$ORACLE_CKPT/pretrain.pt" "$ORACLE_CKPT/optimal.pt"; do
  need_file "$path"
done
if (( ! RETRAIN_DECODER )); then
  need_file "$SOURCE_DECODER_DIR/optimal.pt"
fi
need_dir "$SOURCE_GROUP_DIR"

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

# A symlink makes the unchanged Q baseline visible to the common analyzer
# without copying its completed data or allowing this pipeline to modify it.
if [[ "$GROUP" != "G2-Q" && ! -e "$EXPERIMENT_ROOT/$BASELINE_GROUP" ]]; then
  ln -s "$(cd "$SOURCE_GROUP_DIR" && pwd)" "$EXPERIMENT_ROOT/$BASELINE_GROUP" \
    || [[ -e "$EXPERIMENT_ROOT/$BASELINE_GROUP" ]] \
    || { echo "could not create baseline link $EXPERIMENT_ROOT/$BASELINE_GROUP" >&2; exit 1; }
fi
if [[ ! -e "$RUN/candidate_pool" ]]; then
  ln -s "$(cd "$VOCAB_DIR" && pwd)" "$RUN/candidate_pool"
fi

if [[ "$GROUP" == "G2-Q" ]]; then
  # Do not duplicate or modify the fixed 30,000-row Q data.  The seed changes
  # only training/sampling randomness, which is precisely the replicate test.
  if [[ ! -e "$RUN/generator_data" ]]; then
    ln -s "$(cd "$SOURCE_DATA_DIR" && pwd)" "$RUN/generator_data"
    record_skip prepare_q_data "reused_fixed_source=$(cd "$SOURCE_DATA_DIR" && pwd)"
  elif [[ -f "$RUN/generator_data/processed_data.pkl" ]]; then
    record_skip prepare_q_data existing_shared_source
  else
    echo "Incomplete Q replicate data path: $RUN/generator_data" >&2
    exit 1
  fi
elif [[ -f "$RUN/generator_data/processed_data.pkl" \
      && -f "$RUN/generator_data/mechanism_auxiliary.json" ]]; then
  record_skip prepare_mechanism_auxiliary_data existing_complete_output
elif [[ -e "$RUN/generator_data" ]]; then
  echo "Incomplete MA data output: $RUN/generator_data. It must contain " \
       "processed_data.pkl and mechanism_auxiliary.json before resuming." >&2
  exit 1
else
  run_stage prepare_mechanism_auxiliary_data python3 -u \
    scripts/generation/prepare_mechanism_auxiliary_data.py \
    --source-data-dir "$SOURCE_DATA_DIR" --output-dir "$RUN/generator_data" \
    --mode "$PREP_MODE" --auxiliary-targets "$MA_TARGETS" --seed "$SEED"
fi

if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
  if (( RETRAIN_DECODER )); then
    run_stage write_train_configs python3 -u \
      scripts/generation/write_generator_train_configs.py \
      --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED"
  else
    run_stage write_train_configs python3 -u \
      scripts/generation/write_generator_train_configs.py \
      --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED" \
      --mechanism-auxiliary \
      --mechanism-auxiliary-weight "$MECHANISM_AUX_WEIGHT" \
      --mechanism-auxiliary-hidden-dim "$MECHANISM_AUX_HIDDEN_DIM" \
      --mechanism-auxiliary-targets "$MA_TARGETS"
  fi
else
  record_skip write_train_configs existing_output
fi

# The decoder is retrained only for a new Q replicate.  MA and shuffle then
# share that same-seed Q decoder, isolating the auxiliary-label intervention.
if (( RETRAIN_DECODER )); then
  train_or_resume train_decoder "$RUN/train_decoder" "$RUN/configs/decoder.yaml"
elif [[ ! -e "$RUN/train_decoder" ]]; then
  ln -s "$(cd "$SOURCE_DECODER_DIR" && pwd)" "$RUN/train_decoder"
  record_skip shared_decoder "reused_from=$(cd "$SOURCE_DECODER_DIR" && pwd)"
elif [[ -f "$RUN/train_decoder/optimal.pt" ]]; then
  record_skip shared_decoder existing_link_or_checkpoint
else
  echo "Incomplete shared decoder path: $RUN/train_decoder" >&2
  exit 1
fi

train_or_resume train_diffusion "$RUN/train_diffusion" "$RUN/configs/diffusion.yaml"

if [[ ! -f "$RUN/generated/generated_formulas.json" ]]; then
  run_stage sample_and_score python3 -u scripts/generation/run_conditioned_generation.py \
    --diffusion-config "$RUN/configs/diffusion.yaml" \
    --diffusion-checkpoint "$RUN/train_diffusion/optimal.pt" \
    --decoder-config "$RUN/configs/decoder.yaml" \
    --decoder-checkpoint "$RUN/train_decoder/optimal.pt" \
    --solvents "$VOCAB_DIR/solvents.json" --salts "$VOCAB_DIR/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output "$RUN/generated/generated_formulas.json" \
    --target-conductivity "$TARGET_CONDUCTIVITY" --anion-ratio "$ANION_RATIO" \
    --anion-ratio-min 0.0 --anion-ratio-max 1.0 \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED" \
    --selection-policy joint_target --conductivity-tolerance 2.0 \
    --anion-ratio-tolerance "$ANION_RATIO_TOLERANCE" --joint-anion-weight 1.0
else
  record_skip sample_and_score existing_output
fi

if [[ "$GROUP" == G0-* ]]; then
  ANALYSIS_GROUPS=(G0-Q)
  for arm in G0-MA G0-MA-shuffle G0-MA-sigma G0-MA-r G0-MA-joint; do
    [[ -f "$EXPERIMENT_ROOT/$arm/generated/generated_formulas.json" ]] && ANALYSIS_GROUPS+=("$arm")
  done
else
  ANALYSIS_GROUPS=(G2-Q)
  for arm in G2-MA G2-MA-shuffle G2-MA-sigma G2-MA-r G2-MA-joint; do
    [[ -f "$EXPERIMENT_ROOT/$arm/generated/generated_formulas.json" ]] && ANALYSIS_GROUPS+=("$arm")
  done
fi
command -v flock >/dev/null || { echo "flock is required for analysis-write safety" >&2; exit 1; }
run_stage analyze_generation flock "$EXPERIMENT_ROOT/.analysis.lock" \
  python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$EXPERIMENT_ROOT" --groups "${ANALYSIS_GROUPS[@]}" \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100

printf 'Finished %s. Raw results: %s\n' "$GROUP" "$RUN/generated/generated_formulas.json"
