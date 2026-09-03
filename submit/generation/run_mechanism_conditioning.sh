#!/usr/bin/env bash
# Controlled explicit-mechanism diffusion experiment.
#
# G2-M and G2-M-shuffle reuse the *same existing* 30,000-row G2-Q synthetic
# formulas.  They do not regenerate formulas or call the frozen oracle again:
#   G2-M         retains each formula's [I_sigma, I_r] pair;
#   G2-M-shuffle block-permutes that complete pair only within the same
#                (sigma bin, r bin, solvent-count cohort).
#
# G0-M is the corresponding fixed-Bamboo-vocabulary arm, but it uses the
# current project oracle and a newly trained 4-D diffusion model.  It is not
# the released two-condition Bamboo checkpoint; the name refers only to the
# fixed Bamboo vocabulary baseline.
set -euo pipefail

GROUP="${1:?usage: $0 G2-M|G2-M-shuffle|G0-M}"
case "$GROUP" in
  G2-M)
    PREP_MODE="true"
    BASELINE_GROUP="G2-Q"
    DEFAULT_SOURCE_GROUP="run/generation_interaction_coverage_comparison/G2-Q"
    DEFAULT_SOURCE_DECODER="run/generation_interaction_coverage_comparison/G2-Q/train_decoder"
    ;;
  G2-M-shuffle)
    PREP_MODE="shuffle_within_cell"
    BASELINE_GROUP="G2-Q"
    DEFAULT_SOURCE_GROUP="run/generation_interaction_coverage_comparison/G2-Q"
    DEFAULT_SOURCE_DECODER="run/generation_interaction_coverage_comparison/G2-Q/train_decoder"
    ;;
  G0-M)
    PREP_MODE="true"
    BASELINE_GROUP="G0-Q"
    # The synthetic Q data and vocabulary were retained under the condition
    # folder after the original fixed-Bamboo-vocabulary comparison.
    DEFAULT_SOURCE_GROUP="run/generation_bamboo_composition_comparison/condition_k20_r0p5/G2-b-Q"
    DEFAULT_SOURCE_DECODER="run/generation_bamboo_composition_comparison/G2-b-Q/train_decoder"
    ;;
  *)
    echo "Unknown group: $GROUP (expected G2-M|G2-M-shuffle|G0-M)" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Keep these runs isolated from the original two-condition outputs.  The
# source Q group is linked in read-only as the property-only reference.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_mechanism_conditioning_comparison}"
RUN="$EXPERIMENT_ROOT/$GROUP"
SOURCE_GROUP_DIR="${SOURCE_GROUP_DIR:-$DEFAULT_SOURCE_GROUP}"
SOURCE_DATA_DIR="${SOURCE_DATA_DIR:-$SOURCE_GROUP_DIR/generator_data}"
SOURCE_DECODER_DIR="${SOURCE_DECODER_DIR:-$DEFAULT_SOURCE_DECODER}"
VOCAB_DIR="${VOCAB_DIR:-$SOURCE_GROUP_DIR/candidate_pool}"

TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-500}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
SEED="${SEED:-123}"
TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
MECHANISM_PRIOR_NEIGHBORS="${MECHANISM_PRIOR_NEIGHBORS:-256}"
ANION_RATIO_TOLERANCE="${ANION_RATIO_TOLERANCE:-0.10}"

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
need_dir() { [[ -d "$1" ]] || { echo "Required directory is missing: $1" >&2; exit 1; }; }
for path in "$SOURCE_DATA_DIR/processed_data.pkl" "$SOURCE_DATA_DIR/synthetic_data.json" \
            "$SOURCE_DATA_DIR/dataset_config.yaml" "$SOURCE_DATA_DIR/synthetic_metadata.json" \
            "$VOCAB_DIR/solvents.json" "$VOCAB_DIR/salts.json" \
            "$SOURCE_DECODER_DIR/optimal.pt" "$ORACLE_CONFIG" \
            "$ORACLE_CKPT/pretrain.pt" "$ORACLE_CKPT/optimal.pt"; do
  need_file "$path"
done
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

# Make the property-only baseline visible to the common analyzer without
# copying or modifying its finished outputs.
if [[ ! -e "$EXPERIMENT_ROOT/$BASELINE_GROUP" ]]; then
  # Two M arms may be submitted at the same time.  A second ln can lose the
  # harmless race; accept that only if the baseline appeared meanwhile.
  ln -s "$(cd "$SOURCE_GROUP_DIR" && pwd)" "$EXPERIMENT_ROOT/$BASELINE_GROUP" \
    || [[ -e "$EXPERIMENT_ROOT/$BASELINE_GROUP" ]] \
    || { echo "could not create baseline link $EXPERIMENT_ROOT/$BASELINE_GROUP" >&2; exit 1; }
fi

if [[ ! -e "$RUN/candidate_pool" ]]; then
  ln -s "$(cd "$VOCAB_DIR" && pwd)" "$RUN/candidate_pool"
fi

if [[ -f "$RUN/generator_data/processed_data.pkl" \
      && -f "$RUN/generator_data/mechanism_prior.pt" \
      && -f "$RUN/generator_data/mechanism_conditioning.json" ]]; then
  record_skip prepare_mechanism_data existing_complete_output
elif [[ -e "$RUN/generator_data" ]]; then
  echo "Incomplete mechanism-data output: $RUN/generator_data. " \
       "It must contain processed_data.pkl, mechanism_prior.pt, and " \
       "mechanism_conditioning.json before the run can resume." >&2
  exit 1
else
  run_stage prepare_mechanism_data python3 -u \
    scripts/generation/prepare_mechanism_conditioned_data.py \
    --source-data-dir "$SOURCE_DATA_DIR" --output-dir "$RUN/generator_data" \
    --mode "$PREP_MODE" --seed "$SEED"
fi

if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
  run_stage write_train_configs python3 -u scripts/generation/write_generator_train_configs.py \
    --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED" \
    --mechanism-conditioning
else
  record_skip write_train_configs existing_output
fi

# Decoder training does not consume sigma, r, or mechanism labels.  Reusing
# the same Q decoder removes an otherwise irrelevant stochastic difference;
# the diffusion model is the sole learned intervention in M versus M-shuffle.
if [[ ! -e "$RUN/train_decoder" ]]; then
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
    --diffusion-config "$RUN/configs/diffusion.yaml" --diffusion-checkpoint "$RUN/train_diffusion/optimal.pt" \
    --decoder-config "$RUN/configs/decoder.yaml" --decoder-checkpoint "$RUN/train_decoder/optimal.pt" \
    --solvents "$VOCAB_DIR/solvents.json" --salts "$VOCAB_DIR/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --mechanism-prior "$RUN/generator_data/mechanism_prior.pt" \
    --mechanism-prior-neighbors "$MECHANISM_PRIOR_NEIGHBORS" \
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

if [[ "$GROUP" == G0-M ]]; then
  ANALYSIS_GROUPS=(G0-Q G0-M)
else
  ANALYSIS_GROUPS=(G2-Q)
  for arm in G2-M G2-M-shuffle; do
    [[ -f "$EXPERIMENT_ROOT/$arm/generated/generated_formulas.json" ]] && ANALYSIS_GROUPS+=("$arm")
  done
fi
command -v flock >/dev/null || { echo "flock is required for analysis-write safety" >&2; exit 1; }
run_stage analyze_generation flock "$EXPERIMENT_ROOT/.analysis.lock" \
  python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$EXPERIMENT_ROOT" --groups "${ANALYSIS_GROUPS[@]}" \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100

printf 'Finished %s. Raw results: %s\n' "$GROUP" "$RUN/generated/generated_formulas.json"
