#!/usr/bin/env bash
# One Bamboo molecular-pretraining-based controlled generation replicate.
#
# Usage:
#   bash submit/generation/run_bamboo_pretrain_mechanism_auxiliary_replicate.sh \
#     G1-MA|G2-Q|G2-MA|G2-MA-shuffle 123|456|789
#
# G1-MA and G2-MA each train their own decoder and MA diffusion directly.
# Optional G2-Q is an independent control. G2-MA-shuffle shares G2-MA's decoder.
set -Eeuo pipefail
trap 'status=$?; echo "ERROR: ${GROUP:-unknown} seed ${SEED:-unknown}; line $LINENO; exit=$status; command: $BASH_COMMAND" >&2; exit "$status"' ERR

GROUP="${1:?usage: $0 G1-MA|G2-Q|G2-MA|G2-MA-shuffle SEED}"
SEED="${2:?usage: $0 G1-MA|G2-Q|G2-MA|G2-MA-shuffle SEED}"
case "$GROUP" in G1-MA|G2-Q|G2-MA|G2-MA-shuffle) ;; *)
  echo "Unknown group: $GROUP" >&2; exit 2;; esac
[[ "$SEED" =~ ^(123|456|789)$ ]] || {
  echo "SEED must be one of 123, 456, 789: $SEED" >&2; exit 2;
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-generation-analysis}"
mkdir -p "$MPLCONFIGDIR"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_bamboo_pretrain_electrolyte_v2_comparison}"
# Strict molecular pretraining source. No released-vocabulary additions.
MONO_DATA="${MONO_DATA:-$ROOT/../bamboo_mixer/run/mono_data.json}"
BAMBOO_CATALOG="$EXPERIMENT_ROOT/common/bamboo_pretrain_property_catalog.json"
G1_SHARED="$EXPERIMENT_ROOT/shared_assets/G1_random_bamboo_pretrain"
G2_SHARED="$EXPERIMENT_ROOT/shared_assets/G2_guided_bamboo_pretrain"
G1_ROOT="$EXPERIMENT_ROOT/G1-comp"
G2_ROOT="$EXPERIMENT_ROOT/G2-comp"

POOL_SIZE="${POOL_SIZE:-62}"
SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
OVERSAMPLE_FACTOR="${OVERSAMPLE_FACTOR:-2.0}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-500}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
MECHANISM_AUX_WEIGHT="${MECHANISM_AUX_WEIGHT:-0.05}"
MECHANISM_AUX_HIDDEN_DIM="${MECHANISM_AUX_HIDDEN_DIM:-192}"
G1_VOCAB_SELECTION_SEED="${G1_VOCAB_SELECTION_SEED:-20260908}"
G2_VOCAB_SELECTION_SEED="${G2_VOCAB_SELECTION_SEED:-20260909}"
# Optional experimental constraints only; no task-independent property cutoffs.
# Both arms receive exactly the same explicitly requested gates.
COMMON_MINIMUM_DC="${COMMON_MINIMUM_DC:-}"
COMMON_MAXIMUM_VIS="${COMMON_MAXIMUM_VIS:-}"
COMMON_GATE_ARGS=()
if [[ -n "$COMMON_MINIMUM_DC" ]]; then
  COMMON_GATE_ARGS+=(--minimum-dc "$COMMON_MINIMUM_DC")
fi
if [[ -n "$COMMON_MAXIMUM_VIS" ]]; then
  COMMON_GATE_ARGS+=(--maximum-vis "$COMMON_MAXIMUM_VIS")
fi
# Within that shared universe, G2 prioritizes dielectric constant and
# viscosity over melting point.  These are ranking weights, not model labels.
G2_DC_SCORE_WEIGHT="${G2_DC_SCORE_WEIGHT:-3.0}"
G2_VIS_SCORE_WEIGHT="${G2_VIS_SCORE_WEIGHT:-2.0}"
G2_TM_SCORE_WEIGHT="${G2_TM_SCORE_WEIGHT:-0.5}"
# Keep the aggregate fragment-based anion tendency balanced, matching the
# established non-GDB G2 protocol for a controlled comparison.
G2_ANION_GUIDED_WEIGHT="${G2_ANION_GUIDED_WEIGHT:-0.35}"

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
BAMBOO_SALTS="${BAMBOO_SALTS:-$ROOT/emb_dict/salts.json}"
PROPERTY_CONSTRAINTS="${PROPERTY_CONSTRAINTS:-run/property_fragment_attribution_v2/generation_design_constraints/generation_property_constraints.csv}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
need_dir() { [[ -d "$1" ]] || { echo "Required directory is missing: $1" >&2; exit 1; }; }
source_data_ready() {
  local data_dir="$1"
  [[ -s "$data_dir/processed_data.pkl" && -s "$data_dir/synthetic_data.json" &&
     -s "$data_dir/dataset_config.yaml" && -s "$data_dir/synthetic_metadata.json" ]]
}
auxiliary_data_ready() {
  local data_dir="$1"
  source_data_ready "$data_dir" && [[ -s "$data_dir/mechanism_auxiliary.json" ]]
}
for path in "$MONO_DATA" "$ORACLE_CONFIG" "$ORACLE_CKPT/pretrain.pt" \
            "$ORACLE_CKPT/optimal.pt" "$MONO_CKPT" "$BAMBOO_SOLVENTS" \
            "$BAMBOO_SALTS"; do need_file "$path"; done
if [[ -n "${CONDITIONAL_TASK:-}" ]]; then
  need_file "$CONDITIONAL_TASK"
else
  need_file "$PROPERTY_CONSTRAINTS"
fi
command -v flock >/dev/null || { echo "flock is required for shared-asset safety" >&2; exit 1; }

link_shared() {
  local source="$1" destination="$2" source_abs
  source_abs="$(realpath "$source")"
  if [[ -L "$destination" ]]; then
    [[ "$(realpath "$destination")" == "$source_abs" ]] || {
      echo "Existing link points elsewhere: $destination" >&2; exit 1;
    }
  elif [[ -e "$destination" ]]; then
    echo "Refusing to replace non-link path: $destination" >&2; exit 1
  else
    ln -s "$source_abs" "$destination"
  fi
}

run_stage() {
  local stage="$1" timing="$2" start end start_utc end_utc
  shift 2
  start="$(date +%s)"; "$@"; end="$(date +%s)"
  start_utc="$(date -u -d "@$start" +%FT%TZ)"
  end_utc="$(date -u -d "@$end" +%FT%TZ)"
  printf '%s,%s,%s,%s,%s\n' "$stage" "$start_utc" "$end_utc" \
    "$((end - start))" executed >> "$timing"
}
record_skip() {
  local timing="$1" stage="$2" detail="$3"
  printf '%s,%s,%s,%s,%s\n' "$stage" "$(date -u +%FT%TZ)" \
    "$(date -u +%FT%TZ)" 0 "$detail" >> "$timing"
}
train_or_resume() {
  local stage="$1" timing="$2" work="$3" config="$4"
  if [[ -f "$work/FINISHED" ]]; then
    record_skip "$timing" "$stage" finished
  elif compgen -G "$work/ckpt/ckpt_epoch_*.pt" >/dev/null; then
    run_stage "$stage" "$timing" python3 -u scripts/train/train.py \
      --task formula --conf "$config" --restart
  elif [[ -f "$work/optimal.pt" ]]; then
    echo "Incomplete $stage: optimal.pt exists without FINISHED: $work" >&2; exit 1
  elif [[ -d "$work" ]]; then
    run_stage "$stage" "$timing" python3 -u scripts/train/train.py \
      --task formula --conf "$config" --restart
  else
    run_stage "$stage" "$timing" python3 -u scripts/train/train.py \
      --task formula --conf "$config"
  fi
}

prepare_g1_shared() {
  mkdir -p "$G1_SHARED"
  echo "Waiting for G1 shared-data lock: $G1_SHARED"
  exec 9>"$G1_SHARED/.prepare.lock"; flock 9
  echo "Checking G1 shared data"
  local pool="$G1_SHARED/candidate_pool" source="$G1_SHARED/generator_data_source"
  local aux="$G1_SHARED/generator_data_ma_true" cache="$G1_SHARED/oracle_scored_candidates"
  if [[ ! -f "$pool/solvents.json" ]]; then
    python3 -u scripts/generation/build_candidate_pool.py --mode random \
      --output-dir "$pool" --baseline-solvents "$BAMBOO_SOLVENTS" --salts "$BAMBOO_SALTS" \
      --precomputed-candidate-source "$BAMBOO_CATALOG" --pool-size "$POOL_SIZE" \
      "${COMMON_GATE_ARGS[@]}" \
      --seed "$G1_VOCAB_SELECTION_SEED"
  fi
  if [[ ! -f "$pool/solv_embs.pt" ]]; then
    python3 -u scripts/generation/build_generator_vocab_embeddings.py \
      --solvents "$pool/solvents.json" --salts "$pool/salts.json" \
      --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$pool"
  fi
  if ! source_data_ready "$source"; then
    mkdir -p "$cache"
    python3 -u scripts/generation/build_synthetic_generator_data.py \
      --solvents "$pool/solvents.json" --salts "$pool/salts.json" \
      --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
      --output-dir "$source" --candidate-cache-dir "$cache" \
      --num-formulas "$SYNTHETIC_FORMULAS" --oversample-factor "$OVERSAMPLE_FACTOR" \
      --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
      --label-policy sanitize_keep_all --solvent-count-policy bamboo_empirical_1_to_6 \
      --salt-mixture-mode bamboo_fixed_two_salt --salt-mixture-bins 5 \
      --joint-balance-strategy none --min-nonempty-anion-bins 1 \
      --anion-ratio-min 0 --anion-ratio-max 1 --mechanism-selection matched_random --seed 123
  fi
  if ! auxiliary_data_ready "$aux"; then
    python3 -u scripts/generation/prepare_mechanism_auxiliary_data.py \
      --source-data-dir "$source" --output-dir "$aux" --mode true \
      --auxiliary-targets sigma_r --seed 123
  fi
  echo "G1 shared data ready (existing complete data reused)"
  flock -u 9
}

prepare_g2_shared() {
  mkdir -p "$G2_SHARED"
  echo "Waiting for G2 shared-data lock: $G2_SHARED"
  exec 8>"$G2_SHARED/.prepare.lock"; flock 8
  echo "Checking G2 shared data"
  local pool="$G2_SHARED/candidate_pool" source="$G2_SHARED/generator_data_q"
  local aux="$G2_SHARED/generator_data_ma_true" cache="$G2_SHARED/oracle_scored_candidates"
  if [[ -n "${CONDITIONAL_TASK:-}" ]]; then
    python3 -u scripts/generation/build_candidate_pool.py --mode conditional \
      --conditional-task "$CONDITIONAL_TASK" --output-dir "$pool" \
      --baseline-solvents "$BAMBOO_SOLVENTS" --salts "$BAMBOO_SALTS" \
      --precomputed-candidate-source "$BAMBOO_CATALOG" \
      "${COMMON_GATE_ARGS[@]}" \
      --pool-size "$POOL_SIZE" --seed "$G2_VOCAB_SELECTION_SEED"
  elif [[ ! -f "$pool/solvents.json" ]]; then
    python3 -u scripts/generation/build_candidate_pool.py --mode guided \
      --output-dir "$pool" --baseline-solvents "$BAMBOO_SOLVENTS" --salts "$BAMBOO_SALTS" \
      --precomputed-candidate-source "$BAMBOO_CATALOG" --property-constraints "$PROPERTY_CONSTRAINTS" \
      --guided-selection-strategy coverage_v1 --guided-property-weight 1.0 \
      --guided-dc-score-weight "$G2_DC_SCORE_WEIGHT" \
      --guided-vis-score-weight "$G2_VIS_SCORE_WEIGHT" \
      --guided-tm-score-weight "$G2_TM_SCORE_WEIGHT" \
      --guided-fragment-weight 1.0 --guided-family-coverage-weight 0.7 \
      --guided-scaffold-coverage-weight 0.25 --guided-redundancy-weight 1.0 \
      --anion-guided-weight "$G2_ANION_GUIDED_WEIGHT" \
      "${COMMON_GATE_ARGS[@]}" \
      --pool-size "$POOL_SIZE" --seed "$G2_VOCAB_SELECTION_SEED"
  fi
  if [[ ! -f "$pool/solv_embs.pt" ]]; then
    python3 -u scripts/generation/build_generator_vocab_embeddings.py \
      --solvents "$pool/solvents.json" --salts "$pool/salts.json" \
      --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$pool"
  fi
  if ! source_data_ready "$source"; then
    mkdir -p "$cache"
    python3 -u scripts/generation/build_synthetic_generator_data.py \
      --solvents "$pool/solvents.json" --salts "$pool/salts.json" \
      --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
      --output-dir "$source" --candidate-cache-dir "$cache" \
      --num-formulas "$SYNTHETIC_FORMULAS" --oversample-factor "$OVERSAMPLE_FACTOR" \
      --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
      --label-policy sanitize_keep_all --solvent-count-policy bamboo_empirical_1_to_6 \
      --salt-mixture-mode bamboo_fixed_two_salt --salt-mixture-bins 5 \
      --joint-balance-strategy none --min-nonempty-anion-bins 1 \
      --anion-ratio-min 0 --anion-ratio-max 1 --mechanism-selection matched_random --seed 123
  fi
  if ! auxiliary_data_ready "$aux"; then
    python3 -u scripts/generation/prepare_mechanism_auxiliary_data.py \
      --source-data-dir "$source" --output-dir "$aux" --mode true \
      --auxiliary-targets sigma_r --seed 123
  fi
  echo "G2 shared data ready (existing complete data reused)"
  flock -u 8
}

# Prevent duplicate submissions from writing the same training checkpoints.
mkdir -p "$EXPERIMENT_ROOT/job_locks"
exec 5>"$EXPERIMENT_ROOT/job_locks/${GROUP}_seed${SEED}.lock"
flock -n 5 || { echo "Job already running: $GROUP seed $SEED" >&2; exit 1; }

# Both groups use one shared, target-independent pretraining catalogue.
mkdir -p "$EXPERIMENT_ROOT/common"
exec 6>"$EXPERIMENT_ROOT/common/.catalog.lock"; flock 6
if [[ ! -f "$BAMBOO_CATALOG" ]]; then
  python3 scripts/generation/build_bamboo_pretrain_catalog.py \
    --mono-data "$MONO_DATA" --model-config "$ORACLE_CONFIG" \
    --mono-checkpoint "$MONO_CKPT" --output "$BAMBOO_CATALOG"
fi
flock -u 6

if [[ "$GROUP" == "G1-MA" ]]; then
  mkdir -p "$G1_ROOT"
  prepare_g1_shared
  SEED_ROOT="$G1_ROOT/G1_seed_$SEED"; RUN="$SEED_ROOT/G1-MA"
  POOL="$G1_SHARED/candidate_pool"; SOURCE="$G1_SHARED/generator_data_source"
  DATA="$G1_SHARED/generator_data_ma_true"; RETRAIN_DECODER=1; AUXILIARY=1
else
  mkdir -p "$G2_ROOT"
  prepare_g2_shared
  SEED_ROOT="$G2_ROOT/G2_seed_$SEED"; RUN="$SEED_ROOT/$GROUP"
  POOL="$G2_SHARED/candidate_pool"; SOURCE="$G2_SHARED/generator_data_q"
  Q_RUN="$SEED_ROOT/G2-Q"
  if [[ "$GROUP" == "G2-Q" ]]; then
    DATA="$SOURCE"; RETRAIN_DECODER=1; AUXILIARY=0
  elif [[ "$GROUP" == "G2-MA" ]]; then
    DATA="$G2_SHARED/generator_data_ma_true"; RETRAIN_DECODER=1; AUXILIARY=1
  else
    # Shuffle isolates auxiliary labels, sharing the same-seed factual decoder.
    Q_RUN="$SEED_ROOT/G2-MA"
    need_file "$Q_RUN/train_decoder/FINISHED"
    need_file "$Q_RUN/train_decoder/optimal.pt"
    need_file "$Q_RUN/generated/generated_formulas.json"
    DATA="$RUN/generator_data"; RETRAIN_DECODER=0; AUXILIARY=1
  fi
fi

mkdir -p "$RUN"
TIMING="$RUN/timings.csv"
[[ -f "$TIMING" ]] || printf 'stage,start_utc,end_utc,elapsed_seconds,detail\n' > "$TIMING"
link_shared "$POOL" "$RUN/candidate_pool"
link_shared "$SOURCE" "$RUN/generator_data_source"

if [[ "$GROUP" == "G2-MA-shuffle" ]]; then
  if [[ ! -e "$DATA" ]]; then
    run_stage prepare_shuffled_auxiliary_data "$TIMING" python3 -u \
      scripts/generation/prepare_mechanism_auxiliary_data.py --source-data-dir "$SOURCE" \
      --output-dir "$DATA" --mode shuffle_within_cell --auxiliary-targets sigma_r --seed "$SEED"
  elif [[ -f "$DATA/processed_data.pkl" && -f "$DATA/mechanism_auxiliary.json" ]]; then
    record_skip "$TIMING" prepare_shuffled_auxiliary_data existing_complete_output
  else
    echo "Incomplete shuffled MA data: $DATA" >&2; exit 1
  fi
else
  link_shared "$DATA" "$RUN/generator_data"
  record_skip "$TIMING" prepare_shared_data "reused=$(realpath "$DATA")"
fi

if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
  if (( AUXILIARY )); then
    run_stage write_train_configs "$TIMING" python3 -u scripts/generation/write_generator_train_configs.py \
      --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED" \
      --mechanism-auxiliary --mechanism-auxiliary-weight "$MECHANISM_AUX_WEIGHT" \
      --mechanism-auxiliary-hidden-dim "$MECHANISM_AUX_HIDDEN_DIM" \
      --mechanism-auxiliary-targets sigma_r
  else
    run_stage write_train_configs "$TIMING" python3 -u scripts/generation/write_generator_train_configs.py \
      --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED"
  fi
else
  record_skip "$TIMING" write_train_configs existing_output
fi

if (( RETRAIN_DECODER )); then
  train_or_resume train_decoder "$TIMING" "$RUN/train_decoder" "$RUN/configs/decoder.yaml"
else
  link_shared "$Q_RUN/train_decoder" "$RUN/train_decoder"
  record_skip "$TIMING" shared_q_decoder "reused=$(realpath "$Q_RUN/train_decoder")"
fi
train_or_resume train_diffusion "$TIMING" "$RUN/train_diffusion" "$RUN/configs/diffusion.yaml"

if [[ ! -f "$RUN/generated/generated_formulas.json" ]]; then
  run_stage sample_and_score "$TIMING" python3 -u scripts/generation/run_conditioned_generation.py \
    --diffusion-config "$RUN/configs/diffusion.yaml" --diffusion-checkpoint "$RUN/train_diffusion/optimal.pt" \
    --decoder-config "$RUN/configs/decoder.yaml" --decoder-checkpoint "$RUN/train_decoder/optimal.pt" \
    --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output "$RUN/generated/generated_formulas.json" --target-conductivity "$TARGET_CONDUCTIVITY" \
    --anion-ratio "$ANION_RATIO" --anion-ratio-min 0 --anion-ratio-max 1 \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED" \
    --selection-policy joint_target --conductivity-tolerance 2.0 \
    --anion-ratio-tolerance 0.10 --joint-anion-weight 1.0
else
  record_skip "$TIMING" sample_and_score existing_output
fi

if [[ "$GROUP" == "G1-MA" ]]; then
  ANALYSIS_GROUPS=(G1-MA)
else
  ANALYSIS_GROUPS=()
  for arm in G2-Q G2-MA G2-MA-shuffle; do
    [[ -f "$SEED_ROOT/$arm/generated/generated_formulas.json" ]] && ANALYSIS_GROUPS+=("$arm")
  done
fi
exec 7>"$SEED_ROOT/.analysis.lock"; flock 7
run_stage analyze_seed "$TIMING" python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$SEED_ROOT" --groups "${ANALYSIS_GROUPS[@]}" \
  --target-conductivity "$TARGET_CONDUCTIVITY" --selection-budget 100 \
  --joint-sigma-scale 2 --joint-anion-ratio-scale 0.1

echo "Finished $GROUP seed $SEED: $RUN/generated/generated_formulas.json"
