#!/usr/bin/env bash
# Three-seed G1-MA experiment for the candidate-pool comparison.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 MECHANISM_AUX_WEIGHT=0.05 \
#     bash submit/generation/run_G1_MA_replicate.sh 123
#
# G1 only compares a random candidate vocabulary against the fixed-Bamboo
# (G0) and explanation-screened (G2) vocabularies under the same factual
# interaction-residual auxiliary supervision.  It deliberately does not train
# or report a G1-Q diffusion baseline or a G1-MA-shuffle arm.
#
# A random 62-solvent vocabulary and 30,000 frozen-oracle synthetic formulas
# are built once at seed 123.  All seeds reuse them; each seed independently
# trains its decoder and MA diffusion, so the repeat measures generator
# stochasticity without redrawing the random vocabulary.
set -euo pipefail

SEED="${1:?usage: $0 SEED (for example 123, 456, or 789)}"
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be non-negative: $SEED" >&2; exit 2; }

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-generation-analysis}"
mkdir -p "$MPLCONFIGDIR"

COMP_ROOT="${EXPERIMENT_ROOT:-run/generation_mechanism_auxiliary_comparison/G1-comp}"
SEED_ROOT="$COMP_ROOT/G1_seed_${SEED}"
RUN="$SEED_ROOT/G1-MA"
CANONICAL="$COMP_ROOT/G1_seed_123/G1-MA"
CANONICAL_POOL="$CANONICAL/candidate_pool"
CANONICAL_DATA="$CANONICAL/generator_data_source"
CANONICAL_CACHE="$CANONICAL/oracle_scored_candidates"

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

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/anion_ratio_property_fragment_explanation/ckpt_formula_ft}"
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
BAMBOO_SALTS="${BAMBOO_SALTS:-$ROOT/emb_dict/salts.json}"
SOLVENT_LIKE_POOL="${SOLVENT_LIKE_POOL:-$ROOT/emb_dict/solvent_like_candidates.json}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
for path in "$ORACLE_CONFIG" "$ORACLE_CKPT/optimal.pt" "$ORACLE_CKPT/pretrain.pt" \
            "$MONO_CKPT" "$BAMBOO_SOLVENTS" "$BAMBOO_SALTS" "$SOLVENT_LIKE_POOL"; do
  need_file "$path"
done

mkdir -p "$RUN"
TIMING="$RUN/timings.csv"
[[ -f "$TIMING" ]] || printf 'stage,start_utc,end_utc,elapsed_seconds,detail\n' > "$TIMING"
record_skip() { printf '%s,%s,%s,%s,%s\n' "$1" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0 "$2" >> "$TIMING"; }
run_stage() {
  local stage="$1" start end start_utc end_utc
  shift; start="$(date +%s)"; "$@"; end="$(date +%s)"
  start_utc="$(date -u -d "@$start" +%FT%TZ)"; end_utc="$(date -u -d "@$end" +%FT%TZ)"
  printf '%s,%s,%s,%s,%s\n' "$stage" "$start_utc" "$end_utc" "$((end - start))" executed >> "$TIMING"
}
train_or_resume() {
  local stage="$1" work="$2" config="$3"
  if [[ -f "$work/FINISHED" ]]; then
    record_skip "$stage" finished
  elif [[ -f "$work/optimal.pt" ]]; then
    echo "Incomplete $stage: optimal.pt exists without FINISHED: $work" >&2; exit 1
  elif [[ -d "$work" ]]; then
    run_stage "$stage" python3 -u scripts/train/train.py --task formula --conf "$config" --restart
  else
    run_stage "$stage" python3 -u scripts/train/train.py --task formula --conf "$config"
  fi
}
prepare_canonical_assets() {
  local lock="$COMP_ROOT/.g1_ma_build.lock"
  mkdir -p "$COMP_ROOT"; exec 9>"$lock"; flock 9; mkdir -p "$CANONICAL_POOL"
  if [[ ! -f "$CANONICAL_POOL/solvents.json" ]]; then
    python3 -u scripts/generation/build_candidate_pool.py \
      --mode random --output-dir "$CANONICAL_POOL" \
      --baseline-solvents "$BAMBOO_SOLVENTS" --salts "$BAMBOO_SALTS" \
      --candidate-source "$SOLVENT_LIKE_POOL" --model-config "$ORACLE_CONFIG" \
      --mono-checkpoint "$MONO_CKPT" --pool-size "$POOL_SIZE" --seed 123
  fi
  if [[ ! -f "$CANONICAL_POOL/solv_embs.pt" ]]; then
    python3 -u scripts/generation/build_generator_vocab_embeddings.py \
      --solvents "$CANONICAL_POOL/solvents.json" --salts "$CANONICAL_POOL/salts.json" \
      --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$CANONICAL_POOL"
  fi
  if [[ ! -f "$CANONICAL_DATA/processed_data.pkl" ]]; then
    mkdir -p "$CANONICAL_CACHE"
    python3 -u scripts/generation/build_synthetic_generator_data.py \
      --solvents "$CANONICAL_POOL/solvents.json" --salts "$CANONICAL_POOL/salts.json" \
      --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
      --output-dir "$CANONICAL_DATA" --candidate-cache-dir "$CANONICAL_CACHE" \
      --num-formulas "$SYNTHETIC_FORMULAS" --oversample-factor "$OVERSAMPLE_FACTOR" \
      --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
      --label-policy sanitize_keep_all --solvent-count-policy bamboo_empirical_1_to_6 \
      --salt-mixture-mode bamboo_fixed_two_salt --salt-mixture-bins 5 \
      --joint-balance-strategy none --min-nonempty-anion-bins 1 \
      --anion-ratio-min 0.0 --anion-ratio-max 1.0 \
      --mechanism-selection matched_random --seed 123
  fi
  flock -u 9
}

if [[ "$SEED" == "123" ]]; then
  if [[ ! -f "$CANONICAL_DATA/processed_data.pkl" || ! -f "$CANONICAL_POOL/solv_embs.pt" ]]; then
    run_stage prepare_fixed_random_pool_and_synthetic_data prepare_canonical_assets
  else
    record_skip prepare_fixed_random_pool_and_synthetic_data existing_canonical_assets
  fi
else
  [[ -f "$CANONICAL_DATA/processed_data.pkl" && -f "$CANONICAL_POOL/solv_embs.pt" ]] || {
    echo "Run G1-MA seed 123 first to build the fixed random vocabulary and synthetic data." >&2; exit 1;
  }
fi
if [[ "$RUN" != "$CANONICAL" ]]; then
  [[ -e "$RUN/candidate_pool" ]] || ln -s ../../G1_seed_123/G1-MA/candidate_pool "$RUN/candidate_pool"
  [[ -e "$RUN/generator_data_source" ]] || ln -s ../../G1_seed_123/G1-MA/generator_data_source "$RUN/generator_data_source"
fi

if [[ -f "$RUN/generator_data/processed_data.pkl" && -f "$RUN/generator_data/mechanism_auxiliary.json" ]]; then
  record_skip prepare_mechanism_auxiliary_data existing_complete_output
elif [[ -e "$RUN/generator_data" ]]; then
  echo "Incomplete MA data output: $RUN/generator_data" >&2; exit 1
else
  run_stage prepare_mechanism_auxiliary_data python3 -u scripts/generation/prepare_mechanism_auxiliary_data.py \
    --source-data-dir "$RUN/generator_data_source" --output-dir "$RUN/generator_data" \
    --mode true --auxiliary-targets sigma_r --seed "$SEED"
fi
if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
  run_stage write_train_configs python3 -u scripts/generation/write_generator_train_configs.py \
    --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH" --seed "$SEED" \
    --mechanism-auxiliary --mechanism-auxiliary-weight "$MECHANISM_AUX_WEIGHT" \
    --mechanism-auxiliary-hidden-dim "$MECHANISM_AUX_HIDDEN_DIM" --mechanism-auxiliary-targets sigma_r
else
  record_skip write_train_configs existing_output
fi
train_or_resume train_decoder "$RUN/train_decoder" "$RUN/configs/decoder.yaml"
train_or_resume train_diffusion "$RUN/train_diffusion" "$RUN/configs/diffusion.yaml"
if [[ ! -f "$RUN/generated/generated_formulas.json" ]]; then
  run_stage sample_and_score python3 -u scripts/generation/run_conditioned_generation.py \
    --diffusion-config "$RUN/configs/diffusion.yaml" --diffusion-checkpoint "$RUN/train_diffusion/optimal.pt" \
    --decoder-config "$RUN/configs/decoder.yaml" --decoder-checkpoint "$RUN/train_decoder/optimal.pt" \
    --solvents "$RUN/candidate_pool/solvents.json" --salts "$RUN/candidate_pool/salts.json" \
    --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
    --output "$RUN/generated/generated_formulas.json" --target-conductivity "$TARGET_CONDUCTIVITY" \
    --anion-ratio "$ANION_RATIO" --anion-ratio-min 0.0 --anion-ratio-max 1.0 \
    --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
    --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED" \
    --selection-policy joint_target --conductivity-tolerance 2.0 --anion-ratio-tolerance 0.10 --joint-anion-weight 1.0
else
  record_skip sample_and_score existing_output
fi
run_stage analyze_seed python3 -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root "$SEED_ROOT" --groups G1-MA --target-conductivity "$TARGET_CONDUCTIVITY" \
  --selection-budget 100 --joint-sigma-scale 2 --joint-anion-ratio-scale 0.1
if [[ -f "$COMP_ROOT/G1_seed_123/G1-MA/generated/generated_formulas.json" \
      && -f "$COMP_ROOT/G1_seed_456/G1-MA/generated/generated_formulas.json" \
      && -f "$COMP_ROOT/G1_seed_789/G1-MA/generated/generated_formulas.json" ]]; then
  run_stage aggregate_three_seeds python3 -u scripts/generation/aggregate_generation_replicates.py \
    --replicate-root "$COMP_ROOT" --groups G1-MA --baseline G1-MA \
    --selection-budget 100 --joint-sigma-scale 2 --joint-anion-ratio-scale 0.1 \
    --report-title 'G1 随机候选池交互辅助生成器：多随机种子汇总'
fi
printf 'Finished G1-MA seed %s. Results: %s/generated/generated_formulas.json\n' "$SEED" "$RUN"
