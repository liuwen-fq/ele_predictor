#!/usr/bin/env bash
# Run one controlled G0/G1/G2/G3 formula-generation experiment locally.
#
# G0: released Bamboo generator checkpoint (no retraining)
# G1: random 62-member vocabulary drawn from the fixed solvent-like pool
# G2: soft-constraint 62-member vocabulary drawn from that same pool
# G3: reuse G2's generator and rank outputs by T-c robustness + explanation
#     consistency.  G3 deliberately does not retrain a generator.

set -euo pipefail

GROUP="${1:?usage: $0 G0|G1|G2|G3}"
case "$GROUP" in G0|G1|G2|G3) ;; *) echo "Unknown group: $GROUP" >&2; exit 2 ;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Override these variables in the submission command when a smaller smoke test
# or a larger final experiment is desired.  All non-G3 groups must use the
# same values for a fair comparison.
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_experiments}"
POOL_SIZE="${POOL_SIZE:-62}"
SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
SYNTHETIC_OVERSAMPLE_FACTOR="${SYNTHETIC_OVERSAMPLE_FACTOR:-1.5}"
SYNTHETIC_LABEL_POLICY="${SYNTHETIC_LABEL_POLICY:-strict}"
# The stratified mode balances *input salt-mixture fractions*.  The frozen
# oracle remains the source of the actual anion-ratio label, avoiding an
# unphysical condition/recipe mismatch.
SYNTHETIC_SALT_MIXTURE_MODE="${SYNTHETIC_SALT_MIXTURE_MODE:-random_1_to_2_salt}"
SYNTHETIC_SALT_MIXTURE_BINS="${SYNTHETIC_SALT_MIXTURE_BINS:-5}"
SYNTHETIC_SOLVENT_COUNT_POLICY="${SYNTHETIC_SOLVENT_COUNT_POLICY:-uniform_1_to_3}"
SYNTHETIC_JOINT_BALANCE_STRATEGY="${SYNTHETIC_JOINT_BALANCE_STRATEGY:-hierarchical_anion_then_conductivity}"
SYNTHETIC_MIN_NONEMPTY_ANION_BINS="${SYNTHETIC_MIN_NONEMPTY_ANION_BINS:-1}"
TRAIN_MAX_EPOCH="${TRAIN_MAX_EPOCH:-500}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"
SEED="${SEED:-123}"
# Keep vocabulary selection reproducible without coupling it to diffusion
# training/sampling randomness.  Set this explicitly for each G1 replicate.
CANDIDATE_POOL_SEED="${CANDIDATE_POOL_SEED:-$SEED}"
TARGET_CONDUCTIVITY="${TARGET_CONDUCTIVITY:-20.0}"
ANION_RATIO="${ANION_RATIO:-0.5}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
SELECTION_POLICY="${SELECTION_POLICY:-joint_target}"
SELECTION_CONDUCTIVITY_TOLERANCE="${SELECTION_CONDUCTIVITY_TOLERANCE:-2.0}"
SELECTION_ANION_RATIO_TOLERANCE="${SELECTION_ANION_RATIO_TOLERANCE:-0.10}"
SELECTION_JOINT_ANION_WEIGHT="${SELECTION_JOINT_ANION_WEIGHT:-1.0}"

ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/property_fragment_predict.yaml}"
ORACLE_CKPT="${ORACLE_CKPT:-run/property_fragment_explanation/ckpt_formula_ft}"
MONO_CONFIG="${MONO_CONFIG:-$ORACLE_CONFIG}"
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
# The old ele_predictor repository no longer contains emb_dict.  The clean
# mechanistic repository carries the same 62-solvent/2-salt Bamboo vocabulary
# and is the correct default for G1/G2 and the G0 vocabulary snapshot.
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
BAMBOO_SALTS="${BAMBOO_SALTS:-$ROOT/emb_dict/salts.json}"
BAMBOO_ROOT="${BAMBOO_ROOT:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer}"
BAMBOO_GENERATOR_DIR="${BAMBOO_GENERATOR_DIR:-$BAMBOO_ROOT/ckpts/generator}"
BAMBOO_GENERATE_CONFIG="${BAMBOO_GENERATE_CONFIG:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/config/test_results/generate_config.yaml}"
BAMBOO_SOLV_EMBS="${BAMBOO_SOLV_EMBS:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/solv_embs.pt}"
BAMBOO_SALT_EMBS="${BAMBOO_SALT_EMBS:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/salt_embs.pt}"
PROPERTY_CONSTRAINTS="${PROPERTY_CONSTRAINTS:-run/property_fragment_attribution_v2/generation_design_constraints/generation_property_constraints.csv}"
GUIDED_SELECTION_STRATEGY="${GUIDED_SELECTION_STRATEGY:-coverage_v1}"
GUIDED_PROPERTY_WEIGHT="${GUIDED_PROPERTY_WEIGHT:-1.0}"
GUIDED_FRAGMENT_WEIGHT="${GUIDED_FRAGMENT_WEIGHT:-1.0}"
GUIDED_FAMILY_COVERAGE_WEIGHT="${GUIDED_FAMILY_COVERAGE_WEIGHT:-0.7}"
GUIDED_SCAFFOLD_COVERAGE_WEIGHT="${GUIDED_SCAFFOLD_COVERAGE_WEIGHT:-0.25}"
GUIDED_REDUNDANCY_WEIGHT="${GUIDED_REDUNDANCY_WEIGHT:-1.0}"
# Built once by build_solvent_like_pool.py; G1 and G2 must share it.
SOLVENT_LIKE_POOL="${SOLVENT_LIKE_POOL:-$ROOT/emb_dict/solvent_like_candidates.json}"
SOLVENT_LIKE_AUDIT="${SOLVENT_LIKE_AUDIT:-$ROOT/emb_dict/solvent_like_candidates_audit.json}"
# Only controlled ablations with a documented fixed legacy vocabulary may set this.
ALLOW_FIXED_VOCAB="${ALLOW_FIXED_VOCAB:-0}"

RUN="$EXPERIMENT_ROOT/$GROUP"
mkdir -p "$RUN"
TIMING="$RUN/timings.csv"
if [[ ! -f "$TIMING" ]]; then
  printf 'stage,start_utc,end_utc,elapsed_seconds,detail\n' > "$TIMING"
fi

record_skip() {
  printf '%s,%s,%s,%s,%s\n' "$1" "$(date -u +%FT%TZ)" "$(date -u +%FT%TZ)" 0 "$2" >> "$TIMING"
}

run_stage() {
  local stage="$1"; shift
  local start end elapsed
  start="$(date +%s)"
  "$@"
  end="$(date +%s)"
  elapsed=$((end - start))
  printf '%s,%s,%s,%s,%s\n' "$stage" "$(date -u -d "@$start" +%FT%TZ)" \
    "$(date -u -d "@$end" +%FT%TZ)" "$elapsed" executed >> "$TIMING"
}

train_or_resume() {
  local stage="$1"
  local work="$2"
  local config="$3"
  local checkpoints=()
  shopt -s nullglob
  checkpoints=("$work"/ckpt/ckpt_epoch_*.pt)
  shopt -u nullglob

  if [[ -f "$work/FINISHED" ]]; then
    record_skip "$stage" "finished"
  elif (( ${#checkpoints[@]} > 0 )); then
    echo "Resuming $stage from the latest readable epoch checkpoint."
    run_stage "$stage" python3 -u scripts/train/train.py --task formula \
      --conf "$config" --restart
  elif [[ -f "$work/optimal.pt" ]]; then
    echo "Refusing to treat incomplete $stage as finished: $work/optimal.pt exists, but no FINISHED or epoch checkpoint." >&2
    echo "Restore an epoch checkpoint or remove only this incomplete work directory before restarting." >&2
    exit 1
  else
    run_stage "$stage" python3 -u scripts/train/train.py --task formula \
      --conf "$config"
  fi
}

need_file() {
  [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }
}
need_file "$ORACLE_CONFIG"; need_file "$ORACLE_CKPT/optimal.pt"
need_file "$ORACLE_CKPT/pretrain.pt"
need_file "$BAMBOO_SOLVENTS"; need_file "$BAMBOO_SALTS"

verify_existing_pool_source() {
  local metadata="$1/candidate_pool_metadata.json"
  if [[ "$ALLOW_FIXED_VOCAB" == "1" ]]; then
    return
  fi
  if [[ ! -f "$metadata" ]] || ! grep -Fq "\"candidate_source\": \"$SOLVENT_LIKE_POOL\"" "$metadata"; then
    echo "Existing vocabulary in $1 was not built from the configured solvent-like pool." >&2
    echo "Use a fresh EXPERIMENT_ROOT, or set ALLOW_FIXED_VOCAB=1 only for a documented ablation." >&2
    exit 1
  fi
}

if [[ "$GROUP" == "G0" ]]; then
  # This is intentionally the released Bamboo diffusion/decoder checkpoint.
  # The small candidate-pool file is only a local vocabulary snapshot used by
  # the shared report decoder; it does not train or alter the baseline model.
  need_file "$BAMBOO_GENERATE_CONFIG"
  need_file "$BAMBOO_GENERATOR_DIR/diffusion.pt"
  need_file "$BAMBOO_GENERATOR_DIR/decoder.pt"
  need_file "$BAMBOO_SOLV_EMBS"
  need_file "$BAMBOO_SALT_EMBS"
  POOL="$RUN/candidate_pool"
  G0_SOURCE="$RUN/generator_source.json"
  if [[ -f "$RUN/generated/generated_formulas.json" && ! -f "$G0_SOURCE" ]]; then
    echo "Existing G0 generated result predates the released-checkpoint pipeline." >&2
    echo "Move or remove only $RUN/generated before rerunning G0." >&2
    exit 1
  fi
  if [[ ! -f "$G0_SOURCE" ]]; then
    printf '{\n  "generator_source": "released_bamboo_mixer",\n  "generator_dir": "%s",\n  "generate_config": "%s"\n}\n' \
      "$BAMBOO_GENERATOR_DIR" "$BAMBOO_GENERATE_CONFIG" > "$G0_SOURCE"
  fi
  if [[ ! -f "$POOL/solvents.json" ]]; then
    run_stage vocabulary_snapshot python3 -u scripts/generation/build_candidate_pool.py \
      --mode baseline --output-dir "$POOL" --baseline-solvents "$BAMBOO_SOLVENTS" \
      --salts "$BAMBOO_SALTS" --pool-size "$POOL_SIZE" --seed "$SEED"
  else record_skip vocabulary_snapshot "existing_output"; fi
  record_skip synthetic_formula_oracle "not_used_released_Bamboo_generator"
  record_skip train_diffusion "not_used_released_Bamboo_checkpoint"
  record_skip train_decoder "not_used_released_Bamboo_checkpoint"
  DIFF_CONFIG="$BAMBOO_GENERATE_CONFIG"
  DEC_CONFIG="$BAMBOO_GENERATE_CONFIG"
  DIFF_CKPT="$BAMBOO_GENERATOR_DIR/diffusion.pt"
  DEC_CKPT="$BAMBOO_GENERATOR_DIR/decoder.pt"
  SOLVENTS="$POOL/solvents.json"
  SALTS="$POOL/salts.json"
elif [[ "$GROUP" == "G3" ]]; then
  G2="$EXPERIMENT_ROOT/G2"
  need_file "$G2/train_diffusion/optimal.pt"
  need_file "$G2/train_decoder/optimal.pt"
  need_file "$G2/candidate_pool/solvents.json"
  need_file "$G2/candidate_pool/salts.json"
  record_skip candidate_pool "reused_G2_guided_pool"
  record_skip generator_training "reused_G2_generator"
  DIFF_CONFIG="$G2/configs/diffusion.yaml"
  DEC_CONFIG="$G2/configs/decoder.yaml"
  DIFF_CKPT="$G2/train_diffusion/optimal.pt"
  DEC_CKPT="$G2/train_decoder/optimal.pt"
  SOLVENTS="$G2/candidate_pool/solvents.json"
  SALTS="$G2/candidate_pool/salts.json"
else
  POOL="$RUN/candidate_pool"
  if [[ ! -f "$POOL/solvents.json" ]]; then
    need_file "$SOLVENT_LIKE_POOL"; need_file "$SOLVENT_LIKE_AUDIT"
    if ! grep -Fq '"screen_name": "solvent_like_v1"' "$SOLVENT_LIKE_AUDIT" \
       || grep -Eq '"(hbd_max|hba_max|rotatable_bonds_max|tpsa_max|logp)"' "$SOLVENT_LIKE_AUDIT"; then
      echo "Solvent-like pool audit does not match the current simplified screen: $SOLVENT_LIKE_AUDIT" >&2
      echo "Expected solvent_like_v1 without HBD/HBA/TPSA/rotatable-bond/logP gates." >&2
      exit 1
    fi
    if [[ "$GROUP" == "G1" ]]; then
      need_file "$MONO_CKPT"
      run_stage candidate_pool python3 -u scripts/generation/build_candidate_pool.py \
        --mode random --output-dir "$POOL" --baseline-solvents "$BAMBOO_SOLVENTS" \
        --salts "$BAMBOO_SALTS" --candidate-source "$SOLVENT_LIKE_POOL" \
        --pool-size "$POOL_SIZE" --model-config "$MONO_CONFIG" \
        --mono-checkpoint "$MONO_CKPT" --seed "$CANDIDATE_POOL_SEED"
    else
      need_file "$PROPERTY_CONSTRAINTS"; need_file "$MONO_CKPT"
      run_stage candidate_pool python3 -u scripts/generation/build_candidate_pool.py \
        --mode guided --output-dir "$POOL" --baseline-solvents "$BAMBOO_SOLVENTS" \
        --salts "$BAMBOO_SALTS" --candidate-source "$SOLVENT_LIKE_POOL" \
        --property-constraints "$PROPERTY_CONSTRAINTS" --model-config "$MONO_CONFIG" \
        --mono-checkpoint "$MONO_CKPT" --pool-size "$POOL_SIZE" \
        --seed "$CANDIDATE_POOL_SEED" \
        --guided-selection-strategy "$GUIDED_SELECTION_STRATEGY" \
        --guided-property-weight "$GUIDED_PROPERTY_WEIGHT" \
        --guided-fragment-weight "$GUIDED_FRAGMENT_WEIGHT" \
        --guided-family-coverage-weight "$GUIDED_FAMILY_COVERAGE_WEIGHT" \
        --guided-scaffold-coverage-weight "$GUIDED_SCAFFOLD_COVERAGE_WEIGHT" \
        --guided-redundancy-weight "$GUIDED_REDUNDANCY_WEIGHT"
    fi
  else
    verify_existing_pool_source "$RUN"
    record_skip candidate_pool "existing_output"
  fi

  if [[ ! -f "$POOL/solv_embs.pt" ]]; then
    need_file "$MONO_CKPT"
    run_stage vocabulary_embedding python3 -u scripts/generation/build_generator_vocab_embeddings.py \
      --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
      --model-config "$MONO_CONFIG" --mono-checkpoint "$MONO_CKPT" --output-dir "$POOL"
  else record_skip vocabulary_embedding "existing_output"; fi

  if [[ ! -f "$RUN/generator_data/processed_data.pkl" ]]; then
    run_stage synthetic_formula_oracle python3 -u scripts/generation/build_synthetic_generator_data.py \
      --solvents "$POOL/solvents.json" --salts "$POOL/salts.json" \
      --oracle-config "$ORACLE_CONFIG" --oracle-checkpoint "$ORACLE_CKPT" \
      --output-dir "$RUN/generator_data" --num-formulas "$SYNTHETIC_FORMULAS" \
      --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
      --oversample-factor "$SYNTHETIC_OVERSAMPLE_FACTOR" \
      --label-policy "$SYNTHETIC_LABEL_POLICY" \
      --solvent-count-policy "$SYNTHETIC_SOLVENT_COUNT_POLICY" \
      --salt-mixture-mode "$SYNTHETIC_SALT_MIXTURE_MODE" \
      --salt-mixture-bins "$SYNTHETIC_SALT_MIXTURE_BINS" \
      --joint-balance-strategy "$SYNTHETIC_JOINT_BALANCE_STRATEGY" \
      --min-nonempty-anion-bins "$SYNTHETIC_MIN_NONEMPTY_ANION_BINS" \
      --seed "$SEED"
  else record_skip synthetic_formula_oracle "existing_output"; fi

  if [[ ! -f "$RUN/configs/diffusion.yaml" ]]; then
    run_stage write_train_configs python3 -u scripts/generation/write_generator_train_configs.py \
      --experiment-dir "$RUN" --max-epoch "$TRAIN_MAX_EPOCH"
  else record_skip write_train_configs "existing_output"; fi

  train_or_resume train_diffusion "$RUN/train_diffusion" "$RUN/configs/diffusion.yaml"
  train_or_resume train_decoder "$RUN/train_decoder" "$RUN/configs/decoder.yaml"

  DIFF_CONFIG="$RUN/configs/diffusion.yaml"
  DEC_CONFIG="$RUN/configs/decoder.yaml"
  DIFF_CKPT="$RUN/train_diffusion/optimal.pt"
  DEC_CKPT="$RUN/train_decoder/optimal.pt"
  SOLVENTS="$POOL/solvents.json"
  SALTS="$POOL/salts.json"
fi

if [[ ! -f "$RUN/generated/generated_formulas.json" ]]; then
  cmd=(python3 -u scripts/generation/run_conditioned_generation.py
       --diffusion-config "$DIFF_CONFIG" --diffusion-checkpoint "$DIFF_CKPT"
       --decoder-config "$DEC_CONFIG" --decoder-checkpoint "$DEC_CKPT"
       --solvents "$SOLVENTS" --salts "$SALTS" --oracle-config "$ORACLE_CONFIG"
       --oracle-checkpoint "$ORACLE_CKPT" --output "$RUN/generated/generated_formulas.json"
       --target-conductivity "$TARGET_CONDUCTIVITY" --anion-ratio "$ANION_RATIO"
       --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO"
       --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --seed "$SEED"
       --selection-policy "$SELECTION_POLICY"
       --conductivity-tolerance "$SELECTION_CONDUCTIVITY_TOLERANCE"
       --anion-ratio-tolerance "$SELECTION_ANION_RATIO_TOLERANCE"
       --joint-anion-weight "$SELECTION_JOINT_ANION_WEIGHT")
  if [[ "$GROUP" == "G0" ]]; then
    cmd+=(--decoder-solvent-embeddings "$BAMBOO_SOLV_EMBS"
          --decoder-salt-embeddings "$BAMBOO_SALT_EMBS"
          --released-bamboo-root "$BAMBOO_ROOT")
  fi
  [[ "$GROUP" == "G3" ]] && cmd+=(--robust)
  run_stage sample_and_score "${cmd[@]}"
else record_skip sample_and_score "existing_output"; fi

printf 'Finished %s. Raw results: %s\n' "$GROUP" "$RUN/generated/generated_formulas.json"
