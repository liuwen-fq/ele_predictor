#!/usr/bin/env bash
# Usage: bash submit/run_fragment_explanation_no_fidelity_pipeline.sh [all|train|predict|explain]
set -euo pipefail

cd "$(dirname "$0")/.."
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MODE="${1:-all}"
[[ "$MODE" =~ ^(all|train|predict|explain)$ ]] || {
    echo "Usage: bash submit/run_fragment_explanation_no_fidelity_pipeline.sh [all|train|predict|explain]" >&2
    exit 2
}

SOURCE_BAMBOO_MONO="/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/baseline_comparison/ckpt_mono/optimal.pt"
SOURCE_BAMBOO_FT="/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/baseline_comparison/ckpt_formula_ft/optimal.pt"
SOURCE_FRAGMENT_MONO="/mnt/afs/home/liuwen/git/electrolyte/ele_predictor/run/fragment_sidecar/ckpt_mono/optimal.pt"
RUN_DIR="run/fragment_explanation_no_fidelity"
CKPT_DIR="$RUN_DIR/ckpt_formula_ft"
# Keep explanation artifacts outside RUN_DIR: pre-existing training runs may
# have been created by another account and be read-only.
EXPLAIN_ARTIFACT_DIR="${EXPLAIN_ARTIFACT_DIR:-run/fragment_explanation_artifacts}"
ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-run/fragment_attribution_attachment_aware}"

if [[ "$MODE" == all || "$MODE" == train || "$MODE" == predict ]]; then
    for file in "$SOURCE_BAMBOO_MONO" "$SOURCE_BAMBOO_FT" "$SOURCE_FRAGMENT_MONO"; do
        [[ -f "$file" ]] || { echo "Missing checkpoint: $file" >&2; exit 1; }
    done
    for split in ft_train ft_valid ft_test; do
        [[ -f "run/data/$split/dataset_config.yaml" && \
           -f "run/data/$split/processed_data.pkl" ]] || {
            echo "Missing prepared dataset: run/data/$split" >&2
            exit 1
        }
    done
    mkdir -p run/checkpoints "$CKPT_DIR"
    python3 -u scripts/checkpoint/build_fixed_fragment_mono_checkpoint.py \
        --bamboo-mono "$SOURCE_BAMBOO_MONO" \
        --fragment-mono "$SOURCE_FRAGMENT_MONO" \
        --output run/checkpoints/fixed_fragment_mono.pt
    python3 -u scripts/checkpoint/convert_bamboo_formula_checkpoint.py \
        --input "$SOURCE_BAMBOO_FT" \
        --output run/checkpoints/base_formula_ft.pt
    ln -sfn ../../checkpoints/fixed_fragment_mono.pt "$CKPT_DIR/pretrain.pt"
fi

if [[ "$MODE" == all || "$MODE" == train ]]; then
    python3 -u scripts/train/train.py \
        --conf config/train/fragment_formula_ft_explanation_no_fidelity.yaml \
        --task formula --restart
fi

if [[ "$MODE" == all || "$MODE" == predict ]]; then
    [[ -f "$CKPT_DIR/optimal.pt" ]] || { echo "Missing trained explanation model" >&2; exit 1; }
    python3 -u scripts/test_results/predict.py \
        --conf config/test_results/fragment_predict_explanation_no_fidelity.yaml
fi

if [[ "$MODE" == all || "$MODE" == explain ]]; then
    RESULT="$RUN_DIR/results/ft_test_predictions.json"
    [[ -f "$RESULT" ]] || { echo "Missing prediction result: $RESULT" >&2; exit 1; }
    mkdir -p "$EXPLAIN_ARTIFACT_DIR"
    python3 -u scripts/test_results/evaluate_conductivity.py \
        --input "$RESULT" --output "$EXPLAIN_ARTIFACT_DIR/ft_test_prediction_metrics.json"
    python3 -u scripts/test_results/evaluate_conductivity.py \
        --input "$RESULT" --prediction-key conductivity_explain_pred \
        --output "$EXPLAIN_ARTIFACT_DIR/ft_test_explanation_metrics.json"
    python3 -u scripts/test_results/explain_fragment_contributions.py \
        --input "$RESULT" --output-dir "$EXPLAIN_ARTIFACT_DIR/attention_explanations" \
        --contribution-key fragment_explanations
    python3 -u scripts/test_results/analyze_conductivity_fragment_attributions.py \
        --input "$RESULT" --output-dir "$ATTRIBUTION_DIR"
fi

echo "Finished no-fidelity fragment explanation pipeline: $RUN_DIR"
echo "Other explanation artifacts: $EXPLAIN_ARTIFACT_DIR"
echo "Attachment-aware attribution report: $ATTRIBUTION_DIR"
