#!/usr/bin/env bash
# Usage: bash submit/run_property_fragment_explanation_pipeline.sh [all|train|predict|explain]
set -euo pipefail

cd "$(dirname "$0")/.."
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

MODE="${1:-all}"
[[ "$MODE" =~ ^(all|train|predict|explain)$ ]] || {
    echo "Usage: bash submit/run_property_fragment_explanation_pipeline.sh [all|train|predict|explain]" >&2
    exit 2
}

SOURCE_BAMBOO_MONO="/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/baseline_comparison/ckpt_mono/optimal.pt"
SOURCE_BAMBOO_FT="/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/baseline_comparison/ckpt_formula_ft/optimal.pt"
SOURCE_FRAGMENT_MONO="/mnt/afs/home/liuwen/git/electrolyte/ele_predictor/run/fragment_sidecar/ckpt_mono/optimal.pt"
RUN_DIR="run/property_fragment_explanation"
CKPT_DIR="$RUN_DIR/ckpt_formula_ft"
RESULT="$RUN_DIR/results/ft_test_predictions.json"
ARTIFACT_DIR="${ARTIFACT_DIR:-run/property_fragment_explanation_artifacts}"
ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-run/property_fragment_attribution}"

prepare_checkpoints() {
    for file in "$SOURCE_BAMBOO_MONO" "$SOURCE_BAMBOO_FT" "$SOURCE_FRAGMENT_MONO"; do
        [[ -f "$file" ]] || { echo "Missing checkpoint: $file" >&2; exit 1; }
    done
    for split in ft_train ft_valid ft_test; do
        [[ -f "run/data/$split/dataset_config.yaml" && -f "run/data/$split/processed_data.pkl" ]] || {
            echo "Missing prepared dataset: run/data/$split" >&2; exit 1;
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
}

if [[ "$MODE" == all || "$MODE" == train || "$MODE" == predict ]]; then
    prepare_checkpoints
fi
if [[ "$MODE" == all || "$MODE" == train ]]; then
    python3 -u scripts/train/train.py --conf config/train/property_fragment_explanation.yaml \
        --task formula --restart
fi
if [[ "$MODE" == all || "$MODE" == predict ]]; then
    [[ -f "$CKPT_DIR/optimal.pt" ]] || { echo "Missing trained explanation model" >&2; exit 1; }
    python3 -u scripts/test_results/predict.py --conf config/test_results/property_fragment_predict.yaml
fi
if [[ "$MODE" == all || "$MODE" == explain ]]; then
    [[ -f "$RESULT" ]] || { echo "Missing prediction result: $RESULT" >&2; exit 1; }
    mkdir -p "$ARTIFACT_DIR" "$ATTRIBUTION_DIR"
    python3 -u scripts/test_results/evaluate_conductivity.py --input "$RESULT" \
        --output "$ARTIFACT_DIR/ft_test_prediction_metrics.json"
    python3 -u scripts/test_results/evaluate_conductivity.py --input "$RESULT" \
        --prediction-key conductivity_explain_pred \
        --output "$ARTIFACT_DIR/ft_test_explanation_metrics.json"
    python3 -u scripts/test_results/analyze_property_contributions.py \
        --input "$RESULT" --output-dir "$ARTIFACT_DIR/properties"
    python3 -u scripts/test_results/analyze_conductivity_fragment_attributions.py \
        --input "$RESULT" --output-dir "$ATTRIBUTION_DIR"
fi

echo "Finished property-fragment explanation pipeline: $RUN_DIR"
