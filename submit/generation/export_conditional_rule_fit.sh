#!/usr/bin/env bash
# Explicit fitting data only; never use the default ft_test prediction input.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
DATA="${1:?usage: $0 TRAIN_DATA_DIR [OUTPUT_JSON]}"
OUTPUT="${2:-run/rule_fit/train_predictions.json}"
case "$DATA" in *test*|*valid*) echo 'Use a training/rule-fitting dataset, not test/validation data.' >&2; exit 2;; esac
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$OUTPUT")"
exec python3 scripts/test_results/predict.py \
  --conf "${RULE_FIT_PREDICT_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}" \
  --save-dir "$DATA" --output-path "$OUTPUT"
