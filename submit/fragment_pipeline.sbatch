#!/bin/bash
#SBATCH --job-name=electrolyte-fragment
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=run/fragment_pipeline/slurm-%j.out

set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p run/fragment_pipeline/raw

"$PYTHON_BIN" scripts/prepare_data/split_formula_data.py \
  --input run/formula_pretrain_md_data.json \
  --output_dir run/fragment_pipeline/raw --name md --seed 123 \
  --train_ratio 0.85 --valid_ratio 0.15 --test_ratio 0
"$PYTHON_BIN" scripts/prepare_data/split_formula_data.py \
  --input run/formula_finetune_data.json \
  --output_dir run/fragment_pipeline/raw --name ft --seed 123

"$PYTHON_BIN" scripts/prepare_data/prepare_data.py \
  --conf config/prepare_data/mono_fragment_config.yaml \
  --data_type mono --num_workers "$NUM_WORKERS"

for split in train valid test; do
  if [[ "$split" != "test" ]]; then
    "$PYTHON_BIN" scripts/prepare_data/prepare_data.py \
      --conf "config/prepare_data/fragment_md_${split}.yaml" \
      --data_type formula --num_workers "$NUM_WORKERS"
  fi
  "$PYTHON_BIN" scripts/prepare_data/prepare_data.py \
    --conf "config/prepare_data/fragment_ft_${split}.yaml" \
    --data_type formula --num_workers "$NUM_WORKERS"
done

"$PYTHON_BIN" scripts/train/train.py \
  --conf config/train/mono_fragment_config.yaml --task mono
"$PYTHON_BIN" scripts/train/train.py \
  --conf config/train/fragment_formula_md.yaml --task formula
"$PYTHON_BIN" scripts/train/train.py \
  --conf config/train/fragment_formula_ft.yaml --task formula

ln -sfn "$(pwd)/run/fragment_pipeline/ckpt_mono/optimal.pt" \
  run/fragment_pipeline/ckpt_formula_ft/pretrain.pt
"$PYTHON_BIN" scripts/test_results/predict.py \
  --conf config/test_results/fragment_predict.yaml
"$PYTHON_BIN" scripts/test_results/evaluate_conductivity.py \
  --input run/fragment_pipeline/results/ft_test_predictions.json \
  --output run/fragment_pipeline/results/ft_test_metrics.json

echo "Finished: run/fragment_pipeline/results/ft_test_metrics.json"
