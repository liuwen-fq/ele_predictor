#!/usr/bin/env bash
# Add oracle-predicted anion ratio to existing G0--G3 samples, then redraw.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-/mnt/afs/home/liuwen/miniconda3/envs/bamboo/bin/python3}"
cd "$ROOT"

for group in G0 G1 G2 G3; do
  pool="$group"
  [[ "$group" == "G3" ]] && pool="G2"
  "$PYTHON" -u scripts/generation/enrich_generated_anion_ratio.py \
    --input "run/generation_experiments/$group/generated/generated_formulas.json" \
    --solvents "run/generation_experiments/$pool/candidate_pool/solvents.json" \
    --salts "run/generation_experiments/$pool/candidate_pool/salts.json" \
    --oracle-config config/test_results/property_fragment_predict.yaml \
    --oracle-checkpoint run/property_fragment_explanation/ckpt_formula_ft \
    --batch-size 256
done

"$PYTHON" -u scripts/generation/analyze_generation_experiments.py \
  --experiment-root run/generation_experiments \
  --target-conductivity 20 \
  --selection-budget 100
