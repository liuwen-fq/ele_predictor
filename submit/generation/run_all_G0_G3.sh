#!/usr/bin/env bash
# Serial order is deliberate: G3 reuses G2's trained generator.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
for group in G0 G1 G2 G3; do
  "$HERE/run_generation_experiment.sh" "$group"
done
python3 -u "$(cd "$HERE/../.." && pwd)/scripts/generation/analyze_generation_experiments.py"
