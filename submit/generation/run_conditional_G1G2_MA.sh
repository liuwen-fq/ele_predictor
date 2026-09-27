#!/usr/bin/env bash
# Same MA settings in both groups; condition-adaptive vocabulary only in G2.
# Usage: CUDA_VISIBLE_DEVICES=0 bash ... G1-MA|G2-Q|G2-MA|G2-MA-shuffle SEED TASK.json [--plan]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec python3 scripts/generation/run_conditional_ma.py "$@"
