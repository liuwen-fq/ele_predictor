#!/usr/bin/env bash
# Build the shared G1/G2 candidate universe once before launching experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate bamboo
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_experiments}"
MONO_DATA="${MONO_DATA:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/mono_data.json}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
COMMON_POOL_DIR="${COMMON_POOL_DIR:-$EXPERIMENT_ROOT/common_eligible_pool}"

python3 -u scripts/generation/build_common_eligible_pool.py \
  --mono-data "$MONO_DATA" \
  --bamboo-solvents "$BAMBOO_SOLVENTS" \
  --output "$COMMON_POOL_DIR/eligible_molecules.json" \
  --audit-output "$COMMON_POOL_DIR/eligibility_audit.json"
