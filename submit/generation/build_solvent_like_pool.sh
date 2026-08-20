#!/usr/bin/env bash
#SBATCH --job-name=solvent_like_pool
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=run/solvent_like_pool_%j.out

# Build the fixed G1/G2 solvent-like candidate universe.  This job is
# CPU-only and writes no model checkpoint; it only materializes the reusable
# vocabulary source and its audit in emb_dict/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

MONO_DATA="${MONO_DATA:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer/run/mono_data.json}"
BAMBOO_SOLVENTS="${BAMBOO_SOLVENTS:-$ROOT/emb_dict/solvents.json}"
OUTPUT="${OUTPUT:-$ROOT/emb_dict/solvent_like_candidates.json}"
AUDIT_OUTPUT="${AUDIT_OUTPUT:-$ROOT/emb_dict/solvent_like_candidates_audit.json}"

python3 -u scripts/generation/build_solvent_like_pool.py \
  --mono-data "$MONO_DATA" \
  --bamboo-solvents "$BAMBOO_SOLVENTS" \
  --output "$OUTPUT" \
  --audit-output "$AUDIT_OUTPUT"
