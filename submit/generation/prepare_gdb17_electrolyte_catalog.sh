#!/usr/bin/env bash
# Build the shared GDB17 electrolyte-solvent catalogue once.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash submit/generation/prepare_gdb17_electrolyte_catalog.sh
#
# The result contains a deterministic, structurally screened 20,000-molecule
# catalogue plus frozen mono-property predictions.  It is reused unchanged by
# G1 random selection and G2 guided selection; do not run one copy per seed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_gdb17_mechanism_auxiliary_comparison}"
COMMON_ROOT="$EXPERIMENT_ROOT/common"
GDB17_CSV="${GDB17_CSV:-$ROOT/run/data/GDB17_strictly_uncharged_smiles.csv}"
ORACLE_CONFIG="${ORACLE_CONFIG:-config/test_results/anion_ratio_property_fragment_predict.yaml}"
MONO_CKPT="${MONO_CKPT:-run/checkpoints/fixed_fragment_mono.pt}"
CATALOG_SIZE="${GDB17_CATALOG_SIZE:-20000}"
CATALOG_SEED="${GDB17_CATALOG_SEED:-20260908}"
PROPERTY_BATCH_SIZE="${PROPERTY_BATCH_SIZE:-256}"
CATALOG="$COMMON_ROOT/gdb17_electrolyte_property_catalog.json"
AUDIT="$COMMON_ROOT/gdb17_electrolyte_property_catalog_audit.json"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
for path in "$GDB17_CSV" "$ORACLE_CONFIG" "$MONO_CKPT"; do need_file "$path"; done
command -v flock >/dev/null || { echo "flock is required for shared catalogue safety" >&2; exit 1; }

mkdir -p "$COMMON_ROOT"
exec 9>"$COMMON_ROOT/.gdb17_catalog.lock"
flock 9
if [[ -f "$CATALOG" && -f "$AUDIT" ]]; then
  echo "Reusing completed GDB17 catalogue: $CATALOG"
  exit 0
fi
if [[ -e "$CATALOG" || -e "$AUDIT" ]]; then
  echo "Incomplete GDB17 catalogue output found under $COMMON_ROOT; inspect it rather than overwriting." >&2
  exit 1
fi

python3 -u scripts/generation/build_gdb17_electrolyte_catalog.py \
  --input-csv "$GDB17_CSV" --output "$CATALOG" --audit-output "$AUDIT" \
  --model-config "$ORACLE_CONFIG" --mono-checkpoint "$MONO_CKPT" \
  --catalog-size "$CATALOG_SIZE" --selection-seed "$CATALOG_SEED" \
  --batch-size "$PROPERTY_BATCH_SIZE"

echo "Completed shared GDB17 catalogue: $CATALOG"
