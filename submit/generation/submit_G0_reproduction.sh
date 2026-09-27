#!/usr/bin/env bash
# One group and one seed per invocation, following the existing Bash runners.
# No scheduler submission or automatic resource allocation.
# Usage: CUDA_VISIBLE_DEVICES=0 bash submit/generation/submit_G0_reproduction.sh G0-Q 123 [config.json]
set -euo pipefail
GROUP="${1:?usage: $0 G0-Q|G0-MA|G0-MA-shuffle SEED [config.json]}"
REPLICA_SEED="${2:?missing SEED}"
case "$GROUP" in G0-Q|G0-MA|G0-MA-shuffle) ;; *) echo "Unknown group: $GROUP" >&2; exit 2 ;; esac
[[ "$REPLICA_SEED" =~ ^[0-9]+$ ]] || { echo 'SEED must be a nonnegative integer' >&2; exit 2; }
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG="${3:-config/reproduction/g0_legacy.json}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
python3 scripts/generation/reproduce_g0.py --config "$CONFIG" --group "$GROUP" --seed "$REPLICA_SEED" preflight
ENV_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["environment"]["CONDA_ENV"])' "$CONFIG")"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-g0-${GROUP}-${REPLICA_SEED}-$$}"
mkdir -p "$MPLCONFIGDIR"
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"'
exec python3 scripts/generation/reproduce_g0.py --config "$CONFIG" --group "$GROUP" --seed "$REPLICA_SEED" run
