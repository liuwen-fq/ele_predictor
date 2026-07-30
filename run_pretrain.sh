#!/usr/bin/env bash

# ACP starts this file from a non-interactive shell. Initialize conda
# explicitly, then run from the repository root regardless of the launch path.
set -eu

REPO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/run/fragment_pipeline"
LOG_FILE="$LOG_DIR/acp_pretrain_launcher.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "===== ACP molecular pretraining launcher: $(date '+%F %T') ====="
echo "repo: $REPO_ROOT"
echo "host: $(hostname)"

if [ -f "/mnt/aafs/home/liuwen/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/mnt/aafs/home/liuwen/miniconda3"
elif [ -f "/mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_BASE="/mnt/afs/home/liuwen/miniconda3"
elif command -v conda >/dev/null 2>&1; then
    CONDA_BASE=$(conda info --base 2>/dev/null || true)
else
    echo "ERROR: conda was not found in the mounted miniconda directory or PATH" >&2
    exit 1
fi
echo "conda base: $CONDA_BASE"
if [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "ERROR: conda initialization script was not found under $CONDA_BASE" >&2
    exit 1
fi
. "$CONDA_BASE/etc/profile.d/conda.sh"
if [ ! -d "$CONDA_BASE/envs/bamboo" ]; then
    echo "ERROR: bamboo environment was not found at $CONDA_BASE/envs/bamboo" >&2
    exit 1
fi
echo "activating environment: $CONDA_BASE/envs/bamboo"
conda activate "$CONDA_BASE/envs/bamboo"
echo "python: $(command -v python3)"

# Do not call set_pythonpath.sh here: it uses git rev-parse and exits with
# status 128 when ACP mounts the source tree without its .git metadata.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not-set>}"
echo "ACP rank: ${RANK:-0}/${WORLD_SIZE:-1}"

python3 -u - <<'PY'
import torch

print(f"torch: {torch.__version__}", flush=True)
print(f"cuda_available: {torch.cuda.is_available()}", flush=True)
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available in the training container")
print(f"cuda_device_count: {torch.cuda.device_count()}", flush=True)
print(f"cuda_device_name: {torch.cuda.get_device_name(0)}", flush=True)
PY

# The current train.py initializes a single process and does not consume the
# ACP DDP environment variables. Use one GPU for this molecular pretraining.
exec python3 -u scripts/train/train.py \
    --conf config/train/mono_fragment_config.yaml \
    --task mono
