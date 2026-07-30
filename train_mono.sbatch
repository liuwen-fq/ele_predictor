#!/usr/bin/env bash
#SBATCH --job-name=electrolyte-fragment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=7-00:00:00
#SBATCH --output=/mnt/afs/home/liuwen/git/electrolyte/ele_predictor/run/fragment_pipeline/slurm-%j.out
#SBATCH --error=/mnt/afs/home/liuwen/git/electrolyte/ele_predictor/run/fragment_pipeline/slurm-%j.err

set -euo pipefail

REPO_ROOT="/mnt/afs/home/liuwen/git/electrolyte/ele_predictor"
cd "$REPO_ROOT"

exec /bin/bash "$REPO_ROOT/run_pretrain.sh"
