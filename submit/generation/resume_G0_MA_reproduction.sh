#!/usr/bin/env bash
# Sequential recovery: retain batch size, data, objective and matched decoder.
# Usage: CUDA_VISIBLE_DEVICES=0 bash submit/generation/resume_G0_MA_reproduction.sh [config.json]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
CONFIG="${1:-config/reproduction/g0_legacy.json}"
SEEDS="$(python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); assert all(type(s) is int and s >= 0 for s in c["seeds"]); print(" ".join(map(str,c["seeds"])))' "$CONFIG")"
for seed in $SEEDS; do
  for group in G0-MA G0-MA-shuffle; do
    echo "Resuming $group seed $seed (one task at a time)"
    bash submit/generation/submit_G0_reproduction.sh "$group" "$seed" "$CONFIG"
  done
done
