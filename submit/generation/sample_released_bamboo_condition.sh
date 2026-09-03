#!/usr/bin/env bash
# Recreate G0 in the released Bamboo-Mixer stack: released diffusion/decoder,
# released vocabulary/embeddings, and released formula/mono predictors.
set -euo pipefail

TARGET_CONDUCTIVITY="${1:?usage: $0 TARGET_CONDUCTIVITY ANION_RATIO}"
ANION_RATIO="${2:?usage: $0 TARGET_CONDUCTIVITY ANION_RATIO}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source /mnt/afs/home/liuwen/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-bamboo}"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

SOURCE_ROOT="${SOURCE_ROOT:-run/generation_bamboo_composition_comparison}"
TAG="k${TARGET_CONDUCTIVITY//./p}_r${ANION_RATIO//./p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SOURCE_ROOT}/condition_${TAG}}"
BAMBOO_ROOT="${BAMBOO_ROOT:-/mnt/afs/home/liuwen/git/electrolyte/bamboo_mixer}"
TEMPERATURE_C="${TEMPERATURE_C:-25.0}"
SALT_MOLAR_RATIO="${SALT_MOLAR_RATIO:-0.10}"
GEN_BATCHES="${GEN_BATCHES:-8}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-128}"

need_file() { [[ -f "$1" ]] || { echo "Required file is missing: $1" >&2; exit 1; }; }
for path in "$BAMBOO_ROOT/ckpts/generator/diffusion.pt" "$BAMBOO_ROOT/ckpts/generator/decoder.pt" \
            "$BAMBOO_ROOT/ckpts/formula/optimal.pt" "$BAMBOO_ROOT/ckpts/mono/optimal.pt" \
            "$BAMBOO_ROOT/config/test_results/generate_config.yaml" \
            "$BAMBOO_ROOT/run/solv_embs.pt" "$BAMBOO_ROOT/run/salt_embs.pt" \
            "emb_dict/solvents.json" "emb_dict/salts.json"; do
  need_file "$path"
done

DEST="$OUTPUT_ROOT/G0"
mkdir -p "$DEST/candidate_pool" "$DEST/generated"
# Do not rely on a G2-b candidate-pool symlink: that directory can be rebuilt
# independently, while G0 must retain its released vocabulary snapshot.
cp emb_dict/solvents.json "$DEST/candidate_pool/solvents.json"
cp emb_dict/salts.json "$DEST/candidate_pool/salts.json"

python3 -u scripts/generation/run_released_bamboo_condition.py \
  --bamboo-root "$BAMBOO_ROOT" \
  --output "$DEST/generated/generated_formulas.json" \
  --target-conductivity "$TARGET_CONDUCTIVITY" --anion-ratio "$ANION_RATIO" \
  --temperature-c "$TEMPERATURE_C" --concentration "$SALT_MOLAR_RATIO" \
  --num-batches "$GEN_BATCHES" --batch-size "$GEN_BATCH_SIZE" --overwrite

printf 'Finished native released Bamboo G0 at κ=%s, anion ratio=%s. Output: %s\n' \
  "$TARGET_CONDUCTIVITY" "$ANION_RATIO" "$DEST/generated/generated_formulas.json"
