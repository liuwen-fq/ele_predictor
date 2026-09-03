#!/usr/bin/env bash
# G2-b: fixed Bamboo vocabulary under the Bamboo composition prior.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_bamboo_composition_comparison}"
exec "$ROOT/submit/generation/run_generation_composition_ablation.sh" G2-b
