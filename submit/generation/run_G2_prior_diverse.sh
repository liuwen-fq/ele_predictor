#!/usr/bin/env bash
# New G2 pipeline: coverage-aware soft-prior vocabulary selection.
# It writes to a fresh root by default, so existing G2 results/checkpoints are
# never mixed with the new 62-solvent vocabulary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_G2_prior_diverse_v1}"
export GUIDED_SELECTION_STRATEGY="${GUIDED_SELECTION_STRATEGY:-coverage_v1}"
export GUIDED_PROPERTY_WEIGHT="${GUIDED_PROPERTY_WEIGHT:-1.0}"
export GUIDED_FRAGMENT_WEIGHT="${GUIDED_FRAGMENT_WEIGHT:-1.0}"
export GUIDED_FAMILY_COVERAGE_WEIGHT="${GUIDED_FAMILY_COVERAGE_WEIGHT:-0.7}"
export GUIDED_SCAFFOLD_COVERAGE_WEIGHT="${GUIDED_SCAFFOLD_COVERAGE_WEIGHT:-0.25}"
export GUIDED_REDUNDANCY_WEIGHT="${GUIDED_REDUNDANCY_WEIGHT:-1.0}"

exec "$ROOT/submit/generation/run_generation_experiment.sh" G2
