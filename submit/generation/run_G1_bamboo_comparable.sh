#!/usr/bin/env bash
# G1 with Bamboo-compatible solvent-count distribution and no condition-bin
# reweighting.  The G1 random vocabulary remains the controlled baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_bamboo_comparable_v2_oracle_labels}"
export SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
export SYNTHETIC_OVERSAMPLE_FACTOR="${SYNTHETIC_OVERSAMPLE_FACTOR:-1.0}"
export SYNTHETIC_LABEL_POLICY="${SYNTHETIC_LABEL_POLICY:-sanitize_keep_all}"
export SYNTHETIC_SOLVENT_COUNT_POLICY="${SYNTHETIC_SOLVENT_COUNT_POLICY:-bamboo_empirical_1_to_6}"
export SYNTHETIC_SALT_MIXTURE_MODE="${SYNTHETIC_SALT_MIXTURE_MODE:-bamboo_fixed_two_salt}"
export SYNTHETIC_JOINT_BALANCE_STRATEGY="${SYNTHETIC_JOINT_BALANCE_STRATEGY:-none}"
export SYNTHETIC_MIN_NONEMPTY_ANION_BINS="${SYNTHETIC_MIN_NONEMPTY_ANION_BINS:-1}"

exec "$ROOT/submit/generation/run_generation_experiment.sh" G1
