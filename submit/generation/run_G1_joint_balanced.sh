#!/usr/bin/env bash
# Fresh G1 experiment: random solvent vocabulary, but anion-first balanced
# synthetic conditions and joint conductivity/anion-ratio post-selection.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-run/generation_joint_balanced_v1}"
export SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
export SYNTHETIC_OVERSAMPLE_FACTOR="${SYNTHETIC_OVERSAMPLE_FACTOR:-2.0}"
export SYNTHETIC_LABEL_POLICY="${SYNTHETIC_LABEL_POLICY:-strict}"
export SYNTHETIC_SALT_MIXTURE_MODE="${SYNTHETIC_SALT_MIXTURE_MODE:-stratified_two_salt}"
export SYNTHETIC_SALT_MIXTURE_BINS="${SYNTHETIC_SALT_MIXTURE_BINS:-5}"
export SYNTHETIC_JOINT_BALANCE_STRATEGY="${SYNTHETIC_JOINT_BALANCE_STRATEGY:-hierarchical_anion_then_conductivity}"
export SYNTHETIC_MIN_NONEMPTY_ANION_BINS="${SYNTHETIC_MIN_NONEMPTY_ANION_BINS:-4}"
export SELECTION_POLICY="${SELECTION_POLICY:-joint_target}"
export SELECTION_CONDUCTIVITY_TOLERANCE="${SELECTION_CONDUCTIVITY_TOLERANCE:-2.0}"
export SELECTION_ANION_RATIO_TOLERANCE="${SELECTION_ANION_RATIO_TOLERANCE:-0.10}"
export SELECTION_JOINT_ANION_WEIGHT="${SELECTION_JOINT_ANION_WEIGHT:-1.0}"

exec "$ROOT/submit/generation/run_generation_experiment.sh" G1
