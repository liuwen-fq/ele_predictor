#!/usr/bin/env bash
# G1 uses the same strict, jointly credible synthetic-label policy as G2.
# This isolates the candidate-pool strategy instead of comparing data cleaning.
export SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
export SYNTHETIC_OVERSAMPLE_FACTOR="${SYNTHETIC_OVERSAMPLE_FACTOR:-1.5}"
export SYNTHETIC_LABEL_POLICY="${SYNTHETIC_LABEL_POLICY:-strict}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_generation_experiment.sh" G1
