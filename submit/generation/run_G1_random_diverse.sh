#!/usr/bin/env bash
# Legacy-style G1 control: retain all 30k randomly constructed formulas.
# Invalid oracle labels are sanitized rather than rejected, so this has no
# effect on G0/G2/G3, which retain the shared strict policy by default.
export SYNTHETIC_FORMULAS="${SYNTHETIC_FORMULAS:-30000}"
export SYNTHETIC_OVERSAMPLE_FACTOR="${SYNTHETIC_OVERSAMPLE_FACTOR:-1.0}"
export SYNTHETIC_LABEL_POLICY="${SYNTHETIC_LABEL_POLICY:-sanitize_keep_all}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_generation_experiment.sh" G1
