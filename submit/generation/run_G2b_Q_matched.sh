#!/usr/bin/env bash
# Strict random-selection control for the G2-b interaction-coverage ablation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/submit/generation/run_G2b_interaction_coverage.sh" G2-b-Q
