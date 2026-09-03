#!/usr/bin/env bash
# Interaction-coverage arm for the G2-b matched ablation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/submit/generation/run_G2b_interaction_coverage.sh" G2-b-I
