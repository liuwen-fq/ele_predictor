#!/usr/bin/env bash
# Matched random synthetic-data control for G1-I.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/submit/generation/run_G1G2_interaction_coverage.sh" G1-Q
