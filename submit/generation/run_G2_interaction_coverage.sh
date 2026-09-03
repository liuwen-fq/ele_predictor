#!/usr/bin/env bash
# G2 interaction-coverage retraining (explanation/property-guided vocabulary).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/submit/generation/run_G1G2_interaction_coverage.sh" G2-I
