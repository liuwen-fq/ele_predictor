#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEED="${1:?usage: $0 SEED [config.json]}"
exec bash "$ROOT/submit/generation/submit_G0_reproduction.sh" G0-Q "$SEED" "${2:-config/reproduction/g0_legacy.json}"
