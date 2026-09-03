#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$ROOT/submit/generation/run_mechanism_auxiliary.sh" G2-MA-shuffle
