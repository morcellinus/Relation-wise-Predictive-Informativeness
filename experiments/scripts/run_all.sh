#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
"$PYTHON_BIN" "$ROOT/scripts/verify_bundle.py"
"$ROOT/scripts/run_hcsbm.sh"
"$ROOT/scripts/run_realworld.sh"
"$ROOT/scripts/run_pruning.sh"
