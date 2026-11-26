#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${1:-$ROOT/results/table_3_fixed_1pct}"
OUTPUT_DIR="${2:-$ROOT/results/main/tables/table_3}"

"$PYTHON_BIN" "$ROOT/pruning/analyze_fixed_1pct.py" \
  --input "$INPUT_DIR/fixed_1pct_runs.csv" \
  --output-dir "$OUTPUT_DIR"
