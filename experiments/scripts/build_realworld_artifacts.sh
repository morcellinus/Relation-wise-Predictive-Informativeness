#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${1:-$ROOT/results/realworld_alignment}"
OUTPUT_DIR="${2:-$ROOT/results/main}"

"$PYTHON_BIN" "$ROOT/analysis/plot_realworld.py" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR/figures"
"$PYTHON_BIN" "$ROOT/analysis/plot_alignment.py" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR/figures"
"$PYTHON_BIN" "$ROOT/analysis/destination_aggregation.py" \
  --input-dir "$INPUT_DIR" \
  --data-dir "$ROOT/data/processed" \
  --output-dir "$OUTPUT_DIR/tables/table_2"
