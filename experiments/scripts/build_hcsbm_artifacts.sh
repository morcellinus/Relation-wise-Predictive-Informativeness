#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_DIR="${1:-$ROOT/results/hcsbm}"
OUTPUT_DIR="${2:-$ROOT/results/main/figures}"

"$PYTHON_BIN" "$ROOT/analysis/export_hcsbm_figures.py" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
