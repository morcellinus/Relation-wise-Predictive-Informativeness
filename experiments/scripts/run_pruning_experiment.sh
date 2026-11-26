#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-$ROOT/results/table_3_fixed_1pct}"
if [[ $# -gt 0 ]]; then
  shift
fi
RPI_DIR="$OUTPUT_DIR/operational_rpi"

if [[ ! -s "$RPI_DIR/metrics/rpi_relation.csv" ]]; then
  "$PYTHON_BIN" "$ROOT/realworld/compute_rpi.py" \
    --data-dir "$ROOT/data/processed" \
    --output-dir "$RPI_DIR" \
    --splits 0 1 2 3 4
fi

"$PYTHON_BIN" "$ROOT/pruning/run_fixed_1pct.py" \
  --data-dir "$ROOT/data/processed" \
  --config-dir "$ROOT/configs/realworld_alignment_best_configs" \
  --rpi-dir "$RPI_DIR" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
