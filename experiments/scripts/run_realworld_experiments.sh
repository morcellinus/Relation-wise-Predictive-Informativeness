#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-$ROOT/results/realworld_alignment}"
if [[ $# -gt 0 ]]; then
  shift
fi

"$PYTHON_BIN" "$ROOT/realworld_alignment/run_backbone_alignment.py" \
  --data-dir "$ROOT/data/processed" \
  --best-config-dir "$ROOT/configs/realworld_alignment_best_configs" \
  --output-dir "$OUTPUT_DIR" \
  --use-best-configs \
  --epochs 300 \
  --patience 50 \
  "$@"
