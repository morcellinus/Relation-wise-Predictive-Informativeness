#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${1:-$ROOT/results/hcsbm}"
if [[ $# -gt 0 ]]; then
  shift
fi

"$PYTHON_BIN" "$ROOT/hcsbm/hcbsm_experiment/run_hcsbm_final_rq1_labelspace.py" --output-root "$OUTPUT_DIR" --num-seeds 5 --resume "$@"
"$PYTHON_BIN" "$ROOT/hcsbm/hcbsm_experiment/run_hcsbm_final_rq2_labelspace.py" --output-root "$OUTPUT_DIR" --num-seeds 5 --resume "$@"
"$PYTHON_BIN" "$ROOT/hcsbm/hcbsm_experiment/run_hcsbm_final_rq3_labelspace.py" --output-root "$OUTPUT_DIR" --num-seeds 5 --resume "$@"
