#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/run_pruning_experiment.sh" "$ROOT/results/table_3_fixed_1pct"
"$ROOT/scripts/build_pruning_table.sh" "$ROOT/results/table_3_fixed_1pct" "$ROOT/results/main/tables/table_3"
