#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/run_hcsbm_experiments.sh" "$ROOT/results/hcsbm"
"$ROOT/scripts/build_hcsbm_artifacts.sh" "$ROOT/results/hcsbm" "$ROOT/results/main/figures"
