#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$ROOT/scripts/run_realworld_experiments.sh" "$ROOT/results/realworld_alignment"
"$ROOT/scripts/build_realworld_artifacts.sh" "$ROOT/results/realworld_alignment" "$ROOT/results/main"
