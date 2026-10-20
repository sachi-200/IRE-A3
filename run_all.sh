#!/usr/bin/env bash
# Regenerate every number, figure and table in the report.
#
#   bash run_all.sh            full data       -> results/
#   bash run_all.sh --quick    first 100K vectors -> results_quick/ (< 15 min)
#
# Environment variables (optional):
#   ANN_DATA_DIR   where datasets are stored/downloaded (default: ./data)
#   ANN_THREADS    threads for multi-threaded runs (default: physical cores)
set -euo pipefail
cd "$(dirname "$0")"

QUICK=()
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=(--quick) ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

PY=${PYTHON:-python3}
START=$(date +%s)

echo "== Unit tests =="
$PY -m pytest -q tests

echo "== Part 1: exact baseline =="
$PY part1_exact.py "${QUICK[@]}"

echo "== Part 2: HNSW, IVF-Flat, IVF-PQ =="
$PY part2_bench.py "${QUICK[@]}"
$PY part2_report.py "${QUICK[@]}"

echo "== Collect results =="
$PY collect_results.py "${QUICK[@]}"

echo "Done in $(( $(date +%s) - START ))s"
