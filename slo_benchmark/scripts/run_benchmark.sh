#!/bin/bash
# SLO Scheduling Benchmark Runner
# Run from repo root: ./slo_benchmark/scripts/run_benchmark.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

echo "========================================"
echo "  SLO Scheduling Benchmark"
echo "========================================"

python slo_benchmark/scripts/run_benchmark.py "$@"
