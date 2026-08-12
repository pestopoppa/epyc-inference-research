#!/bin/bash
# Evaluate one canonical arm (python3-only) against the local sandbox backend.
# Usage: run_eval_arm.sh <canonical_runtime|canonical_memory|canonical_normal>
set -euo pipefail
ARM="$1"
export BACKEND_BASE_URL="http://127.0.0.1:8999"
cd /workspace/tmp/effibench-x-upstream
exec nice -n 10 /workspace/tmp/effibench-venv/bin/python evaluate_solution.py evaluate \
    /workspace/tmp/effibench-gate/data/dataset \
    /workspace/tmp/effibench-gate/data/solutions \
    -o /workspace/tmp/effibench-gate/data/evaluation \
    -l python3 \
    -m "$ARM" \
    -p 12 -t 4
