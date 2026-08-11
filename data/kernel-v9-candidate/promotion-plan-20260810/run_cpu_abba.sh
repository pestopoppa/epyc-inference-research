#!/bin/bash
set -euo pipefail

RESEARCH=/workspace/worktrees/v9-promotion-research
PLAN="$RESEARCH/data/kernel-v9-candidate/promotion-plan-20260810"
REGION_LOCK=/workspace/repos/epyc-orchestrator/scripts/region-lock

"$REGION_LOCK" run \
    --regions q0,q1,q2,q3 \
    --role bench \
    --tag v9-promotion-cpu-abba-20260810 \
    -- bash "$PLAN/run_cpu_abba_inner.sh"

python3 "$RESEARCH/scripts/benchmark/compare_v9_promotion_arms.py" \
    --baseline-summary "$PLAN/a1-v8-run/summary.json" \
    --baseline-summary "$PLAN/a2-v8-run/summary.json" \
    --candidate-summary "$PLAN/b1-v9-run/summary.json" \
    --candidate-summary "$PLAN/b2-v9-run/summary.json" \
    --baseline-binary /mnt/raid0/llm/llama.cpp/build/bin/llama-server \
    --candidate-binary /mnt/raid0/llm/llama.cpp-experimental/build-v9-cpu/bin/llama-server \
    --minimum-reps 10 \
    --output "$PLAN/cpu-comparison.json"
