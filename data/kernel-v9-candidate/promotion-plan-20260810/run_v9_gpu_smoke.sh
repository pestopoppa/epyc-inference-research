#!/bin/bash
set -euo pipefail

REGION_LOCK=/workspace/repos/epyc-orchestrator/scripts/region-lock
OPERATOR_RUN=/workspace/worktrees/v9-promotion-research/data/kernel-v9-candidate/promotion-plan-20260810/v9-gpu-smoke-plan/operator_run.sh

"$REGION_LOCK" run \
    --regions q3 \
    --role bench \
    --tag v9-promotion-gpu-functional-20260810 \
    -- bash "$OPERATOR_RUN"
