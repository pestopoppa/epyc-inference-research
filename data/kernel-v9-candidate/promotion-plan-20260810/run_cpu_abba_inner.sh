#!/bin/bash
set -euo pipefail

PLAN=/workspace/worktrees/v9-promotion-research/data/kernel-v9-candidate/promotion-plan-20260810

"$PLAN/a1-v8-plan/operator_run.sh"
"$PLAN/b1-v9-plan/operator_run.sh"
"$PLAN/b2-v9-plan/operator_run.sh"
"$PLAN/a2-v8-plan/operator_run.sh"
