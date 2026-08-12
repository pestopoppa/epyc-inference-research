#!/bin/bash
set -euo pipefail

# Re-run this K35/P-GPU-1 plan inside an approved operator bench window.
# The dry-run preparer only writes this script; it does not execute inference.
# P-GPU-1 caveat: Ratified P-GPU-1 is production-named-kernel only: experimental, candidate, or fork GPU rows remain observation-grade unless MEASUREMENT explicitly permits pre-promotion evidence or retro-certification.
# Default experimental-v7 K35 rows are promotion observations unless the signed protocol says otherwise.
: "${K35_RUN_ID:=k35_stack_context_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
export K35_EXECUTION_BASE="${K35_EXECUTION_BASE:-/workspace/worktrees/v9-promotion-research/data/k35_stack_context_matrix}"
export K35_EXEC_OUTPUT_DIR="${K35_EXEC_OUTPUT_DIR:-${K35_EXECUTION_BASE}/${K35_RUN_ID}}"
cd /workspace/worktrees/v9-promotion-research
/usr/bin/python3 /workspace/worktrees/v9-promotion-research/scripts/benchmark/k35_stack_context_matrix_runner.py --execute --only v9_dsv4_iq3_xxs_dspark_request_nmax0 --only v9_dsv4_iq3_xxs_dspark_request_nmax3 --context 2048 --max-tokens 64 --min-completion-tokens 64 --output-dir "$K35_EXEC_OUTPUT_DIR" --binary /mnt/raid0/llm/llama.cpp/build/bin/llama-server --port-base 19640 --request-timeout 900 --startup-timeout 300 --reps 1 --warmup-discard-policy 'no warm-up requests; no discarded reps; fresh server per arm; quick observation' --cpu-interference-policy 'live production stack resident but idle; operator-requested quick paired observation; not claim-grade' --allow-dirty-host
