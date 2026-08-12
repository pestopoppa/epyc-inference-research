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
/usr/bin/python3 /workspace/worktrees/v9-promotion-research/scripts/benchmark/k35_stack_context_matrix_runner.py --execute --only v9_frontdoor_cpu_native_mtp --only v9_worker_general_cpu_native_mtp --only v9_architect_critic_cpu_native_mtp --only v9_ingest_long_context_cpu_no_spec --context 2048 --max-tokens 128 --min-completion-tokens 128 --output-dir "$K35_EXEC_OUTPUT_DIR" --binary /mnt/raid0/llm/llama.cpp-experimental/build-v9-cpu/bin/llama-server --port-base 19900 --request-timeout 900 --startup-timeout 300 --reps 5 --warmup-discard-policy 'no warm-up requests; no discarded reps; fresh server per rep; graph recapture, if any, is included in the measured row' --cpu-interference-policy 'CPU stack quiet required; runner aborts on AutoPilot or llama workload process blockers unless --allow-dirty-host is explicit' --uptime-waiver operator_20260810_current_boot_pass_stands_reboot_only_on_fail_or_gray
