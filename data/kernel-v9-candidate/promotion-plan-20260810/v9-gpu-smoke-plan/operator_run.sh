#!/bin/bash
set -euo pipefail

# Re-run this K35/P-GPU-1 plan inside an approved operator bench window.
# The dry-run preparer only writes this script; it does not execute inference.
# P-GPU-1 caveat: Ratified P-GPU-1 is production-named-kernel only: experimental, candidate, or fork GPU rows remain observation-grade unless MEASUREMENT explicitly permits pre-promotion evidence or retro-certification.
# Default experimental-v7 K35 rows are promotion observations unless the signed protocol says otherwise.
export K35_EXEC_OUTPUT_DIR="${K35_EXEC_OUTPUT_DIR:-/workspace/worktrees/v9-promotion-research/data/kernel-v9-candidate/promotion-plan-20260810/v9-gpu-smoke-run}"
cd /workspace/worktrees/v9-promotion-research
/usr/bin/python3 /workspace/worktrees/v9-promotion-research/scripts/benchmark/k35_stack_context_matrix_runner.py --execute --only v9_architect_general_gpu_native_mtp --only v9_coder_escalation_gpu_request_no_spec --only v9_worker_vision_gpu_no_spec --context 2048 --max-tokens 128 --min-completion-tokens 128 --output-dir "$K35_EXEC_OUTPUT_DIR" --binary /mnt/raid0/llm/llama.cpp-experimental/build-v9-hip/bin/llama-server --port-base 19900 --request-timeout 900 --startup-timeout 300 --reps 1 --warmup-discard-policy 'no warm-up requests; no discarded reps; fresh server per rep; graph recapture, if any, is included in the measured row' --cpu-interference-policy 'CPU stack quiet required; runner aborts on AutoPilot or llama workload process blockers unless --allow-dirty-host is explicit' --uptime-waiver operator_20260810_current_boot_pass_stands_reboot_only_on_fail_or_gray
