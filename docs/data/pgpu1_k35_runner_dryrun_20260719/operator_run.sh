#!/bin/bash
set -euo pipefail

# Re-run this K35/P-GPU-1 plan inside an approved operator bench window.
# The dry-run preparer only writes this script; it does not execute inference.
: "${K35_RUN_ID:=k35_stack_context_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
export K35_EXECUTION_BASE="${K35_EXECUTION_BASE:-/mnt/raid0/llm/epyc-inference-research/data/k35_stack_context_matrix}"
export K35_EXEC_OUTPUT_DIR="${K35_EXEC_OUTPUT_DIR:-${K35_EXECUTION_BASE}/${K35_RUN_ID}}"
cd /mnt/raid0/llm/epyc-inference-research
/usr/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/k35_stack_context_matrix_runner.py --execute --only frontdoor_gpu_native_mtp --context 8192 --max-tokens 1024 --min-completion-tokens 128 --output-dir "$K35_EXEC_OUTPUT_DIR" --binary /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server --port-base 19100 --request-timeout 900 --startup-timeout 300 --reps 5 --warmup-discard-policy 'no warm-up requests; no discarded reps; fresh server per rep; graph recapture, if any, is included in the measured row' --cpu-interference-policy 'CPU stack quiesced; AutoPilot and unrelated llama workloads stopped before operator-window execution'
