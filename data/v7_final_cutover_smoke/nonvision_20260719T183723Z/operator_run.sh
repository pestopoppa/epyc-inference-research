#!/bin/bash
set -euo pipefail

# Re-run this K35/P-GPU-1 plan inside an approved operator bench window.
# The dry-run preparer only writes this script; it does not execute inference.
# P-GPU-1 caveat: Ratified P-GPU-1 is production-named-kernel only: experimental, candidate, or fork GPU rows remain observation-grade unless MEASUREMENT explicitly permits pre-promotion evidence or retro-certification.
# Default experimental-v7 K35 rows are promotion observations unless the signed protocol says otherwise.
: "${K35_RUN_ID:=k35_stack_context_matrix_$(date -u +%Y%m%dT%H%M%SZ)}"
export K35_EXECUTION_BASE="${K35_EXECUTION_BASE:-/mnt/raid0/llm/epyc-inference-research/data/k35_stack_context_matrix}"
export K35_EXEC_OUTPUT_DIR="${K35_EXEC_OUTPUT_DIR:-${K35_EXECUTION_BASE}/${K35_RUN_ID}}"
cd /mnt/raid0/llm/epyc-inference-research
/usr/bin/python3 /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/k35_stack_context_matrix_runner.py --execute --only frontdoor_gpu_native_mtp --only worker_general_cpu_composed_spec --only architect_general_cpu_native_mtp --only ingest_long_context_cpu_default_experts --context 2048 --max-tokens 256 --min-completion-tokens 64 --output-dir "$K35_EXEC_OUTPUT_DIR" --binary /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server --port-base 19300 --request-timeout 900 --startup-timeout 300 --reps 1 --warmup-discard-policy 'final cutover smoke: no warmup discard; one fresh experimental-v7 server per selected role at 2K context and 256 generated-token cap' --cpu-interference-policy 'AutoPilot stopped; no live llama-server/llama-cli/llama-bench/rocprof/perf/KFD PIDs at preflight; selected roles execute sequentially'
