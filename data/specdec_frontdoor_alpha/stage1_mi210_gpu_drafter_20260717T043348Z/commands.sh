#!/usr/bin/env bash
set -euo pipefail

# Dry-run package only. Review and execute manually in a post-GLM quiet window.
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"
# pinned experimental binary: /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server
# prerequisite strict harness: /mnt/raid0/llm/epyc-inference-research/scripts/benchmark/n5_frontdoor_drafter_retest.sh --strict

# arm: baseline_cpu_target_no_spec
# purpose: CPU target only baseline with speculation disabled
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -t 96 -np 1 -c 8192 -ub 8192 -ngl 0 --device none --host 127.0.0.1 --port 19187 --metrics --slots --jinja --reasoning auto -fa on -ctk q8_0 -ctv q8_0 --spec-type none

# arm: stage1_cpu_target_mi210_external_drafter
# purpose: CPU target plus MI210 external drafter Stage-1 candidate
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -t 96 -np 1 -c 8192 -ub 8192 -ngl 0 --device none --host 127.0.0.1 --port 19188 --metrics --slots --jinja --reasoning auto -fa on -ctk q8_0 -ctv q8_0 -md /mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf --spec-type draft-tree --spec-draft-n-max 1 --spec-draft-p-split 0.05 --spec-draft-device ROCm0 --spec-draft-ngl 99
