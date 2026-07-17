#!/usr/bin/env bash
set -euo pipefail

# Generated dry-run package. Execute with the Python runner for cleanup guarantees.
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"

# arm: gpu_no_spec
# purpose: MI210-resident target baseline with speculation disabled
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -t 96 -np 1 -c 8192 -ub 8192 -ngl 99 --device ROCm0 --host 127.0.0.1 --port 32819 --metrics --slots --jinja --reasoning auto -fa on -ctk q8_0 -ctv q8_0 --spec-type none

# arm: gpu_native_mtp
# purpose: MI210-resident target with native MTP enabled
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -t 96 -np 1 -c 8192 -ub 8192 -ngl 99 --device ROCm0 --host 127.0.0.1 --port 44011 --metrics --slots --jinja --reasoning auto -fa on -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 3

# arm: gpu_external_drafter
# purpose: MI210-resident target plus co-resident external Qwen3.5-0.8B drafter
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf -t 96 -np 1 -c 8192 -ub 8192 -ngl 99 --device ROCm0 --host 127.0.0.1 --port 52979 --metrics --slots --jinja --reasoning auto -fa on -ctk q8_0 -ctv q8_0 -md /mnt/raid0/llm/scratch/n5/Qwen3.5-0.8B-Q8_0.frontdoor-mtp-specials.gguf --spec-type draft-tree --spec-draft-n-max 1 --spec-draft-p-split 0.05 --spec-draft-device ROCm0 --spec-draft-ngl 99
