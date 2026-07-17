#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="data/k11_gemma4_determinism/k11_gemma4_determinism_20260717Tquiet_glm_done"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server"
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"

# run_01
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device ROCm0 --device-draft ROCm0 -ngl 99 --spec-draft-ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30001

# run_02
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device ROCm0 --device-draft ROCm0 -ngl 99 --spec-draft-ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30002

# run_03
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device ROCm0 --device-draft ROCm0 -ngl 99 --spec-draft-ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30003
