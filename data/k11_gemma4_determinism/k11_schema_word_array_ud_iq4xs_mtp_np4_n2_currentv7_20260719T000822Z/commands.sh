#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="/mnt/raid0/llm/epyc-inference-research/data/k11_gemma4_determinism/k11_schema_word_array_ud_iq4xs_mtp_np4_n2_currentv7_20260719T000822Z"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server"
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"

# run_01
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 8192 -fa on -rea off --host 127.0.0.1 --port 30001 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_02
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 8192 -fa on -rea off --host 127.0.0.1 --port 30002 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling
