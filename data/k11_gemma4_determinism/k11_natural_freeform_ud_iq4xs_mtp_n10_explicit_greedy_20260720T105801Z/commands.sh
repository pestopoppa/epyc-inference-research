#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="data/k11_gemma4_determinism/k11_natural_freeform_ud_iq4xs_mtp_n10_explicit_greedy_20260720T105801Z"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server"
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"

# run_01
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30001 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_02
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30002 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_03
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30003 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_04
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30004 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_05
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30005 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_06
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30006 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_07
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30007 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_08
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30008 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_09
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30009 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# run_10
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-UD-IQ4_XS.gguf -np 4 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30010 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling
