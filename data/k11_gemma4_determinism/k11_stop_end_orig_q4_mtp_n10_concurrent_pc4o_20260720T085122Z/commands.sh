#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="data/k11_gemma4_determinism/k11_stop_end_orig_q4_mtp_n10_concurrent_pc4o_20260720T085122Z"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server"
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"

# run_01
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30001 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_02
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30002 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_03
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30003 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_04
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30004 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_05
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30005 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_06
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30006 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_07
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30007 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_08
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30008 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_09
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30009 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99

# run_10
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 4 --device ROCm0 -ngl 99 -t 8 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30010 -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99
