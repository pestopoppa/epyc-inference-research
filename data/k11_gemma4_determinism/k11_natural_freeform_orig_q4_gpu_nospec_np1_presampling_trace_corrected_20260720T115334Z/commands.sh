#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="data/k11_gemma4_determinism/k11_natural_freeform_orig_q4_gpu_nospec_np1_presampling_trace_corrected_20260720T115334Z"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server"
export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin"

# run_01
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30001 --spec-type none

# run_02
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30002 --spec-type none

# run_03
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30003 --spec-type none

# run_04
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30004 --spec-type none

# run_05
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30005 --spec-type none

# run_06
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30006 --spec-type none

# run_07
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30007 --spec-type none

# run_08
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30008 --spec-type none

# run_09
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30009 --spec-type none

# run_10
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf -np 1 --device ROCm0 -ngl 99 -t 96 -ub 512 -c 16384 -fa on -rea off --host 127.0.0.1 --port 30010 --spec-type none
