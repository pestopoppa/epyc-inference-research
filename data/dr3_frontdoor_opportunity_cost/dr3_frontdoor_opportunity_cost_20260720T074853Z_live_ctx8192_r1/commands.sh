#!/bin/bash
set -euo pipefail

# frontdoor alone before DR-3 lease
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 22420 -np 1 -c 9728 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type none

# DR-3 K2 lane active
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin ROCR_VISIBLE_DEVICES=0 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf -md /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf --host 127.0.0.1 --port 22421 -np 1 -c 8192 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type draft-mtp --spec-draft-device ROCm0 --spec-draft-ngl all --spec-draft-n-max 2

# frontdoor after DR-3 eviction/reload
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 22422 -np 1 -c 9728 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type none
