#!/bin/bash
set -euo pipefail

# launch frontdoor
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 19340 -np 1 -c 8192 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type draft-mtp --spec-draft-n-max 3

# launch gemma
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf --host 127.0.0.1 --port 19341 -np 1 -c 8192 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type ngram-mod,draft-mtp -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-draft-n-max 2 --spec-draft-threads 16 --spec-draft-device ROCm0 --spec-draft-ngl 99 --no-mmap

# launch minicpm
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf --mmproj /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf --host 127.0.0.1 --port 19342 -np 1 -c 8192 -t 32 -ub 512 --metrics --slots --jinja --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type none
