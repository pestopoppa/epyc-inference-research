#!/bin/bash
set -euo pipefail
export LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin:/opt/rocm/lib
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export GGML_IQK=1
exec timeout --kill-after=30s 1200s /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-bench \
  -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.Q8_0.gguf \
  -pg 2048,1024 -pg 8192,1024 -pg 32768,512 \
  -r 1 -ngl 99 -dev ROCm0 -fa on -ctk q8_0 -ctv q8_0 \
  -b 4096 -ub 512 -o json --progress
