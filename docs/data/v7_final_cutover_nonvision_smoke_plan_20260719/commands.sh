#!/bin/bash
set -euo pipefail

# frontdoor_gpu_native_mtp nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf --host 127.0.0.1 --port 19300 -np 1 -c 3328 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --spec-type draft-mtp --spec-draft-n-max 3

# worker_general_cpu_composed_spec nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf --host 127.0.0.1 --port 19301 -np 1 -c 3328 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device none -ngl 0 -ctk q8_0 -ctv q8_0 -fa on --spec-type ngram-mod,draft-mtp --no-mmap -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-draft-n-max 2 --spec-draft-threads 16 --spec-draft-device none --spec-draft-ngl 0 --no-op-offload --no-kv-offload

# architect_general_cpu_native_mtp nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 19302 -np 2 -c 6656 -t 96 -ub 8192 --metrics --slots --jinja --reasoning off --device none -ngl 0 -ctk q4_0 -ctv f16 -fa on --spec-type draft-mtp --mlock --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general

# ingest_long_context_cpu_default_experts nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf --host 127.0.0.1 --port 19303 -np 1 -c 3328 -t 96 -ub 8192 --metrics --slots --jinja --reasoning auto --device none -ngl 0 -ctk q4_0 -ctv q4_0 -fa on --spec-type none --mlock --slot-save-path /mnt/raid0/llm/cache/kv_slots/ingest_long_context
