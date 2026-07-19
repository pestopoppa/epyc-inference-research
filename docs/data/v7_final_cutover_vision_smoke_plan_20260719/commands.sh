#!/bin/bash
set -euo pipefail

# worker_vision_cpu_qwen25vl
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf --host 127.0.0.1 --port 19400 -np 2 -c 8192 -t 24 --flash-attn on --device none

# vision_escalation_cpu_qwen25vl_alias
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf --mmproj /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf --host 127.0.0.1 --port 19401 -np 2 -c 8192 -t 24 --flash-attn on --device none

# vision_candidate_mi210_minicpm_o45_q4
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/MiniCPM-o-4_5-Q4_K_M.gguf --mmproj /mnt/raid0/llm/models/MiniCPM-o-4_5-gguf/vision/MiniCPM-o-4_5-vision-F16.gguf --host 127.0.0.1 --port 19402 -np 1 -c 8192 -t 24 --flash-attn on --device ROCm0 --reasoning off
