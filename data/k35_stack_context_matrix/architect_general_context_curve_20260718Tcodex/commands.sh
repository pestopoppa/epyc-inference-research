#!/bin/bash
set -euo pipefail

# architect_general_cpu_native_mtp nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 19100 -np 2 -c 8192 -t 96 -ub 8192 --metrics --slots --jinja --reasoning off --device none -ngl 0 -ctk q4_0 -ctv f16 -fa on --spec-type draft-mtp --mlock --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general

# architect_general_cpu_native_mtp nominal_context=8192 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 19101 -np 2 -c 16384 -t 96 -ub 8192 --metrics --slots --jinja --reasoning off --device none -ngl 0 -ctk q4_0 -ctv f16 -fa on --spec-type draft-mtp --mlock --spec-draft-n-max 4 --slot-save-path /mnt/raid0/llm/cache/kv_slots/architect_general
