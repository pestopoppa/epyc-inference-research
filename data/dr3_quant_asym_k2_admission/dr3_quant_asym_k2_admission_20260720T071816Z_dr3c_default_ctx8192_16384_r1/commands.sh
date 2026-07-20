#!/usr/bin/env bash
set -euo pipefail

# DR-3 live runner command templates only.
# Use scripts/benchmark/dr3_quant_asym_k2_admission_runner.py --execute for evidence.

# template: cpu_baseline_ctx8192
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 22120 -np 1 -c 8192 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type none

# template: combined_k2_ctx8192
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin ROCR_VISIBLE_DEVICES=0 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf -md /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf --host 127.0.0.1 --port 22121 -np 1 -c 8192 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type draft-mtp --spec-draft-device ROCm0 --spec-draft-ngl all --spec-draft-n-max 2

# template: cpu_baseline_ctx16384
LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf --host 127.0.0.1 --port 22122 -np 1 -c 16384 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type none

# template: combined_k2_ctx16384
HIP_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 PATH=/usr/bin:/bin:/opt/rocm/bin ROCR_VISIBLE_DEVICES=0 /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf -md /mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf --host 127.0.0.1 --port 22123 -np 1 -c 16384 -t 96 -ub 1024 --metrics --slots --jinja --reasoning off --device none -ngl 0 --spec-type draft-mtp --spec-draft-device ROCm0 --spec-draft-ngl all --spec-draft-n-max 2
