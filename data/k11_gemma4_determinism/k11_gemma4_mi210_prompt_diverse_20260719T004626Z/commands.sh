#!/usr/bin/env bash
set -euo pipefail

# draft_mtp
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf --host 127.0.0.1 --port 58354 -np 1 -c 8192 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --no-mmap -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# ngram_mod_draft_mtp
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf --host 127.0.0.1 --port 46376 -np 1 -c 8192 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --no-mmap -md /mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf --spec-type ngram-mod,draft-mtp --spec-draft-n-max 2 --device-draft ROCm0 --spec-draft-ngl 99 --no-spec-draft-backend-sampling

# no_spec
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf --host 127.0.0.1 --port 54976 -np 1 -c 8192 -t 96 -ub 512 --metrics --slots --jinja --reasoning off --device ROCm0 -ngl 99 -ctk q8_0 -ctv q8_0 -fa on --no-mmap --spec-type none
