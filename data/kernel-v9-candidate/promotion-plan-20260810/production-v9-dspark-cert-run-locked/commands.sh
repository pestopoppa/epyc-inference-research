#!/bin/bash
set -euo pipefail

# v9_dsv4_q8_dspark_request_nmax0 nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/deepseek-v4-flash-0731/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf --host 127.0.0.1 --port 19630 -np 1 -c 3088 -t 24 --metrics --slots -ub 512 --jinja --reasoning auto --device none -ngl 0 -ctk f16 -ctv f16 -fa on --spec-type draft-dspark -md /mnt/raid0/llm/models/deepseek-v4-flash-0731/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf --spec-draft-n-max 3 --spec-draft-device none --spec-draft-ngl 0 -b 512 --no-repack --no-warmup

# v9_dsv4_q8_dspark_request_nmax3 nominal_context=2048 rep=1
env LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build/bin GGML_IQK=1 ROCR_VISIBLE_DEVICES=-1 HIP_VISIBLE_DEVICES=-1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 numactl --interleave=all /mnt/raid0/llm/llama.cpp/build/bin/llama-server -m /mnt/raid0/llm/models/deepseek-v4-flash-0731/UD-Q8_K_XL/DeepSeek-V4-Flash-0731-UD-Q8_K_XL-00001-of-00005.gguf --host 127.0.0.1 --port 19631 -np 1 -c 3088 -t 24 --metrics --slots -ub 512 --jinja --reasoning auto --device none -ngl 0 -ctk f16 -ctv f16 -fa on --spec-type draft-dspark -md /mnt/raid0/llm/models/deepseek-v4-flash-0731/dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf --spec-draft-n-max 3 --spec-draft-device none --spec-draft-ngl 0 -b 512 --no-repack --no-warmup
