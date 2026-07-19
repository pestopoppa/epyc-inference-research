#!/usr/bin/env bash
set -euo pipefail

# Verifier-only replay: beneficiary candidates are loaded from existing artifacts.
env -i HOME=/tmp LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp-experimental/build-hip/bin OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PATH=/usr/bin:/bin /mnt/raid0/llm/llama.cpp-experimental/build-hip/bin/llama-server -m /mnt/raid0/llm/models/Qwable-v1-GGUF/Qwable-v1.IQ4_XS.gguf --host 127.0.0.1 --port 18941 --device ROCm0 -ngl 99 -t 6 -c 16384 -fa on --no-webui
