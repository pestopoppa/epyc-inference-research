#!/bin/bash
set -euo pipefail
cd /mnt/raid0/llm/epyc-inference-research
DS_E1_KV_TIMESTAMP=20260704T040842Z OUTPUT_DIR=/mnt/raid0/llm/epyc-inference-research/data/dynamic_stack/ds_e1_kv_measurements_20260704T040842Z LLAMA_SERVER=/mnt/raid0/llm/llama.cpp/build/bin/llama-server PORT=8194 THREADS=96 UBATCH=8192 N_PREDICT=32 bash scripts/benchmark/ds_e1_kv_measurements.sh --execute
