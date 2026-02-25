#!/bin/bash
# DeepSeek-R1 Fast Start Script (Restores 5 tokens/sec perf)

set -e

start_monitoring() {
  vmstat 1 >vmstat.log &
  pid_vmstat=$!
  nmon -f -s 1 -c 300 &
  pid_nmon=$!
}

stop_monitoring() {
  kill $pid_vmstat $pid_nmon
}

# === FASTEST SETTINGS ===
unset LLAMA_MLOCK             # mlock disables fast mmap
export LLAMA_CACHE_SIZE=65536 # 64GB = verified fast zone
export OMP_NUM_THREADS=88     # match physical cores
export LLAMA_NUMA=distribute  # better than isolate in your tests

#MODEL_PATH="DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf"
MODEL_PATH="DeepSeek-R1-Q4_K_XL/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf"

start_monitoring

llama-cli \
  -m "$MODEL_PATH" \
  -c 16384 \
  -b 2048 \
  -ub 512 \
  -t $OMP_NUM_THREADS \
  --numa "$LLAMA_NUMA" \
  --rope-freq-base 20000 \
  --rope-freq-scale 0.5 \
  --reasoning-format deepseek \
  --chat-template deepseek \
  -i \
  --log-file "/mnt/raid0/llm/LOGS/deepseek_$(date +%Y%m%d_%H%M%S).log"

# llama-cli -m DeepSeek-R1-Q8_0/DeepSeek-R1.Q8_0-00001-of-00015.gguf --n-gpu-layers 0 -t 96 --ctx-size 8192

stop_monitoring
