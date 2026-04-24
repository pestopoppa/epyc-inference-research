#!/bin/bash
# 32×6t concurrent sweep: 3 physical cores + 3 HT siblings per instance
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

mk_cpuset_32() {
  local i=$1
  # 16 instances per node, each 3 physical + 3 HT siblings
  if [ "$i" -lt 16 ]; then
    local phys_start=$((i * 3))
    local phys_end=$((phys_start + 2))
    local ht_start=$((96 + phys_start))
    local ht_end=$((96 + phys_end))
  else
    local j=$((i - 16))
    local phys_start=$((48 + j * 3))
    local phys_end=$((phys_start + 2))
    local ht_start=$((96 + phys_start))
    local ht_end=$((96 + phys_end))
  fi
  echo "${phys_start}-${phys_end},${ht_start}-${ht_end}"
}

run_32x6() {
  local label=$1 model=$2
  echo "=========================================="
  echo "MODEL: $label ($model)"
  echo "=========================================="
  echo "--- 32x6t concurrent ---"
  local pids=()
  for i in $(seq 0 31); do
    local cpuset=$(mk_cpuset_32 $i)
    taskset -c "$cpuset" "$BIN" -m "$model" -t 6 -p 0 -n 32 -r 2 -o json > "$OUT/${label}-32x6t-inst${i}.json" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local agg=0
  for i in $(seq 0 31); do
    local ts=$(grep '"avg_ts"' "$OUT/${label}-32x6t-inst${i}.json" | tail -1 | awk -F: '{print $2}' | tr -d ' ,')
    [ -n "$ts" ] && agg=$(echo "$agg + $ts" | bc -l)
    printf "  inst%02d: %s t/s\n" $i "$ts"
  done
  echo "  AGGREGATE (32x6t): $agg t/s"
}

run_32x6 qwen36-27b-q8 /mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
run_32x6 qwen36-35b-a3b-q8 /mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
run_32x6 coder32b-q4 /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf
