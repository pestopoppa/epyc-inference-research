#!/bin/bash
# 48×4t concurrent: 2 physical + 2 HT siblings per instance, 48 instances total
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

mk_cpuset_48() {
  local i=$1
  # 24 instances per node, each 2 phys + 2 HT
  if [ "$i" -lt 24 ]; then
    local phys_start=$((i * 2))
    local phys_end=$((phys_start + 1))
    local ht_start=$((96 + phys_start))
    local ht_end=$((96 + phys_end))
  else
    local j=$((i - 24))
    local phys_start=$((48 + j * 2))
    local phys_end=$((phys_start + 1))
    local ht_start=$((96 + phys_start))
    local ht_end=$((96 + phys_end))
  fi
  echo "${phys_start}-${phys_end},${ht_start}-${ht_end}"
}

run_48x4() {
  local label=$1 model=$2
  echo "=========================================="
  echo "MODEL: $label ($model)"
  echo "=========================================="
  echo "--- 48x4t concurrent ---"
  local pids=()
  for i in $(seq 0 47); do
    local cpuset=$(mk_cpuset_48 $i)
    taskset -c "$cpuset" "$BIN" -m "$model" -t 4 -p 0 -n 32 -r 2 -o json > "$OUT/${label}-48x4t-inst${i}.json" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local agg=0
  for i in $(seq 0 47); do
    local ts=$(grep '"avg_ts"' "$OUT/${label}-48x4t-inst${i}.json" | tail -1 | awk -F: '{print $2}' | tr -d ' ,')
    [ -n "$ts" ] && agg=$(echo "$agg + $ts" | bc -l)
  done
  echo "  AGGREGATE (48x4t): $agg t/s"
}

run_48x4 qwen36-27b-q8 /mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
run_48x4 qwen36-35b-a3b-q8 /mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
run_48x4 coder32b-q4 /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf
