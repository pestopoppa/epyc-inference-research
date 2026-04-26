#!/bin/bash
# 4x48t concurrent under NPS4 — ONE instance per NUMA node.
# Each instance: 48 logical (24 phys + 24 SMT) = full node, membind to that node.
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24-nps4/concurrent
mkdir -p "$OUT"
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

MODEL=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

run_4x48() {
  local label=$1 model=$2
  echo "== $label 4x48t (1 per NUMA node, -t 48) =="
  local pids=()
  for i in 0 1 2 3; do
    numactl --cpunodebind=$i --membind=$i "$BIN" -m "$model" -t 48 -p 0 -n 32 -r 2 -o json \
      > "$OUT/${label}-4x48-inst${i}.json" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local agg=0
  for i in 0 1 2 3; do
    local ts=$(grep '"avg_ts"' "$OUT/${label}-4x48-inst${i}.json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
    [ -n "$ts" ] && agg=$(echo "$agg + $ts" | bc -l)
    printf "  inst%d: %s t/s\n" "$i" "${ts:-NA}"
  done
  echo "  AGGREGATE: $agg t/s"
}

run_4x24_phys() {
  local label=$1 model=$2
  echo "== $label 4x24t (1 per node, phys-only) =="
  local pids=()
  local cpusets=("0-23" "24-47" "48-71" "72-95")
  for i in 0 1 2 3; do
    numactl --membind=$i taskset -c "${cpusets[$i]}" "$BIN" -m "$model" -t 24 -p 0 -n 32 -r 2 -o json \
      > "$OUT/${label}-4x24phys-inst${i}.json" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local agg=0
  for i in 0 1 2 3; do
    local ts=$(grep '"avg_ts"' "$OUT/${label}-4x24phys-inst${i}.json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
    [ -n "$ts" ] && agg=$(echo "$agg + $ts" | bc -l)
    printf "  inst%d: %s t/s\n" "$i" "${ts:-NA}"
  done
  echo "  AGGREGATE: $agg t/s"
}

run_4x48 qwen3-coder-30b-a3b-q4 "$MODEL"
run_4x24_phys qwen3-coder-30b-a3b-q4 "$MODEL"
