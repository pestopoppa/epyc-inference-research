#!/bin/bash
# 48x4t under NPS4: 12 instances per NUMA node.
# Per node: 24 phys + 24 SMT = 48 logical, split into 12 instances of 4 logical each (2 phys + 2 SMT).
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24-nps4/concurrent
mkdir -p "$OUT"
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

# Node base CPUs (phys low, HT high)
# node 0 phys 0-23, HT 96-119
# node 1 phys 24-47, HT 120-143
# node 2 phys 48-71, HT 144-167
# node 3 phys 72-95, HT 168-191

mk_cpuset_48x4_nps4() {
  local i=$1
  local node=$((i / 12))
  local j=$((i % 12))          # 0..11 within the node
  local phys_base=$((node * 24 + j * 2))
  local phys_end=$((phys_base + 1))
  local ht_base=$((96 + node * 24 + j * 2))
  local ht_end=$((ht_base + 1))
  echo "${phys_base}-${phys_end},${ht_base}-${ht_end}"
}

run_48x4_nps4() {
  local label=$1 model=$2
  echo "== $label 48x4t NPS4-native =="
  local pids=()
  for i in $(seq 0 47); do
    local cpuset=$(mk_cpuset_48x4_nps4 $i)
    local node=$((i / 12))
    numactl --membind=$node taskset -c "$cpuset" "$BIN" -m "$model" -t 4 -p 0 -n 32 -r 2 -o json \
      > "$OUT/${label}-48x4-inst${i}.json" 2>&1 &
    pids+=($!)
  done
  wait "${pids[@]}"
  local agg=0
  local valid=0
  for i in $(seq 0 47); do
    local ts=$(grep '"avg_ts"' "$OUT/${label}-48x4-inst${i}.json" | head -1 | awk -F: '{print $2}' | tr -d ' ,')
    if [ -n "$ts" ]; then
      agg=$(echo "$agg + $ts" | bc -l)
      valid=$((valid + 1))
    fi
  done
  echo "  valid instances: $valid/48"
  echo "  AGGREGATE: $agg t/s"
}

# 30B-A3B Q4 (worker) — was our canonical baseline model
run_48x4_nps4 qwen3-coder-30b-a3b-q4 /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

# 35B-A3B Q8 — peak under NPS2 at 135 t/s
if [ -f /mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf ]; then
  run_48x4_nps4 qwen36-35b-a3b-q8 /mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
fi
