#!/bin/bash
# Concurrent-load sweep: 4×48t and 8×24t on Coder-32B Q4
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
MODEL=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

# 4×48t concurrent — SMT-aware pairing: each instance gets 48 physical + their HT siblings
# Quarter map (node 0 phys 0-23 + HT 96-119 = Q0A_SMT, etc.):
# But simpler: just do 4 instances covering physical cores 0-47 and 48-95
# Split node-symmetric: Q0_phys = 0-47 (node 0), Q1_phys = 48-95 (node 1)
# For 4×48t, each needs 48 logical threads. So:
#   inst0 = 0-23 + 96-119 (node 0 half A, with HT)
#   inst1 = 24-47 + 120-143 (node 0 half B, with HT)
#   inst2 = 48-71 + 144-167 (node 1 half A, with HT)
#   inst3 = 72-95 + 168-191 (node 1 half B, with HT)
echo "=== 4x48t concurrent (SMT-paired quarters) ==="
for i in 0 1 2 3; do
  case $i in
    0) cpuset="0-23,96-119";;
    1) cpuset="24-47,120-143";;
    2) cpuset="48-71,144-167";;
    3) cpuset="72-95,168-191";;
  esac
  taskset -c "$cpuset" "$BIN" -m "$MODEL" -t 48 -p 0 -n 32 -r 2 -o json > "$OUT/coder32b-q4-4x48t-inst$i.json" 2>&1 &
done
wait
for i in 0 1 2 3; do
  ts=$(grep '"avg_ts"' "$OUT/coder32b-q4-4x48t-inst$i.json" | tail -1 | awk -F: '{print $2}' | tr -d ' ,')
  echo "  inst$i: $ts t/s"
done

# 8×24t concurrent — each instance gets 12 physical + 12 HT on node-local region
# Layout: 8 instances, each 24 logical = 12 physical + their 12 HT siblings
echo ""
echo "=== 8x24t concurrent (SMT-paired eighths) ==="
for i in 0 1 2 3 4 5 6 7; do
  case $i in
    0) cpuset="0-11,96-107";;     # node 0, 1st eighth
    1) cpuset="12-23,108-119";;   # node 0, 2nd eighth
    2) cpuset="24-35,120-131";;   # node 0, 3rd eighth
    3) cpuset="36-47,132-143";;   # node 0, 4th eighth
    4) cpuset="48-59,144-155";;   # node 1, 5th eighth
    5) cpuset="60-71,156-167";;   # node 1, 6th eighth
    6) cpuset="72-83,168-179";;   # node 1, 7th eighth
    7) cpuset="84-95,180-191";;   # node 1, 8th eighth
  esac
  taskset -c "$cpuset" "$BIN" -m "$MODEL" -t 24 -p 0 -n 32 -r 2 -o json > "$OUT/coder32b-q4-8x24t-inst$i.json" 2>&1 &
done
wait
for i in 0 1 2 3 4 5 6 7; do
  ts=$(grep '"avg_ts"' "$OUT/coder32b-q4-8x24t-inst$i.json" | tail -1 | awk -F: '{print $2}' | tr -d ' ,')
  echo "  inst$i: $ts t/s"
done
