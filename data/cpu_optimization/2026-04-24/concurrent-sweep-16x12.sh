#!/bin/bash
# Extended concurrent-load sweep: 16×12t on dense Q8 models + Coder-32B Q4
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

# 16 instances × 12 logical threads each — SMT-paired: 6 physical cores + 6 HT siblings.
# Layout across nodes: 8 instances per node, each spanning 6 physical cores.
# Node 0 physical: 0-47, HT: 96-143. So 8 instances of (6 phys + 6 HT):
#   inst0=0-5,96-101   inst1=6-11,102-107   inst2=12-17,108-113   inst3=18-23,114-119
#   inst4=24-29,120-125 inst5=30-35,126-131 inst6=36-41,132-137   inst7=42-47,138-143
# Node 1 physical: 48-95, HT: 144-191.
#   inst8=48-53,144-149  inst9=54-59,150-155  inst10=60-65,156-161 inst11=66-71,162-167
#   inst12=72-77,168-173 inst13=78-83,174-179 inst14=84-89,180-185 inst15=90-95,186-191

mk_cpuset() {
  local i=$1
  # First 8 on node 0, next 8 on node 1
  if [ "$i" -lt 8 ]; then
    local phys_start=$((i * 6))
    local phys_end=$((phys_start + 5))
    local ht_start=$((96 + phys_start))
    local ht_end=$((96 + phys_end))
  else
    local j=$((i - 8))
    local phys_start=$((48 + j * 6))
    local phys_end=$((phys_start + 5))
    local ht_start=$((96 + phys_start))
    local ht_end=$((96 + phys_end))
  fi
  echo "${phys_start}-${phys_end},${ht_start}-${ht_end}"
}

run_sweep() {
  local label=$1 model=$2
  echo "=========================================="
  echo "MODEL: $label ($model)"
  echo "=========================================="
  for n_inst in 4 8 16; do
    local threads=$((192 / n_inst))
    echo ""
    echo "--- ${n_inst}x${threads}t concurrent ---"
    local pids=()
    for i in $(seq 0 $((n_inst - 1))); do
      if [ "$n_inst" = "4" ]; then
        case $i in
          0) cpuset="0-23,96-119";;
          1) cpuset="24-47,120-143";;
          2) cpuset="48-71,144-167";;
          3) cpuset="72-95,168-191";;
        esac
      elif [ "$n_inst" = "8" ]; then
        case $i in
          0) cpuset="0-11,96-107";;
          1) cpuset="12-23,108-119";;
          2) cpuset="24-35,120-131";;
          3) cpuset="36-47,132-143";;
          4) cpuset="48-59,144-155";;
          5) cpuset="60-71,156-167";;
          6) cpuset="72-83,168-179";;
          7) cpuset="84-95,180-191";;
        esac
      else
        cpuset=$(mk_cpuset $i)
      fi
      taskset -c "$cpuset" "$BIN" -m "$model" -t "$threads" -p 0 -n 32 -r 2 -o json > "$OUT/${label}-${n_inst}x${threads}t-inst${i}.json" 2>&1 &
      pids+=($!)
    done
    wait "${pids[@]}"
    local agg=0
    for i in $(seq 0 $((n_inst - 1))); do
      local ts=$(grep '"avg_ts"' "$OUT/${label}-${n_inst}x${threads}t-inst${i}.json" | tail -1 | awk -F: '{print $2}' | tr -d ' ,')
      [ -n "$ts" ] && agg=$(echo "$agg + $ts" | bc -l)
      printf "  inst%02d: %s t/s\n" $i "$ts"
    done
    echo "  AGGREGATE: $agg t/s"
  done
}

run_sweep qwen36-27b-q8 /mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
run_sweep qwen36-35b-a3b-q8 /mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
run_sweep coder32b-q4 /mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf
