#!/bin/bash
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
MODEL=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-23/thread-sweep
mkdir -p "$OUT"
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

run() {
  local name=$1 cpuset=$2 threads=$3; shift 3
  echo "=== $name: cpuset=$cpuset -t $threads ==="
  taskset -c "$cpuset" "$BIN" -m "$MODEL" -t "$threads" -p 0 -n 64 -r 2 -o json "$@" > "$OUT/$name.json" 2> "$OUT/$name.log"
  tail -6 "$OUT/$name.json"
  echo ""
}

run t024 0-23 24
run t048 0-47 48
run t096 0-95 96
run t144 0-143 144

# 192t: no taskset, --numa distribute, --mlock
echo "=== t192 full machine + numa distribute + mlock ==="
"$BIN" -m "$MODEL" -t 192 -p 0 -n 64 -r 2 --numa distribute -mmp 1 -o json > "$OUT/t192.json" 2> "$OUT/t192.log"
tail -6 "$OUT/t192.json"
