#!/bin/bash
# NPS4 thread sweep — comparable to 2026-04-23 thread-sweep.sh with NUMA-aware variants.
# Under NPS4: node 0=[0-23,96-119], node 1=[24-47,120-143], node 2=[48-71,144-167], node 3=[72-95,168-191]
set -u
cd /mnt/raid0/llm/llama.cpp-experimental
MODEL=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-24-nps4/thread-sweep
mkdir -p "$OUT"
BIN=./build-llamafile-on/bin/llama-bench
export LD_LIBRARY_PATH=./build-llamafile-on/bin

run() {
  local name=$1 ; shift
  echo "=== $name: $* ==="
  "$@" -m "$MODEL" -p 0 -n 64 -r 3 -o json > "$OUT/$name.json" 2> "$OUT/$name.log" && \
    tail -1 "$OUT/$name.json" | head -c 400 && echo ""
  echo ""
}

# Single-node (node 0 only)
run t024-node0         taskset -c 0-23              "$BIN" -t 24
run t048-node0-smt     taskset -c 0-23,96-119       "$BIN" -t 48  # node 0 phys+SMT
run t024-membind0      numactl --cpunodebind=0 --membind=0 "$BIN" -t 24

# 2-node (nodes 0+1)
run t048-nodes01-phys  taskset -c 0-47              "$BIN" -t 48
run t048-membind01     numactl --cpunodebind=0,1 --membind=0,1 "$BIN" -t 48
run t048-interleave01  numactl --interleave=0,1 taskset -c 0-47 "$BIN" -t 48

# All 4 nodes
run t096-all-phys      taskset -c 0-95              "$BIN" -t 96
run t096-membind-all   numactl --cpunodebind=0,1,2,3 --membind=0,1,2,3 "$BIN" -t 96
run t096-interleave    numactl --interleave=all taskset -c 0-95 "$BIN" -t 96
run t096-numa-distrib  "$BIN" -t 96 --numa distribute -mmp 1

# 144t and 192t spans
run t144-interleave    numactl --interleave=all taskset -c 0-143 "$BIN" -t 144
run t192-numa-distrib  "$BIN" -t 192 --numa distribute -mmp 1
