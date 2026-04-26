#!/bin/bash
# CPU24-deeper Script 2 — Per-CCD perf stat
#
# Purpose: measure if specific CCDs/NUMA-quarters carry different counter
# signatures during REAP-246B decode. Tests hypothesis that one or two
# CCDs become hot-spotted while others sit idle (which would imply
# load-imbalance dominates over uniform sync overhead).
#
# EPYC 9655 NPS4 layout (post-revert):
#   Node 0: CPUs 0-23 (CCDs 0-2)
#   Node 1: CPUs 24-47 (CCDs 3-5)
#   Node 2: CPUs 48-71 (CCDs 6-8)
#   Node 3: CPUs 72-95 (CCDs 9-11)
#
# Usage: sudo ./02_perCCD_perfstat.sh

set -euo pipefail

OUTDIR=${1:-/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-26-cpu24/perCCD}
mkdir -p "$OUTDIR"

LDLP=/mnt/raid0/llm/llama.cpp-experimental/build/bin:/opt/AMD/aocc-compiler-5.0.0/lib
BENCH=/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench
MODEL=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf
EVENTS="cycles,instructions,ls_dmnd_fills_from_sys.dram_io_far,ls_dmnd_fills_from_sys.dram_io_near,ls_dmnd_fills_from_sys.remote_cache,ls_dmnd_fills_from_sys.local_all"

echo 3 > /proc/sys/vm/drop_caches

# Launch bench
LD_LIBRARY_PATH=$LDLP \
  numactl --interleave=all --physcpubind=0-95 \
  "$BENCH" -m "$MODEL" -t 96 -fa 1 -p 0 -n 128 -r 1 \
  > "$OUTDIR/bench.log" 2>&1 &
BENCH_PID=$!

# Wait for model load + decode start
sleep 30

# Capture per-CCD perf stat (4-quarter ranges, ~10 sec each, sequential)
for q in 0 1 2 3; do
  cpu_lo=$((q * 24))
  cpu_hi=$((cpu_lo + 23))
  echo "[INFO] perf stat on Node $q (CPUs $cpu_lo-$cpu_hi) for 8s"
  perf stat -e $EVENTS -C $cpu_lo-$cpu_hi -- sleep 8 \
    > "$OUTDIR/node${q}_cpu${cpu_lo}-${cpu_hi}.log" 2>&1
done

wait $BENCH_PID 2>/dev/null || true

# Summarize
echo
echo "=== per-node summary ==="
for q in 0 1 2 3; do
  cpu_lo=$((q * 24)); cpu_hi=$((cpu_lo + 23))
  echo "--- Node $q (CPUs $cpu_lo-$cpu_hi) ---"
  grep -E "cycles|instructions|ls_dmnd|elapsed" "$OUTDIR/node${q}_cpu${cpu_lo}-${cpu_hi}.log" | head -10
  echo
done

echo "=== throughput ==="
tail -4 "$OUTDIR/bench.log" | head -2
