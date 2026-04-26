#!/bin/bash
# CPU24-deeper Script 3 — Per-thread CPU% via pidstat
#
# Purpose: measure thread-level imbalance during REAP-246B decode.
# If sync overhead dominates, we expect thread CPU% to be HIGHLY variable
# (some threads pinned at 100%, others stalling at 30-50% as they wait
# at barriers). Uniform 100% across all 96 threads would refute the
# sync-overhead-dominates hypothesis.
#
# Usage: ./03_thread_imbalance.sh

set -euo pipefail

OUTDIR=${1:-/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-26-cpu24/thread_imbalance}
mkdir -p "$OUTDIR"

LDLP=/mnt/raid0/llm/llama.cpp-experimental/build/bin:/opt/AMD/aocc-compiler-5.0.0/lib
BENCH=/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench
MODEL=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf

sudo -n bash -c 'echo 3 > /proc/sys/vm/drop_caches' >/dev/null 2>&1

LD_LIBRARY_PATH=$LDLP \
  numactl --interleave=all --physcpubind=0-95 \
  "$BENCH" -m "$MODEL" -t 96 -fa 1 -p 0 -n 256 -r 1 \
  > "$OUTDIR/bench.log" 2>&1 &
BENCH_PID=$!

# Wait for model load + decode start
sleep 35

# Capture per-thread CPU% snapshots (1 sec interval, 10 snapshots)
echo "[INFO] capturing pidstat -t for 10s"
pidstat -t -p $BENCH_PID 1 10 > "$OUTDIR/pidstat.log" 2>&1 || true

wait $BENCH_PID 2>/dev/null || true

# Histogram-style summary: count threads in CPU% buckets
echo
echo "=== thread CPU% distribution (last 8 snapshots avg) ==="
awk '/llama-bench$|llama-bench-/ && $7 ~ /[0-9]/ { sum[$3] += $7; n[$3]++ }
     END { for (t in sum) print sum[t]/n[t], t }' "$OUTDIR/pidstat.log" \
  | sort -n | awk '{ if ($1 < 25) b="<25%"; else if ($1 < 50) b="25-50%"; else if ($1 < 75) b="50-75%"; else if ($1 < 95) b="75-95%"; else b=">=95%"; bucket[b]++ } END { for (b in bucket) print b, bucket[b] }' \
  > "$OUTDIR/thread_histogram.txt"
cat "$OUTDIR/thread_histogram.txt"
echo
echo "=== throughput ==="
tail -4 "$OUTDIR/bench.log" | head -2
