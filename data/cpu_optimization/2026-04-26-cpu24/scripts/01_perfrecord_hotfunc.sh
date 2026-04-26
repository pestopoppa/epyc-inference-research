#!/bin/bash
# CPU24-deeper Script 1 — Hot-function profile via perf record
#
# Purpose: identify which symbols dominate CPU time on REAP-246B.
# Confirms the "sync overhead = 96% of parallelism loss" hypothesis from
# the perf-stat counter analysis by showing whether time is concentrated
# in OpenMP/libgomp/barrier symbols vs compute (gemm/quantize) symbols.
#
# Usage: sudo ./01_perfrecord_hotfunc.sh <model_path> <output_dir>
# Default: REAP-246B Q4_K_M
#
# IMPORTANT: must run with NO other heavy CPU/memory workload concurrently
# (concurrent llama-bench will distort both throughput and the profile).

set -euo pipefail

MODEL=${1:-/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf}
OUTDIR=${2:-/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-26-cpu24/perfrecord}
RECORD_SECS=${3:-25}
mkdir -p "$OUTDIR"

LDLP=/mnt/raid0/llm/llama.cpp-experimental/build/bin:/opt/AMD/aocc-compiler-5.0.0/lib
BENCH=/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench

# Pre-bench hygiene
pgrep -af "llama" | grep -v "grep\|zsh\|docker" >&2 || true
echo 3 > /proc/sys/vm/drop_caches
echo 0 > /proc/sys/kernel/numa_balancing
echo "[INFO] state: numa_balancing=$(cat /proc/sys/kernel/numa_balancing) THP=$(cat /sys/kernel/mm/transparent_hugepage/enabled)"

# Launch bench in background, capture PID
LD_LIBRARY_PATH=$LDLP \
  numactl --interleave=all --physcpubind=0-95 \
  "$BENCH" -m "$MODEL" -t 96 -fa 1 -p 0 -n 128 -r 1 \
  > "$OUTDIR/bench_run.log" 2>&1 &
BENCH_PID=$!
echo "[INFO] bench PID=$BENCH_PID"

# Wait for model load (REAP needs ~25s on cold cache)
sleep 30

# Verify bench is in decode phase (still running, presumably past load)
if ! kill -0 $BENCH_PID 2>/dev/null; then
  echo "[ERROR] bench finished before perf record could start" >&2
  exit 1
fi

# Capture decode-phase profile
echo "[INFO] starting perf record for ${RECORD_SECS}s..."
perf record -F 99 -g -p $BENCH_PID -o "$OUTDIR/perf.data" -- sleep $RECORD_SECS
echo "[INFO] perf record done"

wait $BENCH_PID 2>/dev/null || true

# Per-symbol report (sorted by % of samples; top 30, no children to avoid call-graph collapse)
echo "[INFO] generating per-symbol report"
perf report -i "$OUTDIR/perf.data" --stdio --percent-limit=0.5 --no-children > "$OUTDIR/perf_symbols.txt" 2>&1
head -60 "$OUTDIR/perf_symbols.txt"

# Call-graph report (with children collapsed)
echo "[INFO] generating call-graph report"
perf report -i "$OUTDIR/perf.data" --stdio --percent-limit=1.0 > "$OUTDIR/perf_callgraph.txt" 2>&1

# Throughput
echo
echo "[INFO] bench throughput:"
tail -4 "$OUTDIR/bench_run.log" | head -2
