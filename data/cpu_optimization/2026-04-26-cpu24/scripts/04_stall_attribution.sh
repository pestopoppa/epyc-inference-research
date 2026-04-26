#!/bin/bash
# CPU24-deeper Script 4 — Stall attribution via Zen frontend/backend events
#
# Purpose: classify the source of low IPC (0.39 on REAP-246B at 96t).
# AMD Zen 4/5 expose stall-class events that distinguish:
#  - frontend stalls (instruction fetch / decode)
#  - backend stalls (load-store / FP / branch)
#  - dispatch token stalls (resource contention)
# If sync overhead dominates, we expect high time waiting on lock cmpxchg
# loops which show up as backend stalls on memory ops + dispatch token stalls.
#
# Usage: sudo ./04_stall_attribution.sh

set -euo pipefail

OUTDIR=${1:-/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-26-cpu24/stalls}
mkdir -p "$OUTDIR"

LDLP=/mnt/raid0/llm/llama.cpp-experimental/build/bin:/opt/AMD/aocc-compiler-5.0.0/lib
BENCH=/mnt/raid0/llm/llama.cpp-experimental/build/bin/llama-bench
MODEL=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf

# Zen 4/5 stall events (some may not be available; perf will report which)
EVENTS_STALLS="cycles,instructions,\
de_dis_dispatch_token_stalls0.alu_token_stall,\
de_dis_dispatch_token_stalls0.alsq3_token_stall,\
de_dis_dispatch_token_stalls0.alsq2_token_stall,\
de_dis_dispatch_token_stalls0.retire_token_stall,\
de_dis_dispatch_token_stalls0.scaler_phy_reg_token_stall,\
de_dis_dispatch_token_stalls0.fp_phy_reg_token_stall"

EVENTS_BACKEND="cycles,instructions,\
ls_locks.bus_lock,\
ls_locks.spec_lock_hi_spec,\
ls_locks.spec_lock_lo_spec"

EVENTS_FRONTEND="cycles,instructions,\
de_no_dispatch_per_slot.no_ops_from_frontend,\
de_no_dispatch_per_slot.backend_stalls,\
de_no_dispatch_per_slot.smt_contention"

echo 3 > /proc/sys/vm/drop_caches

run_with_events() {
  local name=$1; shift
  local events=$1; shift
  echo "[INFO] running with event set: $name"
  echo 3 > /proc/sys/vm/drop_caches
  perf stat -e $events \
    env LD_LIBRARY_PATH=$LDLP \
    numactl --interleave=all --physcpubind=0-95 \
    "$BENCH" -m "$MODEL" -t 96 -fa 1 -p 0 -n 64 -r 1 \
    > "$OUTDIR/${name}.log" 2>&1
  echo "    exit $?"
}

run_with_events stalls   "$EVENTS_STALLS"
run_with_events backend  "$EVENTS_BACKEND"
run_with_events frontend "$EVENTS_FRONTEND"

echo
echo "=== per-event summaries ==="
for n in stalls backend frontend; do
  echo "--- $n ---"
  grep -E "cycles|instructions|de_dis_dispatch|ls_locks|de_no_dispatch|elapsed" "$OUTDIR/${n}.log" | head -15
  echo
done
