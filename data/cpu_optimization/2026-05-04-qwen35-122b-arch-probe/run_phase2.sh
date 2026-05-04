#!/bin/bash
# Qwen3.5-122B-A10B Probe B — Phase 2: production-wiring revalidation
# Per handoffs/active/qwen35-122b-a10b-arch-class-probe.md "Bonus probe"
# Phase 1 winning env: GGML_NUMA_REPACK_INTERLEAVE=0 (c2)
set -uo pipefail

DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe"
BIN="/mnt/raid0/llm/llama.cpp/build_libomp_pgo_use/bin/llama-bench"
MODEL="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"

cd "$DIR"
SUMMARY="$DIR/phase2_summary.tsv"
echo -e "wiring\tinstance\tnode\tcpu_set\tthreads\tavg_ts\tstddev_ts\tn_runs" > "$SUMMARY"

# NPS4 topology: node N → physical cpus N*24 .. (N+1)*24-1
node_cpus() {
  local node=$1
  local lo=$((node * 24))
  local hi=$((lo + 23))
  echo "${lo}-${hi}"
}

# w1a: 1× single-NUMA-node, isolated to node 0
run_w1a() {
  local out="$DIR/w1a_node0.json"
  local err="$DIR/w1a_node0.stderr"
  echo "[$(date '+%H:%M:%S')] START w1a (1× --cpunodebind=0 --membind=0 -t 24, c2 env)" | tee -a "$DIR/phase2_progress.log"
  local t0=$SECONDS

  GGML_NUMA_REPACK_INTERLEAVE=0 \
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    numactl --cpunodebind=0 --membind=0 -- \
    "$BIN" -m "$MODEL" -t 24 -fa 1 --mmap 0 -p 0 -n 32 -r 5 -o json \
    > "$out" 2>"$err"
  local rc=$?
  local elapsed=$((SECONDS - t0))
  local avg=$(jq -r '.[0].avg_ts // "FAIL"' "$out" 2>/dev/null)
  local sd=$(jq -r '.[0].stddev_ts // ""' "$out" 2>/dev/null)
  local n=$(jq -r '.[0] | (.samples_ts // []) | length' "$out" 2>/dev/null)

  echo "[$(date '+%H:%M:%S')] END   w1a rc=${rc} elapsed=${elapsed}s avg_ts=${avg} +/- ${sd} n=${n}" | tee -a "$DIR/phase2_progress.log"
  echo -e "w1a\t0\t0\t$(node_cpus 0)\t24\t${avg}\t${sd}\t${n}" >> "$SUMMARY"
}

# Concurrent multi-instance launch helper.
# Args: wiring_label, threads_per_instance, list of "node:bench_out_basename" pairs
run_concurrent() {
  local wiring=$1; shift
  local threads=$1; shift
  local pairs=("$@")

  echo "[$(date '+%H:%M:%S')] START ${wiring} (concurrent ${#pairs[@]}× -t ${threads} per-NUMA-node, c2 env)" | tee -a "$DIR/phase2_progress.log"
  local t0=$SECONDS

  # Launch all instances concurrently
  local pids=()
  for pair in "${pairs[@]}"; do
    local node="${pair%%:*}"
    local out_base="${pair##*:}"
    local cpus=$(node_cpus "$node")
    local out="$DIR/${out_base}.json"
    local err="$DIR/${out_base}.stderr"

    GGML_NUMA_REPACK_INTERLEAVE=0 \
      OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
      numactl --cpunodebind="$node" --membind="$node" -- \
      "$BIN" -m "$MODEL" -t "$threads" -fa 1 --mmap 0 -p 0 -n 32 -r 5 -o json \
      > "$out" 2>"$err" &
    pids+=($!)
    echo "  launched node=$node cpus=$cpus PID=$! out=$(basename "$out")" | tee -a "$DIR/phase2_progress.log"
  done

  # Wait for all
  local fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=$((fail + 1))
      echo "  PID $pid FAILED" | tee -a "$DIR/phase2_progress.log"
    fi
  done

  local elapsed=$((SECONDS - t0))
  echo "[$(date '+%H:%M:%S')] END   ${wiring} elapsed=${elapsed}s fail=${fail}" | tee -a "$DIR/phase2_progress.log"

  # Record per-instance results
  local agg=0
  for pair in "${pairs[@]}"; do
    local node="${pair%%:*}"
    local out_base="${pair##*:}"
    local cpus=$(node_cpus "$node")
    local out="$DIR/${out_base}.json"
    local avg=$(jq -r '.[0].avg_ts // "FAIL"' "$out" 2>/dev/null)
    local sd=$(jq -r '.[0].stddev_ts // ""' "$out" 2>/dev/null)
    local n=$(jq -r '.[0] | (.samples_ts // []) | length' "$out" 2>/dev/null)
    echo "  ${out_base}: avg_ts=${avg} +/- ${sd} n=${n}" | tee -a "$DIR/phase2_progress.log"
    echo -e "${wiring}\t${out_base}\t${node}\t${cpus}\t${threads}\t${avg}\t${sd}\t${n}" >> "$SUMMARY"
    if [[ "$avg" != "FAIL" && -n "$avg" ]]; then
      agg=$(awk -v a="$agg" -v v="$avg" 'BEGIN{printf "%.4f", a + v}')
    fi
  done
  echo "  aggregate t/s: ${agg}" | tee -a "$DIR/phase2_progress.log"
}

# w1a single-node single-instance
run_w1a

# w1b 2× concurrent per-NUMA-node (nodes 0 and 1)
run_concurrent "w1b" 24 "0:w1b_node0" "1:w1b_node1"

# w2 4× concurrent per-NUMA-node (nodes 0..3)
run_concurrent "w2" 24 "0:w2_node0" "1:w2_node1" "2:w2_node2" "3:w2_node3"

echo "[$(date '+%H:%M:%S')] === Phase 2 ALL DONE ===" | tee -a "$DIR/phase2_progress.log"
echo
echo "=== SUMMARY ==="
column -t -s$'\t' "$SUMMARY"
