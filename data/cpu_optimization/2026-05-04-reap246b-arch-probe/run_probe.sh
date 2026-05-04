#!/bin/bash
# REAP-246B-A35B Q4_K_M arch-class Probe B
# v5 deployment draft assigns this model to "moe_q4_dram_bound" arch class with env: {}.
# That assignment was based on CPU22 -0.8% (noise) and was never validated under tight
# Probe B methodology. Phase A.2 today found REAP-246B was the WORST regressor under
# Q6_K AVX-512BW (-1.01%, σ 1.47%) — suggesting memory-subsystem sensitivity. This
# probe runs the canonical c0/c1/c2/c3 protocol to either confirm "default v5" or
# discover a winning lever.
set -uo pipefail

DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-reap246b-arch-probe"
BIN="/mnt/raid0/llm/llama.cpp/build_libomp_pgo_use/bin/llama-bench"
MODEL="/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf"

cd "$DIR"
SUMMARY="$DIR/probe_summary.tsv"
echo -e "config\tenv_block\tavg_ts\tstddev_ts\tn_runs" > "$SUMMARY"

run_config() {
  local cfg=$1
  local env_block=$2
  local out="$DIR/${cfg}.json"
  local err="$DIR/${cfg}.stderr"

  echo "[$(date '+%H:%M:%S')] START ${cfg} env=[${env_block}]" | tee -a "$DIR/probe_progress.log"
  local t0=$SECONDS

  # shellcheck disable=SC2086
  env $env_block \
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    numactl --interleave=all -- taskset -c 0-95 \
    "$BIN" -m "$MODEL" -t 96 -fa 1 --mmap 0 \
    -p 0 -n 32 -r 5 \
    -o json > "$out" 2>"$err"
  local rc=$?

  local elapsed=$((SECONDS - t0))
  local avg=$(jq -r '.[0].avg_ts // "FAIL"' "$out" 2>/dev/null)
  local sd=$(jq -r '.[0].stddev_ts // ""' "$out" 2>/dev/null)
  local n=$(jq -r '.[0] | (.samples_ts // []) | length' "$out" 2>/dev/null)

  echo "[$(date '+%H:%M:%S')] END   ${cfg} rc=${rc} elapsed=${elapsed}s avg_ts=${avg} +/- ${sd} n=${n}" | tee -a "$DIR/probe_progress.log"
  echo -e "${cfg}\t${env_block}\t${avg}\t${sd}\t${n}" >> "$SUMMARY"
}

# c0: default v5 (no opt-in env) — current registry assignment
# c1: CPU1 stack — 3-flag stable (was -2.3% Coder per CPU22 closure, but REAP DRAM-bound may differ)
# c2: mbind off — Q6_K AVX A.2 finding suggests REAP is mbind-sensitive (-1.01% under +Q6K AVX could indicate mbind interaction)
# c3: c1 + c2 combined
run_config "c0_default_v5"        ""
run_config "c1_cpu1_stack"        "GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1"
run_config "c2_mbind_off"         "GGML_NUMA_REPACK_INTERLEAVE=0"
run_config "c3_cpu1_plus_mbind"   "GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0"

echo "[$(date '+%H:%M:%S')] === Probe B ALL DONE ===" | tee -a "$DIR/probe_progress.log"
echo
echo "=== SUMMARY ==="
column -t -s$'\t' "$SUMMARY"
