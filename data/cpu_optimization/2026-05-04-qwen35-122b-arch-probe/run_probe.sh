#!/bin/bash
# Qwen3.5-122B-A10B arch-class Probe B
# Per handoffs/active/qwen35-122b-a10b-arch-class-probe.md
# Closes the architect_general slot in v5 deployment draft.
set -uo pipefail

DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-05-04-qwen35-122b-arch-probe"
BIN="/mnt/raid0/llm/llama.cpp/build_libomp_pgo_use/bin/llama-bench"
MODEL="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"

cd "$DIR"
SUMMARY="$DIR/probe_summary.tsv"
echo -e "config\tenv_block\tavg_ts\tstddev_ts\tn_runs" > "$SUMMARY"

# c0: default v5 (no opt-in env)
# c1: CPU1 stack (3-flag stable, no NUMA_WEIGHTS)
# c2: mbind-off
# c3: c1 + c2 combined
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
  local avg_ts=$(jq -r '.[0].avg_ts // "FAIL"' "$out" 2>/dev/null)
  local stddev_ts=$(jq -r '.[0].stddev_ts // ""' "$out" 2>/dev/null)
  local n=$(jq -r '.[0] | (.samples_ts // []) | length' "$out" 2>/dev/null)

  echo "[$(date '+%H:%M:%S')] END   ${cfg} rc=${rc} elapsed=${elapsed}s avg_ts=${avg_ts} +/- ${stddev_ts} n=${n}" | tee -a "$DIR/probe_progress.log"
  echo -e "${cfg}\t${env_block}\t${avg_ts}\t${stddev_ts}\t${n}" >> "$SUMMARY"
}

run_config "c0_default_v5"        ""
run_config "c1_cpu1_stack"        "GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1"
run_config "c2_mbind_off"         "GGML_NUMA_REPACK_INTERLEAVE=0"
run_config "c3_cpu1_plus_mbind"   "GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0"

echo "[$(date '+%H:%M:%S')] === Probe B single-instance ALL DONE ===" | tee -a "$DIR/probe_progress.log"
echo
echo "=== SUMMARY ==="
column -t -s$'\t' "$SUMMARY"
