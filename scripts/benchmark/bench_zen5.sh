#!/bin/bash
# bench_zen5.sh — Systematic Benchmarking for AMD EPYC 9655
# Tests thread counts and NUMA configurations
# Usage: bash bench_zen5.sh [model_path]

set -euo pipefail

# Source environment and logging libraries
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"
# shellcheck source=../utils/agent_log.sh
source "${SCRIPT_DIR}/../utils/agent_log.sh"

agent_session_start "Zen 5 systematic benchmarking"

# Configuration (use variables from env.sh)
MODEL_PATH="${1:-${MODELS_DIR}/DeepSeek-R1-32B-Q4_K_M.gguf}"
LLAMA_BENCH="${LLAMA_CPP_BIN}/llama-bench"
RESULTS_DIR="${LOG_DIR}"
RESULTS_FILE="$RESULTS_DIR/zen5_benchmark_$(date +%Y%m%d_%H%M%S).csv"

# Test parameters
THREAD_COUNTS=(48 64 96 128) # Skip 192 — SMT hurts inference
PROMPT_TOKENS=512
GEN_TOKENS=128
BATCH_SIZE=512

# Ensure prerequisites
mkdir -p "$RESULTS_DIR"

if [[ ! -f "$LLAMA_BENCH" ]]; then
  echo "ERROR: llama-bench not found at $LLAMA_BENCH"
  echo "Build llama.cpp first with: cd /mnt/raid0/llm/llama.cpp && mkdir build && cd build && cmake .. && make -j\$(nproc)"
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "ERROR: Model not found at $MODEL_PATH"
  echo "Usage: $0 /path/to/model.gguf"
  exit 1
fi

# Set critical environment
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

echo "=============================================="
echo "Zen 5 Benchmark Suite"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Threads to test: ${THREAD_COUNTS[*]}"
echo "Results: $RESULTS_FILE"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "=============================================="

agent_task_start "Initialize benchmarking" "Setting up benchmark configuration"
agent_observe "model" "$MODEL_PATH"
agent_observe "thread_counts" "${THREAD_COUNTS[*]}"
agent_observe "results_file" "$RESULTS_FILE"
agent_observe "OMP_NUM_THREADS" "$OMP_NUM_THREADS"
agent_decision "Thread counts exclude 192" "SMT typically hurts inference performance"
agent_task_end "Initialize benchmarking" "success"

# CSV Header
echo "Timestamp,Config,Threads,Prompt_Tokens,Gen_Tokens,Batch,PP_Tok_s,TG_Tok_s,Notes" >"$RESULTS_FILE"

# Function to run benchmark and parse results
run_bench() {
  local config="$1"
  local threads="$2"
  local prefix="$3"

  echo -e "\n>>> Testing: $config, $threads threads..."
  agent_task_start "Benchmark $config @ $threads threads" "Testing thread/NUMA configuration"

  # Drop caches for consistent results
  sync
  echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
  sleep 2

  # Build command
  local cmd="$LLAMA_BENCH -m $MODEL_PATH -t $threads -p $PROMPT_TOKENS -n $GEN_TOKENS -b $BATCH_SIZE"

  # Add numactl prefix if needed
  if [[ -n "$prefix" ]]; then
    cmd="$prefix $cmd"
  fi

  agent_cmd_intent "$cmd" "Running llama-bench with $config config"

  # Run and capture output
  local output
  local start_time
  start_time=$(date +%s)

  if output=$($cmd 2>&1); then
    local end_time
    end_time=$(date +%s)
    local duration
    duration=$((end_time - start_time))

    # Parse results - llama-bench outputs a markdown table
    # Look for the data row (contains the model name)
    local pp_toks
    pp_toks=$(echo "$output" | grep -E "^\|.*\.gguf" | awk -F'|' '{print $7}' | tr -d ' ' | head -1)
    local tg_toks
    tg_toks=$(echo "$output" | grep -E "^\|.*\.gguf" | awk -F'|' '{print $8}' | tr -d ' ' | head -1)

    # Fallback parsing if table format differs
    if [[ -z "$pp_toks" ]]; then
      pp_toks=$(echo "$output" | grep -i "prompt" | grep -oE "[0-9]+\.[0-9]+" | head -1 || echo "N/A")
    fi
    if [[ -z "$tg_toks" ]]; then
      tg_toks=$(echo "$output" | grep -i "generation\|eval" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "N/A")
    fi

    echo "   PP: ${pp_toks:-N/A} t/s, TG: ${tg_toks:-N/A} t/s (${duration}s)"
    echo "$(date +%H:%M:%S),$config,$threads,$PROMPT_TOKENS,$GEN_TOKENS,$BATCH_SIZE,${pp_toks:-N/A},${tg_toks:-N/A},OK" >>"$RESULTS_FILE"

    agent_cmd_result "$cmd" "0" "PP=${pp_toks:-N/A} TG=${tg_toks:-N/A} duration=${duration}s"
    agent_observe "benchmark_result" "$config,$threads,PP=${pp_toks:-N/A},TG=${tg_toks:-N/A}"
    agent_task_end "Benchmark $config @ $threads threads" "success"
  else
    echo "   FAILED"
    echo "$(date +%H:%M:%S),$config,$threads,$PROMPT_TOKENS,$GEN_TOKENS,$BATCH_SIZE,FAIL,FAIL,Error" >>"$RESULTS_FILE"
    agent_cmd_result "$cmd" "1" "FAILED"
    agent_error "Benchmark failed" "$config @ $threads threads"
    agent_task_end "Benchmark $config @ $threads threads" "failure"
  fi
}

# Warmup run (results discarded)
echo -e "\n--- Warmup Run (discarded) ---"
$LLAMA_BENCH -m "$MODEL_PATH" -t 96 -p 64 -n 16 >/dev/null 2>&1 || true

# Test Standard Configuration (no NUMA interleave)
echo -e "\n========== Standard Configuration =========="
for threads in "${THREAD_COUNTS[@]}"; do
  run_bench "Standard" "$threads" ""
done

# Test NUMA Interleaved Configuration
echo -e "\n========== NUMA Interleaved Configuration =========="
for threads in "${THREAD_COUNTS[@]}"; do
  run_bench "Interleaved" "$threads" "numactl --interleave=all"
done

# Summary
echo -e "\n=============================================="
echo "BENCHMARK COMPLETE"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

# Display results
echo -e "\n--- Results Summary ---"
column -t -s',' "$RESULTS_FILE" 2>/dev/null || cat "$RESULTS_FILE"

# Find best configuration
echo -e "\n--- Best Configurations ---"
echo "By Token Generation (TG) speed:"
tail -n +2 "$RESULTS_FILE" | sort -t',' -k8 -rn | head -3 | while IFS=',' read -r ts cfg thr pp gn bat pps tgs notes; do
  echo "  $cfg @ $thr threads: $tgs t/s"
  agent_observe "best_config" "$cfg @ $thr threads: $tgs t/s"
done

agent_session_end "Benchmarking complete. Results: $RESULTS_FILE"
