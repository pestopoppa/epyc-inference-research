#!/bin/bash
# =============================================================================
# OVERNIGHT BENCHMARK SUITE - Complete Model Quality & Speed Testing
# =============================================================================
# Runs ALL benchmark suites with COMPREHENSIVE optimization configurations
#
# Suites:
#   1. Thinking (reasoning, CoT)
#   2. Coder (code generation, debugging)
#   3. VL (vision-language)
#   4. General (instruction following)
#   5. Agentic (tool calling, orchestration)
#   6. Math (mathematical reasoning)
#   7. Long Context (information retrieval across large contexts)
#   8. Instruction Precision (exact format compliance)
#   9. Draft Discovery (automatic draft model optimization)
#
# Optimization Modes (COMPREHENSIVE):
#   - baseline: Standard inference
#   - moe2/4/6/8: MoE expert reduction sweep
#   - spec_k4/8/16/24: Speculative decoding K-value sweep (auto-discovers drafts)
#   - lookup_n3/4/5: Prompt lookup n-gram sweep
#
# Usage:
#   ./run_overnight_benchmark_suite.sh [--suite SUITE] [--skip-slow] [--quick]
#
# Options:
#   --suite SUITE   Run only specified suite (thinking|coder|vl|general|agentic|math|long_context|instruction_precision|draft_discovery|all)
#   --skip-slow     Skip very large models (>100B params)
#   --quick         Quick mode: reduced sweep (moe4 only, K=8,16 only)
#   --dry-run       Show what would run without executing
#
# =============================================================================
set -euo pipefail

# ============================================================================
# SINGLE INSTANCE LOCK - Prevent concurrent benchmark runs
# ============================================================================
RUN_LOCK="/mnt/raid0/llm/tmp/benchmark_running.lock"

# Check if another instance is running
if [[ -f "$RUN_LOCK" ]]; then
  existing_pid=$(cat "$RUN_LOCK" 2>/dev/null)
  if kill -0 "$existing_pid" 2>/dev/null; then
    echo "ERROR: Another benchmark is already running (PID: $existing_pid)"
    echo "If this is stale, remove: $RUN_LOCK"
    exit 1
  else
    echo "Removing stale lock file (PID $existing_pid no longer running)"
    rm -f "$RUN_LOCK"
  fi
fi

# Create lock with our PID
echo $$ >"$RUN_LOCK"

# Ensure lock is removed on exit
cleanup_lock() {
  rm -f "$RUN_LOCK"
}
trap cleanup_lock EXIT

SCRIPT_DIR="$(dirname "$0")"

# Source registry reader for model counts
if [[ -f "$SCRIPT_DIR/lib/registry_reader.sh" ]]; then
  source "$SCRIPT_DIR/lib/registry_reader.sh"
fi

BASE_OUTPUT_DIR="/mnt/raid0/llm/tmp/overnight_benchmark"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$BASE_OUTPUT_DIR/$TIMESTAMP"
LOG_FILE="$RUN_DIR/benchmark.log"
STATUS_FILE="$RUN_DIR/status.txt"

# =============================================================================
# SIMPLE PROGRESS - just print to screen and log to file
# =============================================================================
CURRENT_SUITE=""
SUITE_START_TIME=0
SUITES_PASSED=0
SUITES_FAILED=0
SUITES_RUN=0

# Completed results tracking
COMPLETION_LOG="/mnt/raid0/llm/tmp/benchmark_completions.log"

# Format time helper
format_time() {
  local s=$1
  printf "%02d:%02d:%02d" $((s / 3600)) $(((s % 3600) / 60)) $((s % 60))
}

# No-ops - all output goes to log file only
show_progress() { :; }
clear_progress() { :; }

# Parse arguments
SUITE="all"
SKIP_SLOW=false
DRY_RUN=false
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --skip-slow)
      SKIP_SLOW=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --quick)
      QUICK_MODE=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Export quick mode for rubric scripts to pick up
export BENCHMARK_QUICK_MODE="$QUICK_MODE"

# Setup
mkdir -p "$RUN_DIR"

# Clear completion log for fresh run
: >"$COMPLETION_LOG"

log() {
  local msg
  msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
  # Log to file only - progress display handles all screen output
  echo "$msg" >>"$LOG_FILE"
}

# Log to file only (no screen output)
log_quiet() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >>"$LOG_FILE"
}

run_suite() {
  local suite_name="$1"
  local script="$2"
  local suite_dir="$RUN_DIR/$suite_name"

  mkdir -p "$suite_dir"

  # Update progress state
  CURRENT_SUITE="$suite_name"
  CURRENT_MODEL=""
  SUITE_START_TIME=$(date +%s)

  log_quiet "STARTING SUITE: $suite_name"
  show_progress

  local start_time

  start_time=$(date +%s)

  # Run suite in background and monitor with progress updates
  # Pass BENCHMARK_DRY_RUN so rubric scripts iterate without executing models
  BENCHMARK_DRY_RUN="$DRY_RUN" "$script" >>"$suite_dir/output.log" 2>&1 &
  local suite_pid=$!

  # Update progress while suite runs
  while kill -0 $suite_pid 2>/dev/null; do
    show_progress
    sleep 5
  done

  # Check result
  wait $suite_pid
  local result=$?

  local end_time

  end_time=$(date +%s)
  local duration
  duration=$((end_time - start_time))
  SUITE_START_TIME=0

  local duration_str

  duration_str=$(format_time $duration)
  if [[ $result -eq 0 ]]; then
    log_quiet "COMPLETED: $suite_name ($duration_str)"
    return 0
  else
    log_quiet "FAILED: $suite_name ($duration_str)"
    return 1
  fi
}

# =============================================================================
# DYNAMIC MODEL QUEUE PROCESSING
# =============================================================================
QUEUE_FILE="/mnt/raid0/llm/tmp/benchmark_queue.txt"
QUEUE_LOCK="/mnt/raid0/llm/tmp/benchmark_queue.lock"
QUEUE_DONE="/mnt/raid0/llm/tmp/benchmark_queue_done.txt"
QUEUED_MODELS_PROCESSED=0

# Check if there are models in the queue
has_queued_models() {
  [[ -f "$QUEUE_FILE" ]] && [[ -s "$QUEUE_FILE" ]]
}

# Process all models currently in the queue
process_queue() {
  if ! has_queued_models; then
    return 0
  fi

  log "=========================================="
  log "PROCESSING DYNAMICALLY QUEUED MODELS"
  log "=========================================="

  # Read queue atomically
  local queue_snapshot
  queue_snapshot=$(mktemp)
  (
    flock -x 200
    if [[ -f "$QUEUE_FILE" ]]; then
      cat "$QUEUE_FILE" >"$queue_snapshot"
      # Clear the queue (entries will be moved to done file after processing)
      : >"$QUEUE_FILE"
    fi
  ) 200>"$QUEUE_LOCK"

  if [[ ! -s "$queue_snapshot" ]]; then
    rm -f "$queue_snapshot"
    return 0
  fi

  local count

  count=$(wc -l <"$queue_snapshot")
  log "Found $count model(s) in queue"

  while IFS='|' read -r model_path model_name arch moe_key timestamp; do
    [[ -z "$model_path" ]] && continue

    log ""
    log "Processing queued model: $model_name"
    log "  Path: $model_path"
    log "  Arch: $arch"
    log "  MoE Key: ${moe_key:-none}"
    log "  Queued at: $timestamp"

    if [[ ! -f "$model_path" ]]; then
      log "  ERROR: Model file not found, skipping"
      continue
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
      log "  [DRY RUN] Would run benchmarks for $model_name"
      continue
    fi

    # Run all applicable suites for this model
    run_suites_for_queued_model "$model_path" "$model_name" "$arch" "$moe_key"

    # Mark as processed
    echo "${model_path}|${model_name}|${arch}|${moe_key}|${timestamp}|processed_$(date +%Y%m%d_%H%M%S)" >>"$QUEUE_DONE"
    ((QUEUED_MODELS_PROCESSED++)) || true

  done <"$queue_snapshot"

  rm -f "$queue_snapshot"
  log "Queue processing complete"
}

# Run all benchmark suites for a single queued model
run_suites_for_queued_model() {
  local model_path="$1"
  local model_name="$2"
  local arch="$3"
  local moe_key="$4"

  local model_dir="$RUN_DIR/queued_models/$model_name"
  mkdir -p "$model_dir"

  # Update progress state
  CURRENT_MODEL="$model_name"
  show_progress

  # Build MoE override if needed
  local moe_override=""
  if [[ -n "$moe_key" ]]; then
    moe_override="--override-kv ${moe_key}=int:4"
  fi

  log_quiet "Running benchmark suites for $model_name..."

  # Thinking rubric (if not VL-only model)
  if [[ "$SUITE" == "all" || "$SUITE" == "thinking" ]]; then
    CURRENT_SUITE="queued:thinking"
    show_progress
    "$SCRIPT_DIR/run_thinking_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/thinking.log" 2>&1 || log_quiet "Thinking rubric failed for $model_name"
  fi

  # Coder rubric
  if [[ "$SUITE" == "all" || "$SUITE" == "coder" ]]; then
    CURRENT_SUITE="queued:coder"
    show_progress
    "$SCRIPT_DIR/run_coder_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/coder.log" 2>&1 || log_quiet "Coder rubric failed for $model_name"
  fi

  # General rubric
  if [[ "$SUITE" == "all" || "$SUITE" == "general" ]]; then
    CURRENT_SUITE="queued:general"
    show_progress
    "$SCRIPT_DIR/run_general_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/general.log" 2>&1 || log_quiet "General rubric failed for $model_name"
  fi

  # Math rubric
  if [[ "$SUITE" == "all" || "$SUITE" == "math" ]]; then
    CURRENT_SUITE="queued:math"
    show_progress
    "$SCRIPT_DIR/run_math_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/math.log" 2>&1 || log_quiet "Math rubric failed for $model_name"
  fi

  # Long context rubric
  if [[ "$SUITE" == "all" || "$SUITE" == "long_context" ]]; then
    CURRENT_SUITE="queued:long_context"
    show_progress
    "$SCRIPT_DIR/run_long_context_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/long_context.log" 2>&1 || log_quiet "Long context rubric failed for $model_name"
  fi

  # Instruction precision rubric
  if [[ "$SUITE" == "all" || "$SUITE" == "instruction_precision" ]]; then
    CURRENT_SUITE="queued:instruction_precision"
    show_progress
    "$SCRIPT_DIR/run_instruction_precision_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/instruction_precision.log" 2>&1 || log_quiet "Instruction precision rubric failed for $model_name"
  fi

  # Agentic rubric
  if [[ "$SUITE" == "all" || "$SUITE" == "agentic" ]]; then
    CURRENT_SUITE="queued:agentic"
    show_progress
    "$SCRIPT_DIR/run_agentic_rubric.sh" "$model_path" "$model_name" "$arch" \
      >>"$model_dir/agentic.log" 2>&1 || log_quiet "Agentic rubric failed for $model_name"
  fi

  # Note: VL rubric requires mmproj file, skip for non-VL models
  # VL models should be added manually to the VL benchmark script

  log_quiet "Completed suites for $model_name - Logs: $model_dir/"
  CURRENT_MODEL=""
}

# Start message
echo "Starting overnight benchmark. Log: $LOG_FILE"

# Initialize timing (needed before log() can show progress)
TOTAL_START=$(date +%s)
SUITES_RUN=0
SUITES_PASSED=0
SUITES_FAILED=0
QUEUED_MODELS_PROCESSED=0
CURRENT_SUITE="initializing"

# Write initial config to log (don't use log() yet to avoid progress display)
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] =============================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] OVERNIGHT BENCHMARK SUITE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Started: $(date)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Run ID: $TIMESTAMP"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Output: $RUN_DIR"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Suite: $SUITE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skip slow: $SKIP_SLOW"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Quick mode: $QUICK_MODE"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dry run: $DRY_RUN"
  if [[ "$QUICK_MODE" == "true" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   (Quick mode: MoE=4 only, K=8,16 only, coarse temp sweep)"
  else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')]   (Full mode: MoE=2,4,6,8, K=4,8,16,24,32, binary temp search)"
  fi
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] =============================================="
} >>"$LOG_FILE"

# Show initial progress display
CURRENT_SUITE=""
show_progress

# Record system info
{
  echo "System Information"
  echo "=================="
  echo "Date: $(date)"
  echo "Hostname: $(hostname)"
  echo "CPU: $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
  echo "Cores: $(nproc)"
  echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
  echo ""
  echo "llama.cpp version:"
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli --version 2>&1 | head -5 || echo "N/A"
} >"$RUN_DIR/system_info.txt"

# =============================================================================
# RUN SUITES
# =============================================================================

# Check queue at start (models queued before run began)
process_queue

# Thinking benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "thinking" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "thinking" "$SCRIPT_DIR/run_all_thinking_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi
process_queue # Check for newly queued models

# Coder benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "coder" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "coder" "$SCRIPT_DIR/run_all_coder_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi

# VL benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "vl" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "vl" "$SCRIPT_DIR/run_all_vl_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi
process_queue # Check for newly queued models

# General benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "general" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "general" "$SCRIPT_DIR/run_all_general_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi

# Agentic benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "agentic" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "agentic" "$SCRIPT_DIR/run_all_agentic_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi

# Math benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "math" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "math" "$SCRIPT_DIR/run_all_math_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi
process_queue # Check for newly queued models

# Long Context benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "long_context" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "long_context" "$SCRIPT_DIR/run_all_long_context_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi

# Instruction Precision benchmark
if [[ "$SUITE" == "all" || "$SUITE" == "instruction_precision" ]]; then
  ((SUITES_RUN++)) || true
  if run_suite "instruction_precision" "$SCRIPT_DIR/run_all_instruction_precision_benchmarks.sh"; then
    ((SUITES_PASSED++)) || true
  else
    ((SUITES_FAILED++)) || true
  fi
fi

# Final queue check before speculative decoding
process_queue

# =============================================================================
# DYNAMIC DRAFT DISCOVERY AND OPTIMIZATION
# =============================================================================
# Replaces old hardcoded spec/lookup tests with comprehensive optimization:
#   - Auto-discovers all potential draft models
#   - Tests all compatible target/draft pairs
#   - Sweeps K values: 4, 8, 16, 24, 32
#   - Binary search for optimal temperature (precision: 0.001)
#   - Reports best configurations per target
# =============================================================================
if [[ "$SUITE" == "all" || "$SUITE" == "draft_discovery" || "$SUITE" == "spec_decode" ]]; then
  ((SUITES_RUN++)) || true
  CURRENT_SUITE="draft_discovery"
  SUITE_START_TIME=$(date +%s)
  show_progress

  log_quiet "STARTING: DYNAMIC DRAFT DISCOVERY & OPTIMIZATION"

  DISCOVERY_DIR="$RUN_DIR/draft_discovery"
  mkdir -p "$DISCOVERY_DIR"

  if [[ "$DRY_RUN" == "true" ]]; then
    log_quiet "[DRY RUN] Would run draft discovery"
    ((SUITES_PASSED++)) || true
  else
    # Build command with appropriate flags
    DISCOVERY_CMD="$SCRIPT_DIR/run_draft_discovery.sh"
    if [[ "$QUICK_MODE" == "true" ]]; then
      DISCOVERY_CMD="$DISCOVERY_CMD --quick"
    fi

    log_quiet "Running: $DISCOVERY_CMD"

    # Run in background with progress updates
    $DISCOVERY_CMD >>"$DISCOVERY_DIR/output.log" 2>&1 &
    discovery_pid=$!

    while kill -0 $discovery_pid 2>/dev/null; do
      show_progress
      sleep 5
    done

    wait $discovery_pid
    discovery_result=$?

    SUITE_START_TIME=0

    if [[ $discovery_result -eq 0 ]]; then
      log_quiet "Draft discovery completed successfully"
      ((SUITES_PASSED++)) || true

      # Copy results to run directory
      LATEST_DISCOVERY=$(ls -td /mnt/raid0/llm/tmp/draft_discovery/*/ 2>/dev/null | head -1)
      if [[ -n "$LATEST_DISCOVERY" ]]; then
        cp -r "$LATEST_DISCOVERY"/* "$DISCOVERY_DIR/" 2>/dev/null || true
        log_quiet "Results: $DISCOVERY_DIR"
      fi
    else
      log_quiet "Draft discovery failed - check $DISCOVERY_DIR/output.log"
      ((SUITES_FAILED++)) || true
    fi
  fi
fi

# =============================================================================
# FINAL SUMMARY
# =============================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))
HOURS=$((TOTAL_DURATION / 3600))
MINUTES=$(((TOTAL_DURATION % 3600) / 60))
SECONDS_VAL=$((TOTAL_DURATION % 60))

# One final queue check
CURRENT_SUITE="final_queue_check"
show_progress
process_queue

# Done
CURRENT_SUITE="complete"
echo "Benchmark complete. Passed: $SUITES_PASSED Failed: $SUITES_FAILED Duration: ${HOURS}h ${MINUTES}m ${SECONDS_VAL}s"
echo "Results: $RUN_DIR"

# Log the summary
log_quiet "=============================================="
log_quiet "OVERNIGHT BENCHMARK COMPLETE"
log_quiet "=============================================="
log_quiet "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS_VAL}s"
log_quiet "Suites run: $SUITES_RUN"
log_quiet "Suites passed: $SUITES_PASSED"
log_quiet "Suites failed: $SUITES_FAILED"
if [[ $QUEUED_MODELS_PROCESSED -gt 0 ]]; then
  log_quiet "Dynamically queued models processed: $QUEUED_MODELS_PROCESSED"
fi
log_quiet "Results: $RUN_DIR"
log_quiet "=============================================="

# Process results into permanent storage with JSONL index
if [[ "$DRY_RUN" != "true" ]]; then
  log ""
  log "Processing results into permanent storage..."
  if "$SCRIPT_DIR/process_benchmark_results.sh" --run-id "$TIMESTAMP" >>"$LOG_FILE" 2>&1; then
    log "Results processed successfully"
    log "Permanent storage: /mnt/raid0/llm/claude/benchmarks/results/runs/$TIMESTAMP"
    log "Index: /mnt/raid0/llm/claude/benchmarks/results/index.jsonl"
  else
    log "WARNING: Results processing failed"
  fi
else
  log ""
  log "[DRY RUN] Skipping results processing"
fi

# Generate final report
{
  echo "=============================================="
  echo "OVERNIGHT BENCHMARK FINAL REPORT"
  echo "=============================================="
  echo ""
  echo "Run ID: $TIMESTAMP"
  echo "Started: $(cat "$RUN_DIR/system_info.txt" | grep Date | head -1)"
  echo "Completed: $(date)"
  echo "Duration: ${HOURS}h ${MINUTES}m ${SECONDS_VAL}s"
  echo ""
  echo "Suites: $SUITES_RUN run, $SUITES_PASSED passed, $SUITES_FAILED failed"
  echo ""
  echo "Output Directories:"
  echo "-------------------"
  ls -la "$RUN_DIR" 2>/dev/null || true
  echo ""
  echo "=============================================="
  echo "QUICK LINKS TO RESULTS"
  echo "=============================================="

  for suite in thinking coder vl general agentic math long_context instruction_precision; do
    suite_dir="$RUN_DIR/$suite"
    if [[ -d "$suite_dir" ]]; then
      echo ""
      echo "--- $suite ---"
      echo "Log: $suite_dir/output.log"
      # Find the most recent summary file
      latest_summary=$(find /mnt/raid0/llm/tmp/${suite}_rubric_results/ -name "benchmark_summary_*.txt" 2>/dev/null | sort | tail -1 || true)
      if [[ -n "$latest_summary" ]]; then
        echo "Summary: $latest_summary"
      fi
    fi
  done

  echo ""
  echo "=============================================="
  echo "SPECULATIVE DECODING RESULTS"
  echo "=============================================="
  if [[ -d "$RUN_DIR/speculative_decoding" ]]; then
    for f in "$RUN_DIR"/speculative_decoding/*.txt; do
      if [[ -f "$f" ]]; then
        name=$(basename "$f" .txt)
        speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
        echo "  $name: $speed"
      fi
    done
  else
    echo "  (not run)"
  fi

  echo ""
  echo "=============================================="
  echo "LOOKUP DECODING RESULTS"
  echo "=============================================="
  if [[ -d "$RUN_DIR/lookup_decoding" ]]; then
    for f in "$RUN_DIR"/lookup_decoding/*.txt; do
      if [[ -f "$f" ]]; then
        name=$(basename "$f" .txt)
        speed=$(grep "eval time" "$f" 2>/dev/null | grep -oP '\d+\.\d+ tokens per second' | tail -1 || echo "N/A")
        echo "  $name: $speed"
      fi
    done
  else
    echo "  (not run or llama-lookup not available)"
  fi

} >"$RUN_DIR/FINAL_REPORT.txt"

cat "$RUN_DIR/FINAL_REPORT.txt"

log ""
log "Full report: $RUN_DIR/FINAL_REPORT.txt"
log "Log file: $LOG_FILE"
