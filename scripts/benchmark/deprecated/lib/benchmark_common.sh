#!/bin/bash
# =============================================================================
# Benchmark Common Library
# =============================================================================
# Shared functions for all rubric benchmark scripts
# Provides: metadata, structured output, JSONL indexing
# =============================================================================

# Paths
BENCHMARK_BASE="/mnt/raid0/llm/epyc-inference-research/benchmarks"
PROMPTS_DIR="$BENCHMARK_BASE/prompts"
RESULTS_DIR="$BENCHMARK_BASE/results"
BASELINES_DIR="$BENCHMARK_BASE/baselines"
INDEX_FILE="$RESULTS_DIR/index.jsonl"

# Current prompt version
PROMPT_VERSION="1"

# Get llama.cpp version
get_llama_version() {
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli --version 2>&1 | head -1 | grep -oP 'version: \K\d+' || echo "unknown"
}

# Get llama.cpp commit hash
get_llama_commit() {
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli --version 2>&1 | head -1 | grep -oP '\(\K[a-f0-9]+' || echo "unknown"
}

# Initialize a benchmark run
# Sets up RUN_ID, RUN_DIR, and creates metadata
# Usage: init_benchmark_run "domain" "model_name" "model_path"
init_benchmark_run() {
  local domain="$1"
  local model_name="$2"
  local model_path="$3"

  # Generate run ID
  RUN_ID=$(date +%Y%m%d_%H%M%S)
  RUN_DIR="$RESULTS_DIR/runs/$RUN_ID"

  mkdir -p "$RUN_DIR"

  # Get system info
  local llama_version
  llama_version=$(get_llama_version)
  local llama_commit
  llama_commit=$(get_llama_commit)
  local hostname
  hostname=$(hostname)
  local cpu_model
  cpu_model=$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo "unknown")
  local total_ram
  total_ram=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "unknown")

  # Get model file hash (first 16 chars of sha256, or size if too slow)
  local model_size
  model_size=$(stat -c%s "$model_path" 2>/dev/null || echo "0")
  local model_mtime
  model_mtime=$(stat -c%Y "$model_path" 2>/dev/null || echo "0")

  # Write metadata
  cat >"$RUN_DIR/metadata.json" <<EOF
{
    "run_id": "$RUN_ID",
    "timestamp": "$(date -Iseconds)",
    "domain": "$domain",
    "model": {
        "name": "$model_name",
        "path": "$model_path",
        "size_bytes": $model_size,
        "mtime": $model_mtime
    },
    "system": {
        "hostname": "$hostname",
        "cpu": "$cpu_model",
        "ram": "$total_ram"
    },
    "llama_cpp": {
        "version": "$llama_version",
        "commit": "$llama_commit"
    },
    "prompt_version": "$PROMPT_VERSION",
    "benchmark_version": "1.0"
}
EOF

  echo "$RUN_DIR"
}

# Record a single test result to JSONL index
# Usage: record_result "domain" "model_name" "config" "test_name" "speed_tps" "tokens" "output_file"
record_result() {
  local domain="$1"
  local model_name="$2"
  local config="$3"
  local test_name="$4"
  local speed_tps="$5"
  local tokens="$6"
  local output_file="$7"

  local timestamp

  timestamp=$(date -Iseconds)
  local llama_version
  llama_version=$(get_llama_version)

  # Append to index (atomic write with flock)
  (
    flock -x 200
    echo "{\"timestamp\":\"$timestamp\",\"domain\":\"$domain\",\"model\":\"$model_name\",\"config\":\"$config\",\"test\":\"$test_name\",\"speed_tps\":$speed_tps,\"tokens\":$tokens,\"prompt_version\":$PROMPT_VERSION,\"llama_version\":\"$llama_version\",\"output_file\":\"$output_file\"}" >>"$INDEX_FILE"
  ) 200>"$INDEX_FILE.lock"
}

# Extract speed from llama.cpp output file
# Usage: extract_speed "output_file"
extract_speed() {
  local output_file="$1"
  grep "eval time" "$output_file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0"
}

# Extract token count from llama.cpp output file
extract_tokens() {
  local output_file="$1"
  grep "eval time" "$output_file" 2>/dev/null | grep -oP '\d+(?= runs)' | tail -1 || echo "0"
}

# Extract model response (filter out llama.cpp noise)
extract_response() {
  local output_file="$1"
  grep -v "^llama\|^load\|^print_info\|^common\|^sampler\|^generate\|^system_info\|^main:\|^clip\|^warmup\|^alloc\|^WARN\|^encoding\|^image\|^decoding" "$output_file" 2>/dev/null || true
}

# Create output filename
# Usage: get_output_path "run_dir" "model_name" "config" "test_name"
get_output_path() {
  local run_dir="$1"
  local model_name="$2"
  local config="$3"
  local test_name="$4"
  echo "$run_dir/${model_name}_${config}_${test_name}.txt"
}

# Log with timestamp
log() {
  echo "[$(date '+%H:%M:%S')] $1"
}

# Ensure prompts directory has VERSION file
ensure_prompt_version() {
  echo "$PROMPT_VERSION" >"$PROMPTS_DIR/VERSION"
}

echo "Benchmark library loaded (v1.0, prompts v$PROMPT_VERSION)"
