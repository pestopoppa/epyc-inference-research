#!/bin/bash
# =============================================================================
# Benchmark Results Processor
# =============================================================================
# Processes raw benchmark outputs and creates:
# - Structured metadata (metadata.json)
# - JSONL index entries (appended to index.jsonl)
# - Permanent storage copy
#
# Usage:
#   ./process_benchmark_results.sh [--source DIR] [--run-id ID]
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Paths
BENCHMARK_BASE="${PROJECT_ROOT}/benchmarks"
RESULTS_DIR="$BENCHMARK_BASE/results"
INDEX_FILE="$RESULTS_DIR/index.jsonl"
PROMPTS_DIR="$BENCHMARK_BASE/prompts"

# Defaults
SOURCE_BASE="${TMP_DIR}"
RUN_ID=$(date +%Y%m%d_%H%M%S)
PROMPT_VERSION="1"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --source)
      SOURCE_BASE="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

RUN_DIR="$RESULTS_DIR/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

echo "=============================================="
echo "Processing Benchmark Results"
echo "Run ID: $RUN_ID"
echo "Source: $SOURCE_BASE"
echo "Destination: $RUN_DIR"
echo "=============================================="

# Get system info
get_llama_version() {
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli --version 2>&1 | head -1 | grep -oP 'version: \K\d+' || echo "unknown"
}

get_llama_commit() {
  /mnt/raid0/llm/llama.cpp/build/bin/llama-cli --version 2>&1 | head -1 | grep -oP '\(\K[a-f0-9]+' || echo "unknown"
}

LLAMA_VERSION=$(get_llama_version)
LLAMA_COMMIT=$(get_llama_commit)
HOSTNAME=$(hostname)
CPU_MODEL=$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo "unknown")
TOTAL_RAM=$(free -h 2>/dev/null | grep Mem | awk '{print $2}' || echo "unknown")

# Create run metadata
cat >"$RUN_DIR/metadata.json" <<EOF
{
    "run_id": "$RUN_ID",
    "timestamp": "$(date -Iseconds)",
    "system": {
        "hostname": "$HOSTNAME",
        "cpu": "$CPU_MODEL",
        "ram": "$TOTAL_RAM"
    },
    "llama_cpp": {
        "version": "$LLAMA_VERSION",
        "commit": "$LLAMA_COMMIT"
    },
    "prompt_version": "$PROMPT_VERSION",
    "benchmark_version": "1.0",
    "domains_processed": []
}
EOF

echo "Created metadata: $RUN_DIR/metadata.json"

# Process each domain
DOMAINS_PROCESSED=""

for domain in thinking coder vl general agentic math; do
  source_dir="$SOURCE_BASE/${domain}_rubric_results"

  if [[ ! -d "$source_dir" ]]; then
    echo "Skipping $domain (no results found)"
    continue
  fi

  result_count=$(find "$source_dir" -name "*.txt" -type f 2>/dev/null | wc -l)
  if [[ $result_count -eq 0 ]]; then
    echo "Skipping $domain (no .txt files)"
    continue
  fi

  echo ""
  echo "Processing $domain ($result_count files)..."

  domain_dir="$RUN_DIR/$domain"
  mkdir -p "$domain_dir"

  # Copy all result files
  cp "$source_dir"/*.txt "$domain_dir/" 2>/dev/null || true

  # Generate JSONL entries for each file
  for file in "$domain_dir"/*.txt; do
    [[ -f "$file" ]] || continue

    filename=$(basename "$file" .txt)

    # Parse filename: ModelName_config_testname
    # Examples: Qwen2.5-Math-7B_baseline_t1_q1_arithmetic
    #           Qwen3-30B-A3B_moe4_t2_q1_word_problem

    # Extract model (everything before _baseline or _moe)
    model=$(echo "$filename" | sed -E 's/_(baseline|moe[0-9]+)_.*//')

    # Extract config (baseline or moeN)
    config=$(echo "$filename" | grep -oP '(baseline|moe[0-9]+)' | head -1 || echo "unknown")

    # Extract test name (everything after config_)
    test_name=$(echo "$filename" | sed -E 's/.*_(baseline|moe[0-9]+)_//')

    # Extract speed from file
    speed=$(grep "eval time" "$file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1 || echo "0")

    # Extract token count
    tokens=$(grep "eval time" "$file" 2>/dev/null | grep -oP '\d+(?= runs)' | tail -1 || echo "0")

    # Extract prompt eval tokens
    prompt_tokens=$(grep "prompt eval time" "$file" 2>/dev/null | grep -oP '\d+(?= tokens)' | head -1 || echo "0")

    # Relative path for storage
    rel_path="runs/$RUN_ID/$domain/$(basename "$file")"

    # Append to index (atomic with flock)
    (
      flock -x 200
      echo "{\"timestamp\":\"$(date -Iseconds)\",\"run_id\":\"$RUN_ID\",\"domain\":\"$domain\",\"model\":\"$model\",\"config\":\"$config\",\"test\":\"$test_name\",\"speed_tps\":$speed,\"tokens_generated\":$tokens,\"prompt_tokens\":$prompt_tokens,\"prompt_version\":$PROMPT_VERSION,\"llama_version\":\"$LLAMA_VERSION\",\"output_file\":\"$rel_path\"}" >>"$INDEX_FILE"
    ) 200>"$INDEX_FILE.lock"

  done

  processed_count=$(find "$domain_dir" -name "*.txt" -type f | wc -l)
  echo "  Processed $processed_count files, indexed to JSONL"

  DOMAINS_PROCESSED="$DOMAINS_PROCESSED $domain"
done

# Update metadata with processed domains
DOMAINS_JSON=$(echo $DOMAINS_PROCESSED | tr ' ' '\n' | grep -v '^$' | sed 's/.*/"&"/' | tr '\n' ',' | sed 's/,$//')
sed -i "s/\"domains_processed\": \[\]/\"domains_processed\": [$DOMAINS_JSON]/" "$RUN_DIR/metadata.json"

# Generate summary
echo ""
echo "=============================================="
echo "Processing Complete"
echo "=============================================="
echo "Run ID: $RUN_ID"
echo "Output: $RUN_DIR"
echo "Index: $INDEX_FILE"
echo "Domains: $DOMAINS_PROCESSED"
echo ""

# Count total entries added
new_entries=$(grep -c "\"run_id\":\"$RUN_ID\"" "$INDEX_FILE" 2>/dev/null || echo "0")
total_entries=$(wc -l <"$INDEX_FILE" 2>/dev/null || echo "0")
echo "New index entries: $new_entries"
echo "Total index entries: $total_entries"

# Show sample entry
echo ""
echo "Sample index entry:"
grep "\"run_id\":\"$RUN_ID\"" "$INDEX_FILE" 2>/dev/null | head -1 | python3 -m json.tool 2>/dev/null ||
  grep "\"run_id\":\"$RUN_ID\"" "$INDEX_FILE" 2>/dev/null | head -1
