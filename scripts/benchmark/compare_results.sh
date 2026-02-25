#!/bin/bash
# =============================================================================
# Benchmark Results Comparison Tool
# =============================================================================
# Compare benchmark results across runs, models, or configurations
#
# Usage:
#   ./compare_results.sh --baseline RUN_ID --current RUN_ID
#   ./compare_results.sh --model MODEL_NAME [--domain DOMAIN]
#   ./compare_results.sh --list-runs
#   ./compare_results.sh --summary RUN_ID
#
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

BENCHMARK_BASE="${PROJECT_ROOT}/benchmarks"
RESULTS_DIR="$BENCHMARK_BASE/results"
INDEX_FILE="$RESULTS_DIR/index.jsonl"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
  echo "Usage:"
  echo "  $0 --list-runs                     List all benchmark runs"
  echo "  $0 --summary RUN_ID                Summary of a specific run"
  echo "  $0 --baseline ID --current ID      Compare two runs"
  echo "  $0 --model NAME [--domain DOMAIN]  Show all results for a model"
  echo "  $0 --speed-leaders [--domain D]    Show fastest models per test"
  exit 1
}

list_runs() {
  echo "=============================================="
  echo "Available Benchmark Runs"
  echo "=============================================="

  if [[ ! -f "$INDEX_FILE" ]]; then
    echo "No benchmark results found."
    exit 0
  fi

  echo ""
  echo "Run ID           | Date                | Domains           | Tests"
  echo "-----------------|---------------------|-------------------|------"

  # Get unique run IDs with stats
  jq -r '.run_id' "$INDEX_FILE" 2>/dev/null | sort -u | while read run_id; do
    if [[ -n "$run_id" ]]; then
      timestamp=$(jq -r "select(.run_id==\"$run_id\") | .timestamp" "$INDEX_FILE" | head -1 | cut -d'T' -f1)
      domains=$(jq -r "select(.run_id==\"$run_id\") | .domain" "$INDEX_FILE" | sort -u | tr '\n' ',' | sed 's/,$//')
      count=$(jq -r "select(.run_id==\"$run_id\")" "$INDEX_FILE" | wc -l)
      printf "%-16s | %-19s | %-17s | %s\n" "$run_id" "$timestamp" "$domains" "$count"
    fi
  done
}

run_summary() {
  local run_id="$1"

  echo "=============================================="
  echo "Run Summary: $run_id"
  echo "=============================================="

  # Check if run exists
  if ! grep -q "\"run_id\":\"$run_id\"" "$INDEX_FILE" 2>/dev/null; then
    echo "Run not found: $run_id"
    exit 1
  fi

  # Show metadata if available
  metadata_file="$RESULTS_DIR/runs/$run_id/metadata.json"
  if [[ -f "$metadata_file" ]]; then
    echo ""
    echo "Metadata:"
    cat "$metadata_file" | python3 -m json.tool 2>/dev/null || cat "$metadata_file"
  fi

  echo ""
  echo "Results by Domain:"
  echo ""

  for domain in thinking coder vl general agentic math; do
    count=$(jq -r "select(.run_id==\"$run_id\" and .domain==\"$domain\")" "$INDEX_FILE" 2>/dev/null | wc -l)
    if [[ $count -gt 0 ]]; then
      echo "--- $domain ($count tests) ---"

      # Show speed stats
      speeds=$(jq -r "select(.run_id==\"$run_id\" and .domain==\"$domain\") | .speed_tps" "$INDEX_FILE" 2>/dev/null | grep -v "^0$" | sort -n)
      if [[ -n "$speeds" ]]; then
        min=$(echo "$speeds" | head -1)
        max=$(echo "$speeds" | tail -1)
        avg=$(echo "$speeds" | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
        echo "  Speed: min=${min} t/s, max=${max} t/s, avg=${avg} t/s"
      fi

      # List models tested
      models=$(jq -r "select(.run_id==\"$run_id\" and .domain==\"$domain\") | .model" "$INDEX_FILE" 2>/dev/null | sort -u | tr '\n' ', ' | sed 's/, $//')
      echo "  Models: $models"
      echo ""
    fi
  done
}

compare_runs() {
  local baseline_id="$1"
  local current_id="$2"

  echo "=============================================="
  echo "Comparing Runs"
  echo "Baseline: $baseline_id"
  echo "Current:  $current_id"
  echo "=============================================="

  # Verify both runs exist
  if ! grep -q "\"run_id\":\"$baseline_id\"" "$INDEX_FILE" 2>/dev/null; then
    echo "Baseline run not found: $baseline_id"
    exit 1
  fi
  if ! grep -q "\"run_id\":\"$current_id\"" "$INDEX_FILE" 2>/dev/null; then
    echo "Current run not found: $current_id"
    exit 1
  fi

  echo ""
  echo "Model                          | Test              | Baseline  | Current   | Delta"
  echo "-------------------------------|-------------------|-----------|-----------|--------"

  # Find common tests
  jq -r "select(.run_id==\"$baseline_id\") | \"\(.model)|\(.config)|\(.test)|\(.domain)\"" "$INDEX_FILE" 2>/dev/null | sort -u | while read baseline_key; do
    model=$(echo "$baseline_key" | cut -d'|' -f1)
    config=$(echo "$baseline_key" | cut -d'|' -f2)
    test=$(echo "$baseline_key" | cut -d'|' -f3)
    domain=$(echo "$baseline_key" | cut -d'|' -f4)

    # Get baseline speed
    baseline_speed=$(jq -r "select(.run_id==\"$baseline_id\" and .model==\"$model\" and .config==\"$config\" and .test==\"$test\") | .speed_tps" "$INDEX_FILE" 2>/dev/null | head -1)

    # Get current speed
    current_speed=$(jq -r "select(.run_id==\"$current_id\" and .model==\"$model\" and .config==\"$config\" and .test==\"$test\") | .speed_tps" "$INDEX_FILE" 2>/dev/null | head -1)

    if [[ -n "$current_speed" && "$current_speed" != "null" && "$baseline_speed" != "0" && "$current_speed" != "0" ]]; then
      # Calculate delta
      delta=$(echo "scale=1; (($current_speed - $baseline_speed) / $baseline_speed) * 100" | bc 2>/dev/null || echo "N/A")

      # Color code delta
      if [[ "$delta" != "N/A" ]]; then
        if (($(echo "$delta > 5" | bc -l))); then
          delta_str="${GREEN}+${delta}%${NC}"
        elif (($(echo "$delta < -5" | bc -l))); then
          delta_str="${RED}${delta}%${NC}"
        else
          delta_str="${delta}%"
        fi
      else
        delta_str="N/A"
      fi

      model_short="${model:0:30}"
      test_short="${test:0:17}"
      printf "%-30s | %-17s | %9.1f | %9.1f | %s\n" "$model_short" "$test_short" "$baseline_speed" "$current_speed" "$delta_str"
    fi
  done
}

model_history() {
  local model="$1"
  local domain="${2:-}"

  echo "=============================================="
  echo "Results for Model: $model"
  [[ -n "$domain" ]] && echo "Domain: $domain"
  echo "=============================================="

  local filter="select(.model==\"$model\")"
  [[ -n "$domain" ]] && filter="select(.model==\"$model\" and .domain==\"$domain\")"

  echo ""
  echo "Run ID           | Config    | Test              | Speed (t/s)"
  echo "-----------------|-----------|-------------------|------------"

  jq -r "$filter | \"\(.run_id)|\(.config)|\(.test)|\(.speed_tps)\"" "$INDEX_FILE" 2>/dev/null | sort | while read line; do
    run_id=$(echo "$line" | cut -d'|' -f1)
    config=$(echo "$line" | cut -d'|' -f2)
    test=$(echo "$line" | cut -d'|' -f3)
    speed=$(echo "$line" | cut -d'|' -f4)

    printf "%-16s | %-9s | %-17s | %s\n" "$run_id" "$config" "${test:0:17}" "$speed"
  done
}

speed_leaders() {
  local domain="${1:-}"

  echo "=============================================="
  echo "Speed Leaders"
  [[ -n "$domain" ]] && echo "Domain: $domain"
  echo "=============================================="

  local filter="."
  [[ -n "$domain" ]] && filter="select(.domain==\"$domain\")"

  echo ""
  echo "Test                           | Model                     | Config  | Speed"
  echo "-------------------------------|---------------------------|---------|-------"

  # Get unique tests
  jq -r "$filter | .test" "$INDEX_FILE" 2>/dev/null | sort -u | while read test; do
    # Find fastest for this test
    best=$(jq -r "$filter | select(.test==\"$test\") | \"\(.speed_tps)|\(.model)|\(.config)\"" "$INDEX_FILE" 2>/dev/null | sort -t'|' -k1 -rn | head -1)

    if [[ -n "$best" ]]; then
      speed=$(echo "$best" | cut -d'|' -f1)
      model=$(echo "$best" | cut -d'|' -f2)
      config=$(echo "$best" | cut -d'|' -f3)

      printf "%-30s | %-25s | %-7s | %.1f\n" "${test:0:30}" "${model:0:25}" "$config" "$speed"
    fi
  done
}

# Main
if [[ $# -eq 0 ]]; then
  usage
fi

case "$1" in
  --list-runs)
    list_runs
    ;;
  --summary)
    [[ -z "${2:-}" ]] && usage
    run_summary "$2"
    ;;
  --baseline)
    [[ -z "${2:-}" || -z "${4:-}" ]] && usage
    [[ "$3" != "--current" ]] && usage
    compare_runs "$2" "$4"
    ;;
  --model)
    [[ -z "${2:-}" ]] && usage
    domain=""
    [[ "${3:-}" == "--domain" ]] && domain="${4:-}"
    model_history "$2" "$domain"
    ;;
  --speed-leaders)
    domain=""
    [[ "${2:-}" == "--domain" ]] && domain="${3:-}"
    speed_leaders "$domain"
    ;;
  *)
    usage
    ;;
esac
