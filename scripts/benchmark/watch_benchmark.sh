#!/bin/bash
# =============================================================================
# BENCHMARK PROGRESS WATCHER
# =============================================================================
# Run this in a separate terminal to monitor overnight benchmark progress.
#
# Usage:
#   ./watch_benchmark.sh              # Watch latest run
#   ./watch_benchmark.sh RUN_ID       # Watch specific run
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Colors
readonly BOLD='\033[1m'
readonly DIM='\033[2m'
readonly RESET='\033[0m'
readonly GREEN='\033[32m'
readonly RED='\033[31m'
readonly YELLOW='\033[33m'
readonly CYAN='\033[36m'
readonly BLUE='\033[34m'

BASE_DIR="${TMP_DIR}/overnight_benchmark"

# Get run ID from argument or find latest
if [[ -n "$1" ]]; then
  RUN_ID="$1"
else
  RUN_ID=$(ls -t "$BASE_DIR" 2>/dev/null | head -1)
fi

if [[ -z "$RUN_ID" ]]; then
  echo "No benchmark runs found in $BASE_DIR"
  exit 1
fi

STATUS_FILE="$BASE_DIR/$RUN_ID/status.txt"
LOG_FILE="$BASE_DIR/$RUN_ID/benchmark.log"

if [[ ! -f "$STATUS_FILE" ]]; then
  echo "Status file not found: $STATUS_FILE"
  echo "Benchmark may not be running or run ID is incorrect."
  exit 1
fi

# Format seconds as HH:MM:SS
format_time() {
  local total=$1
  printf "%02d:%02d:%02d" $((total / 3600)) $(((total % 3600) / 60)) $((total % 60))
}

# Progress bar generator
progress_bar() {
  local current=$1
  local total=$2
  local width=25

  [[ $total -eq 0 ]] && {
    printf "[%${width}s]" ""
    return
  }

  local filled

  filled=$((current * width / total))
  local empty
  empty=$((width - filled))

  printf "["
  [[ $filled -gt 0 ]] && printf "%0.s█" $(seq 1 $filled)
  [[ $empty -gt 0 ]] && printf "%0.s░" $(seq 1 $empty)
  printf "]"
}

echo -e "${BOLD}${CYAN}Watching benchmark: $RUN_ID${RESET}"
echo -e "${DIM}Press Ctrl+C to stop watching${RESET}"
echo ""

# Multi-criteria scoring for thinking rubric (0-5 scale)
# Each question has multiple criteria worth different points
score_thinking_result() {
  local result_file="$1"
  local tier="$2"
  local qnum="$3"

  # Extract model output
  local output
  output=$(cat "$result_file" 2>/dev/null | tr '[:upper:]' '[:lower:]')
  local points=0

  # Score based on tier_question with multiple weighted criteria
  case "${tier}_${qnum}" in
    1_1) # Algorithm: insertion sort for small/mostly-sorted (max 5)
      # +2: Recommends insertion sort
      echo "$output" | grep -q "insertion" && ((points += 2))
      # +1: Mentions O(n) best case for sorted
      echo "$output" | grep -qE "o\(n\)|linear|best.case" && ((points += 1))
      # +1: Mentions small n / constant factors
      echo "$output" | grep -qE "small|constant|overhead|10" && ((points += 1))
      # +1: Explains why not quicksort (overhead, pivot)
      echo "$output" | grep -qE "overhead|pivot|partition|overkill" && ((points += 1))
      ;;
    1_2) # Thread safety (max 5)
      # +2: Identifies race condition / thread safety issue
      echo "$output" | grep -qE "race|thread.safe|concurren|synchron" && ((points += 2))
      # +1: Mentions mutex/lock solution
      echo "$output" | grep -qE "mutex|lock|semaphore" && ((points += 1))
      # +1: Mentions atomic operations
      echo "$output" | grep -qE "atomic|cas|compare.and" && ((points += 1))
      # +1: Discusses shared state / critical section
      echo "$output" | grep -qE "shared|critical|protect" && ((points += 1))
      ;;
    2_1) # Mutable default argument bug (max 5)
      # +2: Identifies mutable default argument issue
      echo "$output" | grep -qE "mutable.*default|default.*mutable|shared.*default" && ((points += 2))
      # +1: Mentions dict/list persists between calls
      echo "$output" | grep -qE "persist|reuse|same.*object|shared" && ((points += 1))
      # +1: Suggests None default + create inside
      echo "$output" | grep -qE "none.*default|if.*none|default.*none" && ((points += 1))
      # +1: Explains Python's evaluation timing
      echo "$output" | grep -qE "definition.time|once|evaluated" && ((points += 1))
      ;;
    2_2) # Cache invalidation bug (max 5)
      # +2: Identifies stale cache / invalidation issue
      echo "$output" | grep -qE "stale|invalid|outdated|expired" && ((points += 2))
      # +1: Mentions TTL or expiration
      echo "$output" | grep -qE "ttl|expir|time.to.live|timeout" && ((points += 1))
      # +1: Mentions cache coherence / consistency
      echo "$output" | grep -qE "cohere|consisten|sync" && ((points += 1))
      # +1: Suggests invalidation strategy
      echo "$output" | grep -qE "invalidat|refresh|update|purge" && ((points += 1))
      ;;
    2_3) # API design (max 5)
      # +2: Discusses versioning or backwards compatibility
      echo "$output" | grep -qE "version|backward|compat" && ((points += 2))
      # +1: Mentions deprecation strategy
      echo "$output" | grep -qE "deprecat|sunset|migrat" && ((points += 1))
      # +1: Discusses breaking changes
      echo "$output" | grep -qE "break|contract|semver" && ((points += 1))
      # +1: Mentions documentation / communication
      echo "$output" | grep -qE "document|communicat|changelog" && ((points += 1))
      ;;
    3_1) # Dependency resolution (max 5)
      # +2: Mentions topological sort
      echo "$output" | grep -qE "topolog|topo.sort" && ((points += 2))
      # +1: Discusses DAG / directed acyclic
      echo "$output" | grep -qE "dag|directed.*acyclic|acyclic" && ((points += 1))
      # +1: Mentions cycle detection
      echo "$output" | grep -qE "cycle|circular" && ((points += 1))
      # +1: Discusses DFS/BFS or Kahn's algorithm
      echo "$output" | grep -qE "dfs|bfs|kahn|depth.first|breadth" && ((points += 1))
      ;;
    3_2) # Vector clocks (max 5)
      # +2: Mentions vector clocks or Lamport timestamps
      echo "$output" | grep -qE "vector.clock|lamport" && ((points += 2))
      # +1: Discusses happens-before relationship
      echo "$output" | grep -qE "happen.*before|causal|partial.order" && ((points += 1))
      # +1: Mentions concurrent events detection
      echo "$output" | grep -qE "concurrent|parallel|simultaneous" && ((points += 1))
      # +1: Discusses distributed systems context
      echo "$output" | grep -qE "distribut|node|replica" && ((points += 1))
      ;;
    3_3) # Type system variance (max 5)
      # +2: Mentions covariance/contravariance
      echo "$output" | grep -qE "covariant|contravariant|variance" && ((points += 2))
      # +1: Discusses subtyping / Liskov
      echo "$output" | grep -qE "subtyp|liskov|substitut" && ((points += 1))
      # +1: Mentions input/output position
      echo "$output" | grep -qE "input.*output|producer.*consumer|in.*out" && ((points += 1))
      # +1: Gives correct examples
      echo "$output" | grep -qE "array|list|function|generic" && ((points += 1))
      ;;
    3_4) # Probability / Bayes (max 5)
      # +2: Mentions Bayes theorem or conditional probability
      echo "$output" | grep -qE "bayes|conditional|posterior|prior" && ((points += 2))
      # +1: Shows formula or calculation structure
      echo "$output" | grep -qE "p\(.*\|.*\)|formula|calculat" && ((points += 1))
      # +1: Discusses prior vs posterior
      echo "$output" | grep -qE "prior|posterior|update.*belief" && ((points += 1))
      # +1: Mentions base rate or false positive
      echo "$output" | grep -qE "base.rate|false.positive|sensitiv|specific" && ((points += 1))
      ;;
    *)
      echo "--"
      return
      ;;
  esac

  # Cap at 5
  [[ $points -gt 5 ]] && points=5
  echo "$points/5"
}

# Main watch loop
while true; do
  if [[ ! -f "$STATUS_FILE" ]]; then
    echo -e "${RED}Status file removed - benchmark may have ended${RESET}"
    break
  fi

  # Read status file values safely (avoid sourcing to prevent issues with special chars)
  run_id=$(grep '^run_id=' "$STATUS_FILE" | cut -d= -f2)
  suite=$(grep '^suite=' "$STATUS_FILE" | cut -d= -f2)
  model=$(grep '^model=' "$STATUS_FILE" | cut -d= -f2)
  suites_run=$(grep '^suites_run=' "$STATUS_FILE" | cut -d= -f2)
  suites_passed=$(grep '^suites_passed=' "$STATUS_FILE" | cut -d= -f2)
  suites_failed=$(grep '^suites_failed=' "$STATUS_FILE" | cut -d= -f2)
  queued_processed=$(grep '^queued_processed=' "$STATUS_FILE" | cut -d= -f2)
  elapsed_seconds=$(grep '^elapsed_seconds=' "$STATUS_FILE" | cut -d= -f2)
  quick_mode=$(grep '^quick_mode=' "$STATUS_FILE" | cut -d= -f2)
  dry_run=$(grep '^dry_run=' "$STATUS_FILE" | cut -d= -f2)
  last_update=$(grep '^last_update=' "$STATUS_FILE" | cut -d= -f2-)

  # Infer detailed progress by scanning result files
  # Count unique models, configs, and questions from existing results
  results_base="${TMP_DIR}"

  # Get most recent result file to determine current state
  latest_result=$(ls -t ${results_base}/*_rubric_results/*.txt 2>/dev/null | head -1)

  if [[ -n "$latest_result" ]]; then
    latest_name=$(basename "$latest_result" .txt)
    # Parse: ModelName_config_tX_qY_name.txt
    if [[ "$latest_name" =~ ^(.+)_(baseline|moe[0-9]+|spec_k[0-9]+)_t([0-9]+)_q([0-9]+)_ ]]; then
      current_model="${BASH_REMATCH[1]}"
      current_opt="${BASH_REMATCH[2]}"
      current_tier="${BASH_REMATCH[3]}"
      current_qnum="${BASH_REMATCH[4]}"
      current_question="T${current_tier}Q${current_qnum}"
    fi
  fi

  # Count completed items
  all_results=$(ls ${results_base}/*_rubric_results/*.txt 2>/dev/null)

  # Unique models tested
  models_done=$(echo "$all_results" | xargs -I{} basename {} .txt 2>/dev/null |
    sed 's/_baseline.*//;s/_moe[0-9]*.*//;s/_spec_k[0-9]*.*//;s/_lookup.*//' | sort -u | wc -l)

  # Unique model+config combinations
  configs_done=$(echo "$all_results" | xargs -I{} basename {} .txt 2>/dev/null |
    sed 's/_t[0-9]_q[0-9].*//' | sort -u | wc -l)

  # Total questions completed
  questions_done=$(echo "$all_results" | wc -l)

  # Estimate totals based on quick mode
  if [[ "$quick_mode" == "true" ]]; then
    est_opts_per_model=3 # baseline + 2 spec/moe
    est_questions=9      # 9 tier questions
  else
    est_opts_per_model=5 # baseline + 4 spec/moe
    est_questions=9
  fi

  # Use current model from status file if available, else from results
  [[ -z "$model" ]] && model="$current_model"
  opt="${current_opt:-baseline}"
  question="${current_question:-}"

  # Determine if complete
  if [[ "$suite" == "complete" ]]; then
    clear
    echo -e "${BOLD}${GREEN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                                                              ║"
    echo "║              ✓ BENCHMARK COMPLETE                            ║"
    echo "║                                                              ║"
    echo "╠══════════════════════════════════════════════════════════════╣"
    printf "║  Run ID:   %-48s  ║\n" "$run_id"
    printf "║  Duration: %-48s  ║\n" "$(format_time ${elapsed_seconds:-0})"
    printf "║  Passed:   %-3d    Failed: %-3d    Total: %-3d                 ║\n" "${suites_passed:-0}" "${suites_failed:-0}" "${suites_run:-0}"
    [[ ${queued_processed:-0} -gt 0 ]] && printf "║  Queued models processed: %-32d  ║\n" "$queued_processed"
    printf "║  Results:  %-48s  ║\n" "$BASE_DIR/$run_id"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${RESET}"
    break
  fi

  # Clear screen for update
  clear

  # Calculate totals
  total_suites=9
  [[ "$suite" != *"all"* && "$suite" != "queued"* && "$suite" != "initializing" ]] && total_suites=1
  completed=$((${suites_passed:-0} + ${suites_failed:-0}))

  # Estimate total models (from registry or hardcoded estimate)
  est_total_models=7 # typical thinking benchmark has ~7 models

  # Questions per config: 9 tier questions
  questions_per_config=9

  # Current question within current config
  if [[ -n "$current_model" ]] && [[ -n "$current_opt" ]]; then
    current_config_questions=$(echo "$all_results" | xargs -I{} basename {} .txt 2>/dev/null |
      grep "^${current_model}_${current_opt}_" | wc -l)
  else
    current_config_questions=0
  fi

  # Display header
  echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${RESET}"
  echo -e "${BOLD}${CYAN}║${RESET}  ${BOLD}OVERNIGHT BENCHMARK WATCHER${RESET}                              Run: ${DIM}$run_id${RESET}  ${BOLD}${CYAN}║${RESET}"
  echo -e "${BOLD}${CYAN}╠════════════════════════════════════════════════════════════════════════════╣${RESET}"

  # 4 Progress bars
  # 1. Models (unique models completed)
  pbar_model=$(progress_bar ${models_done:-0} $est_total_models 15)
  model_short="${model:-${current_model:-waiting}}"
  [[ ${#model_short} -gt 25 ]] && model_short="${model_short:0:22}..."
  printf "${BOLD}${CYAN}║${RESET}  ${BOLD}Models:${RESET}    %s %2d/%-2d  ${YELLOW}%-25s${RESET}      ${BOLD}${CYAN}║${RESET}\n" \
    "$pbar_model" "${models_done:-0}" "$est_total_models" "$model_short"

  # 2. Optimizations (configs done for current suite)
  pbar_opt=$(progress_bar ${configs_done:-0} $((est_total_models * est_opts_per_model)) 15)
  printf "${BOLD}${CYAN}║${RESET}  ${BOLD}Configs:${RESET}   %s %2d/%-2d  ${MAGENTA}%-25s${RESET}      ${BOLD}${CYAN}║${RESET}\n" \
    "$pbar_opt" "${configs_done:-0}" "$((est_total_models * est_opts_per_model))" "${opt:-baseline}"

  # 3. Suites
  pbar_suite=$(progress_bar $completed $total_suites 15)
  printf "${BOLD}${CYAN}║${RESET}  ${BOLD}Suites:${RESET}    %s %2d/%-2d  ${CYAN}%-25s${RESET}      ${BOLD}${CYAN}║${RESET}\n" \
    "$pbar_suite" "$completed" "$total_suites" "${suite:-waiting}"

  # 4. Questions (current config progress)
  pbar_q=$(progress_bar ${current_config_questions:-0} $questions_per_config 15)
  printf "${BOLD}${CYAN}║${RESET}  ${BOLD}Questions:${RESET} %s %2d/%-2d  ${GREEN}%-25s${RESET}      ${BOLD}${CYAN}║${RESET}\n" \
    "$pbar_q" "${current_config_questions:-0}" "$questions_per_config" "${question:-}"

  echo -e "${BOLD}${CYAN}╠════════════════════════════════════════════════════════════════════════════╣${RESET}"

  # Stats line with total questions completed
  mode_str=""
  [[ "$quick_mode" == "true" ]] && mode_str="${MAGENTA}[QUICK]${RESET} "
  [[ "$dry_run" == "true" ]] && mode_str="${YELLOW}[DRY]${RESET} "
  printf "${BOLD}${CYAN}║${RESET}  ${mode_str}${GREEN}✓ %d passed${RESET}  ${RED}✗ %d failed${RESET}  ⏱ %s  ${DIM}(%d total tests)${RESET}" \
    "${suites_passed:-0}" "${suites_failed:-0}" "$(format_time ${elapsed_seconds:-0})" "${questions_done:-0}"
  [[ ${queued_processed:-0} -gt 0 ]] && printf "  ${BLUE}⚡%d${RESET}" "$queued_processed"
  printf "   ${BOLD}${CYAN}║${RESET}\n"

  echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${RESET}"
  echo ""

  # Show recent results organized by model and suite
  echo -e "${BOLD}Recent Results:${RESET}"
  echo -e "${DIM}────────────────────────────────────────────────────────────────────────────────────${RESET}"
  printf "  ${DIM}%-20s %-10s %-5s %-3s %-3s %-6s %8s %7s${RESET}\n" "MODEL" "CONFIG" "SUITE" "T" "Q" "SCORE" "SPEED" "ACCEPT"
  echo -e "${DIM}────────────────────────────────────────────────────────────────────────────────────${RESET}"

  results_shown=0
  max_results=15

  # Map directories to suite names
  declare -A SUITE_NAMES=(
    ["/mnt/raid0/llm/tmp/thinking_rubric_results"]="think"
    ["/mnt/raid0/llm/tmp/coder_rubric_results"]="coder"
    ["/mnt/raid0/llm/tmp/general_rubric_results"]="genrl"
    ["/mnt/raid0/llm/tmp/math_rubric_results"]="math"
    ["/mnt/raid0/llm/tmp/agentic_rubric_results"]="agent"
    ["/mnt/raid0/llm/tmp/long_context_rubric_results"]="lctx"
    ["/mnt/raid0/llm/tmp/instruction_precision_rubric_results"]="instr"
  )

  # Collect all recent results across directories
  all_results=$(mktemp)
  for results_dir in "${!SUITE_NAMES[@]}"; do
    [[ ! -d "$results_dir" ]] && continue
    suite_name="${SUITE_NAMES[$results_dir]}"
    for f in "$results_dir"/*.txt; do
      [[ -f "$f" ]] && echo "$(stat -c %Y "$f") $suite_name $f"
    done
  done 2>/dev/null | sort -rn | head -$max_results >"$all_results"

  while read -r mtime suite_name result_file; do
    [[ -z "$result_file" ]] && continue

    # Parse filename: ModelName_config_tier_question.txt
    fname=$(basename "$result_file" .txt)

    # Extract components using regex
    if [[ "$fname" =~ ^(.+)_(baseline|moe[0-9]+|spec_k[0-9]+)_t([0-9]+)_q([0-9]+)_ ]]; then
      model_short="${BASH_REMATCH[1]}"
      config="${BASH_REMATCH[2]}"
      tier="${BASH_REMATCH[3]}"
      qnum="${BASH_REMATCH[4]}"
    else
      # Fallback parsing
      model_short=$(echo "$fname" | cut -d_ -f1)
      config=$(echo "$fname" | cut -d_ -f2)
      tier="?"
      qnum="?"
    fi

    # Truncate model name
    model_short="${model_short:0:20}"
    [[ ${#model_short} -eq 20 ]] && model_short="${model_short:0:17}..."

    # Extract speed
    speed=""
    if grep -q "eval time" "$result_file" 2>/dev/null; then
      speed=$(grep "eval time" "$result_file" 2>/dev/null | grep -oP '\d+\.\d+(?= tokens per second)' | tail -1)
    fi

    # Extract acceptance rate
    accept=""
    if [[ "$config" =~ spec ]]; then
      accept=$(grep -oP 'accept\s*=\s*\K[\d.]+' "$result_file" 2>/dev/null | tail -1)
      [[ -n "$accept" ]] && accept="${accept}%"
    fi

    # Color-code speed (green >100, yellow >50)
    speed_color="$RESET"
    if [[ -n "$speed" ]]; then
      speed_int=${speed%.*}
      if [[ $speed_int -ge 100 ]]; then
        speed_color="$GREEN"
      elif [[ $speed_int -ge 50 ]]; then
        speed_color="$YELLOW"
      fi
    fi

    # Color-code config (spec=cyan, moe=magenta, baseline=dim)
    config_color="$DIM"
    [[ "$config" =~ spec ]] && config_color="$CYAN"
    [[ "$config" =~ moe ]] && config_color="$MAGENTA"

    # Get quality score for thinking suite (0-5 scale)
    score="--"
    score_color="$DIM"
    if [[ "$suite_name" == "think" ]] && [[ "$tier" != "?" ]]; then
      score=$(score_thinking_result "$result_file" "$tier" "$qnum")
      if [[ "$score" =~ ^([0-5])/5$ ]]; then
        score_num="${BASH_REMATCH[1]}"
        case "$score_num" in
          5 | 4) score_color="$GREEN" ;;
          3) score_color="$YELLOW" ;;
          2 | 1) score_color="$RED" ;;
          0) score_color="$RED" ;;
        esac
      fi
    fi

    # Display row
    if [[ -n "$speed" ]]; then
      printf "  %-20s ${config_color}%-10s${RESET} ${CYAN}%-5s${RESET} %-3s %-3s ${score_color}%-6s${RESET} ${speed_color}%7s${RESET} %7s\n" \
        "$model_short" "$config" "$suite_name" "$tier" "$qnum" "$score" "${speed:-N/A}" "${accept:--}"
      ((results_shown++))
    fi
  done <"$all_results"
  rm -f "$all_results"

  if [[ $results_shown -eq 0 ]]; then
    echo -e "  ${DIM}No results yet - waiting for first benchmark to complete...${RESET}"
  fi

  echo ""
  echo -e "${DIM}Last update: $last_update | Log: $LOG_FILE${RESET}"

  sleep 3
done
