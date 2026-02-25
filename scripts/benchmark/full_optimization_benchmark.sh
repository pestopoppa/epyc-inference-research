#!/bin/bash
set -uo pipefail

###############################################################################
# COMPREHENSIVE OPTIMIZATION BENCHMARK SCRIPT
# Tests all optimization techniques against all valid model combinations
#
# Usage: ./full_optimization_benchmark.sh [--dry-run] [--quick] [--force]
#   --dry-run  Show what would be tested without running
#   --quick    Run abbreviated tests (fewer tokens, fewer repeats)
#   --force    Re-run all tests even if results exist in research report
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Configuration (using env vars from env.sh)
LLAMA_DIR="${LLAMA_CPP_BIN}"
LOG_DIR="${LOG_DIR}/benchmarks"
RESEARCH_REPORT="${PROJECT_ROOT}/docs/reference/benchmarks/RESULTS.md"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$LOG_DIR/optimization_results_${TIMESTAMP}.csv"
SUMMARY_FILE="$LOG_DIR/optimization_summary_${TIMESTAMP}.md"
CACHE_DIR="${LLM_ROOT}/tmp/bench_cache"
REGISTRY_PATH="${PROJECT_ROOT}/orchestration/model_registry.yaml"

# Helper function to read timeouts from model_registry.yaml
read_registry_timeout() {
  local category="$1"
  local key="$2"
  local default="$3"
  python3 -c "
import yaml
try:
    with open('$REGISTRY_PATH') as f:
        data = yaml.safe_load(f)
    val = data.get('runtime_defaults', {}).get('timeouts', {}).get('$category', {}).get('$key', $default)
    print(int(val))
except Exception:
    print($default)
" 2>/dev/null || echo "$default"
}

# Timeouts from model_registry.yaml (runtime_defaults.timeouts.benchmark.*)
TIMEOUT_SMALL=$(read_registry_timeout benchmark timeout_small 120)   # <2B models
TIMEOUT_MEDIUM=$(read_registry_timeout benchmark timeout_medium 180) # 2B-14B models
TIMEOUT_LARGE=$(read_registry_timeout benchmark timeout_large 300)   # 14B-72B models
TIMEOUT_HUGE=$(read_registry_timeout benchmark timeout_huge 600)     # 72B-100B models
TIMEOUT_GIANT=$(read_registry_timeout benchmark timeout_giant 1200)  # >100B models (235B, 480B)

# Test parameters
THREADS=96
N_PROMPT=512
N_GEN=128

# Parse arguments
DRY_RUN=false
QUICK=false
FORCE=false
for arg in "$@"; do
  case $arg in
    --dry-run) DRY_RUN=true ;;
    --quick)
      QUICK=true
      N_GEN=64
      ;;
    --force) FORCE=true ;;
  esac
done

# Create directories
mkdir -p "$LOG_DIR" "$CACHE_DIR"

###############################################################################
# TEST PROMPTS
###############################################################################

# Summarization prompt (high overlap - good for lookup)
read -r -d '' PROMPT_SUMMARIZE <<'ENDPROMPT' || true
<|im_start|>user
Summarize this text in 3 bullet points:

The AMD EPYC 9655 processor features 96 cores and 192 threads based on the Zen 5 architecture. It supports DDR5-5600 memory across 12 channels providing approximately 460 GB/s of memory bandwidth with 1.13 TB total capacity. The processor includes true 512-bit AVX-512 support.
<|im_end|>
<|im_start|>assistant
ENDPROMPT

# Code generation prompt (low overlap - good for draft)
read -r -d '' PROMPT_CODE <<'ENDPROMPT' || true
<|im_start|>user
Write a Python function to calculate the nth Fibonacci number using memoization.
<|im_end|>
<|im_start|>assistant
ENDPROMPT

# Code editing prompt (medium overlap)
read -r -d '' PROMPT_EDIT <<'ENDPROMPT' || true
<|im_start|>user
Add error handling to this function:

def divide(a, b):
    return a / b
<|im_end|>
<|im_start|>assistant
ENDPROMPT

###############################################################################
# UTILITY FUNCTIONS
###############################################################################

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_DIR/benchmark.log"
}

# Check if a result already exists in the research report
# Returns 0 (true) if result exists, 1 (false) if not
has_existing_result() {
  # If --force is set, always return "not found"
  $FORCE && return 1

  local model_name="$1"
  local method="$2"
  local draft_name="${3:-}"

  # Build search pattern based on method
  local pattern=""
  case "$method" in
    "baseline")
      # Match model name with t/s in baseline tables
      pattern="$model_name.*t/s"
      ;;
    "lookup")
      # Match "Prompt Lookup" or "lookup" with speed result
      pattern="(Lookup|lookup).*$model_name|$model_name.*(Lookup|lookup)"
      ;;
    "soft_mask")
      # Match "Soft mask" with model
      pattern="(Soft mask|soft.mask).*$model_name|$model_name.*(Soft mask|soft.mask)"
      ;;
    "external_draft")
      # Match model + draft pair
      if [ -n "$draft_name" ]; then
        pattern="$model_name.*$draft_name|$draft_name.*$model_name"
      else
        return 1
      fi
      ;;
    *)
      return 1
      ;;
  esac

  # Search research report
  if grep -qiE "$pattern" "$RESEARCH_REPORT" 2>/dev/null; then
    return 0 # Found
  fi
  return 1 # Not found
}

get_timeout() {
  local model_path="$1"
  local size_bytes
  size_bytes=$(stat -c%s "$model_path" 2>/dev/null || echo 0)
  local size_gb
  size_gb=$((size_bytes / 1024 / 1024 / 1024))

  if [ "$size_gb" -lt 2 ]; then
    echo $TIMEOUT_SMALL
  elif [ "$size_gb" -lt 15 ]; then
    echo $TIMEOUT_MEDIUM
  elif [ "$size_gb" -lt 50 ]; then
    echo $TIMEOUT_LARGE
  elif [ "$size_gb" -lt 100 ]; then
    echo $TIMEOUT_HUGE
  else
    echo $TIMEOUT_GIANT
  fi
}

run_benchmark() {
  local name="$1"
  local cmd="$2"
  local timeout_sec="$3"
  local output_file="$CACHE_DIR/bench_output_$$.txt"

  log "Running: $name" >&2

  if $DRY_RUN; then
    echo "  [DRY-RUN] Would run: ${cmd:0:100}..." >&2
    echo "success,0,0,0"
    return 0
  fi

  # Run with timeout
  if timeout "$timeout_sec" bash -c "$cmd" >"$output_file" 2>&1; then
    # Parse results
    local decoded
    decoded=$(grep -oP 'decoded\s+\d+\s+tokens\s+in\s+[\d.]+\s+seconds,\s+speed:\s+\K[\d.]+' "$output_file" 2>/dev/null | tail -1 || echo "0")
    local accept
    accept=$(grep -oP 'accept\s+=\s+\K[\d.]+' "$output_file" 2>/dev/null | tail -1 || echo "0")
    local n_accept
    n_accept=$(grep -oP 'n_accept\s+=\s+\K\d+' "$output_file" 2>/dev/null | tail -1 || echo "0")

    # Also try llama-bench format
    if [ "$decoded" = "0" ]; then
      decoded=$(grep -oP 'tg\s+\d+:\s+[\d.]+\s+±\s+[\d.]+\s+ms.*\|\s+\K[\d.]+' "$output_file" 2>/dev/null | tail -1 || echo "0")
    fi

    log "  Result: $decoded t/s, $accept% accept" >&2
    echo "success,$decoded,$accept,$n_accept"
  else
    local exit_code=$?
    if [ $exit_code -eq 124 ]; then
      log "  TIMEOUT after ${timeout_sec}s" >&2
      echo "timeout,0,0,0"
    else
      log "  ERROR (exit $exit_code)" >&2
      echo "error,0,0,0"
    fi
  fi

  rm -f "$output_file"
}

###############################################################################
# BENCHMARK FUNCTIONS
###############################################################################

run_baseline() {
  local model_name="$1"
  local model_path="$2"
  local prompt_name="$3"

  [ ! -f "$model_path" ] && {
    echo "skip,0,0,0"
    return
  }

  local timeout
  timeout=$(get_timeout "$model_path")
  local cmd="$LLAMA_DIR/llama-bench -m '$model_path' -t $THREADS -p $N_PROMPT -n $N_GEN -r 1"

  run_benchmark "Baseline: $model_name ($prompt_name)" "$cmd" "$timeout"
}

run_lookup() {
  local model_name="$1"
  local model_path="$2"
  local prompt="$3"
  local prompt_name="$4"

  [ ! -f "$model_path" ] && {
    echo "skip,0,0,0"
    return
  }

  local timeout
  timeout=$(get_timeout "$model_path")
  local cache_file="$CACHE_DIR/lookup_${model_name//[^a-zA-Z0-9]/_}_$$.bin"
  rm -f "$cache_file"

  # Escape prompt for shell
  local escaped_prompt="${prompt//\'/\'\\\'\'}"
  local cmd="$LLAMA_DIR/llama-lookup -m '$model_path' -lcd '$cache_file' -t $THREADS -n $N_GEN -p '$escaped_prompt'"

  local result
  result=$(run_benchmark "Lookup: $model_name ($prompt_name)" "$cmd" "$timeout")
  rm -f "$cache_file"
  echo "$result"
}

run_soft_mask() {
  local model_name="$1"
  local model_path="$2"
  local prompt="$3"
  local prompt_name="$4"
  local override_kv="$5"

  [ ! -f "$model_path" ] && {
    echo "skip,0,0,0"
    return
  }
  [ -z "$override_kv" ] && {
    echo "skip,0,0,0"
    return
  }

  local timeout
  timeout=$(get_timeout "$model_path")
  local escaped_prompt="${prompt//\'/\'\\\'\'}"
  # Use echo to provide empty stdin, --no-cnv to disable conversation mode
  local cmd="echo '' | $LLAMA_DIR/llama-cli -m '$model_path' --override-kv '$override_kv' -t $THREADS -n $N_GEN -p '$escaped_prompt' --no-display-prompt --no-cnv 2>&1 | tail -30"

  run_benchmark "SoftMask: $model_name ($prompt_name)" "$cmd" "$timeout"
}

run_external_draft() {
  local target_name="$1"
  local target_path="$2"
  local draft_name="$3"
  local draft_path="$4"
  local prompt="$5"
  local prompt_name="$6"

  [ ! -f "$target_path" ] && {
    echo "skip,0,0,0"
    return
  }
  [ ! -f "$draft_path" ] && {
    echo "skip,0,0,0"
    return
  }

  local timeout
  timeout=$(get_timeout "$target_path")
  local escaped_prompt="${prompt//\'/\'\\\'\'}"
  local cmd="$LLAMA_DIR/llama-speculative -m '$target_path' -md '$draft_path' --draft-max 16 -t $THREADS -n $N_GEN -p '$escaped_prompt'"

  run_benchmark "ExtDraft: $target_name + $draft_name ($prompt_name)" "$cmd" "$timeout"
}

###############################################################################
# MODEL DATABASE (as simple arrays for reliability)
###############################################################################

# Models to test
MODELS=(
  # Dense 27-32B models
  "Qwen2.5-Coder-32B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-GGUF/Qwen2.5-Coder-32B-Q4_K_M.gguf|dense|Qwen2.5-Coder-0.5B"
  "Qwen3-32B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf|dense|Qwen3-0.6B"
  "DeepSeek-R1-32B|/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf|dense|"
  "Gemma-3-27B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/gemma-3-27B-it-qat-GGUF/gemma-3-27B-it-QAT-Q4_0.gguf|dense|"
  # Dense 70-72B models
  "Meta-Llama-3.1-70B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf|dense|PARD-Llama-1B"
  "Meta-Llama-3-70B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-70B-Instruct-GGUF/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf|dense|PARD-Llama-1B"
  "Hermes-4-70B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Hermes-4-70B-GGUF/Hermes-4-70B-Q4_K_M.gguf|dense|PARD-Llama-1B"
  "DeepSeek-R1-Llama-70B|/mnt/raid0/llm/lmstudio/models/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf|dense|PARD-Llama-1B"
  # NOTE: Qwen2.5-72B shows ~2% spec decode acceptance with ALL draft models tested
  # (Coder-0.5B, Coder-1.5B, Math-1.5B, base-0.5B). Spec decode disabled - use baseline/lookup only.
  "Qwen2.5-72B-Instruct|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-72B-Instruct-GGUF/Qwen2.5-72B-Instruct-Q4_K_M.gguf|dense|"
  "Qwen2.5-Math-72B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Math-72B-Instruct-GGUF/Qwen2.5-Math-72B-Instruct-Q4_K_M.gguf|dense|"
  # MoE 30B models
  "Qwen3-VL-30B-A3B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf|moe|Qwen3-VL-2B|qwen3vlmoe.expert_used_count=int:4"
  "Qwen3-Coder-30B-A3B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|moe||qwen3moe.expert_used_count=int:4"
  # MoE 80B+ models
  "Qwen3-Next-80B-A3B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|moe||qwen3moe.expert_used_count=int:4"
  "Qwen3-235B-A22B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf|moe||qwen3moe.expert_used_count=int:8"
  "Qwen3-Coder-480B-A35B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf|moe||qwen3moe.expert_used_count=int:8"
)

# Draft models
get_draft_path() {
  local draft_name="$1"
  case "$draft_name" in
    "Qwen2.5-Coder-0.5B") echo "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf" ;;
    "Qwen3-0.6B") echo "/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf" ;;
    "Qwen3-VL-2B") echo "/mnt/raid0/llm/lmstudio/models/mradermacher/Qwen3_VL_2B-GGUF/Qwen3_VL_2B.Q4_K_M.gguf" ;;
    "PARD-Llama-1B") echo "/mnt/raid0/llm/lmstudio/models/mradermacher/PARD-Llama-3.2-1B-GGUF/PARD-Llama-3.2-1B.Q8_0.gguf" ;;
    *) echo "" ;;
  esac
}

###############################################################################
# MAIN
###############################################################################

main() {
  log "=========================================="
  log "Starting Full Optimization Benchmark"
  log "=========================================="
  log "DRY_RUN=$DRY_RUN, QUICK=$QUICK, FORCE=$FORCE, N_GEN=$N_GEN"
  log "Results: $RESULTS_FILE"
  log "Checking for existing results in: $RESEARCH_REPORT"
  log ""

  # CSV header
  echo "timestamp,model,method,draft,prompt_type,status,speed_tps,accept_pct,n_accept" >"$RESULTS_FILE"

  local total_tests=0
  local passed_tests=0
  local skipped_tests=0

  # Prompt types to test
  local prompt_types=("summarize" "code" "edit")

  for model_entry in "${MODELS[@]}"; do
    IFS='|' read -r model_name model_path model_type draft_name moe_config <<<"$model_entry"

    if [ ! -f "$model_path" ]; then
      log "SKIP: $model_name (file not found: $model_path)"
      continue
    fi

    log ""
    log "=== Testing: $model_name ($model_type) ==="

    for prompt_type in "${prompt_types[@]}"; do
      local prompt
      case "$prompt_type" in
        "summarize") prompt="$PROMPT_SUMMARIZE" ;;
        "code") prompt="$PROMPT_CODE" ;;
        "edit") prompt="$PROMPT_EDIT" ;;
      esac

      local ts
      ts=$(date '+%Y-%m-%d %H:%M:%S')

      # 1. Baseline (llama-bench)
      if has_existing_result "$model_name" "baseline"; then
        log "  SKIP baseline (already in research report)"
        echo "$ts,$model_name,baseline,none,$prompt_type,skipped,0,0,0" >>"$RESULTS_FILE"
        ((skipped_tests++))
      else
        ((total_tests++))
        result=$(run_baseline "$model_name" "$model_path" "$prompt_type")
        IFS=',' read -r status speed accept n_accept <<<"$result"
        echo "$ts,$model_name,baseline,none,$prompt_type,$status,$speed,$accept,$n_accept" >>"$RESULTS_FILE"
        [ "$status" = "success" ] && ((passed_tests++))
      fi

      # 2. Prompt Lookup
      if has_existing_result "$model_name" "lookup"; then
        log "  SKIP lookup (already in research report)"
        echo "$ts,$model_name,lookup,none,$prompt_type,skipped,0,0,0" >>"$RESULTS_FILE"
        ((skipped_tests++))
      else
        ((total_tests++))
        result=$(run_lookup "$model_name" "$model_path" "$prompt" "$prompt_type")
        IFS=',' read -r status speed accept n_accept <<<"$result"
        echo "$ts,$model_name,lookup,none,$prompt_type,$status,$speed,$accept,$n_accept" >>"$RESULTS_FILE"
        [ "$status" = "success" ] && ((passed_tests++))
      fi

      # 3. Soft Mask (MoE only)
      if [ "$model_type" = "moe" ] && [ -n "$moe_config" ]; then
        if has_existing_result "$model_name" "soft_mask"; then
          log "  SKIP soft_mask (already in research report)"
          echo "$ts,$model_name,soft_mask,none,$prompt_type,skipped,0,0,0" >>"$RESULTS_FILE"
          ((skipped_tests++))
        else
          ((total_tests++))
          result=$(run_soft_mask "$model_name" "$model_path" "$prompt" "$prompt_type" "$moe_config")
          IFS=',' read -r status speed accept n_accept <<<"$result"
          echo "$ts,$model_name,soft_mask,none,$prompt_type,$status,$speed,$accept,$n_accept" >>"$RESULTS_FILE"
          [ "$status" = "success" ] && ((passed_tests++))
        fi
      fi

      # 4. External Draft (if valid pair exists)
      if [ -n "$draft_name" ]; then
        local draft_path
        draft_path=$(get_draft_path "$draft_name")
        if [ -n "$draft_path" ] && [ -f "$draft_path" ]; then
          if has_existing_result "$model_name" "external_draft" "$draft_name"; then
            log "  SKIP external_draft (already in research report)"
            echo "$ts,$model_name,external_draft,$draft_name,$prompt_type,skipped,0,0,0" >>"$RESULTS_FILE"
            ((skipped_tests++))
          else
            ((total_tests++))
            result=$(run_external_draft "$model_name" "$model_path" "$draft_name" "$draft_path" "$prompt" "$prompt_type")
            IFS=',' read -r status speed accept n_accept <<<"$result"
            echo "$ts,$model_name,external_draft,$draft_name,$prompt_type,$status,$speed,$accept,$n_accept" >>"$RESULTS_FILE"
            [ "$status" = "success" ] && ((passed_tests++))
          fi
        fi
      fi
    done
  done

  log ""
  log "=========================================="
  log "Benchmark Complete"
  log "Tests run: $total_tests, Passed: $passed_tests, Skipped (existing): $skipped_tests"
  log "Results: $RESULTS_FILE"
  log "=========================================="

  # Show results
  echo ""
  echo "=== RESULTS ==="
  cat "$RESULTS_FILE"
}

main "$@"
