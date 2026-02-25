#!/bin/bash
# =============================================================================
# BENCHMARK CONSTANTS - Defines all iteration dimensions
# =============================================================================
# Used by overnight benchmark suite for accurate progress tracking and dry-run
#
# Structure:
#   SUITES[]          - All benchmark suite names
#   SUITE_*_MODELS[]  - Models per suite
#   ARCH_*_CONFIGS[]  - Optimization configs per architecture
#   QUESTIONS_PER_RUBRIC - Number of questions per rubric run
# =============================================================================

# =============================================================================
# BENCHMARK SUITES
# =============================================================================
SUITES=(
  "thinking"
  "coder"
  "vl"
  "general"
  "agentic"
  "math"
  "long_context"
  "instruction_precision"
  "draft_discovery"
)

# =============================================================================
# MODELS PER SUITE (matches run_all_*_benchmarks.sh scripts)
# =============================================================================

# Thinking suite models
declare -a SUITE_THINKING_MODELS=(
  "Qwen3-4B-Thinking-Q8|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-4B-Thinking-2507-GGUF/Qwen3-4B-Thinking-2507-Q8_0.gguf|dense"
  "DeepSeek-R1-Distill-7B|/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf|dense"
  "DeepSeek-R1-Distill-14B|/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf|dense"
  "Qwen3-30B-A3B-Thinking-Q4|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q4_K_S.gguf|qwen3moe"
  "Qwen3-30B-A3B-Thinking-Q8|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-30B-A3B-Thinking-2507-GGUF/Qwen3-30B-A3B-Thinking-2507-Q8_0.gguf|qwen3moe"
  "DeepSeek-R1-Distill-32B|/mnt/raid0/llm/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf|dense"
  "QwQ-32B-Q4|/mnt/raid0/llm/models/QwQ-32B-Preview-Q4_K_M.gguf|dense"
)

# Coder suite models
declare -a SUITE_CODER_MODELS=(
  "Qwen3-Coder-30B-A3B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|qwen3moe"
  "Qwen2.5-Coder-32B|/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Q4_K_M.gguf|dense"
)

# VL suite models
declare -a SUITE_VL_MODELS=(
  "Qwen2.5-VL-7B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf|dense"
)

# General suite models
declare -a SUITE_GENERAL_MODELS=(
  "Meta-Llama-3-8B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf|dense"
  "Qwen2.5-7B|/mnt/raid0/llm/lmstudio/models/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf|dense"
)

# Agentic suite models
declare -a SUITE_AGENTIC_MODELS=(
  "Qwen3-Coder-30B-A3B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|qwen3moe"
)

# Math suite models
declare -a SUITE_MATH_MODELS=(
  "Qwen2.5-Math-7B|/mnt/raid0/llm/lmstudio/models/tensorblock/Qwen2.5-Math-7B-Instruct-GGUF/Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf|dense"
  "Qwen2.5-Math-1.5B|/mnt/raid0/llm/lmstudio/models/tensorblock/Qwen2.5-Math-1.5B-Instruct-GGUF/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf|dense"
)

# Long context suite models
declare -a SUITE_LONG_CONTEXT_MODELS=(
  "Qwen2.5-7B|/mnt/raid0/llm/lmstudio/models/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf|dense"
)

# Instruction precision suite models
declare -a SUITE_INSTRUCTION_PRECISION_MODELS=(
  "Qwen3-Coder-30B-A3B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf|qwen3moe"
  "Meta-Llama-3-8B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf|dense"
)

# Draft discovery - special, uses different model structure
declare -a SUITE_DRAFT_DISCOVERY_MODELS=(
  "draft_discovery|na|special"
)

# =============================================================================
# OPTIMIZATION CONFIGURATIONS BY ARCHITECTURE
# =============================================================================

# Dense models: baseline + speculative decoding sweeps (if draft available)
declare -a ARCH_DENSE_CONFIGS_FULL=("baseline" "spec_k4" "spec_k8" "spec_k16" "spec_k24")
declare -a ARCH_DENSE_CONFIGS_QUICK=("baseline" "spec_k8" "spec_k16")
declare -a ARCH_DENSE_CONFIGS_NODRAFT=("baseline")

# MoE models: baseline + expert reduction sweeps
declare -a ARCH_MOE_CONFIGS_FULL=("baseline" "moe2" "moe4" "moe6" "moe8")
declare -a ARCH_MOE_CONFIGS_QUICK=("baseline" "moe4")

# SSM models (qwen3next): baseline + expert reduction ONLY (no spec decode)
declare -a ARCH_SSM_CONFIGS_FULL=("baseline" "moe2" "moe4" "moe6" "moe8")
declare -a ARCH_SSM_CONFIGS_QUICK=("baseline" "moe4")

# Special suites with their own config handling
declare -a ARCH_SPECIAL_CONFIGS=("baseline")

# =============================================================================
# QUESTIONS PER RUBRIC
# =============================================================================
# Standard rubrics have 9 questions (3 tiers x 3 questions, except T3 has 4)
QUESTIONS_PER_RUBRIC=9

# Tier breakdown (for detailed progress)
declare -A QUESTIONS_PER_TIER=(
  ["T1"]=2
  ["T2"]=3
  ["T3"]=4
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

# Get models array for a suite
# Usage: get_suite_models "thinking"
get_suite_models() {
  local suite="$1"
  local varname="SUITE_${suite^^}_MODELS[@]"

  # Handle special cases
  case "$suite" in
    long_context) varname="SUITE_LONG_CONTEXT_MODELS[@]" ;;
    instruction_precision) varname="SUITE_INSTRUCTION_PRECISION_MODELS[@]" ;;
    draft_discovery) varname="SUITE_DRAFT_DISCOVERY_MODELS[@]" ;;
  esac

  if declare -p "${varname%\[@\]}" &>/dev/null 2>&1; then
    echo "${!varname}"
  else
    echo ""
  fi
}

# Get config array for an architecture
# Usage: get_arch_configs "qwen3moe" [quick]
get_arch_configs() {
  local arch="$1"
  local quick="${2:-false}"

  case "$arch" in
    dense)
      if [[ "$quick" == "true" ]]; then
        echo "${ARCH_DENSE_CONFIGS_QUICK[@]}"
      else
        echo "${ARCH_DENSE_CONFIGS_FULL[@]}"
      fi
      ;;
    qwen3moe | mixtral | deepseek2)
      if [[ "$quick" == "true" ]]; then
        echo "${ARCH_MOE_CONFIGS_QUICK[@]}"
      else
        echo "${ARCH_MOE_CONFIGS_FULL[@]}"
      fi
      ;;
    qwen3next)
      if [[ "$quick" == "true" ]]; then
        echo "${ARCH_SSM_CONFIGS_QUICK[@]}"
      else
        echo "${ARCH_SSM_CONFIGS_FULL[@]}"
      fi
      ;;
    special | *)
      echo "${ARCH_SPECIAL_CONFIGS[@]}"
      ;;
  esac
}

# Count total iterations for progress estimation
# Usage: count_total_iterations [quick]
count_total_iterations() {
  local quick="${1:-false}"
  local total=0

  for suite in "${SUITES[@]}"; do
    local models_str
    models_str=$(get_suite_models "$suite")
    [[ -z "$models_str" ]] && continue

    # Parse models
    while IFS='|' read -r name path arch; do
      [[ -z "$name" ]] && continue

      local configs_str
      configs_str=$(get_arch_configs "$arch" "$quick")
      local config_count
      config_count=$(echo "$configs_str" | wc -w)

      # Each model × configs × questions
      total=$((total + config_count * QUESTIONS_PER_RUBRIC))
    done <<<"$models_str"
  done

  echo "$total"
}

# Count models across all suites (unique)
count_total_models() {
  local models=""
  for suite in "${SUITES[@]}"; do
    local models_str
    models_str=$(get_suite_models "$suite")
    [[ -z "$models_str" ]] && continue

    while IFS='|' read -r name path arch; do
      [[ -z "$name" ]] && continue
      models="$models $name"
    done <<<"$models_str"
  done

  echo "$models" | tr ' ' '\n' | sort -u | grep -c . || echo 0
}
