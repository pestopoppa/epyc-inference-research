#!/bin/bash
# =============================================================================
# REGISTRY READER LIBRARY
# =============================================================================
# Provides functions for benchmark scripts to read model configurations
# from the authoritative model registry.
#
# Usage: source lib/registry_reader.sh
# =============================================================================

REGISTRY_FILE="${REGISTRY_FILE:-/mnt/raid0/llm/epyc-inference-research/orchestration/model_registry.yaml}"
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"

# =============================================================================
# REGISTRY ACCESS FUNCTIONS
# =============================================================================

# Check if yq is available, fallback to grep-based parsing
_has_yq() {
  command -v yq &>/dev/null
}

# Get a value from the registry using yq or grep fallback
# Usage: registry_get "models.Qwen2.5-Math-1.5B.path"
registry_get() {
  local key="$1"
  if _has_yq; then
    yq -r ".$key // empty" "$REGISTRY_FILE" 2>/dev/null
  else
    # Fallback: basic grep for simple keys (limited)
    echo "WARNING: yq not available, registry access limited" >&2
    return 1
  fi
}

# Get model path from registry
# Usage: get_model_path "Qwen2.5-Math-1.5B-Instruct"
get_model_path() {
  local model_name="$1"
  registry_get "models.${model_name}.path"
}

# Get model architecture from registry
# Usage: get_model_arch "Qwen2.5-Math-1.5B-Instruct"
get_model_arch() {
  local model_name="$1"
  registry_get "models.${model_name}.arch"
}

# Get launch flags for a model
# Usage: get_launch_flags "Qwen2.5-Math-1.5B-Instruct"
get_launch_flags() {
  local model_name="$1"
  local flags
  flags=$(registry_get "models.${model_name}.launch_config.flags[]")
  echo "$flags" | tr '\n' ' '
}

# Get MoE override key for a model
# Usage: get_moe_override_key "Qwen3-30B-A3B-Thinking"
get_moe_override_key() {
  local model_name="$1"
  registry_get "models.${model_name}.moe_override_key"
}

# Get compatible draft models
# Usage: get_compatible_drafts "Qwen2.5-Coder-32B"
get_compatible_drafts() {
  local model_name="$1"
  registry_get "models.${model_name}.speculative_decoding.compatible_drafts[]"
}

# Check if speculative decoding is viable
# Usage: is_spec_viable "Qwen2.5-Coder-32B"
is_spec_viable() {
  local model_name="$1"
  local viable
  viable=$(registry_get "models.${model_name}.speculative_decoding.viable")
  [[ "$viable" == "true" ]]
}

# =============================================================================
# MODEL DISCOVERY (for scripts that scan all models)
# =============================================================================

# List all registered model names
# Usage: list_registered_models
list_registered_models() {
  if _has_yq; then
    yq -r '.models | keys | .[]' "$REGISTRY_FILE" 2>/dev/null
  else
    echo "ERROR: yq required for listing models" >&2
    return 1
  fi
}

# Get all models with a specific architecture
# Usage: list_models_by_arch "qwen3moe"
list_models_by_arch() {
  local arch="$1"
  if _has_yq; then
    yq -r ".models | to_entries | .[] | select(.value.arch == \"$arch\") | .key" "$REGISTRY_FILE" 2>/dev/null
  else
    echo "ERROR: yq required for filtering models" >&2
    return 1
  fi
}

# =============================================================================
# BUILD COMMAND FROM REGISTRY
# =============================================================================

# Build the complete llama command for a model
# Usage: build_llama_command "Qwen2.5-Math-1.5B-Instruct" [extra_args]
# Returns: Complete command string ready to execute
build_llama_command() {
  local model_name="$1"
  shift
  local extra_args="$*"

  local path

  path=$(get_model_path "$model_name")
  local arch
  arch=$(get_model_arch "$model_name")
  local flags
  flags=$(get_launch_flags "$model_name")
  local moe_key
  moe_key=$(get_moe_override_key "$model_name")

  if [[ -z "$path" ]]; then
    echo "ERROR: Model '$model_name' not found in registry" >&2
    return 1
  fi

  local binary="/mnt/raid0/llm/llama.cpp/build/bin/llama-completion"
  local moe_override=""

  # Build MoE override if needed
  if [[ -n "$moe_key" ]] && [[ "$moe_key" != "null" ]]; then
    moe_override="--override-kv ${moe_key}=int:4"
  fi

  # Build command
  echo "env OMP_NUM_THREADS=1 numactl --interleave=all $binary -m $path $moe_override $flags $extra_args"
}

# Build speculative decoding command
# Usage: build_spec_command "Qwen2.5-Coder-32B" 16 [extra_args]
build_spec_command() {
  local model_name="$1"
  local k_value="${2:-16}"
  shift 2
  local extra_args="$*"

  local path

  path=$(get_model_path "$model_name")
  local drafts
  drafts=$(get_compatible_drafts "$model_name")
  local draft_path
  draft_path=$(echo "$drafts" | head -1)

  if [[ -z "$path" ]] || [[ -z "$draft_path" ]]; then
    echo "ERROR: Model or draft not found in registry" >&2
    return 1
  fi

  local binary="/mnt/raid0/llm/llama.cpp/build/bin/llama-speculative"

  echo "env OMP_NUM_THREADS=1 numactl --interleave=all $binary -m $path -md $draft_path --draft-max $k_value $extra_args"
}

# =============================================================================
# FALLBACK: Direct path resolution (when model not in registry yet)
# =============================================================================

# Resolve a partial path to full path
# Usage: resolve_model_path "tensorblock/Qwen2.5-Math-1.5B-Instruct-GGUF/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf"
resolve_model_path() {
  local partial="$1"

  # Already absolute?
  if [[ "$partial" == /* ]]; then
    echo "$partial"
    return
  fi

  # Prepend base
  local full="$MODEL_BASE/$partial"
  if [[ -f "$full" ]]; then
    echo "$full"
    return
  fi

  # Search for it
  local found
  found=$(find "$MODEL_BASE" -name "$(basename "$partial")" -type f 2>/dev/null | head -1)
  if [[ -n "$found" ]]; then
    echo "$found"
    return
  fi

  echo "ERROR: Could not resolve path: $partial" >&2
  return 1
}

# Detect architecture from model path/name (fallback when not in registry)
# Usage: detect_arch_from_name "Qwen3-30B-A3B-Thinking-2507-Q4_K_S.gguf"
detect_arch_from_name() {
  local name="$1"

  if [[ "$name" == *"Qwen3-"* ]] && [[ "$name" == *"-A"*"B-"* ]] && [[ "$name" != *"Next"* ]] && [[ "$name" != *"VL-"* ]]; then
    echo "qwen3moe"
  elif [[ "$name" == *"Qwen3-VL-"* ]] && [[ "$name" == *"-A"*"B-"* ]]; then
    echo "qwen3vlmoe"
  elif [[ "$name" == *"Qwen3-Next"* ]] || [[ "$name" == *"Qwen3Next"* ]]; then
    echo "qwen3next"
  elif [[ "$name" == *"Mixtral"* ]]; then
    echo "mixtral"
  elif [[ "$name" == *"DeepSeek"* ]] && [[ "$name" == *"MoE"* ]]; then
    echo "deepseek2"
  elif [[ "$name" == *"GLM"* ]] && [[ "$name" == *"-A"*"B"* ]]; then
    echo "glm4moe"
  else
    echo "dense"
  fi
}

# =============================================================================
# INITIALIZATION
# =============================================================================

# Verify registry exists
if [[ ! -f "$REGISTRY_FILE" ]]; then
  echo "WARNING: Registry file not found: $REGISTRY_FILE" >&2
  echo "WARNING: Benchmark scripts will use fallback path resolution" >&2
fi

# Check for yq
if ! _has_yq; then
  echo "WARNING: yq not installed. Install with: sudo apt install yq" >&2
  echo "WARNING: Some registry features will be limited" >&2
fi
