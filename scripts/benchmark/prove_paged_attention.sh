#!/bin/bash
#
# Paged Attention Comprehensive Benchmark Evidence Script
#
# Tests three models (small, medium, large) across three block sizes (64, 128, 256)
# Measures both SPEED (via llama-bench) and MEMORY (via llama-cli verbose output)
#
# Usage: ./prove_paged_attention.sh [output_dir] [--memory-only]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source environment library for path variables
# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Parse arguments
MEMORY_ONLY=false
OUTPUT_DIR=""

for arg in "$@"; do
  case $arg in
    --memory-only)
      MEMORY_ONLY=true
      ;;
    *)
      if [[ -z "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$arg"
      fi
      ;;
  esac
done

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/benchmarks/evidence/$(date +%Y%m%d_%H%M%S)}"
LLAMA_CPP_EXPERIMENTAL="${LLM_ROOT}/llama.cpp-experimental"
LLAMA_BENCH="${LLAMA_CPP_EXPERIMENTAL}/build/bin/llama-bench"
LLAMA_COMPLETION="${LLAMA_CPP_EXPERIMENTAL}/build/bin/llama-completion"
PAGED_MODEL_BASE="${LLM_ROOT}"

# Models from PR (paths from model registry)
SMALL_MODEL="${PAGED_MODEL_BASE}/models/Qwen3-1.7B-Q8_0.gguf"
MEDIUM_MODEL="${PAGED_MODEL_BASE}/models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf"
LARGE_MODEL="${PAGED_MODEL_BASE}/lmstudio/models/lmstudio-community/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-Q4_K_M.gguf"

# Block sizes to test
BLOCK_SIZES=(64 128 256)

# Benchmark parameters
NUM_RUNS=3
PROMPT_TOKENS=1024
GEN_TOKENS=64

# Memory test parameters
MEMORY_PROMPT="Write a detailed story about a robot learning to paint:"
MEMORY_GEN_TOKENS=128

# Colors for terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
  echo -e "[$(date '+%Y-%m-%d %H:%M:%S.%3N')] $1" | tee -a "${OUTPUT_DIR}/full_log.txt"
}

header() {
  echo "" | tee -a "${OUTPUT_DIR}/full_log.txt"
  echo "================================================================================" | tee -a "${OUTPUT_DIR}/full_log.txt"
  log "${GREEN}$1${NC}"
  echo "================================================================================" | tee -a "${OUTPUT_DIR}/full_log.txt"
}

subheader() {
  echo "" | tee -a "${OUTPUT_DIR}/full_log.txt"
  log "${CYAN}--- $1 ---${NC}"
}

# Create output directory
mkdir -p "${OUTPUT_DIR}"
log "Output directory: ${OUTPUT_DIR}"

header "PAGED ATTENTION COMPREHENSIVE BENCHMARK EVIDENCE"
log "Script started"
log "Purpose: Collect ALL evidence for PR #18747 reviewers"

header "1. SYSTEM INFORMATION"

log "Hostname: $(hostname)"
log "Kernel: $(uname -r)"
log "Date/Time: $(date -Iseconds)"

log ""
log "CPU Information:"
lscpu | tee -a "${OUTPUT_DIR}/full_log.txt"

log ""
log "Memory Information:"
free -h | tee -a "${OUTPUT_DIR}/full_log.txt"

log ""
log "NUMA Topology:"
numactl --hardware 2>/dev/null | tee -a "${OUTPUT_DIR}/full_log.txt" || log "numactl not available"

# Save system info
lscpu >"${OUTPUT_DIR}/system_cpu.txt"
free -h >"${OUTPUT_DIR}/system_memory.txt"
cat /proc/meminfo >"${OUTPUT_DIR}/proc_meminfo.txt"

header "2. SOFTWARE VERSIONS"

log "llama.cpp directory: ${LLAMA_CPP_EXPERIMENTAL}"
cd "${LLAMA_CPP_EXPERIMENTAL}"

log "Git commit: $(git log -1 --format='%H')"
log "Git branch: $(git branch --show-current)"
git log -1 --format="%H" >"${OUTPUT_DIR}/git_commit.txt"
# Save full git info to file only (not screen)
git log -1 --format="commit: %H%ndate: %ci%nsubject: %s" >>"${OUTPUT_DIR}/full_log.txt"
git status --short >>"${OUTPUT_DIR}/full_log.txt"

header "3. MODEL VERIFICATION"

verify_model() {
  local model_path="$1"
  local model_name="$2"

  subheader "Verifying ${model_name}"
  log "Path: ${model_path}"

  if [[ -f "${model_path}" ]]; then
    log "Exists: YES"
    log "Size: $(ls -lh "${model_path}" | awk '{print $5}')"
    log "Modified: $(stat -c '%y' "${model_path}")"

    if [[ "${MEMORY_ONLY}" == "true" ]]; then
      log "Skipping SHA256 (--memory-only)"
    else
      log "Computing SHA256..."
      sha256sum "${model_path}" | tee -a "${OUTPUT_DIR}/full_log.txt" "${OUTPUT_DIR}/${model_name}_checksum.txt"
    fi
    return 0
  else
    log "${RED}ERROR: Model not found${NC}"
    return 1
  fi
}

verify_model "${SMALL_MODEL}" "small_qwen3_1.7b" || exit 1
verify_model "${MEDIUM_MODEL}" "medium_deepseek_32b" || exit 1
verify_model "${LARGE_MODEL}" "large_llama_70b" || exit 1

if [[ "${MEMORY_ONLY}" == "false" ]]; then

  header "4. SPEED BENCHMARKS (llama-bench)"

  run_speed_benchmark() {
    local model_path="$1"
    local model_name="$2"
    local threads="$3"
    local use_numactl="$4"

    subheader "Speed: ${model_name}"

    local prefix=""
    if [[ "${use_numactl}" == "yes" ]]; then
      prefix="numactl --interleave=all"
    fi

    # Baseline
    log "Running BASELINE (no paging)..."
    log "Command: ${prefix} ${LLAMA_BENCH} -m ${model_path} -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -t ${threads} -r ${NUM_RUNS}"

    ${prefix} ${LLAMA_BENCH} \
      -m "${model_path}" \
      -p ${PROMPT_TOKENS} \
      -n ${GEN_TOKENS} \
      -t ${threads} \
      -r ${NUM_RUNS} \
      2>&1 | tee -a "${OUTPUT_DIR}/full_log.txt" "${OUTPUT_DIR}/${model_name}_baseline_speed.txt"

    # Test each block size
    for block_size in "${BLOCK_SIZES[@]}"; do
      log ""
      log "Running PAGED (block_size=${block_size})..."
      log "Command: LLAMA_PAGED_ATTN=${block_size} ${prefix} ${LLAMA_BENCH} -m ${model_path} -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -t ${threads} -r ${NUM_RUNS}"

      LLAMA_PAGED_ATTN=${block_size} ${prefix} ${LLAMA_BENCH} \
        -m "${model_path}" \
        -p ${PROMPT_TOKENS} \
        -n ${GEN_TOKENS} \
        -t ${threads} \
        -r ${NUM_RUNS} \
        2>&1 | tee -a "${OUTPUT_DIR}/full_log.txt" "${OUTPUT_DIR}/${model_name}_paged${block_size}_speed.txt"
    done
  }

  # Small model: 16 threads, no numactl
  run_speed_benchmark "${SMALL_MODEL}" "small_qwen3_1.7b" 16 "no"

  # Cool down
  log "Cooling down for 5 seconds..."
  sleep 5

  # Medium model: 96 threads, no numactl
  run_speed_benchmark "${MEDIUM_MODEL}" "medium_deepseek_32b" 96 "no"

  # Cool down
  log "Cooling down for 10 seconds..."
  sleep 10

  # Large model: 96 threads, with numactl
  run_speed_benchmark "${LARGE_MODEL}" "large_llama_70b" 96 "yes"

else
  log "Skipping speed benchmarks (--memory-only flag set)"
fi

header "5. MEMORY BENCHMARKS (llama-cli verbose)"

run_memory_benchmark() {
  local model_path="$1"
  local model_name="$2"
  local threads="$3"
  local use_numactl="$4"

  subheader "Memory: ${model_name}"

  local prefix=""
  if [[ "${use_numactl}" == "yes" ]]; then
    prefix="numactl --interleave=all"
  fi

  local temp_output="${OUTPUT_DIR}/${model_name}_temp.txt"

  # Baseline memory - capture full output, then filter
  log "Running BASELINE memory test..."
  log "Capturing KV cache size from verbose output..."

  # Run with short generation, capture ALL output first
  # NOTE: Redirection order matters! > file 2>&1 captures both stdout AND stderr
  # IMPORTANT: Use llama-completion with -no-cnv to avoid interactive mode hang
  # IMPORTANT: Redirect stdin from /dev/null to prevent any TTY interaction
  timeout 60 ${prefix} ${LLAMA_COMPLETION} \
    -m "${model_path}" \
    -p "${MEMORY_PROMPT}" \
    -n 16 \
    -t ${threads} \
    -no-cnv \
    </dev/null >"${temp_output}" 2>&1 || true

  # Extract memory-related lines (include paged/block info)
  grep -iE "KV|memory|MiB|GiB|cache|buffer|host|paged|enable_block|context" "${temp_output}" >"${OUTPUT_DIR}/${model_name}_baseline_memory.txt" 2>/dev/null || true
  cat "${OUTPUT_DIR}/${model_name}_baseline_memory.txt" >>"${OUTPUT_DIR}/full_log.txt"

  # Show what we captured with key metric
  local baseline_kv
  baseline_kv=$(grep "CPU KV buffer size" "${OUTPUT_DIR}/${model_name}_baseline_memory.txt" 2>/dev/null | head -1 | sed 's/.*= *//')
  log "Captured $(wc -l <"${OUTPUT_DIR}/${model_name}_baseline_memory.txt" 2>/dev/null || echo 0) lines - KV buffer: ${baseline_kv:-?}"

  # Test each block size
  for block_size in "${BLOCK_SIZES[@]}"; do
    log ""
    log "Running PAGED memory test (block_size=${block_size}, max_blocks=100)..."

    LLAMA_PAGED_ATTN=${block_size} LLAMA_PAGED_ATTN_MAX_BLOCKS=100 timeout 60 ${prefix} ${LLAMA_COMPLETION} \
      -m "${model_path}" \
      -p "${MEMORY_PROMPT}" \
      -n 16 \
      -t ${threads} \
      -no-cnv \
      </dev/null >"${temp_output}" 2>&1 || true

    # Extract memory-related lines (include paged/block info)
    grep -iE "KV|memory|MiB|GiB|cache|buffer|host|paged|enable_block|context" "${temp_output}" >"${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" 2>/dev/null || true
    cat "${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" >>"${OUTPUT_DIR}/full_log.txt"

    # Show what we captured with key metrics
    local paged_kv
    paged_kv=$(grep "CPU KV buffer size" "${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" 2>/dev/null | head -1 | sed 's/.*= *//')
    local savings
    savings=$(grep "memory savings" "${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" 2>/dev/null | head -1 | grep -oE '[0-9.]+%')
    log "Captured $(wc -l <"${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" 2>/dev/null || echo 0) lines - KV buffer: ${paged_kv:-?} (${savings:-?} savings)"
  done

  # Clean up temp file
  rm -f "${temp_output}"
}

# Memory tests (smaller token count to complete faster)
log "Note: Memory tests use shorter generation to complete faster"

run_memory_benchmark "${SMALL_MODEL}" "small_qwen3_1.7b" 16 "no"
run_memory_benchmark "${MEDIUM_MODEL}" "medium_deepseek_32b" 96 "no"
run_memory_benchmark "${LARGE_MODEL}" "large_llama_70b" 96 "yes"

header "6. RESULTS SUMMARY"

log "Benchmark completed!"
log ""
log "Output files generated:"
ls -la "${OUTPUT_DIR}/" | tee -a "${OUTPUT_DIR}/full_log.txt"

if [[ "${MEMORY_ONLY}" == "false" ]]; then
  log ""
  log "=== SPEED RESULTS SUMMARY ==="

  for model_name in "small_qwen3_1.7b" "medium_deepseek_32b" "large_llama_70b"; do
    log ""
    log "${model_name}:"

    # Extract tg (generation) speed from each file
    # llama-bench table columns: |model|size|params|backend|threads|test|t/s|
    # With awk -F'|': $1=empty, $2=model, ..., $7=test, $8=t/s
    if [[ -f "${OUTPUT_DIR}/${model_name}_baseline_speed.txt" ]]; then
      baseline_tg=$(grep -E "^\|.*tg${GEN_TOKENS}" "${OUTPUT_DIR}/${model_name}_baseline_speed.txt" | tail -1 | awk -F'|' '{print $8}' | xargs)
      log "  Baseline: ${baseline_tg} t/s"
    fi

    for block_size in "${BLOCK_SIZES[@]}"; do
      if [[ -f "${OUTPUT_DIR}/${model_name}_paged${block_size}_speed.txt" ]]; then
        paged_tg=$(grep -E "^\|.*tg${GEN_TOKENS}" "${OUTPUT_DIR}/${model_name}_paged${block_size}_speed.txt" | tail -1 | awk -F'|' '{print $8}' | xargs)
        log "  Paged-${block_size}: ${paged_tg} t/s"
      fi
    done
  done
fi

log ""
log "=== MEMORY RESULTS SUMMARY ==="

for model_name in "small_qwen3_1.7b" "medium_deepseek_32b" "large_llama_70b"; do
  log ""
  log "${model_name}:"
  # Extract and display KV buffer sizes
  if [[ -f "${OUTPUT_DIR}/${model_name}_baseline_memory.txt" ]]; then
    baseline_kv=$(grep "CPU KV buffer size" "${OUTPUT_DIR}/${model_name}_baseline_memory.txt" | head -1 | sed 's/.*= *//')
    log "  Baseline KV: ${baseline_kv:-not found}"
  fi
  for block_size in "${BLOCK_SIZES[@]}"; do
    if [[ -f "${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" ]]; then
      paged_kv=$(grep "CPU KV buffer size" "${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" | head -1 | sed 's/.*= *//')
      savings=$(grep "memory savings" "${OUTPUT_DIR}/${model_name}_paged${block_size}_memory.txt" | head -1 | grep -oE '[0-9.]+%')
      log "  Paged-${block_size}: ${paged_kv:-not found} (${savings:-?} savings)"
    fi
  done
done

header "7. REPRODUCTION COMMANDS"

log "To reproduce these results:"
log ""
log "1. Clone and build llama.cpp with paged attention:"
log "   git clone https://github.com/ggml-org/llama.cpp"
log "   cd llama.cpp"
log "   git checkout $(cat ${OUTPUT_DIR}/git_commit.txt)"
log "   cmake -B build -DGGML_NATIVE=ON -DGGML_AVX512=ON"
log "   cmake --build build -j"
log ""
log "2. Run speed benchmarks:"
log "   # Baseline"
log "   ./build/bin/llama-bench -m MODEL.gguf -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -r ${NUM_RUNS}"
log "   # Paged"
log "   LLAMA_PAGED_ATTN=64 ./build/bin/llama-bench -m MODEL.gguf -p ${PROMPT_TOKENS} -n ${GEN_TOKENS} -r ${NUM_RUNS}"
log ""
log "3. Run memory benchmarks:"
log "   # Baseline"
log "   ./build/bin/llama-cli -m MODEL.gguf -p 'prompt' -n 128 2>&1 | grep -i 'kv\\|memory\\|cache'"
log "   # Paged with memory savings"
log "   LLAMA_PAGED_ATTN=64 LLAMA_PAGED_ATTN_MAX_BLOCKS=100 ./build/bin/llama-cli -m MODEL.gguf -p 'prompt' -n 128 2>&1 | grep -i 'kv\\|memory\\|cache'"

header "BENCHMARK EVIDENCE COLLECTION COMPLETE"

log "All evidence saved to: ${OUTPUT_DIR}"
log ""
log "Key files for PR response:"
log "  - full_log.txt: Complete timestamped log"
log "  - *_baseline_speed.txt: Raw baseline llama-bench output"
log "  - *_paged*_speed.txt: Raw paged llama-bench output for each block size"
log "  - *_baseline_memory.txt: Baseline KV cache info"
log "  - *_paged_memory.txt: Paged KV cache info (with memory savings)"
log "  - git_commit.txt: Exact commit hash for reproduction"
log ""
log "Script finished at $(date -Iseconds)"
