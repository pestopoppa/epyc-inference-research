#!/bin/bash
set -euo pipefail

# SpecExec Verification Profiling — Phase 1 + Phase 2
#
# Phase 1: Batch verification latency curve for target models
#   - llama-bench -p <batch_sizes> -n 0 across 5 target models
#   - Two NUMA modes: distribute and isolate
#   - 3 repetitions per config
#
# Phase 2: Draft model per-token generation cost
#   - llama-bench -p 0 -n 128 for 9 draft models
#
# Output: CSV files in data/specexec/

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/specexec"
LLAMA_BENCH="/mnt/raid0/llm/llama.cpp/build/bin/llama-bench"
BASE_PATH="/mnt/raid0/llm/lmstudio/models"
THREADS=96
REPS=3

mkdir -p "$DATA_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

drop_caches() {
    log "Dropping page caches..."
    sync
    echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 || log "WARN: Could not drop caches (not root?)"
    sleep 2
}

# ── Phase 1: Target model verification latency curves ──

declare -A TARGET_MODELS=(
    ["Qwen3.5-27B-Q4_K_M"]="$BASE_PATH/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
    ["Qwen2.5-Coder-32B-Q4_K_M"]="$BASE_PATH/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
    ["Qwen3.5-9B-Q4_K_M"]="$BASE_PATH/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
    ["Qwen2.5-7B-f16"]="/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf"
    ["Qwen3.5-0.8B-Q8_0"]="$BASE_PATH/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
)

BATCH_SIZES="1,2,4,8,16,32,64,128,256,512"
NUMA_MODES=("distribute" "isolate")

run_phase1() {
    log "=== PHASE 1: Batch Verification Latency Curves ==="
    for model_name in "${!TARGET_MODELS[@]}"; do
        model_path="${TARGET_MODELS[$model_name]}"
        if [[ ! -f "$model_path" ]]; then
            log "SKIP: $model_name — file not found: $model_path"
            continue
        fi
        for numa in "${NUMA_MODES[@]}"; do
            outfile="$DATA_DIR/phase1_${model_name}_${numa}.csv"
            if [[ -f "$outfile" ]]; then
                log "EXISTS: $outfile — skipping"
                continue
            fi
            drop_caches
            log "Running: $model_name (numa=$numa)"
            "$LLAMA_BENCH" \
                -m "$model_path" \
                -p "$BATCH_SIZES" \
                -n 0 \
                --numa "$numa" \
                -t "$THREADS" \
                -r "$REPS" \
                -o csv > "$outfile" 2>"$DATA_DIR/phase1_${model_name}_${numa}.log"
            log "Done: $outfile ($(wc -l < "$outfile") lines)"
        done
    done
    log "=== PHASE 1 COMPLETE ==="
}

# ── Phase 2: Draft model per-token generation cost ──

declare -A DRAFT_MODELS=(
    ["Qwen3.5-0.8B-Q4_0"]="$BASE_PATH/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q4_0.gguf"
    ["Qwen3.5-0.8B-Q8_0"]="$BASE_PATH/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    ["Qwen2.5-0.5B-Instruct-f16"]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
    ["Qwen2.5-Coder-0.5B-Q8_0"]="$BASE_PATH/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"
    ["Qwen3-Coder-0.75B-Q4_0"]="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
    ["Qwen3-0.6B-Q8_0"]="/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf"
    ["Llama-3.2-1B-Instruct-f16"]="/mnt/raid0/llm/models/Llama-3.2-1B-Instruct-f16.gguf"
    ["Gemma-3-1B-IT-Q8_0"]="/mnt/raid0/llm/models/gemma-3-1b-it-Q8_0.gguf"
    ["DeepSeek-R1-Distill-Qwen-1.5B-Q8_0"]="$BASE_PATH/lmstudio-community/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/DeepSeek-R1-Distill-Qwen-1.5B-Q8_0.gguf"
)

GEN_TOKENS=128

run_phase2() {
    log "=== PHASE 2: Draft Model Per-Token Generation Cost ==="
    outfile="$DATA_DIR/phase2_draft_costs.csv"
    header_written=false

    for model_name in "${!DRAFT_MODELS[@]}"; do
        model_path="${DRAFT_MODELS[$model_name]}"
        if [[ ! -f "$model_path" ]]; then
            log "SKIP: $model_name — file not found: $model_path"
            continue
        fi
        drop_caches
        log "Running: $model_name (generation benchmark, n=$GEN_TOKENS)"
        tmpfile=$(mktemp)
        "$LLAMA_BENCH" \
            -m "$model_path" \
            -p 0 \
            -n "$GEN_TOKENS" \
            --numa distribute \
            -t "$THREADS" \
            -r "$REPS" \
            -o csv > "$tmpfile" 2>"$DATA_DIR/phase2_${model_name}.log"

        if [[ "$header_written" = false ]]; then
            cat "$tmpfile" >> "$outfile"
            header_written=true
        else
            tail -n +2 "$tmpfile" >> "$outfile"
        fi
        rm -f "$tmpfile"
        log "Done: $model_name"
    done
    log "=== PHASE 2 COMPLETE: $outfile ==="
}

# ── Main ──

case "${1:-all}" in
    phase1) run_phase1 ;;
    phase2) run_phase2 ;;
    all)
        run_phase1
        run_phase2
        ;;
    *)
        echo "Usage: $0 [phase1|phase2|all]"
        exit 1
        ;;
esac

log "All done. Results in $DATA_DIR/"
