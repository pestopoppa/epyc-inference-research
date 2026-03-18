#!/bin/bash
# Tree Speculation Server Benchmark
#
# Benchmarks tree speculation in llama-server against linear speculation baseline.
# Tests all 4 target+drafter pairs from specexec Phase 3 profiling for direct comparison.
#
# Sweep: p_split in {0, 0.05, 0.1, 0.2, 0.3}
#   p_split=0 → linear speculation (baseline, tree code not entered)
#   p_split>0 → DySpec tree speculation
#
# Measures: tokens/sec, acceptance rate, tokens per speculation round
#
# Usage: ./bench_tree_speculation_server.sh [--pairs 1,2] [--n-predict 256] [--draft-max 16]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Binary
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

# Output
DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/tree_speculation"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/server_sweep_${TIMESTAMP}.csv"
LOG_DIR_RUN="${DATA_DIR}/logs_${TIMESTAMP}"

# Parameters (overridable via env)
N_PREDICT="${N_PREDICT:-256}"
DRAFT_MAX="${DRAFT_MAX:-16}"
THREADS="${THREADS:-96}"
PORT="${PORT:-8199}"
WARMUP_TOKENS="${WARMUP_TOKENS:-32}"

# p_split values to sweep
P_SPLIT_VALUES=(0 0.05 0.1 0.2 0.3)

# Test prompts — diverse tasks for robust measurement
PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Describe the process of photosynthesis at the molecular level, including the light reactions and Calvin cycle:"
)

# Model pairs (same as specexec Phase 3 profiling)
declare -A PAIR_NAMES
declare -A PAIR_TARGETS
declare -A PAIR_DRAFTS
declare -A PAIR_EXTRA_ARGS  # per-pair extra server flags (e.g., --kv-unified for dense models)

PAIR_NAMES[1]="Qwen2.5-7B-f16+0.5B-f16"
PAIR_TARGETS[1]="/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf"
PAIR_DRAFTS[1]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
PAIR_EXTRA_ARGS[1]="--kv-unified"  # dense: enable multi-path target verification

PAIR_NAMES[2]="Qwen2.5-Coder-32B-Q4KM+0.5B-Q8"
PAIR_TARGETS[2]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
PAIR_DRAFTS[2]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"
PAIR_EXTRA_ARGS[2]="--kv-unified"  # dense

PAIR_NAMES[3]="Qwen3.5-27B-Q4KM+0.8B-Q8"
PAIR_TARGETS[3]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
PAIR_DRAFTS[3]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
PAIR_EXTRA_ARGS[3]=""  # hybrid SSM — no kv-unified

PAIR_NAMES[4]="Qwen3.5-9B-Q4KM+0.8B-Q8"
PAIR_TARGETS[4]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
PAIR_DRAFTS[4]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
PAIR_EXTRA_ARGS[4]=""  # hybrid SSM

PAIR_NAMES[5]="Qwen2.5-Coder-32B-Q4KM+0.5B-f16"
PAIR_TARGETS[5]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
PAIR_DRAFTS[5]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
PAIR_EXTRA_ARGS[5]="--kv-unified"  # dense

PAIR_NAMES[6]="Qwen3.5-122B-A10B-Q4KM+0.8B-Q8"
PAIR_TARGETS[6]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
PAIR_DRAFTS[6]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
PAIR_EXTRA_ARGS[6]=""  # hybrid SSM

PAIR_NAMES[7]="DS-R1-Distill-Qwen-32B-Q6K+0.5B-f16"
PAIR_TARGETS[7]="/mnt/raid0/llm/lmstudio/models/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q6_K.gguf"
PAIR_DRAFTS[7]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
PAIR_EXTRA_ARGS[7]="--kv-unified"  # dense (Qwen2.5 arch)

PAIR_NAMES[8]="Qwen3-235B-A22B-Q4KM+0.6B-Q8"
PAIR_TARGETS[8]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf"
PAIR_DRAFTS[8]="/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf"
PAIR_EXTRA_ARGS[8]="--kv-unified"  # MoE (pure attention, no SSM)

PAIR_NAMES[9]="Qwen3-Coder-480B-A35B-Q4KM+0.75B-Q4"
PAIR_TARGETS[9]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
PAIR_DRAFTS[9]="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
PAIR_EXTRA_ARGS[9]="--kv-unified"  # MoE (pure attention, no SSM)

PAIR_NAMES[10]="Qwen2.5-Coder-32B-f16+0.5B-f16"
PAIR_TARGETS[10]="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf"
PAIR_DRAFTS[10]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
PAIR_EXTRA_ARGS[10]="--kv-unified"  # dense f16 — Phase 4 validation

PAIR_NAMES[11]="Qwen2.5-Coder-32B-Q8_0+0.5B-f16"
PAIR_TARGETS[11]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
PAIR_DRAFTS[11]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
PAIR_EXTRA_ARGS[11]="--kv-unified"  # dense Q8 — verification scaling test

PAIR_NAMES[12]="Qwen3-Next-80B-A3B-Q4KM+Qwen3.5-0.8B-Q8"
PAIR_TARGETS[12]="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"
PAIR_DRAFTS[12]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
PAIR_EXTRA_ARGS[12]=""  # hybrid SSM+MoE — Phase 8 frozen multi-path

PAIR_NAMES[13]="Qwen3.5-35B-A3B-Q8_0+0.8B-Q8"
PAIR_TARGETS[13]="/mnt/raid0/llm/lmstudio/models/jiaojjjjje/Qwen3.5-35B-A3B-abliterated-GGUF/Qwen3.5-35B-A3B-abliterated-Q8_0.gguf"
PAIR_DRAFTS[13]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
PAIR_EXTRA_ARGS[13]=""  # hybrid SSM — Phase 8 frozen multi-path

PAIR_NAMES[14]="Qwen3.5-35B-A3B-Q4KM+0.8B-Q8"
PAIR_TARGETS[14]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
PAIR_DRAFTS[14]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
PAIR_EXTRA_ARGS[14]=""  # hybrid SSM — Phase 8 frozen multi-path

PAIR_NAMES[15]="Qwen3-Coder-30B-A3B-Q4KM+0.75B-Q4"
PAIR_TARGETS[15]="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
PAIR_DRAFTS[15]="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
PAIR_EXTRA_ARGS[15]="--kv-unified"  # MoE (pure attention, no SSM) — frontdoor production model

# Parse arguments
SELECTED_PAIRS="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pairs) SELECTED_PAIRS="$2"; shift 2 ;;
        --n-predict) N_PREDICT="$2"; shift 2 ;;
        --draft-max) DRAFT_MAX="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra PAIRS_TO_RUN <<< "$SELECTED_PAIRS"

mkdir -p "$DATA_DIR" "$LOG_DIR_RUN"

echo "Tree Speculation Server Benchmark"
echo "================================="
echo "n_predict=$N_PREDICT  draft_max=$DRAFT_MAX  threads=$THREADS"
echo "p_split values: ${P_SPLIT_VALUES[*]}"
echo "pairs: ${SELECTED_PAIRS}"
echo "results: $RESULTS_FILE"
echo ""

# CSV header
echo "pair,model_pair,p_split,prompt_idx,tokens_generated,time_ms,tokens_per_sec,draft_accepted,draft_total,acceptance_rate" > "$RESULTS_FILE"

# Function: wait for server health
wait_for_server() {
    local port=$1
    local max_wait="${MAX_SERVER_WAIT:-600}"
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" | grep -q '"status":"ok"' 2>/dev/null; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: server did not start within ${max_wait}s"
            return 1
        fi
    done
}

# Function: send completion request and parse metrics
run_completion() {
    local port=$1
    local prompt="$2"
    local n_predict=$3

    local response
    response=$(curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],
            \"max_tokens\": ${n_predict},
            \"temperature\": 0.0,
            \"stream\": false
        }" 2>/dev/null)

    echo "$response"
}

# Function: get server metrics
get_metrics() {
    local port=$1
    curl -s "http://localhost:${port}/metrics" 2>/dev/null
}

# Function: get slots info for speculation stats
get_slots() {
    local port=$1
    curl -s "http://localhost:${port}/slots" 2>/dev/null
}

# Main benchmark loop
for pair_id in "${PAIRS_TO_RUN[@]}"; do
    pair_name="${PAIR_NAMES[$pair_id]}"
    target="${PAIR_TARGETS[$pair_id]}"
    draft="${PAIR_DRAFTS[$pair_id]}"

    echo "=== Pair $pair_id: $pair_name ==="

    # Verify files exist
    if [ ! -f "$target" ]; then
        echo "  SKIP: target model not found: $target"
        continue
    fi
    if [ ! -f "$draft" ]; then
        echo "  SKIP: draft model not found: $draft"
        continue
    fi

    for p_split in "${P_SPLIT_VALUES[@]}"; do
        label="p_split=$p_split"
        if [ "$p_split" = "0" ]; then
            label="linear (baseline)"
        fi
        echo -n "  $label: "

        SERVER_LOG="${LOG_DIR_RUN}/pair${pair_id}_psplit${p_split}.log"

        # Build server args
        extra_args="${PAIR_EXTRA_ARGS[$pair_id]:-}"
        SERVER_ARGS=(
            -m "$target"
            -md "$draft"
            --draft-max "$DRAFT_MAX"
            --draft-p-split "$p_split"
            -t "$THREADS"
            -np 1
            --port "$PORT"
            -ngl 0
            --metrics
            --slots
            $extra_args
        )

        # Launch server (skip numactl if NUMA not supported)
        if numactl --show &>/dev/null; then
            numactl --interleave=all "$LLAMA_SERVER" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
        else
            "$LLAMA_SERVER" "${SERVER_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
        fi
        SERVER_PID=$!

        if ! wait_for_server "$PORT"; then
            echo "FAILED (server start)"
            kill "$SERVER_PID" 2>/dev/null || true
            wait "$SERVER_PID" 2>/dev/null || true
            continue
        fi

        # Warmup: send a short request to prime KV cache and draft model
        run_completion "$PORT" "Hello" "$WARMUP_TOKENS" > /dev/null 2>&1

        # Reset metrics after warmup
        # (metrics accumulate, so we read before and after)

        total_tokens=0
        total_time_ms=0
        total_draft_accepted=0
        total_draft_total=0

        for prompt_idx in "${!PROMPTS[@]}"; do
            prompt="${PROMPTS[$prompt_idx]}"

            start_ms=$(date +%s%N | cut -b1-13)
            response=$(run_completion "$PORT" "$prompt" "$N_PREDICT")
            end_ms=$(date +%s%N | cut -b1-13)

            elapsed_ms=$((end_ms - start_ms))

            # Parse completion tokens and draft stats from response timings
            parsed=$(echo "$response" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    comp = r.get('usage', {}).get('completion_tokens', 0)
    t = r.get('timings', {})
    da = t.get('draft_n_accepted', 0)
    dt = t.get('draft_n', 0)
    print(f'{comp},{da},{dt}')
except:
    print('0,0,0')
" 2>/dev/null)

            comp_tokens=$(echo "$parsed" | cut -d, -f1)
            draft_accepted=$(echo "$parsed" | cut -d, -f2)
            draft_total=$(echo "$parsed" | cut -d, -f3)

            if [ "$comp_tokens" = "0" ] || [ -z "$comp_tokens" ]; then
                comp_tokens=0
            fi
            if [ "$elapsed_ms" -gt 0 ] && [ "$comp_tokens" -gt 0 ]; then
                tps=$(echo "scale=2; $comp_tokens * 1000 / $elapsed_ms" | bc)
            else
                tps="0"
            fi

            if [ "$draft_total" -gt 0 ]; then
                accept_rate=$(echo "scale=4; $draft_accepted / $draft_total" | bc)
            else
                accept_rate="0"
            fi

            total_tokens=$((total_tokens + comp_tokens))
            total_time_ms=$((total_time_ms + elapsed_ms))
            total_draft_accepted=$((total_draft_accepted + draft_accepted))
            total_draft_total=$((total_draft_total + draft_total))

            echo "$pair_id,$pair_name,$p_split,$prompt_idx,$comp_tokens,$elapsed_ms,$tps,$draft_accepted,$draft_total,$accept_rate" >> "$RESULTS_FILE"
        done

        # Summary for this config
        if [ "$total_time_ms" -gt 0 ] && [ "$total_tokens" -gt 0 ]; then
            avg_tps=$(echo "scale=2; $total_tokens * 1000 / $total_time_ms" | bc)
        else
            avg_tps="0"
        fi
        if [ "$total_draft_total" -gt 0 ]; then
            avg_accept=$(echo "scale=4; $total_draft_accepted / $total_draft_total" | bc)
        else
            avg_accept="0"
        fi
        echo "${avg_tps} t/s  accept=${avg_accept} (${total_draft_accepted}/${total_draft_total})  ${total_tokens} tokens"

        # Stop server
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        sleep 1
    done
    echo ""
done

echo "Results saved to: $RESULTS_FILE"
echo "Server logs: $LOG_DIR_RUN/"
echo ""
echo "=== Summary (averages per config) ==="
python3 -c "
import csv, sys
from collections import defaultdict

data = defaultdict(lambda: {'tokens': 0, 'time_ms': 0, 'accepted': 0, 'total': 0, 'n': 0})
with open('${RESULTS_FILE}') as f:
    for row in csv.DictReader(f):
        key = (row['model_pair'], row['p_split'])
        data[key]['tokens'] += int(row['tokens_generated'])
        data[key]['time_ms'] += int(row['time_ms'])
        data[key]['accepted'] += int(row['draft_accepted'])
        data[key]['total'] += int(row['draft_total'])
        data[key]['n'] += 1

print(f'{'Model Pair':<35} {'p_split':>7} {'t/s':>7} {'Accept%':>8} {'Acc/Tot':>12}')
print('-' * 75)
for (pair, ps), v in sorted(data.items()):
    tps = v['tokens'] * 1000 / v['time_ms'] if v['time_ms'] > 0 else 0
    acc = v['accepted'] / v['total'] * 100 if v['total'] > 0 else 0
    print(f'{pair:<35} {ps:>7} {tps:>7.2f} {acc:>7.1f}% {v[\"accepted\"]:>5}/{v[\"total\"]:>5}')
" 2>/dev/null || echo "(summary requires python3)"
