#!/bin/bash
# Phase 0: KV Cache Quantization Benchmark
#
# Tests existing llama.cpp quantized KV cache support on EPYC 9655.
# No code changes — just CLI flags on existing builds.
#
# Matrix:
#   4 KV configs: f16 (baseline), q8_0/q8_0, q8_0/q4_0, q4_0/q4_0
#   2 models:
#     - Qwen3.5-35B-A3B Q4_K_M (frontdoor, hybrid — 25% attention layers)
#     - Qwen2.5-Coder-32B Q4_K_M (coder, pure attention — max KV impact)
#   3 context lengths: 4096, 16384, 65536
#   Metrics: generation t/s, prompt processing t/s, RSS memory, KV cache size
#
# Usage:
#   ./bench_kv_cache_quant.sh              # Run full matrix
#   ./bench_kv_cache_quant.sh --model q35  # Only Qwen3.5
#   ./bench_kv_cache_quant.sh --model coder # Only Coder
#   ./bench_kv_cache_quant.sh --ctx 4096   # Only 4K context

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

# Models
Q35_MODEL="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
CODER_MODEL="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/kv_cache_quant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/kv_quant_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"
PORT=8190
THREADS=96
# Single-model optimal: numactl --interleave=all + 96t + ub8192
# (192t without NUMA interleave causes cross-NUMA contention — verified A/B 2026-03-19)

# Generation test: 256 tokens output
N_PREDICT=256

# KV cache configurations: label, ctk, ctv
KV_CONFIGS=(
    "f16_f16:f16:f16"
    "q8_q8:q8_0:q8_0"
    "q8_q4:q8_0:q4_0"
    "q4_q4:q4_0:q4_0"
)

# Context lengths to test
CONTEXT_LENGTHS=(4096 16384 65536)

# Prompts: short (speed-focused) and long (context-filling)
SHORT_PROMPT="Write a Python function that implements a concurrent task queue with priority scheduling, worker pools, and graceful shutdown. Include comprehensive error handling, logging, and type hints."

# For context-filling tests, we generate filler text
generate_long_prompt() {
    local target_tokens=$1
    python3 -c "
import sys
# Each 'word' is ~1.3 tokens on average; aim for target
base = '''You are a code reviewer. Below is a large codebase that needs review. Analyze it for bugs, security issues, and performance problems.

'''
# Repeat filler to approximate target token count (1 token ~ 4 chars)
filler = 'The quick brown fox jumps over the lazy dog and finds that the algorithm has O(n^2) complexity which should be optimized to O(n log n) using a divide-and-conquer approach. '
target_chars = int($target_tokens * 3.5)
text = base + (filler * (target_chars // len(filler)))
text = text[:target_chars]
text += '\n\nSummarize the key issues found in the above text in 3 bullet points.'
print(text)
"
}

# Parse CLI args
FILTER_MODEL=""
FILTER_CTX=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) FILTER_MODEL="$2"; shift 2 ;;
        --ctx)   FILTER_CTX="$2"; shift 2 ;;
        *)       echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "Phase 0: KV Cache Quantization Benchmark"
echo "========================================="
echo "Results: $RESULTS_FILE"
echo "Logs:    $LOG_DIR"
echo "Filter:  model=${FILTER_MODEL:-all} ctx=${FILTER_CTX:-all}"
echo ""

# CSV header
echo "model,kv_config,ctk,ctv,context_length,test_type,tokens_generated,prompt_tokens,time_ms,tokens_per_sec,prompt_tps,rss_mb,server_kv_size_mb" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1 max_wait=600 elapsed=0
    while true; do
        local h; h=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "")
        echo "$h" | grep -q '"status":"ok"' && return 0
        sleep 2; elapsed=$((elapsed + 2))
        [ $elapsed -ge $max_wait ] && { echo "TIMEOUT waiting for server on port $port"; return 1; }
    done
}

warmup_server() {
    curl -s "http://localhost:${1}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello, respond with just OK."}],"max_tokens":5,"temperature":0}' > /dev/null 2>&1
    sleep 1
}

get_rss_mb() {
    local pid=$1
    local rss_kb
    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "$rss_kb" ] && [ "$rss_kb" -gt 0 ]; then
        python3 -c "print(f'{$rss_kb / 1024:.1f}')"
    else
        echo "0"
    fi
}

get_server_metrics() {
    # Try to get KV cache info from /slots endpoint
    local port=$1
    local slots_info
    slots_info=$(curl -s "http://localhost:${port}/slots" 2>/dev/null)
    # Extract n_ctx from server props
    local props
    props=$(curl -s "http://localhost:${port}/props" 2>/dev/null)
    echo "$slots_info" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list) and len(data) > 0:
        slot = data[0]
        print(json.dumps({
            'n_ctx': slot.get('n_ctx', 0),
            'n_predict': slot.get('n_predict', 0),
        }))
    else:
        print('{}')
except:
    print('{}')
" 2>/dev/null
}

run_generation_test() {
    local port=$1 prompt="$2" n_predict=$3
    local response
    response=$(curl -s --max-time 600 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],
            \"max_tokens\": ${n_predict},
            \"temperature\": 0.0,
            \"stream\": false
        }" 2>/dev/null)

    # Parse response for tokens and timing
    echo "$response" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    usage = r.get('usage', {})
    completion_tokens = usage.get('completion_tokens', 0)
    prompt_tokens = usage.get('prompt_tokens', 0)
    # timings from x-llama-cpp header or usage
    timings = r.get('timings', {})
    gen_tps = timings.get('predicted_per_second', 0)
    prompt_tps = timings.get('prompt_per_second', 0)
    gen_ms = timings.get('predicted_ms', 0)
    prompt_ms = timings.get('prompt_ms', 0)
    total_ms = int(gen_ms + prompt_ms) if gen_ms > 0 else 0
    print(f'{completion_tokens},{prompt_tokens},{total_ms},{gen_tps:.2f},{prompt_tps:.2f}')
except Exception as e:
    print(f'0,0,0,0.00,0.00')
" 2>/dev/null
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 3
}

# Run a single benchmark configuration
bench_config() {
    local model_label=$1 model_path=$2 kv_label=$3 ctk=$4 ctv=$5 ctx_len=$6

    echo "  --- ${kv_label} | ctx=${ctx_len} ---"

    # Build server args — single-model optimal config
    local server_args="-m $model_path -t $THREADS -np 1 --port $PORT -ngl 0 --flash-attn on -c $ctx_len -ub 8192"
    if [ "$ctk" != "f16" ] || [ "$ctv" != "f16" ]; then
        server_args="$server_args -ctk $ctk -ctv $ctv"
    fi

    local log_file="${LOG_DIR}/${model_label}_${kv_label}_ctx${ctx_len}.log"

    # Launch server with NUMA interleave for optimal cross-node memory access
    numactl --interleave=all $LLAMA_SERVER $server_args > "$log_file" 2>&1 &
    local PID=$!

    if ! wait_for_server $PORT; then
        echo "    FAILED: server did not start"
        kill_servers $PID
        echo "$model_label,$kv_label,$ctk,$ctv,$ctx_len,FAILED,0,0,0,0,0,0,0" >> "$RESULTS_FILE"
        return
    fi

    warmup_server $PORT

    # Get RSS after loading (before generation)
    local rss_mb
    rss_mb=$(get_rss_mb $PID)

    # Extract KV cache size from server log (format: "llama_kv_cache: size = XX.XX MiB")
    local kv_size_mb
    kv_size_mb=$(grep -oP 'llama_kv_cache: size =\s*\K[0-9.]+' "$log_file" 2>/dev/null | tail -1)
    [ -z "$kv_size_mb" ] && kv_size_mb=$(grep -oP 'KV buffer size\s*=\s*\K[0-9.]+' "$log_file" 2>/dev/null | tail -1)
    [ -z "$kv_size_mb" ] && kv_size_mb="0"

    echo "    RSS: ${rss_mb} MB | KV: ${kv_size_mb} MB"

    # Test 1: Short prompt generation (speed test)
    echo -n "    gen test: "
    local gen_result
    gen_result=$(run_generation_test $PORT "$SHORT_PROMPT" $N_PREDICT)
    local gen_tokens gen_prompt_tokens gen_time_ms gen_tps gen_prompt_tps
    gen_tokens=$(echo "$gen_result" | cut -d, -f1)
    gen_prompt_tokens=$(echo "$gen_result" | cut -d, -f2)
    gen_time_ms=$(echo "$gen_result" | cut -d, -f3)
    gen_tps=$(echo "$gen_result" | cut -d, -f4)
    gen_prompt_tps=$(echo "$gen_result" | cut -d, -f5)
    echo "${gen_tps} t/s gen, ${gen_prompt_tps} t/s prompt (${gen_tokens} tok)"

    # Get RSS after generation
    local rss_after
    rss_after=$(get_rss_mb $PID)

    echo "$model_label,$kv_label,$ctk,$ctv,$ctx_len,generation,$gen_tokens,$gen_prompt_tokens,$gen_time_ms,$gen_tps,$gen_prompt_tps,$rss_after,$kv_size_mb" >> "$RESULTS_FILE"

    # Check server is still alive
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "    WARNING: server died after generation test"
        kill_servers $PID
        return
    fi

    # Test 2: Long prompt (prompt processing speed) — fill ~75% of context
    if [ "$ctx_len" -ge 4096 ]; then
        local fill_tokens=$((ctx_len * 3 / 4))
        echo -n "    prefill test (${fill_tokens} tok target): "
        local long_prompt
        long_prompt=$(generate_long_prompt $fill_tokens)
        local prefill_result
        prefill_result=$(run_generation_test $PORT "$long_prompt" 32)
        local pf_tokens pf_prompt_tokens pf_time_ms pf_gen_tps pf_prompt_tps
        pf_tokens=$(echo "$prefill_result" | cut -d, -f1)
        pf_prompt_tokens=$(echo "$prefill_result" | cut -d, -f2)
        pf_time_ms=$(echo "$prefill_result" | cut -d, -f3)
        pf_gen_tps=$(echo "$prefill_result" | cut -d, -f4)
        pf_prompt_tps=$(echo "$prefill_result" | cut -d, -f5)
        echo "${pf_prompt_tps} t/s prompt (${pf_prompt_tokens} prompt tok)"

        local rss_prefill
        rss_prefill=$(get_rss_mb $PID)
        echo "$model_label,$kv_label,$ctk,$ctv,$ctx_len,prefill,$pf_tokens,$pf_prompt_tokens,$pf_time_ms,$pf_gen_tps,$pf_prompt_tps,$rss_prefill,$kv_size_mb" >> "$RESULTS_FILE"
    fi

    # Check server is still alive before reps
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "    WARNING: server died during prefill test"
        kill_servers $PID
        return
    fi

    # Run 2 more generation tests for consistency (repeat short prompt)
    for rep in 2 3; do
        local rep_result
        rep_result=$(run_generation_test $PORT "$SHORT_PROMPT" $N_PREDICT)
        local rep_tps
        rep_tps=$(echo "$rep_result" | cut -d, -f4)
        echo "    gen rep $rep: ${rep_tps} t/s"
        local rep_tokens rep_prompt_tokens rep_time_ms rep_prompt_tps
        rep_tokens=$(echo "$rep_result" | cut -d, -f1)
        rep_prompt_tokens=$(echo "$rep_result" | cut -d, -f2)
        rep_time_ms=$(echo "$rep_result" | cut -d, -f3)
        rep_prompt_tps=$(echo "$rep_result" | cut -d, -f5)
        local rss_rep
        rss_rep=$(get_rss_mb $PID)
        echo "$model_label,$kv_label,$ctk,$ctv,$ctx_len,generation_rep${rep},$rep_tokens,$rep_prompt_tokens,$rep_time_ms,$rep_tps,$rep_prompt_tps,$rss_rep,$kv_size_mb" >> "$RESULTS_FILE"
    done

    kill_servers $PID
}

# ============================================================
# Main benchmark loop
# ============================================================

echo "Starting benchmark at $(date)"
echo ""

# Model 1: Qwen3.5-35B-A3B (hybrid — 25% attention layers)
if [ -z "$FILTER_MODEL" ] || [ "$FILTER_MODEL" = "q35" ]; then
    echo "================================================================"
    echo "=== Qwen3.5-35B-A3B Q4_K_M (hybrid, 25% attention layers)   ==="
    echo "================================================================"
    if [ -f "$Q35_MODEL" ]; then
        for ctx in "${CONTEXT_LENGTHS[@]}"; do
            [ -n "$FILTER_CTX" ] && [ "$FILTER_CTX" != "$ctx" ] && continue
            for kv_cfg in "${KV_CONFIGS[@]}"; do
                IFS=':' read -r kv_label ctk ctv <<< "$kv_cfg"
                bench_config "q35-35B-A3B" "$Q35_MODEL" "$kv_label" "$ctk" "$ctv" "$ctx"
            done
        done
    else
        echo "  SKIP: $Q35_MODEL not found"
    fi
    echo ""
fi

# Model 2: Qwen2.5-Coder-32B (pure attention — max KV impact)
if [ -z "$FILTER_MODEL" ] || [ "$FILTER_MODEL" = "coder" ]; then
    echo "================================================================"
    echo "=== Qwen2.5-Coder-32B Q4_K_M (pure attention, max impact)   ==="
    echo "================================================================"
    if [ -f "$CODER_MODEL" ]; then
        for ctx in "${CONTEXT_LENGTHS[@]}"; do
            [ -n "$FILTER_CTX" ] && [ "$FILTER_CTX" != "$ctx" ] && continue
            for kv_cfg in "${KV_CONFIGS[@]}"; do
                IFS=':' read -r kv_label ctk ctv <<< "$kv_cfg"
                bench_config "q25-coder-32B" "$CODER_MODEL" "$kv_label" "$ctk" "$ctv" "$ctx"
            done
        done
    else
        echo "  SKIP: $CODER_MODEL not found"
    fi
    echo ""
fi

echo "================================================================"
echo "Benchmark complete at $(date)"
echo "Results saved to: $RESULTS_FILE"
echo ""

# Print summary table
echo "=== SUMMARY ==="
python3 -c "
import csv, sys
from collections import defaultdict

results = defaultdict(list)
with open('$RESULTS_FILE') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['test_type'] == 'generation':
            key = (row['model'], row['kv_config'], row['context_length'])
            results[key].append(row)

print(f'{'Model':<20} {'KV Config':<12} {'Context':<8} {'Gen t/s':<10} {'Prompt t/s':<12} {'RSS MB':<10} {'KV MB':<8}')
print('-' * 82)
for key in sorted(results.keys()):
    rows = results[key]
    model, kv_cfg, ctx = key
    gen_tps = float(rows[0]['tokens_per_sec'])
    prompt_tps = float(rows[0]['prompt_tps'])
    rss = rows[0]['rss_mb']
    kv = rows[0]['server_kv_size_mb']
    print(f'{model:<20} {kv_cfg:<12} {ctx:<8} {gen_tps:<10.2f} {prompt_tps:<12.2f} {rss:<10} {kv:<8}')
"
