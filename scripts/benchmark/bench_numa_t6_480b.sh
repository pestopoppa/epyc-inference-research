#!/bin/bash
# T6: NUMA Tree Speculation on Qwen3-Coder-480B-A35B Q4_K_M
#
# Tests tree speculation with NUMA pinning on the largest production model (250GB).
# Can't multi-instance (250GB × 2 > single node). Tests:
#   A) 1×192t with tree spec (dm=48, ps=0.05)
#   B) 1×96t node0 with tree spec
#   C) 1×192t linear spec (dm=48, ps=0) — baseline
#   D) 1×96t node0 linear spec — baseline
#
# Production sweep already showed 1×192t=3.36 t/s, 1×96t=4.08 t/s (linear dm=48).
# This tests whether tree speculation changes the picture with NUMA pinning.

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
TARGET="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_t6_480b"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_t6_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=256
BASE_PORT=8190
DRAFT_MAX=48

NODE0_CPUS="0-47,96-143"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "T6: NUMA Tree Speculation — Qwen3-Coder-480B-A35B Q4_K_M"
echo "==========================================================="
echo "Target: $(basename "$TARGET")"
echo "Drafter: $(basename "$DRAFTER")"
echo "n_predict=$N_PREDICT, draft_max=$DRAFT_MAX"
echo "Results: $RESULTS_FILE"
echo ""

echo "config,threads,cpu_binding,spec_mode,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1
    local max_wait=600
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: server on port $port did not start within ${max_wait}s"
            return 1
        fi
    done
}

run_completion() {
    local port=$1
    local prompt="$2"
    local n_predict=$3
    local start_ms end_ms elapsed_ms tokens tps
    start_ms=$(date +%s%N | cut -b1-13)
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
    end_ms=$(date +%s%N | cut -b1-13)
    elapsed_ms=$((end_ms - start_ms))
    tokens=$(echo "$response" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(r.get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null)
    if [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ]; then
        tps=$(python3 -c "print(f'{$tokens / ($elapsed_ms / 1000):.2f}')")
    else
        tps="0.00"
    fi
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 2
}

warmup_server() {
    local port=$1
    curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1
}

run_config() {
    local config_name=$1
    local threads=$2
    local cpus=$3
    local spec_mode=$4
    local extra_args=$5

    echo "=== Config $config_name: 1×${threads}t, ${cpus}, ${spec_mode} ==="

    if [ "$cpus" = "all" ]; then
        "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $extra_args \
            -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics \
            > "$LOG_DIR/config${config_name}.log" 2>&1 &
    else
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $extra_args \
            -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics \
            > "$LOG_DIR/config${config_name}.log" 2>&1 &
    fi
    local PID=$!

    if ! wait_for_server $BASE_PORT; then
        echo "  FAILED to start"
        kill_servers $PID
        return
    fi
    warmup_server $BASE_PORT
    echo "  Server ready"

    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $BASE_PORT "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$config_name,$threads,$cpus,$spec_mode,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "  prompt $i: ${tps} t/s"
    done

    kill_servers $PID
    echo ""
}

# Verify model files
if [ ! -f "$TARGET" ]; then echo "ERROR: target not found: $TARGET"; exit 1; fi
if [ ! -f "$DRAFTER" ]; then echo "ERROR: drafter not found: $DRAFTER"; exit 1; fi

# Config A: 1×192t tree spec
run_config "A" 192 "all" "tree_dm48_ps005" "--draft-max $DRAFT_MAX --draft-p-split 0.05 --kv-unified"

# Config B: 1×96t node0 tree spec
run_config "B" 96 "$NODE0_CPUS" "tree_dm48_ps005" "--draft-max $DRAFT_MAX --draft-p-split 0.05 --kv-unified"

# Config C: 1×192t linear spec (already tested in production sweep but re-run for consistency)
run_config "C" 192 "all" "linear_dm48" "--draft-max $DRAFT_MAX --kv-unified"

# Config D: 1×96t node0 linear spec
run_config "D" 96 "$NODE0_CPUS" "linear_dm48" "--draft-max $DRAFT_MAX --kv-unified"

# Summary
echo "=== T6 SUMMARY ==="
python3 - "$RESULTS_FILE" << 'PYEOF'
import csv, sys
from collections import defaultdict

results = defaultdict(list)
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for row in reader:
        config = row['config']
        tps = float(row['tokens_per_sec'])
        results[config].append(tps)

print(f"{'Config':<8} {'Threads':<8} {'NUMA':<8} {'Spec':<20} {'Avg t/s':<10}")
print("-" * 54)

labels = {
    'A': ('192', 'all', 'tree dm48 ps=0.05'),
    'B': ('96', 'node0', 'tree dm48 ps=0.05'),
    'C': ('192', 'all', 'linear dm48'),
    'D': ('96', 'node0', 'linear dm48'),
}

baseline = None
for cfg in ['A', 'B', 'C', 'D']:
    vals = results.get(cfg, [])
    if not vals:
        continue
    avg = sum(vals) / len(vals)
    threads, numa, spec = labels[cfg]
    if baseline is None:
        baseline = avg
    speedup = f"({avg/baseline:.2f}x)" if baseline else ""
    print(f"{cfg:<8} {threads:<8} {numa:<8} {spec:<20} {avg:<8.2f} {speedup}")

if results.get('A') and results.get('B'):
    avg_a = sum(results['A']) / len(results['A'])
    avg_b = sum(results['B']) / len(results['B'])
    print(f"\nNUMA node0 tree speedup: {avg_b/avg_a:.2f}x")

if results.get('C') and results.get('D'):
    avg_c = sum(results['C']) / len(results['C'])
    avg_d = sum(results['D']) / len(results['D'])
    print(f"NUMA node0 linear speedup: {avg_d/avg_c:.2f}x")

if results.get('B') and results.get('D'):
    avg_b = sum(results['B']) / len(results['B'])
    avg_d = sum(results['D']) / len(results['D'])
    print(f"Tree vs linear (node0): {avg_b/avg_d:.2f}x")

PYEOF

echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
