#!/bin/bash
# Quick NUMA benchmark: Config C (2-way) and D (4-way) only
# Config A and B results already collected
set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_parallel"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_cd_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_cd_${TIMESTAMP}"

N_PREDICT=256
BASE_PORT=8190

NODE0_CPUS="0-47,96-143"
NODE1_CPUS="48-95,144-191"
NODE0A_CPUS="0-23,96-119"
NODE0B_CPUS="24-47,120-143"
NODE1A_CPUS="48-71,144-167"
NODE1B_CPUS="72-95,168-191"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Describe the process of photosynthesis at the molecular level, including the light reactions and Calvin cycle:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "config,instance,threads,cpu_binding,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

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
    response=$(curl -s --max-time 300 "http://localhost:${port}/v1/chat/completions" \
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

# ============================================================
# Config C: 2 concurrent — 2 instances, one per NUMA node
# ============================================================
echo "=== Config C: 2 concurrent (96 threads each, pinned per node) ==="

PORT_C1=$BASE_PORT
PORT_C2=$((BASE_PORT + 1))

taskset -c "$NODE0_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port $PORT_C1 -ngl 0 --metrics \
    > "$LOG_DIR/configC_node0.log" 2>&1 &
PID_C1=$!

taskset -c "$NODE1_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port $PORT_C2 -ngl 0 --metrics \
    > "$LOG_DIR/configC_node1.log" 2>&1 &
PID_C2=$!

echo "  Waiting for servers..."
wait_for_server $PORT_C1 || { kill_servers $PID_C1 $PID_C2; exit 1; }
wait_for_server $PORT_C2 || { kill_servers $PID_C1 $PID_C2; exit 1; }
echo "  Both servers ready"

# Warmup
curl -s "http://localhost:${PORT_C1}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1 &
WC1=$!
curl -s "http://localhost:${PORT_C2}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1 &
WC2=$!
wait $WC1 $WC2
echo "  Warmup done"

# Sequential per-instance measurement
for i in "${!PROMPTS[@]}"; do
    result1=$(run_completion $PORT_C1 "${PROMPTS[$i]}" "$N_PREDICT")
    result2=$(run_completion $PORT_C2 "${PROMPTS[$i]}" "$N_PREDICT")

    echo "C,node0,96,node0,$i,$result1" >> "$RESULTS_FILE"
    echo "C,node1,96,node1,$i,$result2" >> "$RESULTS_FILE"

    tps1=$(echo "$result1" | cut -d, -f3)
    tps2=$(echo "$result2" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$tps1 + $tps2:.2f}')")
    echo "  prompt $i: node0=${tps1} t/s, node1=${tps2} t/s, aggregate=${agg} t/s"
done

# Concurrent measurement (one prompt)
echo "  --- concurrent measurement ---"
tmpdir=$(mktemp -d)
(run_completion $PORT_C1 "${PROMPTS[0]}" "$N_PREDICT" > "$tmpdir/r1") &
PR1=$!
(run_completion $PORT_C2 "${PROMPTS[0]}" "$N_PREDICT" > "$tmpdir/r2") &
PR2=$!
wait $PR1 $PR2
result1=$(cat "$tmpdir/r1")
result2=$(cat "$tmpdir/r2")
rm -rf "$tmpdir"

echo "C_conc,node0,96,node0,0,$result1" >> "$RESULTS_FILE"
echo "C_conc,node1,96,node1,0,$result2" >> "$RESULTS_FILE"

tps1=$(echo "$result1" | cut -d, -f3)
tps2=$(echo "$result2" | cut -d, -f3)
agg=$(python3 -c "print(f'{$tps1 + $tps2:.2f}')")
echo "  concurrent: node0=${tps1} t/s, node1=${tps2} t/s, aggregate=${agg} t/s"

kill_servers $PID_C1 $PID_C2
echo ""

# ============================================================
# Config D: 4 concurrent — 4 instances, 48 threads each
# ============================================================
echo "=== Config D: 4 concurrent (48 threads each, quarter-machine) ==="

PORT_D1=$BASE_PORT
PORT_D2=$((BASE_PORT + 1))
PORT_D3=$((BASE_PORT + 2))
PORT_D4=$((BASE_PORT + 3))

taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_D1 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n0a.log" 2>&1 &
PID_D1=$!

taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_D2 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n0b.log" 2>&1 &
PID_D2=$!

taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_D3 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n1a.log" 2>&1 &
PID_D3=$!

taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_D4 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n1b.log" 2>&1 &
PID_D4=$!

echo "  Waiting for servers..."
wait_for_server $PORT_D1 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D2 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D3 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D4 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
echo "  All 4 servers ready"

# Warmup
WARMUP_PIDS=()
for port in $PORT_D1 $PORT_D2 $PORT_D3 $PORT_D4; do
    curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1 &
    WARMUP_PIDS+=($!)
done
wait "${WARMUP_PIDS[@]}"
echo "  Warmup done"

for i in "${!PROMPTS[@]}"; do
    result1=$(run_completion $PORT_D1 "${PROMPTS[$i]}" "$N_PREDICT")
    result2=$(run_completion $PORT_D2 "${PROMPTS[$i]}" "$N_PREDICT")
    result3=$(run_completion $PORT_D3 "${PROMPTS[$i]}" "$N_PREDICT")
    result4=$(run_completion $PORT_D4 "${PROMPTS[$i]}" "$N_PREDICT")

    echo "D,q0a,48,node0a,$i,$result1" >> "$RESULTS_FILE"
    echo "D,q0b,48,node0b,$i,$result2" >> "$RESULTS_FILE"
    echo "D,q1a,48,node1a,$i,$result3" >> "$RESULTS_FILE"
    echo "D,q1b,48,node1b,$i,$result4" >> "$RESULTS_FILE"

    tps1=$(echo "$result1" | cut -d, -f3)
    tps2=$(echo "$result2" | cut -d, -f3)
    tps3=$(echo "$result3" | cut -d, -f3)
    tps4=$(echo "$result4" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$tps1 + $tps2 + $tps3 + $tps4:.2f}')")
    echo "  prompt $i: q0a=${tps1}, q0b=${tps2}, q1a=${tps3}, q1b=${tps4}, aggregate=${agg} t/s"
done

kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4
echo ""

# Summary
echo "=== SUMMARY (combine with Config A/B data from earlier run) ==="
echo "Config A (baseline, 192 threads): avg 7.25 t/s"
echo "Config B (single-node, 96 threads): avg 13.39 t/s"
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results saved to: $RESULTS_FILE"
