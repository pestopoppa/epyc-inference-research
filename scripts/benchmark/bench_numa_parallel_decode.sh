#!/bin/bash
# S2: NUMA Parallel Decode Benchmark
#
# Measures aggregate throughput of concurrent single-token decodes across NUMA nodes.
# Hypothesis: 2 instances (one per NUMA node) may outperform 1 instance on all cores
# for models where speculation doesn't work (hybrid recurrent models).
#
# Configs:
#   A) Baseline: 1 instance, 192 threads (all CPUs)
#   B) Single-node: 1 instance, 96 threads on node 0 CPUs (0-47,96-143)
#   C) 2 concurrent: 2 instances, each on one NUMA node (96 threads each)
#   D) 4 concurrent: 4 instances, 48 threads each (quarter-machine)
#
# Note: numactl --membind is blocked in container; using taskset for CPU pinning.
# Memory follows first-touch policy which approximates NUMA binding.
#
# Usage: ./bench_numa_parallel_decode.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_parallel"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_parallel_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT="${N_PREDICT:-256}"
WARMUP_TOKENS=32
BASE_PORT=8190

# NUMA node CPU lists (from lscpu)
NODE0_CPUS="0-47,96-143"   # 96 threads (48 physical cores + HT)
NODE1_CPUS="48-95,144-191"  # 96 threads
# Quarter splits for 4-way
NODE0A_CPUS="0-23,96-119"   # 48 threads (24 physical + HT)
NODE0B_CPUS="24-47,120-143" # 48 threads
NODE1A_CPUS="48-71,144-167" # 48 threads
NODE1B_CPUS="72-95,168-191" # 48 threads

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Describe the process of photosynthesis at the molecular level, including the light reactions and Calvin cycle:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "NUMA Parallel Decode Benchmark"
echo "=============================="
echo "Model: $(basename "$MODEL")"
echo "n_predict=$N_PREDICT"
echo "results: $RESULTS_FILE"
echo ""

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
    for pid in "$@"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "$@"; do
        wait "$pid" 2>/dev/null || true
    done
    sleep 2
}

# ============================================================
# Config A: Baseline — 1 instance, all 192 threads
# ============================================================
echo "=== Config A: Baseline (1 instance, 192 threads) ==="

"$LLAMA_SERVER" -m "$MODEL" -t 192 -np 1 --port $BASE_PORT -ngl 0 --metrics \
    > "$LOG_DIR/configA.log" 2>&1 &
PID_A=$!

if ! wait_for_server $BASE_PORT; then
    echo "  FAILED to start server"
    kill_servers $PID_A
    exit 1
fi

# Warmup
curl -s "http://localhost:${BASE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1

for i in "${!PROMPTS[@]}"; do
    result=$(run_completion $BASE_PORT "${PROMPTS[$i]}" "$N_PREDICT")
    echo "A,1,192,all,$i,$result" >> "$RESULTS_FILE"
    tps=$(echo "$result" | cut -d, -f3)
    echo "  prompt $i: ${tps} t/s"
done

kill_servers $PID_A
echo ""

# ============================================================
# Config B: Single-node — 1 instance, 96 threads on node 0
# ============================================================
echo "=== Config B: Single-node (1 instance, 96 threads, node 0) ==="

taskset -c "$NODE0_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port $BASE_PORT -ngl 0 --metrics \
    > "$LOG_DIR/configB.log" 2>&1 &
PID_B=$!

if ! wait_for_server $BASE_PORT; then
    echo "  FAILED to start server"
    kill_servers $PID_B
    exit 1
fi

curl -s "http://localhost:${BASE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1

for i in "${!PROMPTS[@]}"; do
    result=$(run_completion $BASE_PORT "${PROMPTS[$i]}" "$N_PREDICT")
    echo "B,1,96,node0,$i,$result" >> "$RESULTS_FILE"
    tps=$(echo "$result" | cut -d, -f3)
    echo "  prompt $i: ${tps} t/s"
done

kill_servers $PID_B
echo ""

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

wait_for_server $PORT_C1 || { kill_servers $PID_C1 $PID_C2; exit 1; }
wait_for_server $PORT_C2 || { kill_servers $PID_C1 $PID_C2; exit 1; }

# Warmup both
curl -s "http://localhost:${PORT_C1}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1 &
W1=$!
curl -s "http://localhost:${PORT_C2}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1 &
W2=$!
wait $W1 $W2

# Measure per-instance throughput while both are loaded (sequential queries, tests NUMA contention)
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

# Also measure truly concurrent generation (both generating simultaneously)
echo "  --- concurrent measurement ---"
for i in 0; do
    tmpdir=$(mktemp -d)
    (run_completion $PORT_C1 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r1") &
    PID_R1=$!
    (run_completion $PORT_C2 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r2") &
    PID_R2=$!
    wait $PID_R1 $PID_R2
    result1=$(cat "$tmpdir/r1")
    result2=$(cat "$tmpdir/r2")
    rm -rf "$tmpdir"

    echo "C_conc,node0,96,node0,$i,$result1" >> "$RESULTS_FILE"
    echo "C_conc,node1,96,node1,$i,$result2" >> "$RESULTS_FILE"

    tps1=$(echo "$result1" | cut -d, -f3)
    tps2=$(echo "$result2" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$tps1 + $tps2:.2f}')")
    echo "  concurrent prompt $i: node0=${tps1} t/s, node1=${tps2} t/s, aggregate=${agg} t/s"
done

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

wait_for_server $PORT_D1 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D2 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D3 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D4 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }

# Warmup all
WARMUP_PIDS=()
for port in $PORT_D1 $PORT_D2 $PORT_D3 $PORT_D4; do
    curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1 &
    WARMUP_PIDS+=($!)
done
wait "${WARMUP_PIDS[@]}"

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

# ============================================================
# Summary
# ============================================================
echo "=== SUMMARY ==="
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

# For concurrent configs, we need aggregate per-prompt
# Config A/B: single instance, just average
# Config C: 2 instances per prompt, sum pairs then average
# Config D: 4 instances per prompt, sum quads then average

print(f"{'Config':<12} {'Instances':<10} {'Threads/inst':<13} {'Avg t/s (per inst)':<20} {'Avg Aggregate t/s':<20}")
print("-" * 75)

for config in ['A', 'B', 'C', 'D']:
    vals = results[config]
    if not vals:
        continue

    if config == 'A':
        avg = sum(vals) / len(vals)
        print(f"{'A (all)':<12} {'1':<10} {'192':<13} {avg:<20.2f} {avg:<20.2f}")
    elif config == 'B':
        avg = sum(vals) / len(vals)
        print(f"{'B (node0)':<12} {'1':<10} {'96':<13} {avg:<20.2f} {avg:<20.2f}")
    elif config == 'C':
        # 2 rows per prompt (node0, node1)
        per_prompt_agg = []
        per_inst = []
        for j in range(0, len(vals), 2):
            if j+1 < len(vals):
                per_prompt_agg.append(vals[j] + vals[j+1])
                per_inst.extend([vals[j], vals[j+1]])
        avg_inst = sum(per_inst) / len(per_inst) if per_inst else 0
        avg_agg = sum(per_prompt_agg) / len(per_prompt_agg) if per_prompt_agg else 0
        print(f"{'C (2-way)':<12} {'2':<10} {'96':<13} {avg_inst:<20.2f} {avg_agg:<20.2f}")
    elif config == 'D':
        # 4 rows per prompt
        per_prompt_agg = []
        per_inst = []
        for j in range(0, len(vals), 4):
            if j+3 < len(vals):
                per_prompt_agg.append(sum(vals[j:j+4]))
                per_inst.extend(vals[j:j+4])
        avg_inst = sum(per_inst) / len(per_inst) if per_inst else 0
        avg_agg = sum(per_prompt_agg) / len(per_prompt_agg) if per_prompt_agg else 0
        print(f"{'D (4-way)':<12} {'4':<10} {'48':<13} {avg_inst:<20.2f} {avg_agg:<20.2f}")

# Compute speedups
if results['A']:
    baseline = sum(results['A']) / len(results['A'])
    print(f"\nSpeedups vs baseline (Config A = {baseline:.2f} t/s):")
    for config in ['B', 'C', 'D']:
        vals = results[config]
        if not vals:
            continue
        if config == 'B':
            avg = sum(vals) / len(vals)
            print(f"  {config}: {avg/baseline:.2f}x ({avg:.2f} t/s)")
        elif config == 'C':
            aggs = [vals[j] + vals[j+1] for j in range(0, len(vals), 2) if j+1 < len(vals)]
            avg = sum(aggs) / len(aggs) if aggs else 0
            print(f"  {config}: {avg/baseline:.2f}x aggregate ({avg:.2f} t/s)")
        elif config == 'D':
            aggs = [sum(vals[j:j+4]) for j in range(0, len(vals), 4) if j+3 < len(vals)]
            avg = sum(aggs) / len(aggs) if aggs else 0
            print(f"  {config}: {avg/baseline:.2f}x aggregate ({avg:.2f} t/s)")
PYEOF

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Logs saved to: $LOG_DIR"
