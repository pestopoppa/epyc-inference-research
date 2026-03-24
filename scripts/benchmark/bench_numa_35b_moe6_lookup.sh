#!/bin/bash
# Benchmark: Qwen3.5-35B-A3B 4×48t NUMA with moe6+lookup
#
# Validates the ~78 t/s aggregate estimate for the production frontdoor config.
# Previous NUMA benchmarks used baseline (no accel); this adds moe6+lookup.
#
# Usage: ./bench_numa_35b_moe6_lookup.sh

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_35b_moe6"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_35b_moe6_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT="${N_PREDICT:-256}"
BASE_PORT=8190

# NUMA quarter CPU lists
NODE0A_CPUS="0-23,96-119"
NODE0B_CPUS="24-47,120-143"
NODE1A_CPUS="48-71,144-167"
NODE1B_CPUS="72-95,168-191"

# moe6+lookup flags (production frontdoor config)
ACCEL_FLAGS="--override-kv qwen35moe.expert_used_count=int:6 --lookup"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Describe the process of photosynthesis at the molecular level, including the light reactions and Calvin cycle:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "NUMA 4×48t Benchmark: Qwen3.5-35B-A3B Q4KM + moe6+lookup"
echo "============================================================"
echo "Model: $(basename "$MODEL")"
echo "Accel: moe6+lookup"
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
# Config E: Single instance 48t + moe6+lookup (per-instance baseline)
# ============================================================
echo "=== Config E: Single instance (48t Q0A, moe6+lookup) ==="

taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $BASE_PORT -ngl 0 --metrics --mlock \
    $ACCEL_FLAGS \
    > "$LOG_DIR/configE.log" 2>&1 &
PID_E=$!

if ! wait_for_server $BASE_PORT; then
    echo "  FAILED to start server"
    kill_servers $PID_E
    exit 1
fi

# Warmup
curl -s "http://localhost:${BASE_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
    > /dev/null 2>&1

for i in "${!PROMPTS[@]}"; do
    result=$(run_completion $BASE_PORT "${PROMPTS[$i]}" "$N_PREDICT")
    echo "E,q0a,48,node0a,$i,$result" >> "$RESULTS_FILE"
    tps=$(echo "$result" | cut -d, -f3)
    echo "  prompt $i: ${tps} t/s"
done

kill_servers $PID_E
echo ""

# ============================================================
# Config F: 4×48t NUMA + moe6+lookup (production config)
# ============================================================
echo "=== Config F: 4×48t NUMA (moe6+lookup, mlock) ==="

PORT_F1=$BASE_PORT
PORT_F2=$((BASE_PORT + 1))
PORT_F3=$((BASE_PORT + 2))
PORT_F4=$((BASE_PORT + 3))

taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_F1 -ngl 0 --metrics --mlock \
    $ACCEL_FLAGS \
    > "$LOG_DIR/configF_q0a.log" 2>&1 &
PID_F1=$!

taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_F2 -ngl 0 --metrics --mlock \
    $ACCEL_FLAGS \
    > "$LOG_DIR/configF_q0b.log" 2>&1 &
PID_F2=$!

taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_F3 -ngl 0 --metrics --mlock \
    $ACCEL_FLAGS \
    > "$LOG_DIR/configF_q1a.log" 2>&1 &
PID_F3=$!

taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 48 -np 1 --port $PORT_F4 -ngl 0 --metrics --mlock \
    $ACCEL_FLAGS \
    > "$LOG_DIR/configF_q1b.log" 2>&1 &
PID_F4=$!

echo "  Waiting for 4 instances to load (with mlock, ~80 GB total)..."
wait_for_server $PORT_F1 || { kill_servers $PID_F1 $PID_F2 $PID_F3 $PID_F4; exit 1; }
echo "  Instance 1 ready"
wait_for_server $PORT_F2 || { kill_servers $PID_F1 $PID_F2 $PID_F3 $PID_F4; exit 1; }
echo "  Instance 2 ready"
wait_for_server $PORT_F3 || { kill_servers $PID_F1 $PID_F2 $PID_F3 $PID_F4; exit 1; }
echo "  Instance 3 ready"
wait_for_server $PORT_F4 || { kill_servers $PID_F1 $PID_F2 $PID_F3 $PID_F4; exit 1; }
echo "  All 4 instances ready"

# Warmup all
WARMUP_PIDS=()
for port in $PORT_F1 $PORT_F2 $PORT_F3 $PORT_F4; do
    curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1 &
    WARMUP_PIDS+=($!)
done
wait "${WARMUP_PIDS[@]}"

# Sequential per-instance measurement (checks per-instance throughput under 4-way memory pressure)
echo "  --- sequential per-instance ---"
for i in "${!PROMPTS[@]}"; do
    result1=$(run_completion $PORT_F1 "${PROMPTS[$i]}" "$N_PREDICT")
    result2=$(run_completion $PORT_F2 "${PROMPTS[$i]}" "$N_PREDICT")
    result3=$(run_completion $PORT_F3 "${PROMPTS[$i]}" "$N_PREDICT")
    result4=$(run_completion $PORT_F4 "${PROMPTS[$i]}" "$N_PREDICT")

    echo "F,q0a,48,node0a,$i,$result1" >> "$RESULTS_FILE"
    echo "F,q0b,48,node0b,$i,$result2" >> "$RESULTS_FILE"
    echo "F,q1a,48,node1a,$i,$result3" >> "$RESULTS_FILE"
    echo "F,q1b,48,node1b,$i,$result4" >> "$RESULTS_FILE"

    tps1=$(echo "$result1" | cut -d, -f3)
    tps2=$(echo "$result2" | cut -d, -f3)
    tps3=$(echo "$result3" | cut -d, -f3)
    tps4=$(echo "$result4" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$tps1 + $tps2 + $tps3 + $tps4:.2f}')")
    echo "  prompt $i: q0a=${tps1}, q0b=${tps2}, q1a=${tps3}, q1b=${tps4}, aggregate=${agg} t/s"
done

# Concurrent measurement (all 4 generating simultaneously)
echo "  --- concurrent measurement ---"
for i in "${!PROMPTS[@]}"; do
    tmpdir=$(mktemp -d)
    (run_completion $PORT_F1 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r1") &
    PID_R1=$!
    (run_completion $PORT_F2 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r2") &
    PID_R2=$!
    (run_completion $PORT_F3 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r3") &
    PID_R3=$!
    (run_completion $PORT_F4 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r4") &
    PID_R4=$!
    wait $PID_R1 $PID_R2 $PID_R3 $PID_R4
    result1=$(cat "$tmpdir/r1")
    result2=$(cat "$tmpdir/r2")
    result3=$(cat "$tmpdir/r3")
    result4=$(cat "$tmpdir/r4")
    rm -rf "$tmpdir"

    echo "F_conc,q0a,48,node0a,$i,$result1" >> "$RESULTS_FILE"
    echo "F_conc,q0b,48,node0b,$i,$result2" >> "$RESULTS_FILE"
    echo "F_conc,q1a,48,node1a,$i,$result3" >> "$RESULTS_FILE"
    echo "F_conc,q1b,48,node1b,$i,$result4" >> "$RESULTS_FILE"

    tps1=$(echo "$result1" | cut -d, -f3)
    tps2=$(echo "$result2" | cut -d, -f3)
    tps3=$(echo "$result3" | cut -d, -f3)
    tps4=$(echo "$result4" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$tps1 + $tps2 + $tps3 + $tps4:.2f}')")
    echo "  concurrent prompt $i: q0a=${tps1}, q0b=${tps2}, q1a=${tps3}, q1b=${tps4}, aggregate=${agg} t/s"
done

kill_servers $PID_F1 $PID_F2 $PID_F3 $PID_F4
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

print(f"{'Config':<16} {'Instances':<10} {'Avg t/s (per inst)':<22} {'Avg Aggregate t/s':<20}")
print("-" * 68)

for config, label in [('E', 'E (1×48t)'), ('F', 'F (4×48t seq)'), ('F_conc', 'F (4×48t conc)')]:
    vals = results[config]
    if not vals:
        continue

    if config == 'E':
        avg = sum(vals) / len(vals)
        print(f"{label:<16} {'1':<10} {avg:<22.2f} {avg:<20.2f}")
    else:
        per_prompt_agg = []
        per_inst = []
        for j in range(0, len(vals), 4):
            if j+3 < len(vals):
                per_prompt_agg.append(sum(vals[j:j+4]))
                per_inst.extend(vals[j:j+4])
        avg_inst = sum(per_inst) / len(per_inst) if per_inst else 0
        avg_agg = sum(per_prompt_agg) / len(per_prompt_agg) if per_prompt_agg else 0
        print(f"{label:<16} {'4':<10} {avg_inst:<22.2f} {avg_agg:<20.2f}")

# Speedup
if results['E']:
    baseline = sum(results['E']) / len(results['E'])
    print(f"\nBaseline (single 48t moe6+lu): {baseline:.2f} t/s")
    for config, label in [('F', '4×48t seq'), ('F_conc', '4×48t conc')]:
        vals = results[config]
        if not vals:
            continue
        aggs = [sum(vals[j:j+4]) for j in range(0, len(vals), 4) if j+3 < len(vals)]
        avg = sum(aggs) / len(aggs) if aggs else 0
        print(f"  {label}: {avg:.2f} t/s ({avg/baseline:.2f}x)")
    print(f"\nExpected: ~19.6 t/s per instance, ~78 t/s aggregate")
PYEOF

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Logs saved to: $LOG_DIR"
