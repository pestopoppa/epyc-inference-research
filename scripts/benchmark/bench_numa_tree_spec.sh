#!/bin/bash
# T5: NUMA 4-Way Tree Speculation Benchmark
#
# Tests NUMA-pinned parallel instances with tree speculation on Qwen2.5-Coder-32B f16.
# Builds on S2 finding that 4×48t gives 6.9x aggregate on hybrid — tests if same applies to dense + tree.
#
# Configs:
#   A) 1×192t with tree spec (existing benchmark reference)
#   B) 1×96t single-node with tree spec
#   C) 2×96t per-node with tree spec
#   D) 4×48t quarter with tree spec
#   E) 4×48t quarter WITHOUT tree spec (baseline for comparison)
#
# Model: Qwen2.5-Coder-32B f16 (~65GB) + Qwen2.5-0.5B f16 drafter (~1GB)
# Tree settings: dm=32, p_split=0.05 (optimal from Phase 4 validation: +12.2%)

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
TARGET="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf"
DRAFTER="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_tree_spec"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_tree_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=256
BASE_PORT=8190
DRAFT_MAX=32
P_SPLIT=0.05

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

echo "NUMA Tree Speculation Benchmark (T5)"
echo "====================================="
echo "Target: $(basename "$TARGET")"
echo "Drafter: $(basename "$DRAFTER")"
echo "Tree: dm=$DRAFT_MAX, p_split=$P_SPLIT"
echo "n_predict=$N_PREDICT"
echo "Results: $RESULTS_FILE"
echo ""

echo "config,instance,threads,cpu_binding,tree,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

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

run_prompts() {
    local config=$1
    local instance=$2
    local threads=$3
    local binding=$4
    local tree=$5
    local port=$6

    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $port "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$config,$instance,$threads,$binding,$tree,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "    prompt $i: ${tps} t/s"
    done
}

TREE_ARGS="--draft-max $DRAFT_MAX --draft-p-split $P_SPLIT --kv-unified"

# Verify model files exist
if [ ! -f "$TARGET" ]; then echo "ERROR: target not found: $TARGET"; exit 1; fi
if [ ! -f "$DRAFTER" ]; then echo "ERROR: drafter not found: $DRAFTER"; exit 1; fi

# ============================================================
# Config A: 1×192t with tree spec
# ============================================================
echo "=== Config A: 1×192t with tree spec ==="

"$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $TREE_ARGS \
    -t 192 -np 1 --port $BASE_PORT -ngl 0 --metrics \
    > "$LOG_DIR/configA.log" 2>&1 &
PID_A=$!

wait_for_server $BASE_PORT || { kill_servers $PID_A; exit 1; }
warmup_server $BASE_PORT
echo "  Server ready"
run_prompts "A" "1" "192" "all" "tree" $BASE_PORT

kill_servers $PID_A
echo ""

# ============================================================
# Config B: 1×96t single-node with tree spec
# ============================================================
echo "=== Config B: 1×96t single-node with tree spec ==="

taskset -c "$NODE0_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $TREE_ARGS \
    -t 96 -np 1 --port $BASE_PORT -ngl 0 --metrics \
    > "$LOG_DIR/configB.log" 2>&1 &
PID_B=$!

wait_for_server $BASE_PORT || { kill_servers $PID_B; exit 1; }
warmup_server $BASE_PORT
echo "  Server ready"
run_prompts "B" "1" "96" "node0" "tree" $BASE_PORT

kill_servers $PID_B
echo ""

# ============================================================
# Config D: 4×48t quarter with tree spec (test this before C since it's the key experiment)
# ============================================================
echo "=== Config D: 4×48t quarter with tree spec ==="

PORT_D1=$BASE_PORT
PORT_D2=$((BASE_PORT + 1))
PORT_D3=$((BASE_PORT + 2))
PORT_D4=$((BASE_PORT + 3))

taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $TREE_ARGS \
    -t 48 -np 1 --port $PORT_D1 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n0a.log" 2>&1 &
PID_D1=$!

taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $TREE_ARGS \
    -t 48 -np 1 --port $PORT_D2 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n0b.log" 2>&1 &
PID_D2=$!

taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $TREE_ARGS \
    -t 48 -np 1 --port $PORT_D3 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n1a.log" 2>&1 &
PID_D3=$!

taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" $TREE_ARGS \
    -t 48 -np 1 --port $PORT_D4 -ngl 0 --metrics \
    > "$LOG_DIR/configD_n1b.log" 2>&1 &
PID_D4=$!

echo "  Waiting for 4 servers..."
wait_for_server $PORT_D1 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D2 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D3 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }
wait_for_server $PORT_D4 || { kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4; exit 1; }

WARMUP_PIDS=()
for port in $PORT_D1 $PORT_D2 $PORT_D3 $PORT_D4; do
    warmup_server $port &
    WARMUP_PIDS+=($!)
done
wait "${WARMUP_PIDS[@]}"
echo "  All 4 servers ready"

for i in "${!PROMPTS[@]}"; do
    r1=$(run_completion $PORT_D1 "${PROMPTS[$i]}" "$N_PREDICT")
    r2=$(run_completion $PORT_D2 "${PROMPTS[$i]}" "$N_PREDICT")
    r3=$(run_completion $PORT_D3 "${PROMPTS[$i]}" "$N_PREDICT")
    r4=$(run_completion $PORT_D4 "${PROMPTS[$i]}" "$N_PREDICT")

    echo "D,q0a,48,node0a,tree,$i,$r1" >> "$RESULTS_FILE"
    echo "D,q0b,48,node0b,tree,$i,$r2" >> "$RESULTS_FILE"
    echo "D,q1a,48,node1a,tree,$i,$r3" >> "$RESULTS_FILE"
    echo "D,q1b,48,node1b,tree,$i,$r4" >> "$RESULTS_FILE"

    t1=$(echo "$r1" | cut -d, -f3)
    t2=$(echo "$r2" | cut -d, -f3)
    t3=$(echo "$r3" | cut -d, -f3)
    t4=$(echo "$r4" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$t1 + $t2 + $t3 + $t4:.2f}')")
    echo "    prompt $i: q0a=${t1}, q0b=${t2}, q1a=${t3}, q1b=${t4}, aggregate=${agg} t/s"
done

kill_servers $PID_D1 $PID_D2 $PID_D3 $PID_D4
echo ""

# ============================================================
# Config E: 4×48t quarter WITHOUT tree spec (linear baseline)
# ============================================================
echo "=== Config E: 4×48t quarter, linear speculation (no tree) ==="

taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" \
    --draft-max $DRAFT_MAX -t 48 -np 1 --port $PORT_D1 -ngl 0 --metrics \
    > "$LOG_DIR/configE_n0a.log" 2>&1 &
PID_E1=$!

taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" \
    --draft-max $DRAFT_MAX -t 48 -np 1 --port $PORT_D2 -ngl 0 --metrics \
    > "$LOG_DIR/configE_n0b.log" 2>&1 &
PID_E2=$!

taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" \
    --draft-max $DRAFT_MAX -t 48 -np 1 --port $PORT_D3 -ngl 0 --metrics \
    > "$LOG_DIR/configE_n1a.log" 2>&1 &
PID_E3=$!

taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$TARGET" -md "$DRAFTER" \
    --draft-max $DRAFT_MAX -t 48 -np 1 --port $PORT_D4 -ngl 0 --metrics \
    > "$LOG_DIR/configE_n1b.log" 2>&1 &
PID_E4=$!

echo "  Waiting for 4 servers..."
wait_for_server $PORT_D1 || { kill_servers $PID_E1 $PID_E2 $PID_E3 $PID_E4; exit 1; }
wait_for_server $PORT_D2 || { kill_servers $PID_E1 $PID_E2 $PID_E3 $PID_E4; exit 1; }
wait_for_server $PORT_D3 || { kill_servers $PID_E1 $PID_E2 $PID_E3 $PID_E4; exit 1; }
wait_for_server $PORT_D4 || { kill_servers $PID_E1 $PID_E2 $PID_E3 $PID_E4; exit 1; }

WARMUP_PIDS=()
for port in $PORT_D1 $PORT_D2 $PORT_D3 $PORT_D4; do
    warmup_server $port &
    WARMUP_PIDS+=($!)
done
wait "${WARMUP_PIDS[@]}"
echo "  All 4 servers ready"

for i in "${!PROMPTS[@]}"; do
    r1=$(run_completion $PORT_D1 "${PROMPTS[$i]}" "$N_PREDICT")
    r2=$(run_completion $PORT_D2 "${PROMPTS[$i]}" "$N_PREDICT")
    r3=$(run_completion $PORT_D3 "${PROMPTS[$i]}" "$N_PREDICT")
    r4=$(run_completion $PORT_D4 "${PROMPTS[$i]}" "$N_PREDICT")

    echo "E,q0a,48,node0a,linear,$i,$r1" >> "$RESULTS_FILE"
    echo "E,q0b,48,node0b,linear,$i,$r2" >> "$RESULTS_FILE"
    echo "E,q1a,48,node1a,linear,$i,$r3" >> "$RESULTS_FILE"
    echo "E,q1b,48,node1b,linear,$i,$r4" >> "$RESULTS_FILE"

    t1=$(echo "$r1" | cut -d, -f3)
    t2=$(echo "$r2" | cut -d, -f3)
    t3=$(echo "$r3" | cut -d, -f3)
    t4=$(echo "$r4" | cut -d, -f3)
    agg=$(python3 -c "print(f'{$t1 + $t2 + $t3 + $t4:.2f}')")
    echo "    prompt $i: q0a=${t1}, q0b=${t2}, q1a=${t3}, q1b=${t4}, aggregate=${agg} t/s"
done

kill_servers $PID_E1 $PID_E2 $PID_E3 $PID_E4
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

configs = [
    ('A', '1×192t tree', 1),
    ('B', '1×96t node0 tree', 1),
    ('D', '4×48t tree', 4),
    ('E', '4×48t linear', 4),
]

print(f"{'Config':<22} {'Per-inst avg t/s':<18} {'Aggregate t/s':<18}")
print("-" * 58)

baseline_agg = None
for cfg_id, label, n_inst in configs:
    vals = results[cfg_id]
    if not vals:
        continue
    avg_inst = sum(vals) / len(vals)
    if n_inst == 1:
        agg = avg_inst
    else:
        # N instances per prompt, average the per-prompt aggregates
        per_prompt_agg = [sum(vals[j:j+n_inst]) for j in range(0, len(vals), n_inst)]
        agg = sum(per_prompt_agg) / len(per_prompt_agg) if per_prompt_agg else 0

    if baseline_agg is None:
        baseline_agg = agg

    speedup = f"({agg/baseline_agg:.1f}x)" if baseline_agg else ""
    print(f"{label:<22} {avg_inst:<18.2f} {agg:<14.2f} {speedup}")

PYEOF

echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
