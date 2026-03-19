#!/bin/bash
# NUMA Production Model Sweep
#
# Tests NUMA pinning impact across ALL production models.
# For each model: 1×192t (baseline) vs 1×96t (single-node) vs best parallel config.
# Includes speculation where applicable.
#
# This is the definitive NUMA characterization for the production stack.

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_production"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_prod_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=256
BASE_PORT=8190

NODE0_CPUS="0-47,96-143"
NODE0A_CPUS="0-23,96-119"
NODE0B_CPUS="24-47,120-143"
NODE1A_CPUS="48-71,144-167"
NODE1B_CPUS="72-95,168-191"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "NUMA Production Model Sweep"
echo "============================"
echo "n_predict=$N_PREDICT"
echo "Results: $RESULTS_FILE"
echo ""

echo "model,config,instance,threads,cpu_binding,spec,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

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

# Run a single-instance benchmark
bench_single() {
    local model_name=$1
    local config=$2
    local target=$3
    local drafter=$4
    local threads=$5
    local cpus=$6
    local extra_args=$7
    local spec_label=$8

    local port=$BASE_PORT
    local log_name="${model_name}_${config}"

    echo "  --- $config ($threads threads, $cpus) ---"

    if [ "$cpus" = "all" ]; then
        "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
            -t "$threads" -np 1 --port $port -ngl 0 --metrics \
            > "$LOG_DIR/${log_name}.log" 2>&1 &
    else
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
            -t "$threads" -np 1 --port $port -ngl 0 --metrics \
            > "$LOG_DIR/${log_name}.log" 2>&1 &
    fi
    local PID=$!

    if ! wait_for_server $port; then
        echo "    FAILED to start"
        kill_servers $PID
        return
    fi
    warmup_server $port

    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $port "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$model_name,$config,1,$threads,$cpus,$spec_label,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "    prompt $i: ${tps} t/s"
    done

    kill_servers $PID
}

# Run a 4-way benchmark
bench_quad() {
    local model_name=$1
    local config=$2
    local target=$3
    local drafter=$4
    local extra_args=$5
    local spec_label=$6

    echo "  --- $config (4×48 threads, quarter-machine) ---"

    local PORT1=$BASE_PORT
    local PORT2=$((BASE_PORT + 1))
    local PORT3=$((BASE_PORT + 2))
    local PORT4=$((BASE_PORT + 3))

    taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 48 -np 1 --port $PORT1 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n0a.log" 2>&1 &
    local PID1=$!
    taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 48 -np 1 --port $PORT2 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n0b.log" 2>&1 &
    local PID2=$!
    taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 48 -np 1 --port $PORT3 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n1a.log" 2>&1 &
    local PID3=$!
    taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 48 -np 1 --port $PORT4 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n1b.log" 2>&1 &
    local PID4=$!

    echo "    Loading 4 instances..."
    wait_for_server $PORT1 || { kill_servers $PID1 $PID2 $PID3 $PID4; return; }
    wait_for_server $PORT2 || { kill_servers $PID1 $PID2 $PID3 $PID4; return; }
    wait_for_server $PORT3 || { kill_servers $PID1 $PID2 $PID3 $PID4; return; }
    wait_for_server $PORT4 || { kill_servers $PID1 $PID2 $PID3 $PID4; return; }

    local WP=()
    for p in $PORT1 $PORT2 $PORT3 $PORT4; do warmup_server $p & WP+=($!); done
    wait "${WP[@]}"
    echo "    All ready"

    for i in "${!PROMPTS[@]}"; do
        local r1 r2 r3 r4
        r1=$(run_completion $PORT1 "${PROMPTS[$i]}" "$N_PREDICT")
        r2=$(run_completion $PORT2 "${PROMPTS[$i]}" "$N_PREDICT")
        r3=$(run_completion $PORT3 "${PROMPTS[$i]}" "$N_PREDICT")
        r4=$(run_completion $PORT4 "${PROMPTS[$i]}" "$N_PREDICT")

        echo "$model_name,$config,q0a,48,node0a,$spec_label,$i,$r1" >> "$RESULTS_FILE"
        echo "$model_name,$config,q0b,48,node0b,$spec_label,$i,$r2" >> "$RESULTS_FILE"
        echo "$model_name,$config,q1a,48,node1a,$spec_label,$i,$r3" >> "$RESULTS_FILE"
        echo "$model_name,$config,q1b,48,node1b,$spec_label,$i,$r4" >> "$RESULTS_FILE"

        local t1 t2 t3 t4 agg
        t1=$(echo "$r1" | cut -d, -f3); t2=$(echo "$r2" | cut -d, -f3)
        t3=$(echo "$r3" | cut -d, -f3); t4=$(echo "$r4" | cut -d, -f3)
        agg=$(python3 -c "print(f'{$t1 + $t2 + $t3 + $t4:.2f}')")
        echo "    prompt $i: q0a=${t1}, q0b=${t2}, q1a=${t3}, q1b=${t4}, agg=${agg} t/s"
    done

    kill_servers $PID1 $PID2 $PID3 $PID4
}

# ============================================================
# Model 1: Qwen3-Coder-30B-A3B Q4_K_M (frontdoor, MoE, 16GB)
# ============================================================
echo "================================================================"
echo "=== MODEL 1: Qwen3-Coder-30B-A3B Q4_K_M (frontdoor, 16 GB) ==="
echo "================================================================"

M1_TARGET="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
M1_DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

bench_single "30B-A3B" "A_192t" "$M1_TARGET" "$M1_DRAFTER" 192 "all" "--draft-max 32 --kv-unified" "linear_dm32"
bench_single "30B-A3B" "B_96t_node0" "$M1_TARGET" "$M1_DRAFTER" 96 "$NODE0_CPUS" "--draft-max 32 --kv-unified" "linear_dm32"
bench_quad   "30B-A3B" "D_4x48t" "$M1_TARGET" "$M1_DRAFTER" "--draft-max 32 --kv-unified" "linear_dm32"
echo ""

# ============================================================
# Model 2: Qwen3-235B-A22B Q4_K_M (architect_general, MoE, ~130GB)
# Too large for 4-way (4×130=520GB, tight), test 1×192 vs 1×96
# ============================================================
echo "================================================================"
echo "=== MODEL 2: Qwen3-235B-A22B Q4_K_M (architect, ~130 GB)    ==="
echo "================================================================"

M2_TARGET="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf"
M2_DRAFTER="/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf"

bench_single "235B-A22B" "A_192t" "$M2_TARGET" "$M2_DRAFTER" 192 "all" "--draft-max 32 --kv-unified" "linear_dm32"
bench_single "235B-A22B" "B_96t_node0" "$M2_TARGET" "$M2_DRAFTER" 96 "$NODE0_CPUS" "--draft-max 32 --kv-unified" "linear_dm32"
echo ""

# ============================================================
# Model 3: Qwen3-Coder-480B-A35B Q4_K_M (architect_coding, MoE, ~250GB)
# Way too large for multi-instance. Test 1×192 vs 1×96 only.
# ============================================================
echo "================================================================"
echo "=== MODEL 3: Qwen3-Coder-480B-A35B Q4_K_M (coding, ~250 GB) ==="
echo "================================================================"

M3_TARGET="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
M3_DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

bench_single "480B-A35B" "A_192t" "$M3_TARGET" "$M3_DRAFTER" 192 "all" "--draft-max 48 --kv-unified" "linear_dm48"
bench_single "480B-A35B" "B_96t_node0" "$M3_TARGET" "$M3_DRAFTER" 96 "$NODE0_CPUS" "--draft-max 48 --kv-unified" "linear_dm48"
echo ""

# ============================================================
# Summary
# ============================================================
echo "=== FULL SUMMARY ==="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
