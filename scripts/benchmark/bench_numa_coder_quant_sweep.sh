#!/bin/bash
# NUMA Coder Quant Throughput Sweep
#
# Tests 4x48t NUMA throughput on all 3 coder variants with optimal spec configs.
# For each variant: 1x192t baseline, then 4x48t NUMA quarters.
# Tree spec tested on f16 and Q8_0 only (net-negative on Q4_K_M).
#
# Variants:
#   - f16 (65 GB x 4 = 260 GB)
#   - Q8_0 (33 GB x 4 = 132 GB)
#   - Q4_K_M (18.5 GB x 4 = 74 GB)
#
# Draft: Qwen2.5-Coder-0.5B-Q8_0
# Output: data/numa_coder_quant/

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_coder_quant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=128
BASE_PORT=8180

# NUMA bindings
NODE0A_CPUS="0-23,96-119"
NODE0B_CPUS="24-47,120-143"
NODE1A_CPUS="48-71,144-167"
NODE1B_CPUS="72-95,168-191"

# Models
CODER_F16="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf"
CODER_Q8="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
CODER_Q4="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Optimal spec configs (UPDATE after bench_sweep_coder_spec_params.sh results)
# NUMA-mode values (48t quarter)
F16_DM=24;  F16_PS=0.3
Q8_DM=24;   Q8_PS=0.3
Q4_DM=24;   Q4_PS=0

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Design a microservice architecture for a real-time chat application with message persistence and delivery guarantees:"
    "Write a Rust implementation of a lock-free concurrent queue using compare-and-swap operations:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

RESULTS_FILE="${DATA_DIR}/numa_coder_quant_${TIMESTAMP}.csv"
echo "variant,config,instance,threads,cpu_binding,spec,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

echo "NUMA Coder Quant Throughput Sweep"
echo "=================================="
echo "n_predict=$N_PREDICT"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# Helper Functions
# ============================================================

wait_for_server() {
    local port=$1
    local max_wait=${2:-600}
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

warmup_server() {
    local port=$1
    curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1
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
    for pid in "$@"; do kill -9 "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    # Kill any remaining servers on benchmark ports
    for p in $BASE_PORT $((BASE_PORT+1)) $((BASE_PORT+2)) $((BASE_PORT+3)); do
        local pids
        pids=$(lsof -ti :$p 2>/dev/null || true)
        [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
    done
    sleep 2
}

# 1x192t single-instance benchmark
bench_single() {
    local variant=$1
    local config=$2
    local target=$3
    local dm=$4
    local ps=$5
    local spec_label=$6

    local port=$BASE_PORT

    local spec_args="--draft-max $dm --lookup"
    if [ "$ps" != "0" ]; then
        spec_args="$spec_args --kv-unified --draft-p-split $ps"
    fi

    echo "  --- $config (192 threads, all cores) ---"
    "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args \
        -t 192 -np 1 --port $port -ngl 0 --metrics \
        > "$LOG_DIR/${variant}_${config}.log" 2>&1 &
    local PID=$!

    if ! wait_for_server $port; then
        echo "    FAILED to start"
        kill_servers $PID
        return
    fi
    warmup_server $port

    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $port "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$variant,$config,1,192,all,$spec_label,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "    prompt $i: ${tps} t/s"
    done

    kill_servers $PID
}

# 4x48t NUMA quarter benchmark
bench_quad() {
    local variant=$1
    local config=$2
    local target=$3
    local dm=$4
    local ps=$5
    local spec_label=$6

    echo "  --- $config (4x48 threads, quarter-machine) ---"

    local PORT1=$BASE_PORT
    local PORT2=$((BASE_PORT + 1))
    local PORT3=$((BASE_PORT + 2))
    local PORT4=$((BASE_PORT + 3))

    local spec_args="--draft-max $dm --lookup"
    if [ "$ps" != "0" ]; then
        spec_args="$spec_args --kv-unified --draft-p-split $ps"
    fi

    taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args \
        -t 48 -np 1 --port $PORT1 -ngl 0 --metrics > "$LOG_DIR/${variant}_${config}_n0a.log" 2>&1 &
    local PID1=$!
    taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args \
        -t 48 -np 1 --port $PORT2 -ngl 0 --metrics > "$LOG_DIR/${variant}_${config}_n0b.log" 2>&1 &
    local PID2=$!
    taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args \
        -t 48 -np 1 --port $PORT3 -ngl 0 --metrics > "$LOG_DIR/${variant}_${config}_n1a.log" 2>&1 &
    local PID3=$!
    taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args \
        -t 48 -np 1 --port $PORT4 -ngl 0 --metrics > "$LOG_DIR/${variant}_${config}_n1b.log" 2>&1 &
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

        echo "$variant,$config,q0a,48,node0a,$spec_label,$i,$r1" >> "$RESULTS_FILE"
        echo "$variant,$config,q0b,48,node0b,$spec_label,$i,$r2" >> "$RESULTS_FILE"
        echo "$variant,$config,q1a,48,node1a,$spec_label,$i,$r3" >> "$RESULTS_FILE"
        echo "$variant,$config,q1b,48,node1b,$spec_label,$i,$r4" >> "$RESULTS_FILE"

        local t1 t2 t3 t4 agg
        t1=$(echo "$r1" | cut -d, -f3); t2=$(echo "$r2" | cut -d, -f3)
        t3=$(echo "$r3" | cut -d, -f3); t4=$(echo "$r4" | cut -d, -f3)
        agg=$(python3 -c "print(f'{$t1 + $t2 + $t3 + $t4:.2f}')")
        echo "    prompt $i: q0a=${t1}, q0b=${t2}, q1a=${t3}, q1b=${t4}, agg=${agg} t/s"
    done

    kill_servers $PID1 $PID2 $PID3 $PID4
}

# ============================================================
# VARIANT 1: f16 (65 GB)
# ============================================================
echo "================================================================"
echo "=== VARIANT 1: Coder 32B f16 (65 GB x 4 = 260 GB)           ==="
echo "================================================================"

bench_single "f16" "A_192t_spec" "$CODER_F16" "$F16_DM" "$F16_PS" "spec_dm${F16_DM}_ps${F16_PS}"
bench_quad   "f16" "D_4x48t_spec" "$CODER_F16" "$F16_DM" "$F16_PS" "spec_dm${F16_DM}_ps${F16_PS}"

# Compare tree vs linear
if [ "$F16_PS" != "0" ]; then
    bench_quad "f16" "D_4x48t_linear" "$CODER_F16" "$F16_DM" "0" "linear_dm${F16_DM}"
fi
echo ""

# ============================================================
# VARIANT 2: Q8_0 (33 GB)
# ============================================================
echo "================================================================"
echo "=== VARIANT 2: Coder 32B Q8_0 (33 GB x 4 = 132 GB)         ==="
echo "================================================================"

bench_single "q8" "A_192t_spec" "$CODER_Q8" "$Q8_DM" "$Q8_PS" "spec_dm${Q8_DM}_ps${Q8_PS}"
bench_quad   "q8" "D_4x48t_spec" "$CODER_Q8" "$Q8_DM" "$Q8_PS" "spec_dm${Q8_DM}_ps${Q8_PS}"

if [ "$Q8_PS" != "0" ]; then
    bench_quad "q8" "D_4x48t_linear" "$CODER_Q8" "$Q8_DM" "0" "linear_dm${Q8_DM}"
fi
echo ""

# ============================================================
# VARIANT 3: Q4_K_M (18.5 GB)
# ============================================================
echo "================================================================"
echo "=== VARIANT 3: Coder 32B Q4_K_M (18.5 GB x 4 = 74 GB)      ==="
echo "================================================================"

bench_single "q4km" "A_192t_spec" "$CODER_Q4" "$Q4_DM" "$Q4_PS" "spec_dm${Q4_DM}_ps${Q4_PS}"
bench_quad   "q4km" "D_4x48t_spec" "$CODER_Q4" "$Q4_DM" "$Q4_PS" "spec_dm${Q4_DM}_ps${Q4_PS}"
echo ""

# ============================================================
# Summary
# ============================================================
echo "================================================================"
echo "=== NUMA CODER QUANT SWEEP COMPLETE ==="
echo "================================================================"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
echo ""

export RESULTS_FILE
python3 << 'PYEOF'
import csv, sys, os

results_file = os.environ.get("RESULTS_FILE", "")
if not results_file or not os.path.exists(results_file):
    print("No results file found")
    sys.exit()

rows = []
with open(results_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

# Group by (variant, config)
groups = {}
for row in rows:
    key = (row["variant"], row["config"])
    tps = float(row["tokens_per_sec"])
    instance = row["instance"]
    prompt_idx = row["prompt_idx"]
    groups.setdefault(key, {}).setdefault(prompt_idx, {})[instance] = tps

print("\n=== Throughput Summary ===\n")
print(f"{'Variant':<10} {'Config':<20} {'Per-Instance':>14} {'Aggregate':>12}")
print("-" * 60)

for (variant, config) in sorted(groups.keys()):
    prompts = groups[(variant, config)]
    if any("q0a" in instances for instances in prompts.values()):
        # Multi-instance: compute per-instance avg and aggregate
        all_per_instance = []
        all_aggregate = []
        for pidx, instances in prompts.items():
            inst_vals = [v for k, v in instances.items() if v > 0]
            if inst_vals:
                all_per_instance.extend(inst_vals)
                all_aggregate.append(sum(inst_vals))
        avg_per = sum(all_per_instance) / len(all_per_instance) if all_per_instance else 0
        avg_agg = sum(all_aggregate) / len(all_aggregate) if all_aggregate else 0
        print(f"{variant:<10} {config:<20} {avg_per:>12.2f}   {avg_agg:>10.2f}")
    else:
        # Single instance
        all_tps = [v for instances in prompts.values() for v in instances.values() if v > 0]
        avg = sum(all_tps) / len(all_tps) if all_tps else 0
        print(f"{variant:<10} {config:<20} {avg:>12.2f}   {'n/a':>10}")

print()
PYEOF
