#!/bin/bash
# NUMA Production Model Sweep
#
# Deterministic NUMA throughput characterization for production models.
#
# HARDWARE: AMD EPYC 9655, 96 physical cores (192 logical), 2 NUMA nodes, 1.1TB RAM
#
# NUMA LAUNCH RULES (learned the hard way 2026-04-17):
#   1. ALWAYS use --mlock (prevents page eviction, forces private resident pages)
#   2. Single-instance uses numactl --interleave=all (spread across both nodes)
#   3. Multi-instance uses numactl --cpunodebind=N --membind=N (pin CPU AND memory)
#   4. Quarter-machine uses numactl --membind=N + taskset -c <cpus> (membind to parent node)
#   5. NEVER use bare taskset — it pins CPU but not memory, causing cross-NUMA thrash
#   6. Always use 96 threads (physical cores only, SMT adds contention on bandwidth-bound workloads)
#   7. Multi-instance: load SEQUENTIALLY (concurrent mlock crashes)
#
# CONFIGS:
#   A: 1×96t, numactl --interleave=all     (baseline, best single-request latency)
#   B: 1×96t, numactl --cpunodebind=0      (single NUMA node)
#   C: 2×96t, one per NUMA node            (2x throughput if model fits per-node)
#   D: 4×48t, quarter-machine              (4x throughput if model fits per-quarter)
#
# USAGE:
#   bash bench_numa_production_sweep.sh              # run all models
#   bash bench_numa_production_sweep.sh --model 4    # run only model 4
#
# OUTPUT: CSV with incremental writes (safe to kill mid-run, no data loss)

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_production"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_prod_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=256
BASE_PORT=8190

# NUMA topology
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
)

# Parse args
ONLY_MODEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) ONLY_MODEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "NUMA Production Model Sweep"
echo "============================"
echo "n_predict=$N_PREDICT"
echo "Results: $RESULTS_FILE"
echo ""

echo "model,config,instance,threads,cpu_binding,spec,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

# ============================================================
# Helpers
# ============================================================

wait_for_server() {
    local port=$1
    local max_wait=600
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "    ERROR: server on port $port did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "    port $port ready (${elapsed}s)"
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
        tps=$(python3 -c "print(f'{int($tokens) / (int($elapsed_ms) / 1000):.2f}')")
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
    curl -s --max-time 120 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1
}

run_prompts() {
    local port=$1
    local model_name=$2
    local config=$3
    local instance=$4
    local threads=$5
    local cpu_binding=$6
    local spec_label=$7

    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion "$port" "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$model_name,$config,$instance,$threads,$cpu_binding,$spec_label,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "    prompt $i: ${tps} t/s"
    done
}

# ============================================================
# Config A: 1×96t, interleave across both NUMA nodes
# Best single-request latency — memory spread across both controllers
# ============================================================
bench_config_A() {
    local model_name=$1 target=$2 extra_args=$3 spec_label=$4
    local drafter=${5:-}

    echo "  --- Config A: 1×96t interleave ---"

    numactl --interleave=all "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 96 -np 1 --port $BASE_PORT -ngl 0 --mlock --metrics \
        > "$LOG_DIR/${model_name}_A.log" 2>&1 &
    local PID=$!

    if ! wait_for_server $BASE_PORT; then
        echo "    Config A FAILED"
        kill_servers $PID
        return
    fi
    warmup_server $BASE_PORT
    run_prompts $BASE_PORT "$model_name" "A_1x96t_interleave" "1" "96" "interleave" "$spec_label"
    kill_servers $PID
}

# ============================================================
# Config B: 1×96t, pinned to NUMA node 0 (CPU + memory)
# Tests single-node bandwidth vs interleave
# ============================================================
bench_config_B() {
    local model_name=$1 target=$2 extra_args=$3 spec_label=$4
    local drafter=${5:-}

    echo "  --- Config B: 1×96t node0 ---"

    numactl --cpunodebind=0 --membind=0 "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 96 -np 1 --port $BASE_PORT -ngl 0 --mlock --metrics \
        > "$LOG_DIR/${model_name}_B.log" 2>&1 &
    local PID=$!

    if ! wait_for_server $BASE_PORT; then
        echo "    Config B FAILED"
        kill_servers $PID
        return
    fi
    warmup_server $BASE_PORT
    run_prompts $BASE_PORT "$model_name" "B_1x96t_node0" "1" "96" "node0" "$spec_label"
    kill_servers $PID
}

# ============================================================
# Config C: 2×96t, one per NUMA node (sequential load)
# Each instance gets its own node's CPU + memory
# ============================================================
bench_config_C() {
    local model_name=$1 target=$2 extra_args=$3 spec_label=$4
    local drafter=${5:-}

    echo "  --- Config C: 2×96t dual-node ---"

    local PORT1=$BASE_PORT
    local PORT2=$((BASE_PORT + 1))

    # Instance 1: node0
    numactl --cpunodebind=0 --membind=0 "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 96 -np 1 --port $PORT1 -ngl 0 --mlock --metrics \
        > "$LOG_DIR/${model_name}_C_n0.log" 2>&1 &
    local PID1=$!
    echo "    loading instance 1 (node0)..."
    if ! wait_for_server $PORT1; then
        echo "    Config C instance 1 FAILED"
        kill_servers $PID1
        return
    fi

    # Instance 2: node1 (sequential — wait for instance 1 first)
    numactl --cpunodebind=1 --membind=1 "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
        -t 96 -np 1 --port $PORT2 -ngl 0 --mlock --metrics \
        > "$LOG_DIR/${model_name}_C_n1.log" 2>&1 &
    local PID2=$!
    echo "    loading instance 2 (node1)..."
    if ! wait_for_server $PORT2; then
        echo "    Config C instance 2 FAILED"
        kill_servers $PID1 $PID2
        return
    fi

    warmup_server $PORT1 & warmup_server $PORT2 & wait
    echo "    both ready"

    for i in "${!PROMPTS[@]}"; do
        local r1 r2
        r1=$(run_completion $PORT1 "${PROMPTS[$i]}" "$N_PREDICT")
        r2=$(run_completion $PORT2 "${PROMPTS[$i]}" "$N_PREDICT")

        echo "$model_name,C_2x96t,n0,96,node0,$spec_label,$i,$r1" >> "$RESULTS_FILE"
        echo "$model_name,C_2x96t,n1,96,node1,$spec_label,$i,$r2" >> "$RESULTS_FILE"

        local t1 t2 agg
        t1=$(echo "$r1" | cut -d, -f3); t2=$(echo "$r2" | cut -d, -f3)
        agg=$(python3 -c "print(f'{float($t1) + float($t2):.2f}')")
        echo "    prompt $i: n0=${t1}, n1=${t2}, agg=${agg} t/s"
    done

    kill_servers $PID1 $PID2
}

# ============================================================
# Config D: 4×48t, quarter-machine (sequential load)
# membind to parent NUMA node, taskset to CPU quarter
# ============================================================
bench_config_D() {
    local model_name=$1 target=$2 extra_args=$3 spec_label=$4
    local drafter=${5:-}

    echo "  --- Config D: 4×48t quarter-machine ---"

    local PORT1=$BASE_PORT
    local PORT2=$((BASE_PORT + 1))
    local PORT3=$((BASE_PORT + 2))
    local PORT4=$((BASE_PORT + 3))

    local QUARTER_CPUS=("$NODE0A_CPUS" "$NODE0B_CPUS" "$NODE1A_CPUS" "$NODE1B_CPUS")
    local QUARTER_MEMBIND=(0 0 1 1)
    local QUARTER_PORTS=($PORT1 $PORT2 $PORT3 $PORT4)
    local QUARTER_NAMES=(n0a n0b n1a n1b)
    local QUARTER_PIDS=()

    # Sequential loading
    for q in 0 1 2 3; do
        numactl --membind="${QUARTER_MEMBIND[$q]}" taskset -c "${QUARTER_CPUS[$q]}" \
            "$LLAMA_SERVER" -m "$target" ${drafter:+-md "$drafter"} $extra_args \
            -t 48 -np 1 --port "${QUARTER_PORTS[$q]}" -ngl 0 --mlock --metrics \
            > "$LOG_DIR/${model_name}_D_${QUARTER_NAMES[$q]}.log" 2>&1 &
        QUARTER_PIDS+=($!)
        echo "    loading instance $((q+1)) (${QUARTER_NAMES[$q]})..."
        if ! wait_for_server "${QUARTER_PORTS[$q]}"; then
            echo "    Config D instance $((q+1)) FAILED"
            kill_servers "${QUARTER_PIDS[@]}"
            return
        fi
    done

    for p in "${QUARTER_PORTS[@]}"; do warmup_server "$p" & done; wait
    echo "    all ready"

    for i in "${!PROMPTS[@]}"; do
        local TPS_PARTS=""
        for q in 0 1 2 3; do
            local r
            r=$(run_completion "${QUARTER_PORTS[$q]}" "${PROMPTS[$i]}" "$N_PREDICT")
            echo "$model_name,D_4x48t,${QUARTER_NAMES[$q]},48,${QUARTER_NAMES[$q]},$spec_label,$i,$r" >> "$RESULTS_FILE"
            local t
            t=$(echo "$r" | cut -d, -f3)
            TPS_PARTS="${TPS_PARTS} ${QUARTER_NAMES[$q]}=${t}"
        done
        echo "    prompt $i:${TPS_PARTS}"
    done

    kill_servers "${QUARTER_PIDS[@]}"
}

# ============================================================
# MODEL DEFINITIONS
# Add new models here. Each model specifies which configs to run.
# ============================================================

# --- Model 1: Qwen3-Coder-30B-A3B Q4_K_M (frontdoor, MoE, ~16GB) ---
run_model_1() {
    echo "================================================================"
    echo "=== MODEL 1: Qwen3-Coder-30B-A3B Q4_K_M (frontdoor, 16 GB) ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    local DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
    local ARGS="--draft-max 32 --kv-unified"
    local LABEL="linear_dm32"

    bench_config_A "30B-A3B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    bench_config_B "30B-A3B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    bench_config_D "30B-A3B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    echo ""
}

# --- Model 2: Qwen3-235B-A22B Q4_K_M (architect_general, MoE, ~130GB) ---
run_model_2() {
    echo "================================================================"
    echo "=== MODEL 2: Qwen3-235B-A22B Q4_K_M (architect, ~130 GB)    ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf"
    local DRAFTER="/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf"
    local ARGS="--draft-max 32 --kv-unified"
    local LABEL="linear_dm32"

    bench_config_A "235B-A22B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    bench_config_B "235B-A22B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    bench_config_C "235B-A22B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    echo ""
}

# --- Model 3: Qwen3-Coder-480B-A35B Q4_K_M (architect_coding, MoE, ~250GB) ---
run_model_3() {
    echo "================================================================"
    echo "=== MODEL 3: Qwen3-Coder-480B-A35B Q4_K_M (coding, ~250 GB) ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
    local DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
    local ARGS="--draft-max 48 --kv-unified"
    local LABEL="linear_dm48"

    bench_config_A "480B-A35B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    bench_config_B "480B-A35B" "$TARGET" "$ARGS" "$LABEL" "$DRAFTER"
    # Too large for C or D (250GB × 2 = 500GB per node, won't fit in ~560GB with KV)
    echo ""
}

# --- Model 4: MiniMax-M2.7 UD-Q4_K_XL (MoE 230B-A10B, ~132GB) ---
run_model_4() {
    echo "================================================================"
    echo "=== MODEL 4: MiniMax-M2.7 UD-Q4_K_XL (MoE 230B-A10B, ~132 GB) ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/models/MiniMax-M2.7-GGUF/UD-Q4_K_XL/MiniMax-M2.7-UD-Q4_K_XL-00001-of-00004.gguf"
    local ARGS="--spec-type ngram-simple --draft-max 64"
    local LABEL="ngram_dm64"

    bench_config_A "M2.7-Q4XL" "$TARGET" "$ARGS" "$LABEL"
    bench_config_B "M2.7-Q4XL" "$TARGET" "$ARGS" "$LABEL"
    bench_config_C "M2.7-Q4XL" "$TARGET" "$ARGS" "$LABEL"
    bench_config_D "M2.7-Q4XL" "$TARGET" "$ARGS" "$LABEL"
    echo ""
}

# --- Model 5: MiniMax-M2.7 Q8_0 (MoE 230B-A10B, ~227GB) ---
run_model_5() {
    echo "================================================================"
    echo "=== MODEL 5: MiniMax-M2.7 Q8_0 (MoE 230B-A10B, ~227 GB)    ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/models/MiniMax-M2.7-GGUF/Q8_0/MiniMax-M2.7-Q8_0-00001-of-00006.gguf"
    local ARGS="--spec-type ngram-simple --draft-max 64"
    local LABEL="ngram_dm64"

    bench_config_A "M2.7-Q8" "$TARGET" "$ARGS" "$LABEL"
    bench_config_B "M2.7-Q8" "$TARGET" "$ARGS" "$LABEL"
    bench_config_C "M2.7-Q8" "$TARGET" "$ARGS" "$LABEL"
    # D skipped: 4×227 = 908GB, too tight with KV overhead
    echo ""
}

# --- Model 6: Qwen3.6-35B-A3B Q4_K_M (~22GB) ---
run_model_6() {
    echo "================================================================"
    echo "=== MODEL 6: Qwen3.6-35B-A3B Q4_K_M (~22 GB)               ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
    local ARGS=""
    local LABEL="baseline"

    bench_config_A "Qwen3.6-Q4" "$TARGET" "$ARGS" "$LABEL"
    bench_config_B "Qwen3.6-Q4" "$TARGET" "$ARGS" "$LABEL"
    bench_config_C "Qwen3.6-Q4" "$TARGET" "$ARGS" "$LABEL"
    bench_config_D "Qwen3.6-Q4" "$TARGET" "$ARGS" "$LABEL"
    echo ""
}

# --- Model 7: Qwen3.6-35B-A3B Q8_0 (~37GB) ---
run_model_7() {
    echo "================================================================"
    echo "=== MODEL 7: Qwen3.6-35B-A3B Q8_0 (~37 GB)                 ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf"
    local ARGS=""
    local LABEL="baseline"

    bench_config_A "Qwen3.6-Q8" "$TARGET" "$ARGS" "$LABEL"
    bench_config_B "Qwen3.6-Q8" "$TARGET" "$ARGS" "$LABEL"
    bench_config_C "Qwen3.6-Q8" "$TARGET" "$ARGS" "$LABEL"
    bench_config_D "Qwen3.6-Q8" "$TARGET" "$ARGS" "$LABEL"
    echo ""
}

# --- Model 8: Bonsai-8B (~1.2GB) ---
run_model_8() {
    echo "================================================================"
    echo "=== MODEL 8: Bonsai-8B (~1.2 GB)                           ==="
    echo "================================================================"

    local TARGET="/mnt/raid0/llm/models/Bonsai-8B.gguf"
    local ARGS=""
    local LABEL="baseline"

    bench_config_A "Bonsai-8B" "$TARGET" "$ARGS" "$LABEL"
    bench_config_B "Bonsai-8B" "$TARGET" "$ARGS" "$LABEL"
    bench_config_C "Bonsai-8B" "$TARGET" "$ARGS" "$LABEL"
    bench_config_D "Bonsai-8B" "$TARGET" "$ARGS" "$LABEL"
    echo ""
}

# ============================================================
# Run selected or all models
# ============================================================

if [ -n "$ONLY_MODEL" ]; then
    run_model_"$ONLY_MODEL"
else
    run_model_1
    run_model_2
    run_model_3
    run_model_4
    run_model_5
    run_model_6
    run_model_7
    run_model_8
fi

# ============================================================
# Summary
# ============================================================
echo "=== FULL SUMMARY ==="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
