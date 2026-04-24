#!/bin/bash
# NUMA Sweep: MiniMax M2.7 (Q8_0 + Q4_K_XL) + SuperGemma4 (31b + 26b)
#
# Tests NUMA pinning impact for 4 newly downloaded models.
# Configs per model:
#   - 1×192t interleave (baseline)
#   - 1×96t single-node
#   - 2×96t dual-node (where model fits 2×)
#   - 4×48t quarter-machine (where model fits 4×)
#
# All models use --spec-type ngram-simple (no draft model).
# Models are run sequentially — one at a time.

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_new_models"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/numa_new_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

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
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "NUMA New Models Sweep"
echo "====================="
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
    local threads=$4
    local cpus=$5
    local extra_args=$6
    local spec_label=$7

    local port=$BASE_PORT
    local log_name="${model_name}_${config}"

    echo "  --- $config ($threads threads, $cpus) ---"

    if [ "$cpus" = "all" ]; then
        numactl --interleave=all "$LLAMA_SERVER" -m "$target" $extra_args \
            -t "$threads" -np 1 --port $port -ngl 0 --metrics \
            > "$LOG_DIR/${log_name}.log" 2>&1 &
    else
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$target" $extra_args \
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

# Run a 2-way benchmark
bench_dual() {
    local model_name=$1
    local config=$2
    local target=$3
    local extra_args=$4
    local spec_label=$5

    echo "  --- $config (2×96 threads, dual-node) ---"

    local PORT1=$BASE_PORT
    local PORT2=$((BASE_PORT + 1))

    taskset -c "$NODE0_CPUS" "$LLAMA_SERVER" -m "$target" $extra_args \
        -t 96 -np 1 --port $PORT1 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n0.log" 2>&1 &
    local PID1=$!
    echo "    Loading instance 1 (node0)..."
    wait_for_server $PORT1 || { kill_servers $PID1; return; }

    taskset -c "$NODE1_CPUS" "$LLAMA_SERVER" -m "$target" $extra_args \
        -t 96 -np 1 --port $PORT2 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n1.log" 2>&1 &
    local PID2=$!
    echo "    Loading instance 2 (node1)..."
    wait_for_server $PORT2 || { kill_servers $PID1 $PID2; return; }

    warmup_server $PORT1 & warmup_server $PORT2 & wait
    echo "    Both ready"

    for i in "${!PROMPTS[@]}"; do
        local r1 r2
        r1=$(run_completion $PORT1 "${PROMPTS[$i]}" "$N_PREDICT")
        r2=$(run_completion $PORT2 "${PROMPTS[$i]}" "$N_PREDICT")

        echo "$model_name,$config,n0,96,node0,$spec_label,$i,$r1" >> "$RESULTS_FILE"
        echo "$model_name,$config,n1,96,node1,$spec_label,$i,$r2" >> "$RESULTS_FILE"

        local t1 t2 agg
        t1=$(echo "$r1" | cut -d, -f3); t2=$(echo "$r2" | cut -d, -f3)
        agg=$(python3 -c "print(f'{$t1 + $t2:.2f}')")
        echo "    prompt $i: n0=${t1}, n1=${t2}, agg=${agg} t/s"
    done

    kill_servers $PID1 $PID2
}

# Run a 4-way benchmark
bench_quad() {
    local model_name=$1
    local config=$2
    local target=$3
    local extra_args=$4
    local spec_label=$5

    echo "  --- $config (4×48 threads, quarter-machine) ---"

    local PORT1=$BASE_PORT
    local PORT2=$((BASE_PORT + 1))
    local PORT3=$((BASE_PORT + 2))
    local PORT4=$((BASE_PORT + 3))

    taskset -c "$NODE0A_CPUS" "$LLAMA_SERVER" -m "$target" $extra_args \
        -t 48 -np 1 --port $PORT1 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n0a.log" 2>&1 &
    local PID1=$!
    echo "    Loading instance 1 (node0a)..."
    wait_for_server $PORT1 || { kill_servers $PID1; return; }

    taskset -c "$NODE0B_CPUS" "$LLAMA_SERVER" -m "$target" $extra_args \
        -t 48 -np 1 --port $PORT2 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n0b.log" 2>&1 &
    local PID2=$!
    echo "    Loading instance 2 (node0b)..."
    wait_for_server $PORT2 || { kill_servers $PID1 $PID2; return; }

    taskset -c "$NODE1A_CPUS" "$LLAMA_SERVER" -m "$target" $extra_args \
        -t 48 -np 1 --port $PORT3 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n1a.log" 2>&1 &
    local PID3=$!
    echo "    Loading instance 3 (node1a)..."
    wait_for_server $PORT3 || { kill_servers $PID1 $PID2 $PID3; return; }

    taskset -c "$NODE1B_CPUS" "$LLAMA_SERVER" -m "$target" $extra_args \
        -t 48 -np 1 --port $PORT4 -ngl 0 --metrics > "$LOG_DIR/${model_name}_${config}_n1b.log" 2>&1 &
    local PID4=$!
    echo "    Loading instance 4 (node1b)..."
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
# Model paths
# ============================================================

M27_Q8="/mnt/raid0/llm/models/MiniMax-M2.7-GGUF/Q8_0/MiniMax-M2.7-Q8_0-00001-of-00006.gguf"
M27_Q4="/mnt/raid0/llm/models/MiniMax-M2.7-GGUF/UD-Q4_K_XL/MiniMax-M2.7-UD-Q4_K_XL-00001-of-00004.gguf"
SG31="/mnt/raid0/llm/models/SuperGemma4-31b-abliterated.Q4_K_M.gguf"
SG26="/mnt/raid0/llm/models/supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf"

DRAFT_MAX_VALUES=(16 32 48 64 96 128)

# ============================================================
# PHASE 1: draft-max sweep at 1×192t interleave per model
# Find optimal draft-max, then use it in Phase 2 NUMA sweep.
# ============================================================

if [ "${1:-}" != "--phase2-only" ]; then

echo ""
echo "================================================================"
echo "=== PHASE 1: draft-max sweep (1×192t interleave, ngram-simple) ==="
echo "================================================================"
echo ""

draft_max_sweep() {
    local model_name=$1
    local target=$2

    for dm in "${DRAFT_MAX_VALUES[@]}"; do
        local dm_args="--spec-type ngram-simple --draft-max $dm"
        local config="dm${dm}_192t"
        echo "  --- $model_name draft-max=$dm ---"

        local port=$BASE_PORT
        local log_name="${model_name}_dmsweep_${dm}"

        numactl --interleave=all "$LLAMA_SERVER" -m "$target" $dm_args \
            -t 192 -np 1 --port $port -ngl 0 --metrics \
            > "$LOG_DIR/${log_name}.log" 2>&1 &
        local PID=$!

        if ! wait_for_server $port; then
            echo "    FAILED to start"
            kill_servers $PID
            continue
        fi
        warmup_server $port

        for i in "${!PROMPTS[@]}"; do
            result=$(run_completion $port "${PROMPTS[$i]}" "$N_PREDICT")
            echo "$model_name,$config,1,192,all,ngram_dm${dm},$i,$result" >> "$RESULTS_FILE"
            tps=$(echo "$result" | cut -d, -f3)
            echo "    dm=$dm prompt $i: ${tps} t/s"
        done

        kill_servers $PID
    done
}

echo "--- M2.7 Q8_0 draft-max sweep ---"
draft_max_sweep "M2.7-Q8" "$M27_Q8"
echo ""

echo "--- M2.7 Q4_K_XL draft-max sweep ---"
draft_max_sweep "M2.7-Q4XL" "$M27_Q4"
echo ""

echo "--- SuperGemma4-31b draft-max sweep ---"
draft_max_sweep "SG4-31b" "$SG31"
echo ""

echo "--- SuperGemma4-26b draft-max sweep ---"
draft_max_sweep "SG4-26b" "$SG26"
echo ""

echo "================================================================"
echo "=== PHASE 1 COMPLETE — check results to pick best draft-max  ==="
echo "=== per model, then set BEST_DM_* below and run --phase2-only ==="
echo "================================================================"

fi  # end phase1

# ============================================================
# PHASE 2: NUMA sweep with best draft-max per model
# Set these after Phase 1 results. Defaults to 64 if not tuned.
# ============================================================

BEST_DM_M27_Q8=${BEST_DM_M27_Q8:-64}
BEST_DM_M27_Q4=${BEST_DM_M27_Q4:-64}
BEST_DM_SG31=${BEST_DM_SG31:-64}
BEST_DM_SG26=${BEST_DM_SG26:-64}

if [ "${1:-}" != "--phase1-only" ]; then

echo ""
echo "================================================================"
echo "=== PHASE 2: NUMA sweep (best draft-max per model)            ==="
echo "=== M2.7-Q8: dm=$BEST_DM_M27_Q8, M2.7-Q4XL: dm=$BEST_DM_M27_Q4 ==="
echo "=== SG4-31b: dm=$BEST_DM_SG31, SG4-26b: dm=$BEST_DM_SG26       ==="
echo "================================================================"
echo ""

# Model 1: MiniMax M2.7 Q8_0 (~227 GB) — 1×192, 1×96, 2×96
echo "================================================================"
echo "=== MODEL 1: MiniMax-M2.7 Q8_0 (MoE 230B-A10B, ~227 GB)    ==="
echo "================================================================"
M27Q8_ARGS="--spec-type ngram-simple --draft-max $BEST_DM_M27_Q8"

bench_single "M2.7-Q8" "A_192t_interleave" "$M27_Q8" 192 "all" "$M27Q8_ARGS" "ngram_dm${BEST_DM_M27_Q8}"
bench_single "M2.7-Q8" "B_96t_node0" "$M27_Q8" 96 "$NODE0_CPUS" "$M27Q8_ARGS" "ngram_dm${BEST_DM_M27_Q8}"
bench_dual   "M2.7-Q8" "C_2x96t" "$M27_Q8" "$M27Q8_ARGS" "ngram_dm${BEST_DM_M27_Q8}"
echo ""

# Model 2: MiniMax M2.7 UD-Q4_K_XL (~132 GB) — all configs
echo "================================================================"
echo "=== MODEL 2: MiniMax-M2.7 UD-Q4_K_XL (MoE 230B-A10B, ~132 GB) ==="
echo "================================================================"
M27Q4_ARGS="--spec-type ngram-simple --draft-max $BEST_DM_M27_Q4"

bench_single "M2.7-Q4XL" "A_192t_interleave" "$M27_Q4" 192 "all" "$M27Q4_ARGS" "ngram_dm${BEST_DM_M27_Q4}"
bench_single "M2.7-Q4XL" "B_96t_node0" "$M27_Q4" 96 "$NODE0_CPUS" "$M27Q4_ARGS" "ngram_dm${BEST_DM_M27_Q4}"
bench_dual   "M2.7-Q4XL" "C_2x96t" "$M27_Q4" "$M27Q4_ARGS" "ngram_dm${BEST_DM_M27_Q4}"
bench_quad   "M2.7-Q4XL" "D_4x48t" "$M27_Q4" "$M27Q4_ARGS" "ngram_dm${BEST_DM_M27_Q4}"
echo ""

# Model 3: SuperGemma4-31b (~18 GB) — all configs
echo "================================================================"
echo "=== MODEL 3: SuperGemma4-31b-abliterated Q4_K_M (~18 GB)    ==="
echo "================================================================"
SG31_ARGS="--spec-type ngram-simple --draft-max $BEST_DM_SG31"

bench_single "SG4-31b" "A_192t_interleave" "$SG31" 192 "all" "$SG31_ARGS" "ngram_dm${BEST_DM_SG31}"
bench_single "SG4-31b" "B_96t_node0" "$SG31" 96 "$NODE0_CPUS" "$SG31_ARGS" "ngram_dm${BEST_DM_SG31}"
bench_dual   "SG4-31b" "C_2x96t" "$SG31" "$SG31_ARGS" "ngram_dm${BEST_DM_SG31}"
bench_quad   "SG4-31b" "D_4x48t" "$SG31" "$SG31_ARGS" "ngram_dm${BEST_DM_SG31}"
echo ""

# Model 4: SuperGemma4-26b (~16 GB) — all configs
echo "================================================================"
echo "=== MODEL 4: SuperGemma4-26b-uncensored Q4_K_M (~16 GB)     ==="
echo "================================================================"
SG26_ARGS="--spec-type ngram-simple --draft-max $BEST_DM_SG26"

bench_single "SG4-26b" "A_192t_interleave" "$SG26" 192 "all" "$SG26_ARGS" "ngram_dm${BEST_DM_SG26}"
bench_single "SG4-26b" "B_96t_node0" "$SG26" 96 "$NODE0_CPUS" "$SG26_ARGS" "ngram_dm${BEST_DM_SG26}"
bench_dual   "SG4-26b" "C_2x96t" "$SG26" "$SG26_ARGS" "ngram_dm${BEST_DM_SG26}"
bench_quad   "SG4-26b" "D_4x48t" "$SG26" "$SG26_ARGS" "ngram_dm${BEST_DM_SG26}"
echo ""

fi  # end phase2

# ============================================================
# Summary
# ============================================================
echo "=== FULL SUMMARY ==="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
