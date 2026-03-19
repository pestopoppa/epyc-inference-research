#!/bin/bash
# NUMA Sweep: ALL Qwen3.5 Hybrid Models
#
# Tests NUMA pinning across remaining Qwen3.5 models:
#   - 27B Q4_K_M (16 GB, dense hybrid) — 4×48t possible
#   - 122B-A10B Q4_K_M (~69 GB, MoE hybrid) — 2×96t possible
#   - 397B-A17B Q4_K_XL (~205 GB, MoE hybrid) — single instance only
#
# For each: 1×192t vs 1×96t node0. Plus multi-instance where model size permits.
# Also tests with AR drafter (Qwen3.5-0.8B Q8_0) at best NUMA config.

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
DRAFTER="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_qwen35_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/q35_sweep_${TIMESTAMP}.csv"
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
    "Explain the theory of general relativity in detail, covering spacetime curvature:"
    "Implement a concurrent hash map in C++ using fine-grained locking:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "NUMA Sweep: ALL Qwen3.5 Hybrid Models"
echo "======================================="
echo "Results: $RESULTS_FILE"
echo ""

echo "model,config,instance,threads,cpu_binding,spec,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1 max_wait=600 elapsed=0
    while true; do
        local h; h=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "")
        echo "$h" | grep -q '"status":"ok"' && return 0
        sleep 2; elapsed=$((elapsed + 2))
        [ $elapsed -ge $max_wait ] && { echo "TIMEOUT port $port"; return 1; }
    done
}

run_completion() {
    local port=$1 prompt="$2" n_predict=$3
    local start_ms end_ms elapsed_ms tokens tps
    start_ms=$(date +%s%N | cut -b1-13)
    local response
    response=$(curl -s --max-time 600 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":${n_predict},\"temperature\":0.0,\"stream\":false}" 2>/dev/null)
    end_ms=$(date +%s%N | cut -b1-13)
    elapsed_ms=$((end_ms - start_ms))
    tokens=$(echo "$response" | python3 -c "import json,sys;
try: print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))
except: print(0)" 2>/dev/null)
    [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ] && tps=$(python3 -c "print(f'{$tokens/($elapsed_ms/1000):.2f}')") || tps="0.00"
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 2
}

warmup_server() {
    curl -s "http://localhost:${1}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"temperature":0}' > /dev/null 2>&1
}

bench_single() {
    local model_label=$1 model_path=$2 config=$3 threads=$4 cpus=$5 extra_args=$6 spec_label=$7
    echo "  --- $config ($threads threads, $cpus, $spec_label) ---"
    if [ "$cpus" = "all" ]; then
        "$LLAMA_SERVER" -m "$model_path" $extra_args -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics > "$LOG_DIR/${model_label}_${config}.log" 2>&1 &
    else
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$model_path" $extra_args -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics > "$LOG_DIR/${model_label}_${config}.log" 2>&1 &
    fi
    local PID=$!
    wait_for_server $BASE_PORT || { kill_servers $PID; return; }
    warmup_server $BASE_PORT
    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $BASE_PORT "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$model_label,$config,1,$threads,$cpus,$spec_label,$i,$result" >> "$RESULTS_FILE"
        echo "    prompt $i: $(echo "$result" | cut -d, -f3) t/s"
    done
    kill_servers $PID
}

bench_quad() {
    local model_label=$1 model_path=$2 config=$3 extra_args=$4 spec_label=$5
    echo "  --- $config (4×48t, $spec_label) ---"
    local PIDS=()
    local PORTS=($BASE_PORT $((BASE_PORT+1)) $((BASE_PORT+2)) $((BASE_PORT+3)))
    local CPUS=("$NODE0A_CPUS" "$NODE0B_CPUS" "$NODE1A_CPUS" "$NODE1B_CPUS")
    local LABELS=(node0a node0b node1a node1b)
    for i in 0 1 2 3; do
        taskset -c "${CPUS[$i]}" "$LLAMA_SERVER" -m "$model_path" $extra_args -t 48 -np 1 --port ${PORTS[$i]} -ngl 0 --metrics > "$LOG_DIR/${model_label}_${config}_${LABELS[$i]}.log" 2>&1 &
        PIDS+=($!)
    done
    echo "    Loading 4 instances..."
    for port in "${PORTS[@]}"; do
        wait_for_server $port || { kill_servers "${PIDS[@]}"; return; }
    done
    for port in "${PORTS[@]}"; do warmup_server $port; done
    echo "    All ready"
    for pi in "${!PROMPTS[@]}"; do
        local results=()
        for j in 0 1 2 3; do
            results+=($(run_completion ${PORTS[$j]} "${PROMPTS[$pi]}" "$N_PREDICT"))
            echo "$model_label,$config,${LABELS[$j]},48,${CPUS[$j]},$spec_label,$pi,${results[$j]}" >> "$RESULTS_FILE"
        done
        local agg=0
        for r in "${results[@]}"; do
            local t=$(echo "$r" | cut -d, -f3)
            agg=$(python3 -c "print(f'{$agg+$t:.2f}')")
        done
        echo "    prompt $pi: agg=$agg t/s"
    done
    kill_servers "${PIDS[@]}"
}

# ============================================================
# Model 1: Qwen3.5-27B Q4_K_M (16 GB, dense hybrid — SHOULD GET 4-WAY BENEFIT)
# ============================================================
echo "================================================================"
echo "=== Qwen3.5-27B Q4_K_M (16 GB, dense hybrid)                ==="
echo "================================================================"
M1="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
if [ -f "$M1" ]; then
    bench_single "q35-27B" "$M1" "A_192t" 192 "all" "" "no_spec"
    bench_single "q35-27B" "$M1" "B_96t" 96 "$NODE0_CPUS" "" "no_spec"
    bench_quad   "q35-27B" "$M1" "D_4x48t" "" "no_spec"
    bench_single "q35-27B" "$M1" "B_96t_draft" 96 "$NODE0_CPUS" "-md $DRAFTER --draft-max 16" "draft_dm16"
else
    echo "  SKIP: model not found"
fi
echo ""

# ============================================================
# Model 2: Qwen3.5-122B-A10B Q4_K_M (~69 GB, MoE hybrid — 2-WAY POSSIBLE)
# ============================================================
echo "================================================================"
echo "=== Qwen3.5-122B-A10B Q4_K_M (~69 GB, MoE hybrid)           ==="
echo "================================================================"
M2="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
if [ -f "$M2" ]; then
    bench_single "q35-122B" "$M2" "A_192t" 192 "all" "" "no_spec"
    bench_single "q35-122B" "$M2" "B_96t" 96 "$NODE0_CPUS" "" "no_spec"
    bench_single "q35-122B" "$M2" "B_96t_draft" 96 "$NODE0_CPUS" "-md $DRAFTER --draft-max 16" "draft_dm16"
else
    echo "  SKIP: model not found"
fi
echo ""

# ============================================================
# Model 3: Qwen3.5-397B-A17B Q4_K_XL (~205 GB, MoE hybrid — SINGLE INSTANCE)
# ============================================================
echo "================================================================"
echo "=== Qwen3.5-397B-A17B Q4_K_XL (~205 GB, MoE hybrid)         ==="
echo "================================================================"
M3="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-397B-A17B-GGUF/UD-Q4_K_XL/Qwen3.5-397B-A17B-UD-Q4_K_XL-00001-of-00006.gguf"
if [ -f "$M3" ]; then
    bench_single "q35-397B" "$M3" "A_192t" 192 "all" "" "no_spec"
    bench_single "q35-397B" "$M3" "B_96t" 96 "$NODE0_CPUS" "" "no_spec"
else
    echo "  SKIP: model not found"
fi
echo ""

# Summary
echo "=== FULL SUMMARY ==="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
