#!/bin/bash
# Coder Quant Spec Param Sweep
#
# Sweeps (draft_max, p_split) for each coder variant. Server restarts per config
# (per-request speculative.n_max is unreliable).
#
# Variants:
#   - f16 (65 GB, 9 shards): dm sweep + p_split sweep (tree beneficial)
#   - Q8_0 (33 GB): dm sweep + p_split sweep (tree marginal)
#   - Q4_K_M (18.5 GB): dm sweep only (tree net-negative, p_split=0)
#
# Two modes per variant: 192t + 48t NUMA quarter
# Draft: Qwen2.5-Coder-0.5B-Q8_0
# Output: data/coder_spec_sweep/

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/coder_spec_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=128
N_REQUESTS=10
BASE_PORT=8180

NODE0A_CPUS="0-23,96-119"

CODER_F16="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf"
CODER_Q8="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
CODER_Q4="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Write a Rust implementation of a lock-free concurrent queue using compare-and-swap operations:"
    "Design a microservice architecture for a real-time chat application with message persistence:"
    "Implement a B+ tree in Python with bulk loading, range queries, and page splitting:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "Coder Quant Spec Param Sweep"
echo "============================="
echo "n_predict=$N_PREDICT, n_requests=$N_REQUESTS"
echo "Timestamp: $TIMESTAMP"
echo ""

# ============================================================
# Helper Functions (identical to bench_sweep_spec_params.sh)
# ============================================================

wait_for_server() {
    local port=$1 max_wait=${2:-600} elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 2; elapsed=$((elapsed + 2))
        [ $elapsed -ge $max_wait ] && { echo "ERROR: port $port timeout"; return 1; }
    done
}

warmup_server() {
    curl -s "http://localhost:${1}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' > /dev/null 2>&1
}

run_completion() {
    local port=$1 prompt="$2" n_predict=$3
    local start_ms=$(date +%s%N | cut -b1-13)
    local response=$(curl -s --max-time 600 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":${n_predict},\"temperature\":0.0,\"stream\":false}" 2>/dev/null)
    local end_ms=$(date +%s%N | cut -b1-13) elapsed_ms=$(($(date +%s%N | cut -b1-13) - start_ms))
    elapsed_ms=$((end_ms - start_ms))
    local tokens=$(echo "$response" | python3 -c "import json,sys;r=json.load(sys.stdin);print(r.get('usage',{}).get('completion_tokens',0))" 2>/dev/null)
    local tps="0.00"
    [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ] && tps=$(python3 -c "print(f'{$tokens/($elapsed_ms/1000):.2f}')")
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill -9 "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    local port_pids
    port_pids=$(lsof -ti :$BASE_PORT 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    sleep 2
}

run_sweep_requests() {
    local port=$1 variant=$2 mode=$3 threads=$4 dm=$5 ps=$6 results_file=$7
    local tps_values=()
    for ((r=0; r<N_REQUESTS; r++)); do
        local pidx=$((r % ${#PROMPTS[@]}))
        result=$(run_completion $port "${PROMPTS[$pidx]}" "$N_PREDICT")
        tps_values+=($(echo "$result" | cut -d, -f3))
        echo "$variant,$mode,$threads,$dm,$ps,$r,$result" >> "$results_file"
    done
    python3 -c "
import sys
vals=sorted([float(v) for v in sys.argv[1:] if float(v)>0])
if not vals: print('0.00,0.00,0.00'); sys.exit()
n=len(vals); print(f'{sum(vals)/n:.2f},{vals[n//2]:.2f},{vals[int(n*0.95)]:.2f}')
" "${tps_values[@]}"
}

# Sweep one variant in one mode. Returns "best_dm best_ps" on last line.
sweep_variant() {
    local variant=$1 target=$2 threads=$3 cpus=$4 mode=$5 do_tree=$6 results_file=$7
    local dm_values=(8 16 24 32 48) best_dm=24 best_avg=0

    echo "  [${mode} — dm sweep, p_split=0]"
    for dm in "${dm_values[@]}"; do
        echo "    dm=$dm ..."
        local spec_args="--draft-max $dm --draft-p-split 0 --lookup --flash-attn on"
        if [ "$cpus" = "all" ]; then
            "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                > "$LOG_DIR/${variant}_${mode}_dm${dm}.log" 2>&1 &
        else
            taskset -c "$cpus" "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                > "$LOG_DIR/${variant}_${mode}_dm${dm}.log" 2>&1 &
        fi
        local PID=$!
        if ! wait_for_server $BASE_PORT; then echo "      FAILED"; kill_servers $PID; continue; fi
        warmup_server $BASE_PORT
        local stats=$(run_sweep_requests $BASE_PORT "$variant" "$mode" "$threads" "$dm" "0" "$results_file")
        local avg=$(echo "$stats" | cut -d, -f1)
        echo "      avg=${avg} t/s"
        [ "$(python3 -c "print(1 if $avg > $best_avg else 0)")" = "1" ] && { best_avg="$avg"; best_dm=$dm; }
        kill_servers $PID
    done
    echo "    Best dm=$best_dm ($best_avg t/s)"

    local best_ps=0
    if [ "$do_tree" = "yes" ]; then
        echo ""
        echo "  [${mode} — p_split sweep at dm=$best_dm (baseline=$best_avg t/s)]"
        for ps in 0.05 0.1 0.3; do
            echo "    ps=$ps ..."
            local spec_args="--draft-max $best_dm --kv-unified --lookup --draft-p-split $ps --flash-attn on"
            if [ "$cpus" = "all" ]; then
                "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                    > "$LOG_DIR/${variant}_${mode}_dm${best_dm}_ps${ps}.log" 2>&1 &
            else
                taskset -c "$cpus" "$LLAMA_SERVER" -m "$target" -md "$DRAFT" $spec_args -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                    > "$LOG_DIR/${variant}_${mode}_dm${best_dm}_ps${ps}.log" 2>&1 &
            fi
            local PID=$!
            if ! wait_for_server $BASE_PORT; then echo "      FAILED"; kill_servers $PID; continue; fi
            warmup_server $BASE_PORT
            local stats=$(run_sweep_requests $BASE_PORT "$variant" "$mode" "$threads" "$best_dm" "$ps" "$results_file")
            local avg=$(echo "$stats" | cut -d, -f1)
            echo "      avg=${avg} t/s"
            [ "$(python3 -c "print(1 if $avg > $best_avg else 0)")" = "1" ] && { best_avg="$avg"; best_ps="$ps"; }
            kill_servers $PID
        done
        echo "    Best ps=$best_ps ($best_avg t/s)"
    fi
    echo "$best_dm $best_ps"
}

# ============================================================
# Run all variants
# ============================================================

for variant_info in "f16:$CODER_F16:yes" "q8:$CODER_Q8:yes" "q4km:$CODER_Q4:no"; do
    IFS=: read -r vname vpath vtree <<< "$variant_info"
    echo "================================================================"
    echo "=== Coder 32B $vname ==="
    echo "================================================================"

    V_RESULTS="${DATA_DIR}/coder_${vname}_${TIMESTAMP}.csv"
    echo "variant,mode,threads,draft_max,p_split,request_idx,tokens_generated,time_ms,tokens_per_sec" > "$V_RESULTS"

    echo "--- 192t mode ---"
    v_192t=$(sweep_variant "coder_${vname}" "$vpath" 192 "all" "192t" "$vtree" "$V_RESULTS")
    v_192t_best=$(echo "$v_192t" | tail -1)
    echo ""
    echo "--- NUMA 48t mode ---"
    v_numa=$(sweep_variant "coder_${vname}" "$vpath" 48 "$NODE0A_CPUS" "numa_48t" "$vtree" "$V_RESULTS")
    v_numa_best=$(echo "$v_numa" | tail -1)
    echo ""
    echo ">>> $vname: 192t=${v_192t_best}, NUMA=${v_numa_best}"
    echo ""
done

echo "================================================================"
echo "=== CODER SPEC SWEEP COMPLETE ==="
echo "================================================================"
echo "Results: $DATA_DIR"
echo "Logs: $LOG_DIR"
