#!/bin/bash
# S5: NUMA Prefill Benchmark for Qwen3-Next-80B-A3B (ingest_long_context)
#
# Phase 1: Baseline NUMA characterization before pipeline implementation.
# Tests prefill speed (prompt processing) and decode speed at various context lengths.
#
# Configs:
#   A) 1×192t all CPUs (current default)
#   B) 1×96t node 0 (single-node pinned)
#   C) 2×96t (one per NUMA node, aggregate throughput)
#
# Context lengths: short (~50 tokens), 4K, 8K
# Metrics: prompt_per_second (prefill), predicted_per_second (decode), TTFT

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_s5_prefill"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/s5_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

BASE_PORT=8190
NODE0_CPUS="0-47,96-143"
NODE1_CPUS="48-95,144-191"

# Generate prompts of different context lengths
# Short: simple question (~50 tokens)
# 4K: repeated technical content to fill context
# 8K: even more content
PROMPT_SHORT="Write a Python function to implement binary search."

# Generate long prompts by repeating content
generate_long_prompt() {
    local target_tokens=$1
    local base="The following is a detailed technical specification for a distributed computing system. "
    base+="Each node in the cluster communicates via message passing using Protocol Buffers. "
    base+="The consensus algorithm uses a modified Raft protocol with pre-vote extensions. "
    base+="Leader election timeout is randomized between 150-300ms to prevent split votes. "
    base+="Log replication uses pipeline mode for better throughput on high-latency networks. "
    base+="Snapshot transfer uses chunked streaming with checksums for integrity. "
    base+="Configuration changes use joint consensus for safe membership transitions. "
    base+="The storage engine uses LSM trees with write-ahead logging for durability. "
    base+="Compaction runs in background with rate limiting to avoid impact on reads. "
    base+="Read queries can be served from followers using linearizable reads via ReadIndex. "
    local result=""
    local current=0
    while [ $current -lt $target_tokens ]; do
        result+="$base "
        current=$((current + 100))  # ~100 tokens per repetition
    done
    result+=" Based on the above specifications, write a summary of the key design decisions."
    echo "$result"
}

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "S5: NUMA Prefill Benchmark — Qwen3-Next-80B-A3B Q4_K_M"
echo "========================================================"
echo "Model: $(basename "$MODEL") (46 GB, hybrid MoE+DeltaNet)"
echo "Results: $RESULTS_FILE"
echo ""

echo "config,threads,cpu_binding,context_size,prompt_tokens,predicted_tokens,prompt_ms,predicted_ms,prompt_tps,predicted_tps,total_ms" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1 max_wait=600 elapsed=0
    while true; do
        local health
        health=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "")
        if echo "$health" | grep -q '"status":"ok"'; then return 0; fi
        sleep 2; elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then echo "TIMEOUT port $port"; return 1; fi
    done
}

run_prefill_test() {
    local port=$1
    local prompt="$2"
    local n_predict=$3
    local response
    response=$(curl -s --max-time 600 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":${n_predict},\"temperature\":0.0,\"stream\":false}" 2>/dev/null)

    # Extract detailed timings
    python3 -c "
import json, sys
try:
    r = json.loads('''$response'''.replace(\"'''\", ''))
except:
    try:
        r = json.load(sys.stdin)
    except:
        print('0,0,0,0,0,0,0')
        sys.exit()

t = r.get('timings', {})
u = r.get('usage', {})
prompt_n = t.get('prompt_n', u.get('prompt_tokens', 0))
predicted_n = t.get('predicted_n', u.get('completion_tokens', 0))
prompt_ms = t.get('prompt_ms', 0)
predicted_ms = t.get('predicted_ms', 0)
prompt_tps = t.get('prompt_per_second', 0)
predicted_tps = t.get('predicted_per_second', 0)
total_ms = prompt_ms + predicted_ms
print(f'{prompt_n},{predicted_n},{prompt_ms:.1f},{predicted_ms:.1f},{prompt_tps:.2f},{predicted_tps:.2f},{total_ms:.1f}')
" <<< "$response" 2>/dev/null || echo "0,0,0,0,0,0,0"
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 2
}

warmup_server() {
    curl -s "http://localhost:${1}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"temperature":0}' \
        > /dev/null 2>&1
}

# Pre-generate long prompts
echo "Generating test prompts..."
PROMPT_4K=$(generate_long_prompt 4000)
PROMPT_8K=$(generate_long_prompt 8000)
echo "  Short: ~50 tokens"
echo "  4K: ~4000 tokens ($(echo "$PROMPT_4K" | wc -c) chars)"
echo "  8K: ~8000 tokens ($(echo "$PROMPT_8K" | wc -c) chars)"
echo ""

N_PREDICT=64  # Small output — we're measuring prefill, not generation

run_config() {
    local config=$1 threads=$2 cpus=$3

    echo "=== Config $config: 1×${threads}t, $cpus ==="

    if [ "$cpus" = "all" ]; then
        "$LLAMA_SERVER" -m "$MODEL" -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics \
            > "$LOG_DIR/config${config}.log" 2>&1 &
    else
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$MODEL" -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics \
            > "$LOG_DIR/config${config}.log" 2>&1 &
    fi
    local PID=$!

    wait_for_server $BASE_PORT || { kill_servers $PID; return; }
    warmup_server $BASE_PORT
    echo "  Server ready"

    for ctx_label in "short" "4K" "8K"; do
        local prompt
        case $ctx_label in
            short) prompt="$PROMPT_SHORT";;
            4K) prompt="$PROMPT_4K";;
            8K) prompt="$PROMPT_8K";;
        esac

        echo -n "  $ctx_label context: "
        local result
        result=$(run_prefill_test $BASE_PORT "$prompt" $N_PREDICT)
        echo "$config,$threads,$cpus,$ctx_label,$result" >> "$RESULTS_FILE"

        local prompt_tps predicted_tps prompt_n
        prompt_n=$(echo "$result" | cut -d, -f1)
        prompt_tps=$(echo "$result" | cut -d, -f5)
        predicted_tps=$(echo "$result" | cut -d, -f6)
        echo "prompt=$prompt_n tokens @ ${prompt_tps} t/s prefill, ${predicted_tps} t/s decode"
    done

    kill_servers $PID
    echo ""
}

# Verify model exists
if [ ! -f "$MODEL" ]; then echo "ERROR: model not found: $MODEL"; exit 1; fi

# Config A: 1×192t all CPUs
run_config "A" 192 "all"

# Config B: 1×96t node 0
run_config "B" 96 "$NODE0_CPUS"

# Config C: 2×96t (one per NUMA node) — measure aggregate
echo "=== Config C: 2×96t (one per NUMA node) ==="

PORT_C1=$BASE_PORT
PORT_C2=$((BASE_PORT + 1))

taskset -c "$NODE0_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port $PORT_C1 -ngl 0 --metrics \
    > "$LOG_DIR/configC_node0.log" 2>&1 &
PID_C1=$!

taskset -c "$NODE1_CPUS" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port $PORT_C2 -ngl 0 --metrics \
    > "$LOG_DIR/configC_node1.log" 2>&1 &
PID_C2=$!

echo "  Loading 2 instances..."
wait_for_server $PORT_C1 || { kill_servers $PID_C1 $PID_C2; echo "FAILED"; exit 1; }
wait_for_server $PORT_C2 || { kill_servers $PID_C1 $PID_C2; echo "FAILED"; exit 1; }
warmup_server $PORT_C1
warmup_server $PORT_C2
echo "  Both servers ready"

for ctx_label in "short" "4K" "8K"; do
    c_prompt=""
    case $ctx_label in
        short) c_prompt="$PROMPT_SHORT";;
        4K) c_prompt="$PROMPT_4K";;
        8K) c_prompt="$PROMPT_8K";;
    esac

    echo -n "  $ctx_label context: "
    result1=$(run_prefill_test $PORT_C1 "$c_prompt" $N_PREDICT)
    result2=$(run_prefill_test $PORT_C2 "$c_prompt" $N_PREDICT)

    echo "C_node0,96,$NODE0_CPUS,$ctx_label,$result1" >> "$RESULTS_FILE"
    echo "C_node1,96,$NODE1_CPUS,$ctx_label,$result2" >> "$RESULTS_FILE"

    ptps1=$(echo "$result1" | cut -d, -f5)
    ptps2=$(echo "$result2" | cut -d, -f5)
    dtps1=$(echo "$result1" | cut -d, -f6)
    dtps2=$(echo "$result2" | cut -d, -f6)
    pn1=$(echo "$result1" | cut -d, -f1)
    echo "n0: prefill=${ptps1} t/s, decode=${dtps1} t/s | n1: prefill=${ptps2} t/s, decode=${dtps2} t/s (prompt=$pn1 tokens)"
done

kill_servers $PID_C1 $PID_C2
echo ""

# Summary
echo "=== S5 SUMMARY ==="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
