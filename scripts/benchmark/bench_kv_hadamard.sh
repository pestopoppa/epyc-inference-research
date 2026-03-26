#!/bin/bash
# Phase 1: Hadamard-Smoothed KV Cache Benchmark
#
# Compares Hadamard-smoothed q4_0 vs plain q4_0 vs f16 baseline
# using the same experimental build (build-hadamard).
#
# Model: Qwen2.5-Coder-32B Q4_K_M (pure attention — max KV impact)
# Context: 4096
# Server: numactl --interleave=all, 96t, -ub 8192, --flash-attn on

set -u

export LD_LIBRARY_PATH="/mnt/raid0/llm/llama.cpp-experimental/build-hadamard/bin"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp-experimental/build-hadamard/bin/llama-server"
MODEL="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/kv_cache_quant"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/hadamard_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_hadamard_${TIMESTAMP}"
PORT=8191
THREADS=96
CTX=4096
N_PREDICT=256
N_REPS=3

PROMPT="Write a Python function that implements a concurrent task queue with priority scheduling, worker pools, and graceful shutdown. Include comprehensive error handling, logging, and type hints."

# Configs: label, ctk, ctv, extra_flags
CONFIGS=(
    "f16_baseline:f16:f16:"
    "q4_plain:q4_0:q4_0:"
    "q8k_q4v_plain:q8_0:q4_0:"
    "q4_hadamard:q4_0:q4_0:--kv-hadamard"
    "q8k_q4v_hadamard:q8_0:q4_0:--kv-hadamard"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "Phase 1: Hadamard KV Benchmark"
echo "==============================="
echo "Results: $RESULTS_FILE"
echo ""

echo "config,ctk,ctv,hadamard,rep,tokens,time_ms,gen_tps,prompt_tps,rss_mb,kv_mb" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1 max_wait=300 elapsed=0
    while true; do
        curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"' && return 0
        sleep 2; elapsed=$((elapsed + 2))
        [ $elapsed -ge $max_wait ] && { echo "TIMEOUT"; return 1; }
    done
}

run_test() {
    local port=$1
    curl -s --max-time 300 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$PROMPT" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":${N_PREDICT},\"temperature\":0.0,\"stream\":false}" 2>/dev/null | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    u = r.get('usage', {})
    t = r.get('timings', {})
    print(f'{u.get(\"completion_tokens\",0)},{int(t.get(\"predicted_ms\",0)+t.get(\"prompt_ms\",0))},{t.get(\"predicted_per_second\",0):.2f},{t.get(\"prompt_per_second\",0):.2f}')
except:
    print('0,0,0.00,0.00')
" 2>/dev/null
}

for cfg_str in "${CONFIGS[@]}"; do
    IFS=':' read -r label ctk ctv extra <<< "$cfg_str"
    hadamard="no"
    [[ "$extra" == *"hadamard"* ]] && hadamard="yes"

    echo "--- $label (ctk=$ctk ctv=$ctv hadamard=$hadamard) ---"

    server_args="-m $MODEL -t $THREADS -np 1 --port $PORT -ngl 0 --flash-attn on -c $CTX -ub 8192"
    [ "$ctk" != "f16" ] && server_args="$server_args -ctk $ctk -ctv $ctv"
    [ -n "$extra" ] && server_args="$server_args $extra"

    log_file="${LOG_DIR}/${label}.log"
    numactl --interleave=all $LLAMA_SERVER $server_args > "$log_file" 2>&1 &
    PID=$!

    if ! wait_for_server $PORT; then
        echo "  FAILED to start"
        kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 2
        continue
    fi

    # Warmup
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model":"t","messages":[{"role":"user","content":"Hi"}],"max_tokens":5,"temperature":0}' > /dev/null 2>&1
    sleep 1

    rss_kb=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    rss_mb=$(python3 -c "print(f'{${rss_kb:-0}/1024:.1f}')")
    kv_mb=$(grep -oP 'llama_kv_cache: size =\s*\K[0-9.]+' "$log_file" 2>/dev/null | tail -1)
    [ -z "$kv_mb" ] && kv_mb="0"

    for rep in $(seq 1 $N_REPS); do
        result=$(run_test $PORT)
        gen_tps=$(echo "$result" | cut -d, -f3)
        echo "  rep $rep: $gen_tps t/s"
        echo "$label,$ctk,$ctv,$hadamard,$rep,$result,$rss_mb,$kv_mb" >> "$RESULTS_FILE"
    done

    kill $PID 2>/dev/null; wait $PID 2>/dev/null; sleep 3
done

echo ""
echo "=== SUMMARY ==="
python3 -c "
import csv
from collections import defaultdict

data = defaultdict(list)
with open('$RESULTS_FILE') as f:
    for row in csv.DictReader(f):
        data[row['config']].append(float(row['gen_tps']))

print(f'{\"Config\":<25} {\"Avg Gen t/s\":<12} {\"Min\":<8} {\"Max\":<8}')
print('-' * 55)
for cfg in data:
    vals = data[cfg]
    print(f'{cfg:<25} {sum(vals)/len(vals):<12.2f} {min(vals):<8.2f} {max(vals):<8.2f}')
"
echo ""
echo "Done: $RESULTS_FILE"
