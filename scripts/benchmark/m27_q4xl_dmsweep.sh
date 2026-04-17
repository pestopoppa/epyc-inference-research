#!/bin/bash
set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/models/MiniMax-M2.7-GGUF/UD-Q4_K_XL/MiniMax-M2.7-UD-Q4_K_XL-00001-of-00004.gguf"
RESULTS="/mnt/raid0/llm/epyc-inference-research/data/numa_new_models/m27_q4xl_dmsweep.csv"
PORT=8190

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
)

echo "model,draft_max,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS"

for dm in 0 16 32 48 64 96 128; do
    echo "=== draft-max=$dm ==="

    if [ "$dm" -eq 0 ]; then
        SPEC_ARGS=""
        label="baseline"
    else
        SPEC_ARGS="--spec-type ngram-simple --draft-max $dm"
        label="ngram_dm${dm}"
    fi

    numactl --interleave=all "$LLAMA_SERVER" -m "$MODEL" $SPEC_ARGS \
        -t 96 -np 1 --port $PORT -ngl 0 --metrics \
        > /mnt/raid0/llm/epyc-inference-research/data/numa_new_models/m27_q4xl_dm${dm}.log 2>&1 &
    PID=$!

    elapsed=0
    while ! curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge 600 ]; then
            echo "  FAILED to start"
            kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
            continue 2
        fi
    done
    echo "  server ready (${elapsed}s)"

    # Warmup
    curl -s --max-time 120 "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' > /dev/null 2>&1

    for i in "${!PROMPTS[@]}"; do
        start_ms=$(date +%s%N | cut -b1-13)
        response=$(curl -s --max-time 600 "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"test\",
                \"messages\": [{\"role\": \"user\", \"content\": $(echo "${PROMPTS[$i]}" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],
                \"max_tokens\": 256,
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
        echo "  prompt $i: ${tps} t/s ($tokens tokens)"
        echo "$label,$dm,$i,$tokens,$elapsed_ms,$tps" >> "$RESULTS"
    done

    kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
    sleep 2
    echo ""
done

echo "=== DONE ==="
cat "$RESULTS"
