#!/bin/bash
set -euo pipefail

# Sweep draft_max for ngram-simple speculation
# Usage: ./sweep_draft_max.sh <model_path> [chat_template]

MODEL="${1:?Usage: sweep_draft_max.sh <model_path> [chat_template]}"
TEMPLATE="${2:-}"  # optional: e.g., "chatml"
PORT=8899
LLAMA=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
DRAFT_VALUES="0 16 32 64 128 256"
PROMPT="Write a detailed explanation of how neural networks learn through backpropagation. Cover the forward pass, loss computation, gradient calculation, and weight updates. Include mathematical intuition."
MAX_TOKENS=500
REPS=3  # repetitions per draft_max value

echo "=== draft_max sweep ==="
echo "Model: $MODEL"
echo "Template: ${TEMPLATE:-default}"
echo "Prompt tokens: ~30, Gen tokens: $MAX_TOKENS, Reps: $REPS"
echo ""

for DM in $DRAFT_VALUES; do
    # Kill any existing server
    pkill -9 -f "llama-server.*$PORT" 2>/dev/null || true
    sleep 2

    # Build command
    CMD="numactl --interleave=all $LLAMA -m $MODEL -t 96 --port $PORT -ngl 0 --no-warmup"
    if [ -n "$TEMPLATE" ]; then
        CMD="$CMD --chat-template $TEMPLATE"
    fi
    if [ "$DM" -gt 0 ]; then
        CMD="$CMD --spec-type ngram-simple --draft-max $DM"
        LABEL="dm=$DM"
    else
        LABEL="baseline"
    fi

    # Start server
    eval "$CMD" > /tmp/sweep_server.log 2>&1 &
    SERVER_PID=$!

    # Wait for healthy
    HEALTHY=false
    for i in $(seq 1 120); do
        if curl -s http://localhost:$PORT/health 2>/dev/null | grep -q '"status":"ok"'; then
            HEALTHY=true
            break
        fi
        sleep 1
    done

    if [ "$HEALTHY" = false ]; then
        echo "$LABEL: FAILED (server didn't start)"
        kill -9 $SERVER_PID 2>/dev/null || true
        continue
    fi

    # Run reps and collect speeds
    SPEEDS=""
    for rep in $(seq 1 $REPS); do
        RESULT=$(curl -s http://localhost:$PORT/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"test\",
                \"messages\": [{\"role\": \"user\", \"content\": \"$PROMPT\"}],
                \"max_tokens\": $MAX_TOKENS,
                \"temperature\": 0.1,
                \"chat_template_kwargs\": {\"enable_thinking\": false}
            }" 2>/dev/null)

        SPEED=$(echo "$RESULT" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f\"{d['timings']['predicted_per_second']:.2f}\")
except:
    print('0')
" 2>/dev/null)
        SPEEDS="$SPEEDS $SPEED"
    done

    # Calculate average
    AVG=$(echo "$SPEEDS" | python3 -c "
import sys
vals = [float(x) for x in sys.stdin.read().split() if float(x) > 0]
if vals:
    avg = sum(vals) / len(vals)
    print(f'{avg:.2f}')
else:
    print('0')
")

    echo "$LABEL: ${AVG} t/s  (runs:$SPEEDS)"

    kill -9 $SERVER_PID 2>/dev/null || true
done

echo ""
echo "=== sweep complete ==="
