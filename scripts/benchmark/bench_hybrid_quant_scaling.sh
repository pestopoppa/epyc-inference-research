#!/bin/bash
# Hybrid Quantization Scaling Test
#
# Tests whether Qwen3.5 hybrid decode speed changes with quantization.
# Hypothesis: if recurrent state update dominates, Q4→Q8→f16 barely changes decode t/s.
# Quick test: load each quant, generate 128 tokens, measure decode speed.

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/hybrid_quant_scaling"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/quant_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"
PORT=8190

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "Hybrid Quantization Scaling Test"
echo "================================"
echo ""

echo "model,quant,size_gb,threads,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1 max_wait=600 elapsed=0
    while true; do
        local h; h=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "")
        echo "$h" | grep -q '"status":"ok"' && return 0
        sleep 2; elapsed=$((elapsed + 2))
        [ $elapsed -ge $max_wait ] && return 1
    done
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 2
}

bench_model() {
    local label=$1 quant=$2 model_path=$3 size=$4 threads=$5

    echo "=== $label $quant ($size) @ ${threads}t ==="

    taskset -c "0-47,96-143" "$LLAMA_SERVER" -m "$model_path" -t "$threads" -np 1 --port $PORT -ngl 0 \
        > "$LOG_DIR/${label}_${quant}.log" 2>&1 &
    local PID=$!

    wait_for_server $PORT || { echo "  FAILED"; kill_servers $PID; return; }

    # Warmup
    curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":10,"temperature":0}' > /dev/null 2>&1

    # Test with 2 prompts, 128 tokens each (enough for stable decode measurement)
    for prompt in "Write a Python binary search implementation with detailed comments:" "Explain quantum computing fundamentals:"; do
        local start_ms end_ms elapsed_ms tokens tps
        start_ms=$(date +%s%N | cut -b1-13)
        local response
        response=$(curl -s --max-time 300 "http://localhost:${PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":128,\"temperature\":0.0,\"stream\":false}" 2>/dev/null)
        end_ms=$(date +%s%N | cut -b1-13)
        elapsed_ms=$((end_ms - start_ms))
        tokens=$(echo "$response" | python3 -c "import json,sys;
try: print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))
except: print(0)" 2>/dev/null)
        [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ] && tps=$(python3 -c "print(f'{$tokens/($elapsed_ms/1000):.2f}')") || tps="0.00"
        echo "  ${tps} t/s ($tokens tokens)"
        echo "$label,$quant,$size,$threads,$tokens,$elapsed_ms,$tps" >> "$RESULTS_FILE"
    done

    kill_servers $PID
}

# 9B at Q4, Q6, Q8 (all fit on one node easily)
bench_model "Qwen3.5-9B" "Q4_K_M" "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf" "5.3G" 96
bench_model "Qwen3.5-9B" "Q6_K"   "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q6_K.gguf" "7.0G" 96
bench_model "Qwen3.5-9B" "Q8_0"   "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf" "8.9G" 96

# 27B at Q4, Q6
bench_model "Qwen3.5-27B" "Q4_K_M" "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf" "16G" 96
bench_model "Qwen3.5-27B" "Q6_K"   "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q6_K.gguf" "21G" 96

# 35B-A3B at Q4 (already have) vs Q8
bench_model "Qwen3.5-35B-A3B" "Q4_K_M" "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf" "19G" 96
bench_model "Qwen3.5-35B-A3B" "Q8_0" "/mnt/raid0/llm/lmstudio/models/jiaojjjjje/Qwen3.5-35B-A3B-abliterated-GGUF/Qwen3.5-35B-A3B-abliterated-Q8_0.gguf" "35G" 96

echo ""
echo "=== SUMMARY ==="
python3 - "$RESULTS_FILE" << 'PYEOF'
import csv
from collections import defaultdict

groups = defaultdict(list)
with open("$RESULTS_FILE") as f:
    for row in csv.DictReader(f):
        tps = float(row['tokens_per_sec'])
        if tps > 0:
            groups[(row['model'], row['quant'], row['size_gb'])].append(tps)

print(f"{'Model':<20} {'Quant':<10} {'Size':<8} {'Avg t/s':<10} {'vs Q4'}")
print("-" * 58)

q4_speeds = {}
for (model, quant, size), vals in sorted(groups.items()):
    avg = sum(vals) / len(vals)
    if 'Q4' in quant:
        q4_speeds[model] = avg
    ratio = f"{avg/q4_speeds.get(model,avg):.2f}x" if model in q4_speeds else "—"
    print(f"{model:<20} {quant:<10} {size:<8} {avg:<10.2f} {ratio}")

PYEOF
echo ""
echo "Results: $RESULTS_FILE"
