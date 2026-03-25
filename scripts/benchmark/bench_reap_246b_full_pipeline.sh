#!/bin/bash
# REAP-246B Full Pipeline: Convert FP8 → GGUF Q4_K_M → Speed Sweep
#
# 50% pruned Qwen3-Coder-480B → 246B. Same 35B active params.
# This is the interesting REAP test: 120 GB RAM savings (250→~130 GB)
# enables concurrent large-model deployment.
#
# Prerequisites: FP8 safetensors downloaded to /mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-FP8/
#
# Usage: ./bench_reap_246b_full_pipeline.sh

set -euo pipefail

LLAMA_CPP="/mnt/raid0/llm/llama.cpp"
LLAMA_SERVER="${LLAMA_CPP}/build/bin/llama-server"
FP8_DIR="/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-FP8"
GGUF_OUT="/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf"
DRAFT="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/reap_246b_phase1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/reap_246b_phase1_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

PORT=8196
CPUS="0-47,96-143"  # NUMA node 0
THREADS=96
N_PREDICT=256

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature:"
    "Implement a concurrent hash map in C++ using fine-grained locking:"
)

# ============================================================
# Step 1: Verify download complete
# ============================================================
echo "Step 1: Verify FP8 download"
if [ ! -d "$FP8_DIR" ]; then
    echo "ERROR: FP8 directory not found: $FP8_DIR"
    exit 1
fi

# Check for safetensors files
SHARD_COUNT=$(ls "$FP8_DIR"/*.safetensors 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No safetensors files found in $FP8_DIR"
    echo "Download may still be in progress. Check: du -sh $FP8_DIR"
    exit 1
fi
echo "  Found $SHARD_COUNT safetensors shards"
echo "  Total size: $(du -sh "$FP8_DIR" | cut -f1)"

# ============================================================
# Step 2: Convert FP8 → GGUF Q4_K_M
# ============================================================
echo ""
echo "Step 2: Convert FP8 → GGUF Q4_K_M"

if [ -f "$GGUF_OUT" ]; then
    echo "  GGUF already exists: $GGUF_OUT ($(du -sh "$GGUF_OUT" | cut -f1))"
    echo "  Skipping conversion"
else
    echo "  Converting... (this will take 15-30 minutes)"
    cd "$LLAMA_CPP"
    python3 convert_hf_to_gguf.py "$FP8_DIR" --outtype q4_k_m -o "$GGUF_OUT" 2>&1 | tail -10

    if [ ! -f "$GGUF_OUT" ]; then
        echo "ERROR: Conversion failed — GGUF not created"
        exit 1
    fi
    echo "  Created: $GGUF_OUT ($(du -sh "$GGUF_OUT" | cut -f1))"
fi

MODEL="$GGUF_OUT"

# ============================================================
# Step 3: Speed Sweep (same as 363B Phase 1)
# ============================================================
echo ""
echo "Step 3: Speed Sweep"
echo "==================="
echo "Model: $(basename "$MODEL")"
echo "Draft: $(basename "$DRAFT")"
echo "n_predict=$N_PREDICT, threads=$THREADS"

mkdir -p "$DATA_DIR" "$LOG_DIR"
echo "config,draft_max,p_split,lookup,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1
    local max_wait=900
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: server did not start within ${max_wait}s"
            return 1
        fi
        if [ $((elapsed % 60)) -eq 0 ]; then echo "  ... loading ($((elapsed/60))m)"; fi
    done
    echo "  Server ready (${elapsed}s)"
}

run_completion() {
    local port=$1
    local prompt="$2"
    local n_predict=$3
    local start_ms end_ms elapsed_ms tokens tps

    start_ms=$(date +%s%N | cut -b1-13)
    local response
    response=$(curl -s --max-time 300 "http://localhost:${port}/v1/chat/completions" \
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

kill_server() {
    kill "$1" 2>/dev/null || true
    wait "$1" 2>/dev/null || true
    sleep 3
}

run_config() {
    local label="$1" dm="$2" ps="$3" lookup="$4"
    local extra_flags=""

    if [ "$dm" != "0" ]; then
        extra_flags="-md $DRAFT --draft-max $dm --draft-p-split $ps --kv-unified"
    fi
    if [ "$lookup" = "yes" ]; then extra_flags="$extra_flags --lookup"; fi

    echo "=== $label (dm=$dm, ps=$ps, lookup=$lookup) ==="

    taskset -c "$CPUS" "$LLAMA_SERVER" -m "$MODEL" -t $THREADS -np 1 --port $PORT -ngl 0 --mlock \
        $extra_flags > "$LOG_DIR/${label}.log" 2>&1 &
    local PID=$!

    if ! wait_for_server $PORT; then kill_server $PID; return; fi

    curl -s "http://localhost:${PORT}/v1/chat/completions" -H "Content-Type: application/json" \
        -d '{"model":"t","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' > /dev/null 2>&1

    local total_tps=0 count=0
    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $PORT "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$label,$dm,$ps,$lookup,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "  prompt $i: ${tps} t/s"
        if [ "$tps" != "0.00" ]; then
            total_tps=$(python3 -c "print($total_tps + $tps)")
            count=$((count + 1))
        fi
    done

    if [ $count -gt 0 ]; then
        echo "  Average: $(python3 -c "print(f'{$total_tps / $count:.2f}')") t/s"
    fi
    kill_server $PID
    echo ""
}

# Test matrix
run_config "baseline" 0 0 no
for dm in 8 16 24 32 48; do
    run_config "spec_dm${dm}_linear" $dm 0 no
done
run_config "spec_dm24_lookup" 24 0 yes

# ============================================================
# Summary
# ============================================================
echo "=== SUMMARY ==="
python3 - "$RESULTS_FILE" << 'PYEOF'
import csv, sys
from collections import defaultdict

results = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        tps = float(row['tokens_per_sec'])
        if tps > 0:
            results[row['config']].append(tps)

print(f"{'Config':<30} {'Avg t/s':<10} {'Samples':<8}")
print("-" * 48)

best_config, best_tps = "", 0
for config in sorted(results):
    vals = results[config]
    avg = sum(vals) / len(vals)
    print(f"{config:<30} {avg:<10.2f} {len(vals):<8}")
    if avg > best_tps:
        best_tps, best_config = avg, config

print(f"\nBest: {best_config} at {best_tps:.2f} t/s")
print(f"\nComparison:")
print(f"  Production 480B (unpruned):  7.0 t/s (dm=24, ps=0)")
print(f"  REAP-363B (25% pruned):      6.54 t/s")
print(f"  REAP-246B (50% pruned):      {best_tps:.2f} t/s")
if best_tps > 0:
    print(f"  REAP-246B vs 480B:           {best_tps/7.0*100:.1f}%")
    print(f"  RAM savings:                 ~120 GB (250 → ~130 GB)")
PYEOF

echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
