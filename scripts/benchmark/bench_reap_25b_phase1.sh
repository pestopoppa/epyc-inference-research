#!/bin/bash
# REAP-25B Phase 1: Speed Optimization
#
# Tests: baseline, spec decode sweep (dm × ps), lookup, NUMA 4×48t
# Comparison targets: worker 30B-A3B (39.1 t/s), frontdoor 35B moe6 (12.7 t/s)
#
# Usage: ./bench_reap_25b_phase1.sh

set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/lmstudio/models/bartowski/cerebras_Qwen3-Coder-REAP-25B-A3B-GGUF/cerebras_Qwen3-Coder-REAP-25B-A3B-Q4_K_M.gguf"
DRAFT="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/reap_25b_phase1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/reap_phase1_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

PORT=8196
CPUS="0-23,96-119"  # NUMA quarter 0A
THREADS=48
N_PREDICT=256

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature:"
    "Implement a concurrent hash map in C++ using fine-grained locking:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "REAP-25B Phase 1: Speed Optimization"
echo "======================================"
echo "Model: $(basename "$MODEL") (15 GB, pure MoE, REAP 25% pruned)"
echo "Draft: $(basename "$DRAFT")"
echo "n_predict=$N_PREDICT, threads=$THREADS"
echo "Results: $RESULTS_FILE"
echo ""

echo "config,draft_max,p_split,lookup,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1
    local max_wait=300
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then
            echo "  ERROR: server on port $port did not start within ${max_wait}s"
            return 1
        fi
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
    response=$(curl -s --max-time 180 "http://localhost:${port}/v1/chat/completions" \
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
    local pid=$1
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 2
}

run_config() {
    local label="$1"
    local dm="$2"
    local ps="$3"
    local lookup="$4"
    local extra_flags=""

    # Build flags
    if [ "$dm" != "0" ]; then
        extra_flags="-md $DRAFT --draft-max $dm --draft-p-split $ps --kv-unified"
    fi
    if [ "$lookup" = "yes" ]; then
        extra_flags="$extra_flags --lookup"
    fi

    echo "=== $label (dm=$dm, ps=$ps, lookup=$lookup) ==="

    taskset -c "$CPUS" "$LLAMA_SERVER" -m "$MODEL" -t $THREADS -np 1 --port $PORT -ngl 0 --mlock \
        $extra_flags \
        > "$LOG_DIR/${label}.log" 2>&1 &
    local PID=$!

    if ! wait_for_server $PORT; then
        echo "  FAILED to start"
        kill_server $PID
        return
    fi

    # Warmup
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"t","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1

    local total_tps=0
    local count=0
    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion $PORT "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$label,$dm,$ps,$lookup,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "  prompt $i: ${tps} t/s"

        if [ "$tps" != "0.00" ]; then
            total_tps=$(python3 -c "print($total_tps + $tps)")
            count=$((count + 1))
        else
            # Check if server crashed
            if ! kill -0 $PID 2>/dev/null; then
                echo "  SERVER CRASHED — aborting config"
                return
            fi
        fi
    done

    if [ $count -gt 0 ]; then
        avg=$(python3 -c "print(f'{$total_tps / $count:.2f}')")
        echo "  Average: ${avg} t/s"
    fi

    kill_server $PID
    echo ""
}

# ============================================================
# Test Matrix
# ============================================================

# 1. Baseline: no spec, no lookup
run_config "baseline" 0 0 no

# 2. Spec decode sweep: dm={8,16,24,32,48}, ps=0 (linear only)
for dm in 8 16 24 32 48; do
    run_config "spec_dm${dm}_linear" $dm 0 no
done

# 3. Tree speculation: dm={16,32}, ps={0.05,0.1} (pure MoE should benefit)
for dm in 16 32; do
    for ps in 0.05 0.1; do
        ps_label=$(echo $ps | tr '.' '_')
        run_config "spec_dm${dm}_tree_ps${ps_label}" $dm $ps no
    done
done

# 4. Lookup: best dm from above + lookup
# Use dm=8 as initial guess (same as worker base)
run_config "spec_dm8_lookup" 8 0 yes
run_config "spec_dm16_lookup" 16 0 yes

# 5. Best config + lookup + tree (if tree helped)
run_config "spec_dm16_tree_ps005_lookup" 16 0.05 yes
run_config "spec_dm32_tree_ps005_lookup" 32 0.05 yes

# ============================================================
# Summary
# ============================================================
echo "=== SUMMARY ==="
python3 - "$RESULTS_FILE" << 'PYEOF'
import csv, sys
from collections import defaultdict

results = defaultdict(list)
with open(sys.argv[1]) as f:
    reader = csv.DictReader(f)
    for row in reader:
        config = row['config']
        tps = float(row['tokens_per_sec'])
        if tps > 0:
            results[config].append(tps)

print(f"{'Config':<35} {'Avg t/s':<10} {'Samples':<8}")
print("-" * 53)

best_config = ""
best_tps = 0
for config in sorted(results.keys()):
    vals = results[config]
    avg = sum(vals) / len(vals) if vals else 0
    print(f"{config:<35} {avg:<10.2f} {len(vals):<8}")
    if avg > best_tps:
        best_tps = avg
        best_config = config

print(f"\nBest: {best_config} at {best_tps:.2f} t/s")
print(f"\nComparison:")
print(f"  Worker 30B-A3B (unpruned):    39.1 t/s (dm=8, ps=0)")
print(f"  Frontdoor 35B moe6:           12.7 t/s (4×48t agg: 50.8)")
print(f"  REAP-25B best:                {best_tps:.2f} t/s")
if best_tps > 0:
    print(f"  REAP vs worker:               {best_tps/39.1*100:.1f}% of worker speed")
    print(f"  REAP 4×48t estimate:          {best_tps*4:.1f} t/s aggregate")
PYEOF

echo ""
echo "Results saved to: $RESULTS_FILE"
echo "Logs saved to: $LOG_DIR"
