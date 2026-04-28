#!/bin/bash
# Phase 1.1.A measurement: --spec-numa-quarters 4 (primary ctx pinned to 24t quarter 0)
# vs Phase 1.0 baseline (96t full-machine, was 6.80 t/s linear).
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-numa-quarter-pin-phase-1-1a
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
TGT=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
DFT=/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf

export LD_LIBRARY_PATH=$BIN

cd $OUT
ulimit -c 0

PROMPTS=(
'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
)

run_config() {
    local TAG=$1
    local NUMA_Q=$2
    local THREADS=$3
    local PSPLIT=$4

    echo "=== Config: $TAG (numa-q=$NUMA_Q, threads=$THREADS, psplit=$PSPLIT) ==="
    date

    # Start server
    $BIN/llama-server -m $TGT -md $DFT -t $THREADS -c 4096 -fa 1 \
        --spec-numa-quarters $NUMA_Q \
        --draft-max 24 --draft-min 4 \
        --port 18099 \
        > srv_${TAG}.log 2>&1 &
    SRV_PID=$!

    # Wait for /health (35B Q8 takes ~60-90s to load)
    for i in $(seq 1 180); do
        if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
            echo "  server ready after ${i}s"
            sleep 60 # warmup
            break
        fi
        sleep 1
    done

    # Run 3 prompts × 3 reps
    for r in 0 1 2; do
        for p in 0 1 2; do
            PROMPT="${PROMPTS[$p]}"
            curl -s http://localhost:18099/completion \
                -H 'Content-Type: application/json' \
                -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" '{prompt: $p, n_predict: 64, temperature: 0.0, p_split: $ps}')" \
                > comp_${TAG}_p${p}_r${r}.json 2>&1
            sleep 1
        done
    done

    # Kill server
    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

# K=4 → primary ctx pinned to threads [0, 24); 3 idle aux ctxs on quarters 1-3
run_config "k4_linear" 4 96 0

echo "=== ALL DONE ==="
