#!/bin/bash
# Final canonical measurement — Phase 1.1 dispatcher v1 vs K=1 baseline
# on Qwen3.6-35B-A3B-Q8_0 hybrid Delta Net + Qwen3-1.7B-Q8_0 drafter.
# 3 prompts × 2 reps each per config. n_predict=64, p_split=0.05, temperature=0.0.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-state-sync-cost-probe
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
TGT=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
DFT=/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

PROMPTS=(
'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
)

run_config() {
    local TAG=$1
    local NUMA_Q=$2
    local PSPLIT=$3

    echo "=== Config: $TAG (numa-q=$NUMA_Q, psplit=$PSPLIT) ==="
    date

    numactl --interleave=all $BIN/llama-server \
        -m "$TGT" -md "$DFT" \
        -t 96 -c 4096 -fa 1 \
        --spec-numa-quarters $NUMA_Q \
        --draft-max 24 --draft-min 4 \
        --port 18099 \
        > srv_final_${TAG}.log 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 180); do
        if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
            echo "  ready after ${i}s"
            sleep 30
            break
        fi
        sleep 1
    done

    for r in 0 1; do
        for p in 0 1 2; do
            PROMPT="${PROMPTS[$p]}"
            echo "  --- p${p}_r${r} at $(date +%H:%M:%S) ---"
            curl -s http://localhost:18099/completion \
                -H 'Content-Type: application/json' \
                -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" '{prompt: $p, n_predict: 64, temperature: 0.0, p_split: $ps}')" \
                > comp_final_${TAG}_p${p}_r${r}.json 2>&1
            sleep 1
        done
    done

    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

run_config "k1"  1 0.05
run_config "k4"  4 0.05

echo ""
echo "=== Aggregate results ==="
for TAG in k1 k4; do
    echo "--- $TAG ---"
    for f in comp_final_${TAG}_*.json; do
        TPS=$(jq -r '.timings.predicted_per_second // "n/a"' "$f" 2>/dev/null)
        ACC=$(jq -r '"\(.timings.draft_n_accepted // "?")/\(.timings.draft_n // "?")"' "$f" 2>/dev/null)
        echo "  $f: $TPS t/s, accept=$ACC"
    done
done

echo ""
echo "=== K-parallel verify hit count (k4 only) ==="
grep -c "numa K-parallel verify" srv_final_k4.log || echo "0"
