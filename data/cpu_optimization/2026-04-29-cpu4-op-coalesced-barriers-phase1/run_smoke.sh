#!/bin/bash
# CPU4 Phase 1 smoke test — verify GGML_BARRIER_COALESCE=1 doesn't crash
# and produces output. Quick sanity check before heavier measurement.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-cpu4-op-coalesced-barriers-phase1
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

PROMPT='Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'

run_smoke() {
    local TAG=$1
    local COALESCE=$2

    echo "=== smoke: $TAG (GGML_BARRIER_COALESCE=$COALESCE) ==="
    date

    GGML_BARRIER_COALESCE=$COALESCE numactl --interleave=all $BIN/llama-server \
        -m "$CODER" \
        -t 96 -c 4096 -fa 1 \
        --port 18099 \
        > srv_smoke_${TAG}.log 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
            echo "  ready after ${i}s"
            sleep 30
            break
        fi
        sleep 1
    done

    curl -s http://localhost:18099/completion \
        -H 'Content-Type: application/json' \
        -d "$(jq -n --arg p "$PROMPT" '{prompt: $p, n_predict: 64, temperature: 0.0, seed: 4242}')" \
        > comp_smoke_${TAG}.json 2>&1

    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

run_smoke "off" 0
run_smoke "on"  1

echo ""
echo "=== smoke results ==="
for TAG in off on; do
    echo "--- COALESCE=$TAG ---"
    TPS=$(jq -r '.timings.predicted_per_second // "n/a"' comp_smoke_${TAG}.json 2>/dev/null)
    HEAD=$(jq -r '.content // "(empty)"' comp_smoke_${TAG}.json 2>/dev/null | head -c 80 | tr '\n' ' ')
    echo "  TPS: $TPS"
    echo "  HEAD: $HEAD"
done

echo ""
echo "=== content identity (deterministic seed: should be IDENTICAL) ==="
A=$(jq -r '.content' comp_smoke_off.json 2>/dev/null)
B=$(jq -r '.content' comp_smoke_on.json 2>/dev/null)
if [ "$A" = "$B" ]; then
    echo "  ✅ IDENTICAL — coalesce path produces bit-exact output as off"
else
    echo "  ❌ DIFFER — coalesce path BROKE OUTPUT (likely race condition)"
    echo "  off: ${A:0:120}"
    echo "  on : ${B:0:120}"
fi

echo ""
echo "completed at $(date)"
