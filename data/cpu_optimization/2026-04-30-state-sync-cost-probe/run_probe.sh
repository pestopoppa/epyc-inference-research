#!/bin/bash
# Phase 1.1 Slice B.5 — measure cost of one-shot numa_state_sync at
# SLOT_STATE_GENERATING transition on Qwen3.6-35B-A3B-Q8_0 (hybrid Delta Net).
# This probes the binding architectural risk for the dispatcher gate before
# committing to the parallel-decode build-out. The dispatcher itself is still
# pass-through at this commit — the only new behavior is the one-shot sync.
#
# Output: srv_probe.log shows "numa one-shot state sync: ..." per request.
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

# K=4 — primary ctx pinned to threads [0, 24); 3 aux ctxs on quarters 1-3
echo "=== boot server K=4 ==="
date

numactl --interleave=all $BIN/llama-server \
    -m "$TGT" -md "$DFT" \
    -t 96 -c 4096 -fa 1 \
    --spec-numa-quarters 4 \
    --draft-max 24 --draft-min 4 \
    --port 18099 \
    > srv_probe.log 2>&1 &
SRV_PID=$!

echo "  server pid=$SRV_PID"

for i in $(seq 1 180); do
    if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
        echo "  server ready after ${i}s"
        sleep 30   # warmup
        break
    fi
    sleep 1
done

# 3 prompts × 2 reps each — each request triggers a fresh SLOT_STATE_GENERATING transition
for r in 0 1; do
    for p in 0 1 2; do
        PROMPT="${PROMPTS[$p]}"
        echo "--- request p${p}_r${r} at $(date +%H:%M:%S) ---"
        curl -s http://localhost:18099/completion \
            -H 'Content-Type: application/json' \
            -d "$(jq -n --arg p "$PROMPT" '{prompt: $p, n_predict: 32, temperature: 0.0, p_split: 0.05}')" \
            > comp_p${p}_r${r}.json 2>&1
        sleep 2
    done
done

echo "=== kill server ==="
kill -INT $SRV_PID 2>/dev/null
sleep 3
kill -KILL $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null
echo "  done at $(date)"

echo ""
echo "=== state-sync timing summary ==="
grep "numa one-shot state sync" srv_probe.log || echo "(no sync lines found)"
