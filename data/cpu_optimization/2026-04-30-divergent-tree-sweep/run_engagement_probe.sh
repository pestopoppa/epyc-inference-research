#!/bin/bash
# Phase 1.1 sweep — engagement probe.
# Find a (p_split, temperature, prompt) configuration where the dispatcher's
# K-parallel block actually engages (numa_alt_paths populates with >=1 alt
# path on at least some rounds).
#
# We launch K=4 once with --verbose and hit it with multiple
# (prompt, p_split, temperature) requests, then grep the server log for the
# DBG-level "numa_select_top_k_alt_paths" line emitted in update_batch.
# A line with "alt=N paths" where N >= 1 means the dispatcher engaged
# K-parallel verify on that round.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-divergent-tree-sweep
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
TGT=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
DFT=/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

echo "=== boot K=4 --verbose ==="
date

numactl --interleave=all $BIN/llama-server \
    -m "$TGT" -md "$DFT" \
    -t 96 -c 4096 -fa 1 \
    --spec-numa-quarters 4 \
    --draft-max 24 --draft-min 4 \
    --port 18099 \
    --verbose \
    > srv_engage.log 2>&1 &
SRV_PID=$!
echo "  pid=$SRV_PID"

for i in $(seq 1 180); do
    if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
        echo "  ready after ${i}s"
        sleep 30
        break
    fi
    sleep 1
done

# Test matrix: 5 prompts × 4 (p_split, temp) configs.
# Key levers:
#   - LOW p_split keeps more drafter candidates (=> more tree branches).
#   - HIGHER target temperature increases drafter/target disagreement
#     (=> alt paths can win).
declare -a PROMPTS=(
    'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
    'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
    'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
    'Write a haiku about quantum entanglement.'
    'A philosophical question: what does it mean to be conscious? Answer in three sentences.'
)
declare -a PROMPT_NAMES=(binary lru moving haiku conscious)

declare -a CONFIGS=(
    '0.05 0.0'
    '0.001 0.0'
    '0.05 0.7'
    '0.001 0.7'
)
declare -a CONFIG_NAMES=(p005_t0 p001_t0 p005_t7 p001_t7)

for ci in "${!CONFIGS[@]}"; do
    cfg=(${CONFIGS[$ci]})
    PSPLIT=${cfg[0]}
    TEMP=${cfg[1]}
    CNAME=${CONFIG_NAMES[$ci]}
    for pi in "${!PROMPTS[@]}"; do
        PROMPT="${PROMPTS[$pi]}"
        PNAME=${PROMPT_NAMES[$pi]}
        echo "  --- $CNAME / $PNAME at $(date +%H:%M:%S) ---"
        curl -s http://localhost:18099/completion \
            -H 'Content-Type: application/json' \
            -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" --argjson tp "$TEMP" \
                  '{prompt: $p, n_predict: 32, temperature: $tp, p_split: $ps}')" \
            > comp_engage_${CNAME}_${PNAME}.json 2>&1
        sleep 1
    done
done

kill -INT $SRV_PID 2>/dev/null
sleep 3
kill -KILL $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null

echo ""
echo "=== alt_paths engagement summary ==="
echo "(rows with alt > 0 = K-parallel verify actually triggered)"
grep "numa_select_top_k_alt_paths" srv_engage.log | head -50

echo ""
echo "=== K-parallel verify hit count ==="
grep -c "numa K-parallel verify" srv_engage.log

echo ""
echo "=== completed at $(date) ==="
