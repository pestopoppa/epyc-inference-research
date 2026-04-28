#!/bin/bash
# MAB tree-shape selector — Phase 0' under sampling regime (temp=0.7).
#
# Phase 0 (2026-04-29) closed NO-GO at temperature=0.0 because the verifier
# collapses tree to greedy path: byte-identical outputs for linear vs tree,
# tree adds wasted compute on non-greedy branches. Closure scope explicitly
# does NOT generalize to higher-temperature sampling (where the verifier
# might accept non-greedy branches).
#
# This Phase 0' re-runs the same probe but with temperature=0.7 to test
# whether the tree mechanism shows gain in the sampling regime.
#
# Same models, same prompts, same builds, same drafter as Phase 0 — only
# temperature is changed.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-mab-phase-0-prime-sampling
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
DFT_CODER=/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf
REAP=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

PROMPTS=(
'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
)

# Each cell: 1 model × 1 shape × 3 prompts × 3 reps; total = 2 models × 2 shapes × 9 = 36 requests.
run_cell() {
    local MODEL_TAG=$1
    local TGT=$2
    local DFT=$3
    local SHAPE_TAG=$4
    local PSPLIT=$5

    echo "=== $MODEL_TAG / $SHAPE_TAG (temp=0.7, p_split=$PSPLIT) ==="
    date

    numactl --interleave=all $BIN/llama-server \
        -m "$TGT" -md "$DFT" \
        -t 96 -c 4096 -fa 1 \
        --port 18099 \
        > srv_${MODEL_TAG}_${SHAPE_TAG}.log 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
            echo "  ready after ${i}s"
            sleep 30
            break
        fi
        sleep 1
    done

    for r in 0 1 2; do
        for p in 0 1 2; do
            PROMPT="${PROMPTS[$p]}"
            echo "  --- ${MODEL_TAG}/${SHAPE_TAG} p${p}_r${r} at $(date +%H:%M:%S) ---"
            curl -s http://localhost:18099/completion \
                -H 'Content-Type: application/json' \
                -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" \
                      '{prompt: $p, n_predict: 64, temperature: 0.7, top_k: 40, top_p: 0.95, seed: 4242, p_split: $ps}')" \
                > comp_${MODEL_TAG}_${SHAPE_TAG}_p${p}_r${r}.json 2>&1
            sleep 1
        done
    done

    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

# Coder-30B Q4_K_M
run_cell "coder" "$CODER" "$DFT_CODER" "linear" 0
run_cell "coder" "$CODER" "$DFT_CODER" "tree"   0.05

# REAP-246B Q4_K_M (drafter same as Coder per Phase 0; vocab-compatible variant)
if [ -f "$REAP" ]; then
    run_cell "reap" "$REAP" "$DFT_CODER" "linear" 0
    run_cell "reap" "$REAP" "$DFT_CODER" "tree"   0.05
else
    echo "REAP not at $REAP — skipping"
fi

echo ""
echo "=== aggregate ==="
for MODEL_TAG in coder reap; do
    for SHAPE_TAG in linear tree; do
        echo "--- $MODEL_TAG / $SHAPE_TAG (temp=0.7) ---"
        for f in comp_${MODEL_TAG}_${SHAPE_TAG}_*.json; do
            [ -f "$f" ] || continue
            TPS=$(jq -r '.timings.predicted_per_second // "n/a"' "$f" 2>/dev/null)
            ACC=$(jq -r '"\(.timings.draft_n_accepted // "?")/\(.timings.draft_n // "?")"' "$f" 2>/dev/null)
            HEAD=$(jq -r '.content // "(empty)"' "$f" 2>/dev/null | head -c 60 | tr '\n' ' ')
            echo "  $f: $TPS t/s, accept=$ACC, content head: $HEAD"
        done
    done
done

echo ""
echo "completed at $(date)"
