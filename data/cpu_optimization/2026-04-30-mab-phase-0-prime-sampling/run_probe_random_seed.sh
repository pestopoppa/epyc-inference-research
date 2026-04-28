#!/bin/bash
# MAB Phase 0' addendum — re-run on Coder with RANDOM seed (no fixed seed).
# The original Phase 0' run used seed=4242 fixed, which made the verifier
# deterministic regardless of tree branching choices. This addendum uses
# seed=-1 (let llama-server pick a random seed per request) to exercise
# true sampling stochasticity.
#
# Goal: distinguish "tree is dead at temp=0.7 even with stochastic sampling"
# (NO-GO confirmed for sampling regime) from "tree was dead only because of
# the deterministic seed" (Phase 0' result was misleading).
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-mab-phase-0-prime-sampling
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
DFT_CODER=/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

PROMPTS=(
'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
)

run_cell() {
    local SHAPE_TAG=$1
    local PSPLIT=$2

    echo "=== coder / $SHAPE_TAG (temp=0.7, p_split=$PSPLIT, seed=random) ==="
    date

    numactl --interleave=all $BIN/llama-server \
        -m "$CODER" -md "$DFT_CODER" \
        -t 96 -c 4096 -fa 1 \
        --port 18099 \
        > srv_rand_coder_${SHAPE_TAG}.log 2>&1 &
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
            echo "  --- coder/${SHAPE_TAG}/rand p${p}_r${r} at $(date +%H:%M:%S) ---"
            curl -s http://localhost:18099/completion \
                -H 'Content-Type: application/json' \
                -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" \
                      '{prompt: $p, n_predict: 64, temperature: 0.7, top_k: 40, top_p: 0.95, seed: -1, p_split: $ps}')" \
                > comp_rand_coder_${SHAPE_TAG}_p${p}_r${r}.json 2>&1
            sleep 1
        done
    done

    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

run_cell "linear" 0
run_cell "tree"   0.05

echo ""
echo "=== aggregate (random-seed, temp=0.7) ==="
for SHAPE_TAG in linear tree; do
    echo "--- coder / $SHAPE_TAG ---"
    for f in comp_rand_coder_${SHAPE_TAG}_*.json; do
        [ -f "$f" ] || continue
        TPS=$(jq -r '.timings.predicted_per_second // "n/a"' "$f" 2>/dev/null)
        ACC=$(jq -r '"\(.timings.draft_n_accepted // "?")/\(.timings.draft_n // "?")"' "$f" 2>/dev/null)
        HEAD=$(jq -r '.content // "(empty)"' "$f" 2>/dev/null | head -c 60 | tr '\n' ' ')
        echo "  $f: $TPS t/s, accept=$ACC | head: $HEAD"
    done
done

echo ""
echo "=== content diff: linear vs tree (random seeds — should DIFFER) ==="
for prompt in p0 p1 p2; do
    for rep in r0 r1 r2; do
        LIN=$(jq -r '.content[0:80]' comp_rand_coder_linear_${prompt}_${rep}.json 2>/dev/null)
        TREE=$(jq -r '.content[0:80]' comp_rand_coder_tree_${prompt}_${rep}.json 2>/dev/null)
        if [ "$LIN" = "$TREE" ]; then
            echo "  ${prompt}_${rep}: IDENTICAL (suspicious — should differ at temp=0.7 random seed)"
        else
            echo "  ${prompt}_${rep}: DIFFER"
        fi
    done
done
echo ""
echo "completed at $(date)"
