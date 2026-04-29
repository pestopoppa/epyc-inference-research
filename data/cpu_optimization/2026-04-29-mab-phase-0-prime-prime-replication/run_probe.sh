#!/bin/bash
# MAB tree-shape selector — Phase 0'' (high-rep replication of the
# random-seed sampling-regime signal found in Phase 0').
#
# Phase 0' (2026-04-30) found Coder random-seed temp=0.7: tree +9.6%
# over linear at n=9 (p≈0.23, NOT significant). This Phase 0''
# replicates with n=30 reps per cell on BOTH Coder and REAP to either
# clear the +9.6% signal at p<0.05 (GO for Phase 1 implementation) or
# show the signal collapses (NO-GO closes the sampling-regime branch).
#
# Method: same setup as Phase 0' random-seed addendum (random per-
# request seed, temp=0.7, top_k=40, top_p=0.95, --draft-max=24
# --draft-min=4) except n=30 instead of 3 reps per (model, shape,
# prompt) cell.
#
# Total: 2 models × 2 shapes × 3 prompts × 30 reps = 360 requests.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-mab-phase-0-prime-prime-replication
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
DFT=/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf
REAP=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

PROMPTS=(
'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
)

N_REPS=30

run_cell() {
    local MODEL_TAG=$1
    local TGT=$2
    local SHAPE_TAG=$3
    local PSPLIT=$4

    echo "=== $MODEL_TAG / $SHAPE_TAG (temp=0.7, p_split=$PSPLIT, seed=random, n_reps=$N_REPS) ==="
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

    for r in $(seq 0 $((N_REPS - 1))); do
        for p in 0 1 2; do
            PROMPT="${PROMPTS[$p]}"
            curl -s http://localhost:18099/completion \
                -H 'Content-Type: application/json' \
                -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" \
                      '{prompt: $p, n_predict: 64, temperature: 0.7, top_k: 40, top_p: 0.95, seed: -1, p_split: $ps}')" \
                > comp_${MODEL_TAG}_${SHAPE_TAG}_p${p}_r${r}.json 2>&1
        done
        # one short status line every 5 reps
        if [ $((r % 5)) -eq 4 ]; then
            echo "  ${MODEL_TAG}/${SHAPE_TAG} reps 0-${r} done at $(date +%H:%M:%S)"
        fi
    done

    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

# Coder cells
run_cell "coder" "$CODER" "linear" 0
run_cell "coder" "$CODER" "tree"   0.05

# REAP cells (same drafter as Coder per prior probes)
if [ -f "$REAP" ]; then
    run_cell "reap" "$REAP" "linear" 0
    run_cell "reap" "$REAP" "tree"   0.05
else
    echo "REAP not found at $REAP — skipping"
fi

echo ""
echo "=== aggregate (n=$N_REPS per cell) ==="
for MODEL_TAG in coder reap; do
    for SHAPE_TAG in linear tree; do
        echo "--- $MODEL_TAG / $SHAPE_TAG ---"
        FILES=$(ls comp_${MODEL_TAG}_${SHAPE_TAG}_p*_r*.json 2>/dev/null)
        [ -z "$FILES" ] && { echo "  (no files)"; continue; }
        STATS=$(jq -s 'map(.timings.predicted_per_second) | {n: length, mean: (add/length), std: ((add/length) as $m | (map(. - $m | . * .) | add / length | sqrt))}' $FILES)
        echo "  $STATS"
    done
done

echo ""
echo "=== per-prompt × shape breakdown ==="
for MODEL_TAG in coder reap; do
    for prompt in p0 p1 p2; do
        L_MEAN=$(jq -s 'map(.timings.predicted_per_second) | add/length' comp_${MODEL_TAG}_linear_${prompt}_*.json 2>/dev/null)
        T_MEAN=$(jq -s 'map(.timings.predicted_per_second) | add/length' comp_${MODEL_TAG}_tree_${prompt}_*.json 2>/dev/null)
        if [ -n "$L_MEAN" ] && [ -n "$T_MEAN" ]; then
            DELTA=$(echo "scale=4; ($T_MEAN - $L_MEAN) / $L_MEAN * 100" | bc 2>/dev/null)
            echo "  $MODEL_TAG / $prompt: linear=$L_MEAN  tree=$T_MEAN  Δ=${DELTA}%"
        fi
    done
done

echo ""
echo "=== paired t-test (per-prompt-per-rep linear vs tree) ==="
# Generate a simple paired-test summary in Python via inline awk; if Python is available, we'll do proper t-test.
python3 - << 'PYEOF'
import os, json, math
def load(model, shape):
    out = {}
    for prompt in ['p0', 'p1', 'p2']:
        for rep in range(30):
            p = f"comp_{model}_{shape}_{prompt}_r{rep}.json"
            if not os.path.exists(p):
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
                out[(prompt, rep)] = d['timings']['predicted_per_second']
            except Exception:
                pass
    return out

for model in ['coder', 'reap']:
    L = load(model, 'linear')
    T = load(model, 'tree')
    keys = sorted(set(L.keys()) & set(T.keys()))
    if not keys:
        print(f"  {model}: no paired data")
        continue
    diffs = [T[k] - L[k] for k in keys]
    n = len(diffs)
    if n < 2:
        print(f"  {model}: n={n} too few")
        continue
    mean_d = sum(diffs) / n
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = mean_d / se if se > 0 else 0
    # rough one-sided p-value via normal approx (n>=20 OK)
    z = abs(t)
    # Approximate two-sided p using complementary error function
    p_approx = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    pct = (mean_d / (sum(L[k] for k in keys) / n)) * 100
    print(f"  {model}: n={n} paired,  Δ_mean={mean_d:.3f} t/s ({pct:+.2f}%), SD={sd:.3f}, t={t:.3f}, p≈{p_approx:.4f}")
PYEOF

echo ""
echo "completed at $(date)"
