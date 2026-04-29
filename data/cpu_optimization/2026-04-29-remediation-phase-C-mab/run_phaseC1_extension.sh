#!/bin/bash
# Remediation Phase C.1 — Extension after gate FLIP
# Phase C n=90 showed +2.72% (was -3.97%), p=0.237 (not significant).
# Extend with 30 more reps per Coder cell (cumulative n=60 per cell, n=180 paired)
# AND add REAP cells (n=30 per cell, n=90 paired) to check generalization.

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-remediation-phase-C-mab
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

# Reps 30..59 (extension layer for Coder)
run_cell_extension() {
    local MODEL_TAG=$1
    local TGT=$2
    local SHAPE_TAG=$3
    local PSPLIT=$4
    local R_START=$5
    local R_END=$6

    echo "=== $MODEL_TAG / $SHAPE_TAG (extension reps $R_START-$R_END) ==="
    date

    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    numactl --interleave=all $BIN/llama-server \
        -m "$TGT" -md "$DFT" \
        -t 96 -c 4096 -fa 1 \
        --port 18099 \
        > srv_${MODEL_TAG}_${SHAPE_TAG}_ext.log 2>&1 &
    SRV_PID=$!

    for i in $(seq 1 240); do
        if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
            echo "  ready after ${i}s"
            sleep 30
            break
        fi
        sleep 1
    done

    for r in $(seq $R_START $R_END); do
        for p in 0 1 2; do
            PROMPT="${PROMPTS[$p]}"
            curl -s http://localhost:18099/completion \
                -H 'Content-Type: application/json' \
                -d "$(jq -n --arg p "$PROMPT" --argjson ps "$PSPLIT" \
                      '{prompt: $p, n_predict: 64, temperature: 0.7, top_k: 40, top_p: 0.95, seed: -1, p_split: $ps}')" \
                > comp_${MODEL_TAG}_${SHAPE_TAG}_p${p}_r${r}.json 2>&1
        done
        if [ $((r % 5)) -eq 4 ]; then
            echo "  ${MODEL_TAG}/${SHAPE_TAG} reps up to $r done at $(date +%H:%M:%S)"
        fi
    done

    kill -INT $SRV_PID 2>/dev/null
    sleep 3
    kill -KILL $SRV_PID 2>/dev/null
    wait $SRV_PID 2>/dev/null
    echo "  done at $(date)"
}

# Coder extension: reps 30..59 (renames to coder_ prefix; original was un-prefixed)
# (Original Phase C used filenames comp_linear_p* and comp_tree_p*; rename for clarity)
echo "=== Renaming original Phase C files for Coder ==="
for f in comp_linear_p*_r*.json; do
    [ -f "$f" ] && mv -n "$f" "comp_coder_${f#comp_}"
done
for f in comp_tree_p*_r*.json; do
    [ -f "$f" ] && mv -n "$f" "comp_coder_${f#comp_}"
done

# Coder extension reps 30..59
run_cell_extension "coder" "$CODER" "linear" 0    30 59
run_cell_extension "coder" "$CODER" "tree"   0.05 30 59

# REAP fresh n=30
if [ -f "$REAP" ]; then
    run_cell_extension "reap" "$REAP" "linear" 0    0 29
    run_cell_extension "reap" "$REAP" "tree"   0.05 0 29
else
    echo "REAP not found at $REAP — skipping"
fi

echo ""
echo "=== Phase C.1 cumulative analysis ==="
python3 - << 'PYEOF'
import os, json, math, glob

def paired_test(model, n_max):
    L = {}; T = {}
    for prompt in ['p0', 'p1', 'p2']:
        for rep in range(n_max):
            for shape, store in [('linear', L), ('tree', T)]:
                p = f"comp_{model}_{shape}_{prompt}_r{rep}.json"
                if os.path.exists(p):
                    try:
                        d = json.load(open(p))
                        store[(prompt, rep)] = d['timings']['predicted_per_second']
                    except: pass
    keys = sorted(set(L.keys()) & set(T.keys()))
    if not keys:
        return None
    diffs = [T[k] - L[k] for k in keys]
    n = len(diffs)
    mean_d = sum(diffs) / n
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0
    se = sd / math.sqrt(n) if n > 0 else 0
    t = mean_d / se if se > 0 else 0
    z = abs(t)
    p_approx = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    L_mean = sum(L[k] for k in keys) / n
    T_mean = sum(T[k] for k in keys) / n
    pct = (mean_d / L_mean) * 100 if L_mean > 0 else 0
    return n, L_mean, T_mean, mean_d, sd, t, p_approx, pct

for model, n_max in [('coder', 60), ('reap', 30)]:
    r = paired_test(model, n_max)
    if r is None:
        print(f"{model}: no paired data")
        continue
    n, L_m, T_m, md, sd, t, p, pct = r
    print(f"{model}: n={n}  linear={L_m:.3f}  tree={T_m:.3f}  Δ={md:+.3f} ({pct:+.2f}%)  SD={sd:.3f}  t={t:.3f}  p≈{p:.4f}")
PYEOF

echo ""
echo "completed at $(date)"
