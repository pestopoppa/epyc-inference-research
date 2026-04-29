#!/bin/bash
# Remediation Phase C — MAB tree-shape selector Phase 0'' n=90 paired re-test
# under FULL CANONICAL recipe (OMP env + interleave + --mmap 0 implied).
# Original n=90 paired t-test on Coder showed -3.97% (p=0.012) for tree vs linear.
# Suspect was missing OMP env stack — tree branching may have more barrier
# overhead than linear, so broken-OMP regime would be asymmetric.
#
# Scope: Coder only (where original effect was found). 30 reps × 3 prompts ×
# 2 shapes = 180 requests. If gate flips, extend to REAP.

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-remediation-phase-C-mab
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
DFT=/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf

export LD_LIBRARY_PATH=$BIN
mkdir -p "$OUT"
cd "$OUT"
ulimit -c 0

PROMPTS=(
'Write a Python function to find the binary search of an integer in a sorted list. Return -1 if not found.'
'Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.'
'Write a Python function that computes the moving average of a CSV column over a window of N rows.'
)

N_REPS=30

run_cell() {
    local SHAPE_TAG=$1
    local PSPLIT=$2

    echo "=== coder / $SHAPE_TAG (temp=0.7, p_split=$PSPLIT, seed=random, n_reps=$N_REPS) ==="
    date

    # Apply FULL canonical recipe: OMP env + numactl --interleave=all
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    numactl --interleave=all $BIN/llama-server \
        -m "$CODER" -md "$DFT" \
        -t 96 -c 4096 -fa 1 \
        --port 18099 \
        > srv_${SHAPE_TAG}.log 2>&1 &
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
                > comp_${SHAPE_TAG}_p${p}_r${r}.json 2>&1
        done
        if [ $((r % 5)) -eq 4 ]; then
            echo "  ${SHAPE_TAG} reps 0-${r} done at $(date +%H:%M:%S)"
        fi
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
echo "=== paired t-test (Coder linear vs tree, n=90 paired) ==="
python3 - << 'PYEOF'
import os, json, math

def load(shape):
    out = {}
    for prompt in ['p0', 'p1', 'p2']:
        for rep in range(30):
            p = f"comp_{shape}_{prompt}_r{rep}.json"
            if not os.path.exists(p):
                continue
            try:
                with open(p) as f:
                    d = json.load(f)
                out[(prompt, rep)] = d['timings']['predicted_per_second']
            except Exception:
                pass
    return out

L = load('linear')
T = load('tree')
keys = sorted(set(L.keys()) & set(T.keys()))
if not keys:
    print("no paired data")
else:
    diffs = [T[k] - L[k] for k in keys]
    n = len(diffs)
    mean_d = sum(diffs) / n
    sd = math.sqrt(sum((d - mean_d) ** 2 for d in diffs) / (n - 1))
    se = sd / math.sqrt(n)
    t = mean_d / se if se > 0 else 0
    z = abs(t)
    p_approx = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    pct = (mean_d / (sum(L[k] for k in keys) / n)) * 100
    L_mean = sum(L[k] for k in keys) / n
    T_mean = sum(T[k] for k in keys) / n
    print(f"linear mean: {L_mean:.3f} t/s")
    print(f"tree mean:   {T_mean:.3f} t/s")
    print(f"paired n={n}, Δ_mean={mean_d:+.3f} ({pct:+.2f}%), SD={sd:.3f}, t={t:.3f}, p≈{p_approx:.4f}")
PYEOF

echo ""
echo "completed at $(date)"
