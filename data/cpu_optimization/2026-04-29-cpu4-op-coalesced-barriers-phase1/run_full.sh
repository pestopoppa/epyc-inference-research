#!/bin/bash
# CPU4 Phase 1 full measurement — GGML_BARRIER_COALESCE off vs on, on
# 3 sync-bound Q4_K_M models. Both PPL bit-exact (chunk-3 quick gate)
# AND 5-rep canonical tg64.
#
# Pre-condition: smoke test (run_smoke.sh) PASSED with identical output
# at COALESCE=0 vs =1.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-cpu4-op-coalesced-barriers-phase1
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
NEXT80=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
REAP=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf

WIKI=/mnt/raid0/llm/data/wiki.test.raw

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

# PPL chunk-3 quick gate (covers prefill + a few generation steps; bit-exact
# diagnostic catches any silent reordering bugs).
ppl_run() {
    local TAG=$1
    local MODEL=$2
    local COALESCE=$3
    echo "=== PPL: $TAG (COALESCE=$COALESCE) ==="
    date
    GGML_BARRIER_COALESCE=$COALESCE numactl --interleave=all $BIN/llama-perplexity \
        -m "$MODEL" -f "$WIKI" \
        -t 96 -fa 1 --chunks 3 \
        > ppl_${TAG}_c${COALESCE}.log 2>&1 || true
    echo "  done at $(date)"
}

# tg64 5-rep canonical (taskset -c 0-95 -t 96 -fa 1, no env vars except COALESCE)
bench_run() {
    local TAG=$1
    local MODEL=$2
    local COALESCE=$3
    echo "=== bench: $TAG tg64 -r 5 (COALESCE=$COALESCE) ==="
    date
    GGML_BARRIER_COALESCE=$COALESCE taskset -c 0-95 $BIN/llama-bench \
        -m "$MODEL" -t 96 -fa 1 \
        -p 0 -n 64 -r 5 \
        > bench_${TAG}_c${COALESCE}.log 2>&1
    echo "  done at $(date)"
}

# PPL pass first (correctness gate must pass before we trust throughput).
for COALESCE in 0 1; do
    ppl_run "coder" "$CODER" $COALESCE
    ppl_run "reap"  "$REAP"  $COALESCE
done

echo ""
echo "=== PPL chunk-3 comparison (must be BIT-EXACT for Phase 1 to ship) ==="
for MODEL in coder reap; do
    P0=$(grep -E "^\s*\[3\]" ppl_${MODEL}_c0.log | tail -1 | awk '{print $NF}')
    P1=$(grep -E "^\s*\[3\]" ppl_${MODEL}_c1.log | tail -1 | awk '{print $NF}')
    if [ "$P0" = "$P1" ]; then
        echo "  $MODEL: chunk-3 PPL=${P0}  ✅ BIT-EXACT"
    else
        echo "  $MODEL: chunk-3 PPL c0=${P0} vs c1=${P1}  ❌ DRIFT — coalesce path broken"
    fi
done

# tg64 5-rep canonical on all 3 sync-bound models
for COALESCE in 0 1; do
    bench_run "coder"   "$CODER"  $COALESCE
    bench_run "next80"  "$NEXT80" $COALESCE
    bench_run "reap"    "$REAP"   $COALESCE
done

echo ""
echo "=== tg64 5-rep results (mean ± std) ==="
for MODEL in coder next80 reap; do
    for COALESCE in 0 1; do
        F="bench_${MODEL}_c${COALESCE}.log"
        # llama-bench outputs CSV-ish format on the last line
        if [ -f "$F" ]; then
            T=$(grep -E "^\| " $F | tail -1 | awk -F'|' '{print $7}' | tr -d ' ')
            echo "  $MODEL c$COALESCE: $T"
        fi
    done
done

echo ""
echo "=== Δ tg64 (c1 vs c0) per model ==="
python3 - << 'PYEOF'
import re, os, sys
models = ['coder', 'next80', 'reap']
for m in models:
    pairs = {}
    for c in (0, 1):
        f = f'bench_{m}_c{c}.log'
        if not os.path.exists(f):
            continue
        # Parse llama-bench's last data row: |  ...  |  TPS ± std  |
        rows = [ln for ln in open(f) if ln.strip().startswith('|') and 'tg' in ln]
        if not rows:
            # Fallback: any data row
            rows = [ln for ln in open(f) if ln.strip().startswith('|') and 'pp' not in ln and '----' not in ln]
        if rows:
            cols = [c.strip() for c in rows[-1].split('|')]
            for col in cols:
                m2 = re.match(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', col)
                if m2:
                    pairs[c] = (float(m2.group(1)), float(m2.group(2)))
                    break
    if 0 in pairs and 1 in pairs:
        tps0, sd0 = pairs[0]
        tps1, sd1 = pairs[1]
        delta = (tps1 - tps0) / tps0 * 100
        print(f'  {m}: c0={tps0:.2f}±{sd0:.2f}  c1={tps1:.2f}±{sd1:.2f}  Δ={delta:+.2f}%')
PYEOF

echo ""
echo "completed at $(date)"
