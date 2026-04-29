#!/bin/bash
# Replicate Coder tg64 measurement at higher n to determine whether the
# initial +20.5% delta is real signal or measurement noise.
# n=5 reps showed ±33% CV; need n>=30 to detect <5% effects.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-cpu4-op-coalesced-barriers-phase1
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

# Alternate the configs to control for system drift
for trial in 1 2 3; do
    for COALESCE in 0 1; do
        echo "=== trial $trial / coder tg64 -r 5 (COALESCE=$COALESCE) ==="
        date
        GGML_BARRIER_COALESCE=$COALESCE taskset -c 0-95 $BIN/llama-bench \
            -m "$CODER" -t 96 -fa 1 \
            -p 0 -n 64 -r 5 \
            > bench_replicate_coder_t${trial}_c${COALESCE}.log 2>&1
        echo "  done at $(date)"
    done
done

echo ""
echo "=== aggregated over 3 alternated trials ==="
python3 - << 'PYEOF'
import re, os, glob
def extract(f):
    if not os.path.exists(f):
        return None
    for ln in open(f):
        if ln.startswith('|') and 'tg' in ln:
            parts = [p.strip() for p in ln.split('|')]
            for p in parts:
                m = re.match(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', p)
                if m:
                    return (float(m.group(1)), float(m.group(2)))
    return None

vals = {0: [], 1: []}
for trial in (1, 2, 3):
    for c in (0, 1):
        v = extract(f'bench_replicate_coder_t{trial}_c{c}.log')
        if v:
            vals[c].append(v[0])
            print(f'  trial {trial} c{c}: {v[0]:.2f} ± {v[1]:.2f}')
import statistics
for c in (0, 1):
    if vals[c]:
        m = statistics.mean(vals[c])
        sd = statistics.stdev(vals[c]) if len(vals[c]) > 1 else 0
        print(f'  COALESCE={c} aggregate over trials: mean={m:.2f}, sd={sd:.2f}')
if len(vals[0]) >= 2 and len(vals[1]) >= 2:
    m0 = statistics.mean(vals[0]); m1 = statistics.mean(vals[1])
    sd0 = statistics.stdev(vals[0]); sd1 = statistics.stdev(vals[1])
    delta = (m1 - m0) / m0 * 100
    print(f'  Δ tg64 (c1 vs c0): {delta:+.2f}%')
PYEOF

echo ""
echo "completed at $(date)"
