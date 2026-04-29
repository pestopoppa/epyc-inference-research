#!/bin/bash
# Replicate the Nemotron +13.43% headline finding at n=30.
# Also replicate gemma-4-31B -12% regression to verify it's real.
# Probe A used n=5 with baseline std 0.64 → uncertain at p<0.05.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-multi-arch-coverage
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
HYBRID=/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf
DENSE_Q4=/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

run_cell() {
    local TAG=$1
    local MODEL=$2
    local CONFIG=$3
    local ENV_PREFIX=$4

    echo "=== $TAG / $CONFIG (n=30) ==="
    date

    if [ -z "$ENV_PREFIX" ]; then
        taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 0 -n 64 -r 30 \
            > rep30_${TAG}_${CONFIG}.log 2>&1
    else
        env $ENV_PREFIX taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 0 -n 64 -r 30 \
            > rep30_${TAG}_${CONFIG}.log 2>&1
    fi
    echo "  done at $(date)"
}

# Nemotron: replicate c0 (baseline) vs c3 (CPU1 + CPU2-off)
run_cell "nemotron_9b" "$HYBRID" "c0" ""
run_cell "nemotron_9b" "$HYBRID" "c3" "GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0"

# gemma-31B: replicate c0 (baseline) vs c1 (CPU1 alone, the specific failing config)
run_cell "gemma_31b" "$DENSE_Q4" "c0" ""
run_cell "gemma_31b" "$DENSE_Q4" "c1" "GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1"

echo ""
echo "=== high-rep replication summary (n=30) ==="
python3 - << 'PYEOF'
import os, re, math

def extract_tg(f):
    if not os.path.exists(f):
        return None
    for ln in open(f):
        if ln.startswith('|') and 'tg' in ln:
            for col in [c.strip() for c in ln.split('|')]:
                m = re.match(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', col)
                if m:
                    return (float(m.group(1)), float(m.group(2)))
    return None

cells = [
    ('nemotron_9b', 'c0', 'baseline (default)'),
    ('nemotron_9b', 'c3', 'CPU1 + CPU2-off'),
    ('gemma_31b',   'c0', 'baseline (default)'),
    ('gemma_31b',   'c1', 'CPU1 alone'),
]

results = {}
for tag, cfg, desc in cells:
    f = f'rep30_{tag}_{cfg}.log'
    v = extract_tg(f)
    if v:
        tps, sd = v
        n = 30
        sem = sd / math.sqrt(n)
        ci_low = tps - 1.96 * sem
        ci_high = tps + 1.96 * sem
        results[(tag, cfg)] = (tps, sd, ci_low, ci_high)
        print(f'  {tag:<14} {cfg:<6} {desc:<25} {tps:.3f} ± {sd:.3f}  (95% CI: [{ci_low:.3f}, {ci_high:.3f}])')

print()
print('=== paired comparison (n=30) ===')
for tag, c_base, c_test in [('nemotron_9b', 'c0', 'c3'), ('gemma_31b', 'c0', 'c1')]:
    base = results.get((tag, c_base))
    test = results.get((tag, c_test))
    if base and test:
        tps_b, sd_b, _, _ = base
        tps_t, sd_t, _, _ = test
        delta = (tps_t - tps_b) / tps_b * 100
        sem_b = sd_b / math.sqrt(30)
        sem_t = sd_t / math.sqrt(30)
        sem_diff = math.sqrt(sem_b**2 + sem_t**2)
        # unpaired t-test for unequal variance (Welch's), df ~ large
        t = (tps_t - tps_b) / sem_diff if sem_diff > 0 else 0
        # approx p via normal
        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
        print(f'  {tag} {c_test} vs {c_base}: Δ = {delta:+.2f}%  t = {t:.3f}  p ≈ {p:.4f}')
PYEOF

echo ""
echo "completed at $(date)"
