#!/bin/bash
# Multi-arch coverage probe — test CPU1 stack + CPU2 mbind on:
#   1. Qwen3.6-27B-Q8_0 (dense Q8, BW-bound)
#   2. Nemotron-Nano-9B-v2-Q8_0 (hybrid SSM)
#   3. gemma-4-31B-it-Q4_K_M (dense Q4_K_M)
#
# Hypothesis: existing tools (CPU1, CPU2 mbind) were tested ONLY on Q4_K_M
# MoE class. Different bottleneck class on dense / hybrid SSM may make
# them NET POSITIVE there.
#
# Configs:
#   c0 = baseline (no env flags, default GGML_NUMA_REPACK_INTERLEAVE=1)
#   c1 = CPU1 stack only (CCD_POOLS + CCD_WORK_DIST + BARRIER_LOCAL_BETWEEN_OPS)
#   c2 = CPU2 mbind off (GGML_NUMA_REPACK_INTERLEAVE=0)
#   c3 = CPU1 + CPU2-off
#
# 5-rep canonical: taskset -c 0-95 -t 96 -fa 1, no other env vars.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-multi-arch-coverage
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

DENSE_Q8=/mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
HYBRID=/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf
DENSE_Q4=/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

# Verify zero processes before start (CPU20 protocol)
if pgrep -af llama-server | grep -v zsh > /dev/null 2>&1; then
    echo "ERROR: pre-existing llama-server detected, aborting"
    pgrep -af llama-server | grep -v zsh
    exit 1
fi

run_cell() {
    local TAG=$1
    local MODEL=$2
    local CONFIG=$3       # c0, c1, c2, c3

    local ENV_PREFIX=""
    case $CONFIG in
        c0) ENV_PREFIX="" ;;
        c1) ENV_PREFIX="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1" ;;
        c2) ENV_PREFIX="GGML_NUMA_REPACK_INTERLEAVE=0" ;;
        c3) ENV_PREFIX="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0" ;;
    esac

    echo "=== $TAG / $CONFIG (env: ${ENV_PREFIX:-default}) ==="
    date

    if [ -z "$ENV_PREFIX" ]; then
        taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 0 -n 64 -r 5 \
            > bench_${TAG}_${CONFIG}.log 2>&1
    else
        env $ENV_PREFIX taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 0 -n 64 -r 5 \
            > bench_${TAG}_${CONFIG}.log 2>&1
    fi
    echo "  done at $(date)"
}

# Run all configs sequentially per model
for MODEL_TAG_PATH in "qwen36_27b $DENSE_Q8" "nemotron_9b $HYBRID" "gemma_31b $DENSE_Q4"; do
    set -- $MODEL_TAG_PATH
    TAG=$1
    MODEL=$2
    if [ ! -f "$MODEL" ]; then
        echo "SKIP $TAG: $MODEL not found"
        continue
    fi
    for CONFIG in c0 c1 c2 c3; do
        run_cell "$TAG" "$MODEL" "$CONFIG"
    done
done

echo ""
echo "=== aggregate results ==="
python3 - << 'PYEOF'
import os, re, glob

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

models = ['qwen36_27b', 'nemotron_9b', 'gemma_31b']
configs = ['c0', 'c1', 'c2', 'c3']
config_label = {'c0': 'baseline', 'c1': 'CPU1', 'c2': 'CPU2-off', 'c3': 'CPU1+CPU2off'}

print(f"{'Model':<14} {'Config':<14} {'tg64 t/s':<20} {'Δ vs baseline':<15}")
print('-' * 70)
for m in models:
    base_v = None
    for c in configs:
        v = extract_tg(f'bench_{m}_{c}.log')
        if v is None:
            print(f"  {m:<14} {config_label[c]:<14} (missing)")
            continue
        tps, sd = v
        if c == 'c0':
            base_v = tps
            delta = '(baseline)'
        else:
            delta = f"{(tps - base_v) / base_v * 100:+.2f}%"
        print(f"  {m:<14} {config_label[c]:<14} {tps:.2f} ± {sd:.2f}     {delta}")
    print()
PYEOF

echo ""
echo "completed at $(date)"
