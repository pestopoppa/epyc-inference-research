#!/bin/bash
# Workload-shape coverage probe — test prefill (pp512, pp2048) and decode
# (tg32, tg64) on the production Q4_K_M sync-bound class with current
# stack flags. Different op-chain shape may show different optimization
# opportunities than tg-only measurements.
#
# Models: Qwen3-Coder-30B-A3B-Q4_K_M (canonical sync-bound MoE) +
#         Qwen3.6-35B-A3B-Q8_0 (frontdoor BW-bound MoE) +
#         Qwen3.6-27B-Q8_0 (dense Q8, BW-bound)
#
# Configs: c0 baseline (no env flags), c_full (CPU1 stack + CPU2 mbind on),
#          c_coalesce (+ GGML_BARRIER_COALESCE=1 from CPU4 Phase 1).
#
# Test types: pp512, pp2048, tg64 (all in single bench run).
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-workload-shape-coverage
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
Q8_FRONT=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
DENSE_Q8=/mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

if pgrep -af llama-server | grep -v zsh > /dev/null 2>&1; then
    echo "ERROR: pre-existing llama-server detected, aborting"
    exit 1
fi

run_cell() {
    local TAG=$1
    local MODEL=$2
    local CONFIG=$3

    local ENV_PREFIX=""
    case $CONFIG in
        c0)         ENV_PREFIX="" ;;
        c_full)     ENV_PREFIX="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1" ;;
        c_coalesce) ENV_PREFIX="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_BARRIER_COALESCE=1" ;;
    esac

    echo "=== $TAG / $CONFIG (env: ${ENV_PREFIX:-default}) ==="
    date

    if [ -z "$ENV_PREFIX" ]; then
        taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 512,2048 -n 64 -r 5 \
            > bench_${TAG}_${CONFIG}.log 2>&1
    else
        env $ENV_PREFIX taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 512,2048 -n 64 -r 5 \
            > bench_${TAG}_${CONFIG}.log 2>&1
    fi
    echo "  done at $(date)"
}

for MODEL_TAG_PATH in "coder $CODER" "q8_front $Q8_FRONT" "dense_q8 $DENSE_Q8"; do
    set -- $MODEL_TAG_PATH
    TAG=$1
    MODEL=$2
    if [ ! -f "$MODEL" ]; then
        echo "SKIP $TAG: $MODEL not found"
        continue
    fi
    for CONFIG in c0 c_full c_coalesce; do
        run_cell "$TAG" "$MODEL" "$CONFIG"
    done
done

echo ""
echo "=== aggregate (pp512, pp2048, tg64) ==="
python3 - << 'PYEOF'
import os, re, glob

def extract_all(f):
    if not os.path.exists(f):
        return {}
    out = {}
    for ln in open(f):
        if not ln.startswith('|'):
            continue
        cols = [c.strip() for c in ln.split('|')]
        # llama-bench format: | model | size | params | backend | threads | fa | test | t/s |
        if len(cols) < 9 or '----' in ln:
            continue
        # try to find test name (pp512, pp2048, tg64)
        test = None
        for c in cols:
            if re.match(r'^(pp|tg)\d+$', c):
                test = c
                break
        if not test:
            continue
        # find tps
        for c in cols:
            m = re.match(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', c)
            if m:
                out[test] = (float(m.group(1)), float(m.group(2)))
                break
    return out

models = ['coder', 'q8_front', 'dense_q8']
configs = ['c0', 'c_full', 'c_coalesce']
tests = ['pp512', 'pp2048', 'tg64']

for m in models:
    print(f'\n=== {m} ===')
    base = extract_all(f'bench_{m}_c0.log')
    for c in configs:
        d = extract_all(f'bench_{m}_{c}.log')
        for t in tests:
            v = d.get(t)
            if v:
                tps, sd = v
                if c == 'c0':
                    print(f'  {c:<14} {t:<8} {tps:.2f} ± {sd:.2f}')
                else:
                    base_v = base.get(t)
                    if base_v:
                        delta = (tps - base_v[0]) / base_v[0] * 100
                        print(f'  {c:<14} {t:<8} {tps:.2f} ± {sd:.2f}    Δ={delta:+.2f}%')
                    else:
                        print(f'  {c:<14} {t:<8} {tps:.2f} ± {sd:.2f}')
PYEOF

echo ""
echo "completed at $(date)"
