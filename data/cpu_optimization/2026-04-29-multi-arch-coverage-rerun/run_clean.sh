#!/bin/bash
# Multi-arch coverage CLEAN RE-RUN (n=15) after discovering possible
# poisoning of the first-pass measurements (3 other claude sessions
# active during 2026-04-29 09:01-09:19; Probe A first-pass +13.43%
# Nemotron headline didn't replicate at n=30).
#
# Per-cell pgrep guard: abort if any other agent's llama-* process
# appears mid-run.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-multi-arch-coverage-rerun
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

DENSE_Q8=/mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
HYBRID=/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf
DENSE_Q4=/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf

export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

check_clean() {
    local context=$1
    local procs=$(pgrep -af "llama-bench|llama-cli|llama-server|llama-perplex|llama-mtmd-cli" | grep -v zsh)
    if [ -n "$procs" ]; then
        echo "ABORT [$context]: foreign llama process detected!" | tee -a guard_violations.log
        echo "$procs" | tee -a guard_violations.log
        return 1
    fi
    return 0
}

if ! check_clean "pre-flight"; then exit 1; fi

echo "=== START $(date) ==="
echo "Other claude sessions active: $(ps -ef | grep -E 'claude\b' | grep -v grep | wc -l)"
uptime

run_cell() {
    local TAG=$1
    local MODEL=$2
    local CONFIG=$3
    local N_REPS=15

    if ! check_clean "before $TAG/$CONFIG"; then return 1; fi

    local ENV_PREFIX=""
    case $CONFIG in
        c0) ENV_PREFIX="" ;;
        c1) ENV_PREFIX="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1" ;;
        c2) ENV_PREFIX="GGML_NUMA_REPACK_INTERLEAVE=0" ;;
        c3) ENV_PREFIX="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0" ;;
    esac

    echo "=== $TAG / $CONFIG (n=$N_REPS, env: ${ENV_PREFIX:-default}) ==="
    date

    if [ -z "$ENV_PREFIX" ]; then
        taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 0 -n 64 -r $N_REPS \
            > clean_${TAG}_${CONFIG}.log 2>&1
    else
        env $ENV_PREFIX taskset -c 0-95 $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 \
            -p 0 -n 64 -r $N_REPS \
            > clean_${TAG}_${CONFIG}.log 2>&1
    fi
    echo "  done at $(date)"

    if ! check_clean "after $TAG/$CONFIG"; then
        echo "WARNING: foreign process appeared during/after $TAG/$CONFIG; cell may be poisoned"
        return 1
    fi

    sleep 5  # stabilize between cells
}

# Order: smallest model first (cheapest to verify clean), then larger.
for MODEL_TAG_PATH in "nemotron_9b $HYBRID" "qwen36_27b $DENSE_Q8" "gemma_31b $DENSE_Q4"; do
    set -- $MODEL_TAG_PATH
    TAG=$1
    MODEL=$2
    if [ ! -f "$MODEL" ]; then
        echo "SKIP $TAG: $MODEL not found"
        continue
    fi
    for CONFIG in c0 c1 c2 c3; do
        run_cell "$TAG" "$MODEL" "$CONFIG" || echo "  cell flagged but continuing"
    done
done

echo ""
echo "=== aggregate (n=15, clean re-run) ==="
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

models = ['nemotron_9b', 'qwen36_27b', 'gemma_31b']
configs = ['c0', 'c1', 'c2', 'c3']
config_label = {'c0': 'baseline', 'c1': 'CPU1', 'c2': 'CPU2-off', 'c3': 'CPU1+CPU2off'}

print(f"{'Model':<14} {'Config':<14} {'tg64 t/s (n=15)':<22} {'95% CI':<22} {'Δ vs baseline':<15}")
print('-' * 95)
for m in models:
    base_v = None
    base_sd = None
    for c in configs:
        v = extract_tg(f'clean_{m}_{c}.log')
        if v is None:
            print(f"  {m:<14} {config_label[c]:<14} (missing)")
            continue
        tps, sd = v
        sem = sd / math.sqrt(15)
        ci_low = tps - 1.96 * sem
        ci_high = tps + 1.96 * sem
        if c == 'c0':
            base_v = tps
            base_sd = sd
            delta = '(baseline)'
        else:
            delta_pct = (tps - base_v) / base_v * 100
            # Welch's t-test
            sem_diff = math.sqrt((sd**2 + base_sd**2) / 15)
            t_stat = (tps - base_v) / sem_diff if sem_diff > 0 else 0
            z = abs(t_stat)
            p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
            delta = f"{delta_pct:+.2f}% (p≈{p:.3f})"
        print(f"  {m:<14} {config_label[c]:<14} {tps:.3f} ± {sd:.3f}    [{ci_low:.2f}, {ci_high:.2f}]    {delta}")
    print()
PYEOF

echo ""
echo "=== END $(date) ==="
