#!/bin/bash
# Multi-arch coverage CANONICAL re-run (post-reboot, post-OMP-fix).
# 3 archs × 4 configs × n=15 reps under FULL canonical recipe:
#   OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active
#   numactl --interleave=all -- taskset -c 0-95
#   llama-bench -t 96 -fa 1 --mmap 0 -p 0 -n 64 -r 15
#
# Per-cell pgrep guard: abort if foreign llama-* process appears.

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-multi-arch-coverage-canonical
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

DENSE_Q8=/mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
HYBRID=/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf
DENSE_Q4=/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf

mkdir -p "$OUT"
export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

check_clean() {
    local context=$1
    local procs=$(pgrep -af "llama-bench|llama-cli|llama-server|llama-perplex|llama-mtmd-cli" | grep -v zsh | grep -v $$)
    if [ -n "$procs" ]; then
        echo "ABORT [$context]: foreign llama process detected!" | tee -a guard_violations.log
        echo "$procs" | tee -a guard_violations.log
        return 1
    fi
    return 0
}

if ! check_clean "pre-flight"; then exit 1; fi

echo "=== START $(date) ==="
echo "Lib md5: $(md5sum $BIN/libggml-cpu.so.0.9.11 | awk '{print $1}')"
echo "Build hash: $(readelf -n $BIN/llama-bench 2>/dev/null | grep 'Build ID' | awk '{print $3}')"
echo

run_cell() {
    local TAG=$1
    local MODEL=$2
    local CONFIG=$3
    local N_REPS=15

    if ! check_clean "before $TAG/$CONFIG"; then return 1; fi

    local CONFIG_ENV=""
    case $CONFIG in
        c0) CONFIG_ENV="" ;;
        c1) CONFIG_ENV="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1" ;;
        c2) CONFIG_ENV="GGML_NUMA_REPACK_INTERLEAVE=0" ;;
        c3) CONFIG_ENV="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0" ;;
    esac

    echo "=== $TAG / $CONFIG (n=$N_REPS, env: ${CONFIG_ENV:-default}) ==="
    date

    # FULL CANONICAL stack: OMP env + numactl --interleave=all + --mmap 0
    env $CONFIG_ENV \
        OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
        numactl --interleave=all -- taskset -c 0-95 \
        $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 --mmap 0 \
        -p 0 -n 64 -r $N_REPS \
        > canon_${TAG}_${CONFIG}.log 2>&1
    grep -E "qwen3|nemotron|gemma" canon_${TAG}_${CONFIG}.log | tail -1
    echo "  done at $(date)"

    if ! check_clean "after $TAG/$CONFIG"; then
        echo "WARNING: foreign process appeared during/after $TAG/$CONFIG; cell may be poisoned"
        return 1
    fi

    sleep 3
}

# Order: smallest model first.
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

echo
echo "=== aggregate (canonical n=15) ==="
python3 - << 'PYEOF'
import os, re, math

def extract_tg(f):
    if not os.path.exists(f):
        return None
    for ln in open(f):
        if ln.startswith('|') and 'tg' in ln and 'test' not in ln:
            for col in [c.strip() for c in ln.split('|')]:
                m = re.match(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', col)
                if m:
                    return (float(m.group(1)), float(m.group(2)))
    return None

models = ['nemotron_9b', 'qwen36_27b', 'gemma_31b']
configs = ['c0', 'c1', 'c2', 'c3']
config_label = {'c0': 'baseline', 'c1': 'CPU1 stack', 'c2': 'CPU2 mbind off', 'c3': 'CPU1+CPU2off'}

print(f"{'Model':<14} {'Config':<16} {'tg64 t/s (n=15)':<22} {'Δ vs baseline':<15}")
print('-' * 80)
for m in models:
    base_v = None
    for c in configs:
        v = extract_tg(f'canon_{m}_{c}.log')
        if v is None:
            print(f"  {m:<14} {config_label[c]:<16} (missing)")
            continue
        tps, sd = v
        if c == 'c0':
            base_v = tps
            delta = '(baseline)'
        else:
            delta = f'{(tps-base_v)/base_v*100:+.2f}%' if base_v else 'n/a'
        print(f"  {m:<14} {config_label[c]:<16} {tps:.2f} ± {sd:.2f}{'':<8} {delta}")
    print()

PYEOF

echo
echo "=== completed $(date) ==="
