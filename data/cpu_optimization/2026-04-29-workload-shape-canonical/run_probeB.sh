#!/bin/bash
# Probe B — Workload-shape coverage under FULL canonical recipe.
# 3 models × 3 workloads (pp512, pp2048, tg64) × 3 configs (c0, c2, c3) × n=5
# = 27 cells. Tests whether CPU1 / CPU2-mbind effects depend on
# prefill-vs-decode regime.
#
# c1 (CPU1 alone) dropped from this probe per multi-arch findings:
# c2 (mbind-off) consistently dominated c1 alone in tg64.

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-workload-shape-canonical
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin

NEMOTRON=/mnt/raid0/llm/models/Nemotron-Nano-9B-v2-GGUF/nvidia_NVIDIA-Nemotron-Nano-9B-v2-Q8_0.gguf
DENSE_Q8=/mnt/raid0/llm/models/Qwen3.6-27B-Q8_0.gguf
DENSE_Q4=/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf

mkdir -p "$OUT"
export LD_LIBRARY_PATH=$BIN
cd "$OUT"
ulimit -c 0

check_clean() {
    local procs=$(pgrep -af "llama-bench|llama-cli|llama-server|llama-perplex|llama-mtmd-cli" | grep -v zsh | grep -v $$)
    if [ -n "$procs" ]; then
        echo "ABORT: foreign llama process detected!"
        echo "$procs"
        return 1
    fi
    return 0
}

if ! check_clean; then exit 1; fi

echo "=== Probe B START $(date) ==="
echo "Lib md5: $(md5sum $BIN/libggml-cpu.so.0.9.11 | awk '{print $1}')"
echo

run_cell() {
    local TAG=$1
    local MODEL=$2
    local CONFIG=$3
    local SHAPE_FLAGS=$4
    local SHAPE_TAG=$5
    local N_REPS=5

    local CONFIG_ENV=""
    case $CONFIG in
        c0) CONFIG_ENV="" ;;
        c2) CONFIG_ENV="GGML_NUMA_REPACK_INTERLEAVE=0" ;;
        c3) CONFIG_ENV="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0" ;;
    esac

    echo "=== $TAG / $SHAPE_TAG / $CONFIG (env: ${CONFIG_ENV:-default}) ==="
    date

    env $CONFIG_ENV \
        OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
        numactl --interleave=all -- taskset -c 0-95 \
        $BIN/llama-bench -m "$MODEL" -t 96 -fa 1 --mmap 0 \
        $SHAPE_FLAGS -r $N_REPS \
        > probeB_${TAG}_${SHAPE_TAG}_${CONFIG}.log 2>&1
    grep -E "qwen3|nemotron|gemma" probeB_${TAG}_${SHAPE_TAG}_${CONFIG}.log | tail -2
    echo "  done at $(date)"
    sleep 2
}

for MODEL_TAG_PATH in "nemotron_9b $NEMOTRON" "qwen36_27b $DENSE_Q8" "gemma_31b $DENSE_Q4"; do
    set -- $MODEL_TAG_PATH
    TAG=$1
    MODEL=$2
    if [ ! -f "$MODEL" ]; then
        echo "SKIP $TAG: $MODEL not found"
        continue
    fi

    for CONFIG in c0 c2 c3; do
        # tg64: pure decode (no prefill)
        run_cell "$TAG" "$MODEL" "$CONFIG" "-p 0 -n 64" "tg64"

        # pp512: prefill 512 tokens (no decode)
        run_cell "$TAG" "$MODEL" "$CONFIG" "-p 512 -n 0" "pp512"

        # pp2048: prefill 2048 tokens (no decode)
        run_cell "$TAG" "$MODEL" "$CONFIG" "-p 2048 -n 0" "pp2048"
    done
done

echo
echo "=== aggregate (Probe B canonical n=5) ==="
python3 - << 'PYEOF'
import os, re

def extract(f, test_kind):
    if not os.path.exists(f):
        return None
    for ln in open(f):
        if ln.startswith('|') and test_kind in ln and 'test' not in ln:
            for col in [c.strip() for c in ln.split('|')]:
                m = re.match(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', col)
                if m:
                    return (float(m.group(1)), float(m.group(2)))
    return None

models = ['nemotron_9b', 'qwen36_27b', 'gemma_31b']
shapes = [('tg64', 'tg64'), ('pp512', 'pp512'), ('pp2048', 'pp2048')]
configs = ['c0', 'c2', 'c3']

print(f"{'Model':<14} {'Shape':<8} {'c0 baseline':<22} {'c2 mbind-off':<22} {'c3 CPU1+mbind-off':<22}")
print('-' * 100)
for m in models:
    for shape, test in shapes:
        row = [m, shape]
        base_v = None
        for c in configs:
            f = f'probeB_{m}_{shape}_{c}.log'
            v = extract(f, test)
            if v is None:
                row.append('(missing)')
            else:
                tps, sd = v
                if c == 'c0':
                    base_v = tps
                    row.append(f'{tps:.2f} ± {sd:.2f}')
                else:
                    delta = (tps - base_v) / base_v * 100 if base_v else 0
                    row.append(f'{tps:.2f} ± {sd:.2f} ({delta:+.1f}%)')
        print(f"  {row[0]:<14} {row[1]:<8} {row[2]:<22} {row[3]:<22} {row[4]:<22}")
    print()
PYEOF

echo
echo "=== completed $(date) ==="
