#!/bin/bash
# Hybrid SSM follow-up — does Probe B's +8.9% pp512 c3 finding on
# Nemotron-9B generalize to Qwen3-Next-80B-A3B Q4_K_M (other Hybrid SSM
# in production lineup)?
#
# 1 model × 3 configs × 3 shapes × n=5 = 9 cells.

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-hybrid-ssm-next80b-followup
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
NEXT80=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf

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

echo "=== Qwen3-Next-80B Hybrid SSM follow-up START $(date) ==="
echo "Lib md5: $(md5sum $BIN/libggml-cpu.so.0.9.11 | awk '{print $1}')"
echo

run_cell() {
    local CONFIG=$1
    local SHAPE_FLAGS=$2
    local SHAPE_TAG=$3
    local N_REPS=5

    local CONFIG_ENV=""
    case $CONFIG in
        c0) CONFIG_ENV="" ;;
        c2) CONFIG_ENV="GGML_NUMA_REPACK_INTERLEAVE=0" ;;
        c3) CONFIG_ENV="GGML_CCD_POOLS=1 GGML_CCD_WORK_DIST=1 GGML_BARRIER_LOCAL_BETWEEN_OPS=1 GGML_NUMA_REPACK_INTERLEAVE=0" ;;
    esac

    echo "=== next80 / $SHAPE_TAG / $CONFIG (env: ${CONFIG_ENV:-default}) ==="
    date

    env $CONFIG_ENV \
        OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
        numactl --interleave=all -- taskset -c 0-95 \
        $BIN/llama-bench -m "$NEXT80" -t 96 -fa 1 --mmap 0 \
        $SHAPE_FLAGS -r $N_REPS \
        > followup_${SHAPE_TAG}_${CONFIG}.log 2>&1
    grep -E "qwen3next" followup_${SHAPE_TAG}_${CONFIG}.log | tail -2
    echo "  done at $(date)"
    sleep 2
}

# Order: tg64 first (fastest sanity), then pp512 (the headline shape), then pp2048
for CONFIG in c0 c2 c3; do
    run_cell "$CONFIG" "-p 0 -n 64"  "tg64"
    run_cell "$CONFIG" "-p 512 -n 0" "pp512"
    run_cell "$CONFIG" "-p 2048 -n 0" "pp2048"
done

echo
echo "=== aggregate (Qwen3-Next-80B Hybrid SSM follow-up, n=5) ==="
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

shapes = [('tg64', 'tg64'), ('pp512', 'pp512'), ('pp2048', 'pp2048')]
configs = ['c0', 'c2', 'c3']

print(f"{'Shape':<8} {'c0 baseline':<22} {'c2 mbind-off':<22} {'c3 CPU1+mbind-off':<22}")
print('-' * 80)
for shape, test in shapes:
    row = [shape]
    base_v = None
    for c in configs:
        f = f'followup_{shape}_{c}.log'
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
    print(f"  {row[0]:<8} {row[1]:<22} {row[2]:<22} {row[3]:<22}")

print()
print("Reference (Nemotron-9B, Probe B):")
print("  tg64    12.69 baseline   12.83 (+1.1%)        12.96 (+2.1%)")
print("  pp512   317.32 baseline  340.72 (+7.4%)       345.57 (+8.9%)  <<-- headline")
print("  pp2048  323.76 baseline  333.83 (+3.1%)       335.67 (+3.7%)")
PYEOF

echo
echo "=== completed $(date) ==="
