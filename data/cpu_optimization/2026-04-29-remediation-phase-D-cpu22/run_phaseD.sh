#!/bin/bash
# Remediation Phase D — CPU22 work-stealing re-test under canonical
# Original 2026-04-28 measurement WAS already done with full canonical
# (OMP env + numactl --interleave=all + --mmap 0). So this Phase is
# a verification that the closure stands; not expected to flip.
# Original result: Coder -2.3%, Next-80B -0.3% (NS), REAP -0.8% (NS).
# Gate threshold ≥10%; not met.

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-remediation-phase-D-cpu22
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf
NEXT=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
REAP=/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf

mkdir -p "$OUT"
cd "$OUT"
ulimit -c 0
export LD_LIBRARY_PATH=$BIN

echo "=== Phase D — CPU22 work-stealing re-test ==="
echo "Date: $(date -Iseconds)"
echo "Lib md5: $(md5sum $BIN/libggml-cpu.so.0.9.11 | awk '{print $1}')"
echo

bench_run() {
    local TAG=$1
    local MODEL=$2
    local STEAL=$3
    echo "--- $TAG (GGML_EP_WORK_STEALING=$STEAL) at $(date) ---"
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    GGML_EP_WORK_STEALING=$STEAL \
        numactl --interleave=all -- taskset -c 0-95 \
        $BIN/llama-bench \
        -m "$MODEL" -t 96 -fa 1 --mmap 0 \
        -p 0 -n 64 -r 5 \
        > bench_${TAG}_steal${STEAL}.log 2>&1
    grep -E "qwen3moe|qwen3next" bench_${TAG}_steal${STEAL}.log | tail -1
    echo
}

# Coder-30B (most negative in original at -2.3%)
bench_run "coder" "$CODER" 0
bench_run "coder" "$CODER" 1

# Next-80B (was neutral)
if [ -f "$NEXT" ]; then
    bench_run "next80" "$NEXT" 0
    bench_run "next80" "$NEXT" 1
else
    echo "Next-80B not at expected path — skipping"
fi

# REAP-246B (was neutral)
if [ -f "$REAP" ]; then
    bench_run "reap" "$REAP" 0
    bench_run "reap" "$REAP" 1
fi

echo
echo "=== summary ==="
python3 - << 'PYEOF'
import re, os
def parse(tag, steal):
    f = f'bench_{tag}_steal{steal}.log'
    if not os.path.exists(f): return None
    rows = [ln for ln in open(f) if ln.strip().startswith('|') and ('tg' in ln) and 'test' not in ln]
    if not rows: return None
    m = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', rows[-1])
    return (float(m.group(1)), float(m.group(2))) if m else None

for tag in ['coder', 'next80', 'reap']:
    p0 = parse(tag, 0)
    p1 = parse(tag, 1)
    if p0 and p1:
        delta = (p1[0] - p0[0]) / p0[0] * 100
        print(f'{tag}: env=0 {p0[0]:.2f}±{p0[1]:.2f}  env=1 {p1[0]:.2f}±{p1[1]:.2f}  Δ={delta:+.2f}%')
    elif p0:
        print(f'{tag}: env=0 {p0[0]:.2f} (env=1 missing)')
    else:
        print(f'{tag}: missing data')
PYEOF

echo
echo "completed at $(date)"
