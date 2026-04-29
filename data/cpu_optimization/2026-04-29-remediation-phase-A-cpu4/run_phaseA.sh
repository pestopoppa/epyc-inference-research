#!/bin/bash
# Remediation Phase A — CPU4 op-coalesced barriers re-test
# under FULL CANONICAL recipe (OMP env + interleave + --mmap 0).
# Original Phase 1 measured -19.7% on Coder; suspect was broken OMP env
# (recipe was missing OMP_PROC_BIND/PLACES/WAIT_POLICY + numactl
# --interleave=all + --mmap 0).

set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-29-remediation-phase-A-cpu4
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
CODER=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf

cd "$OUT"
ulimit -c 0
export LD_LIBRARY_PATH=$BIN

echo "=== Phase A — CPU4 coalesce re-test (Coder-30B Q4_K_M tg64, n=5) ==="
echo "Date: $(date -Iseconds)"
echo "Build: $BIN/llama-bench"
echo "Lib md5: $(md5sum $BIN/libggml-cpu.so.0.9.11 | awk '{print $1}')"
echo

# Full canonical recipe + COALESCE env toggle.
bench_run() {
    local TAG=$1
    local COALESCE=$2
    echo "--- bench: $TAG (COALESCE=$COALESCE) at $(date) ---"
    OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active \
    GGML_BARRIER_COALESCE=$COALESCE \
        numactl --interleave=all -- taskset -c 0-95 \
        $BIN/llama-bench \
        -m "$CODER" -t 96 -fa 1 --mmap 0 \
        -p 0 -n 64 -r 5 \
        > bench_coder_c${COALESCE}.log 2>&1
    grep "qwen3moe" bench_coder_c${COALESCE}.log | tail -1
    echo
}

# Run sequentially: c0 (control), c1 (treatment), c0 (re-control to detect drift)
bench_run "coder" 0
bench_run "coder" 1
bench_run "coder_recheck" 0

echo
echo "=== summary ==="
for tag in coder_c0 coder_c1 coder_recheck_c0; do
    F="bench_${tag}.log"
    [ -f "$F" ] && grep "qwen3moe" "$F" | tail -1 | awk -F'|' '{printf "%-25s %s\n", "'"$tag"'", $10}'
done

echo
python3 - << 'PYEOF'
import re, os
for tag in ['coder_c0', 'coder_c1', 'coder_recheck_c0']:
    f = f'bench_{tag}.log'
    if not os.path.exists(f):
        continue
    rows = [ln for ln in open(f) if ln.strip().startswith('|') and 'tg' in ln]
    if rows:
        m = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', rows[-1])
        if m:
            print(f'{tag}: {m.group(1)} ± {m.group(2)}')

# Compute Δ
import re
def parse(tag):
    f = f'bench_{tag}.log'
    if not os.path.exists(f): return None
    rows = [ln for ln in open(f) if ln.strip().startswith('|') and 'tg' in ln]
    if not rows: return None
    m = re.search(r'(\d+\.\d+)\s*±\s*(\d+\.\d+)', rows[-1])
    return (float(m.group(1)), float(m.group(2))) if m else None

c0 = parse('coder_c0')
c1 = parse('coder_c1')
c0r = parse('coder_recheck_c0')
if c0 and c1:
    d1 = (c1[0] - c0[0]) / c0[0] * 100
    print(f'\nΔ c1 vs c0:        {d1:+.2f}%')
if c0 and c0r:
    d2 = (c0r[0] - c0[0]) / c0[0] * 100
    print(f'Drift c0_re vs c0: {d2:+.2f}% (drift control)')
PYEOF

echo
echo "completed at $(date)"
