#!/bin/bash
set -euo pipefail

# Six identical production-v8 observations under one outer q0-q3 region claim:
# four consecutive runs, 180 seconds idle, then two more. bench_canonical.sh's
# own per-run lock is deliberately skipped because the caller holds the same
# lock continuously across both the measurements and the rest interval.

OUT_DIR="${1:?usage: run_rest_recovery.sh OUTPUT_DIR}"
MODEL="/mnt/raid0/llm/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
SOURCE_ROOT="/mnt/raid0/llm/llama.cpp"
BINARY="${SOURCE_ROOT}/build/bin/llama-bench"
LIBRARY_PATH="${SOURCE_ROOT}/build/bin"
BENCH="/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/bench_canonical.sh"
REGION_LOCK="/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock"

mkdir -p "$OUT_DIR"

"$REGION_LOCK" status > "${OUT_DIR}/claim_status_at_start.txt"
{
    date --iso-8601=seconds
    uptime
    cat /proc/loadavg
    cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor
    cat /sys/kernel/mm/transparent_hugepage/enabled
    cat /sys/kernel/mm/transparent_hugepage/defrag
    cat /proc/sys/kernel/numa_balancing
} > "${OUT_DIR}/host_state_at_start.txt"

run_one() {
    local index="$1"
    local stem="${OUT_DIR}/anchor_${index}"
    date --iso-8601=seconds > "${stem}.started_at"
    CANONICAL_SKIP_REGION_LOCK=1 "$BENCH" \
        --model "$MODEL" \
        --binary "$BINARY" \
        --source-root "$SOURCE_ROOT" \
        --library-path "$LIBRARY_PATH" \
        -p 512 -n 128 -r 5 \
        -- -o json \
        > "${stem}.json" 2> "${stem}.stderr"
    date --iso-8601=seconds > "${stem}.ended_at"
    printf 'completed run %s\n' "$index"
}

for index in 1 2 3 4; do
    run_one "$index"
done

date --iso-8601=seconds > "${OUT_DIR}/rest.started_at"
sleep 180
date --iso-8601=seconds > "${OUT_DIR}/rest.ended_at"

for index in 5 6; do
    run_one "$index"
done

"$REGION_LOCK" status > "${OUT_DIR}/claim_status_at_end.txt"

(
    cd "$OUT_DIR"
    sha256sum anchor_* claim_status_* host_state_* rest.* > SHA256SUMS
)
