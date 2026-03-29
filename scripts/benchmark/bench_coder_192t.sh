#!/bin/bash
# Full dm x p_split Sweep — Qwen2.5-Coder-32B (Q4_K_M, Q8_0, f16)
#
# Two thread configs per variant:
#   192t: numactl --interleave=all, 192 threads, ub 8192
#   48t:  taskset -c 0-23,96-119, 48 threads, ub 512
#
# Sweep plan per (variant, mode):
#   Phase 1: dm in [8, 16, 24, 32, 48] at p_split=0 → find best dm
#   Phase 2: p_split in [0, 0.05, 0.1, 0.2, 0.3] at best dm → find best ps
#
# Total: 3 variants x 2 modes x (5 dm + 5 ps) = 60 server starts
# Each start: load + warmup(7x256) + 10 requests(128tok) + kill
# Estimated runtime: ~3-4 hours
#
# Run: bash scripts/benchmark/bench_coder_192t.sh 2>&1 | tee /tmp/bench_coder_sweep.log

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"
RESP_FILE="/tmp/sweep_response.json"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/coder_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"
CSV_FILE="${DATA_DIR}/coder_sweep_${TIMESTAMP}.csv"

N_PREDICT=128
N_REQUESTS=10
PORT=8180

# Model paths
CODER_Q4KM="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
CODER_Q8="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
CODER_F16="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf"
CODER_DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# NUMA quarter binding (first quarter)
NUMA_QUARTER_CPUS="0-23,96-119"

# dm and p_split sweep values
DM_VALUES=(8 16 24 32 48)
PS_VALUES=(0 0.05 0.1 0.2 0.3)

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Design a microservice architecture for a real-time chat application with message persistence and delivery guarantees:"
    "Write a Rust implementation of a lock-free concurrent queue using compare-and-swap operations:"
)

# ============================================================
# Cleanup trap
# ============================================================

cleanup() {
    echo ""
    echo "Cleaning up..."
    lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    rm -f "$RESP_FILE"
}
trap cleanup EXIT

# ============================================================
# Helper Functions
# ============================================================

wait_for_server() {
    local max_wait=${1:-600}
    local elapsed=0
    echo -n "    waiting for port $PORT..."
    while ! curl -s "http://localhost:${PORT}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then
            echo " TIMEOUT (${max_wait}s)"
            return 1
        fi
        if [ $((elapsed % 20)) -eq 0 ]; then echo -n " ${elapsed}s..."; fi
    done
    echo " ready (${elapsed}s)"
}

warmup_server() {
    echo -n "      warmup: " >&2
    for w in 1 2 3 4 5 6 7; do
        curl -s -o /dev/null "http://localhost:${PORT}/completion" \
            -H "Content-Type: application/json" \
            -d '{"prompt":"Write a comprehensive Python implementation of a red-black tree with insert, delete, search, and rebalance operations. Include type hints and docstrings:","n_predict":256,"temperature":0}'
        echo -n "." >&2
    done
    echo " done" >&2
}

run_completion() {
    local prompt="$1"
    local n_predict=$2

    local escaped_prompt
    escaped_prompt=$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')

    local start_ms end_ms
    start_ms=$(date +%s%N | cut -b1-13)
    curl -s --max-time 600 -o "$RESP_FILE" "http://localhost:${PORT}/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":${escaped_prompt},\"n_predict\":${n_predict},\"temperature\":0.0,\"cache_prompt\":false}"
    end_ms=$(date +%s%N | cut -b1-13)
    local elapsed_ms=$((end_ms - start_ms))

    python3 -c "
import json
try:
    with open('$RESP_FILE','rb') as f: r=json.load(f)
    tokens = r.get('timings',{}).get('predicted_n', 0)
    tps = r.get('timings',{}).get('predicted_per_second', 0)
    print(f'{tokens},{$elapsed_ms},{tps:.2f}')
except:
    print('0,${elapsed_ms},0.00')
" 2>/dev/null
}

kill_server() {
    lsof -ti :$PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
    sleep 3
}

# Start server with given parameters
# Args: model_path draft_max p_split mode_name variant_name
start_server() {
    local model_path="$1"
    local dm="$2"
    local ps="$3"
    local mode="$4"
    local log_label="$5"
    local max_wait="${6:-600}"

    local log_file="${LOG_DIR}/${log_label}_dm${dm}_ps${ps}.log"

    if [ "$mode" = "192t" ]; then
        numactl --interleave=all "$LLAMA_SERVER" -m "$model_path" -md "$CODER_DRAFT" \
            --draft-max "$dm" --draft-p-split "$ps" --lookup -fa on \
            -t 192 -np 1 --port $PORT -ngl 0 -ub 8192 \
            > "$log_file" 2>&1 &
    elif [ "$mode" = "48t" ]; then
        taskset -c "$NUMA_QUARTER_CPUS" "$LLAMA_SERVER" -m "$model_path" -md "$CODER_DRAFT" \
            --draft-max "$dm" --draft-p-split "$ps" --lookup -fa on \
            -t 48 -np 1 --port $PORT -ngl 0 -ub 512 \
            > "$log_file" 2>&1 &
    fi

    wait_for_server "$max_wait"
}

# Run N_REQUESTS and return avg tps; also writes CSV rows
# Args: csv_label
# Returns: avg_tps (printed to stdout)
run_measurement() {
    local csv_label="$1"
    local tps_values=()

    echo -n "      requests: " >&2
    for ((r=0; r<N_REQUESTS; r++)); do
        local pidx=$((r % ${#PROMPTS[@]}))
        local result
        result=$(run_completion "${PROMPTS[$pidx]}" "$N_PREDICT")
        local tokens wall tps
        tokens=$(echo "$result" | cut -d, -f1)
        wall=$(echo "$result" | cut -d, -f2)
        tps=$(echo "$result" | cut -d, -f3)
        tps_values+=("$tps")
        echo "$csv_label,$r,$tokens,$wall,$tps" >> "$CSV_FILE"
        echo -n "${tps} " >&2
    done
    echo "" >&2

    # Return average tps
    python3 -c "
vals = [float(v) for v in [$(IFS=,; echo "${tps_values[*]}")] if v > 0]
if vals:
    print(f'{sum(vals)/len(vals):.2f}')
else:
    print('0.00')
"
}

# Sweep dm values, return best dm
# Args: model_path mode variant_label max_wait
# Prints: best_dm to stdout
sweep_dm() {
    local model_path="$1"
    local mode="$2"
    local variant_label="$3"
    local max_wait="${4:-600}"

    local best_dm=8
    local best_avg="0.00"

    echo ""
    echo "    --- Phase 1: dm sweep at p_split=0 ---"

    for dm in "${DM_VALUES[@]}"; do
        echo ""
        echo "    [dm=$dm] Starting server..."

        local csv_label="${variant_label},${mode},${mode/t/},${dm},0"
        local log_label="${variant_label}_${mode}"

        if start_server "$model_path" "$dm" "0" "$mode" "$log_label" "$max_wait"; then
            warmup_server
            local avg
            avg=$(run_measurement "$csv_label")
            echo "    [dm=$dm] avg=${avg} t/s"

            # Compare with best
            local is_better
            is_better=$(python3 -c "print(1 if float('${avg}') > float('${best_avg}') else 0)")
            if [ "$is_better" = "1" ]; then
                best_dm=$dm
                best_avg=$avg
            fi
        else
            echo "    [dm=$dm] FAILED to start server"
        fi

        kill_server
    done

    echo ""
    echo "    >>> Phase 1 winner: dm=${best_dm} (${best_avg} t/s)"
    echo "$best_dm"
}

# Sweep p_split values at a fixed dm, return best ps
# Args: model_path mode variant_label best_dm max_wait
# Prints: best_ps to stdout (last line)
sweep_ps() {
    local model_path="$1"
    local mode="$2"
    local variant_label="$3"
    local best_dm="$4"
    local max_wait="${5:-600}"

    local best_ps="0"
    local best_avg="0.00"

    echo ""
    echo "    --- Phase 2: p_split sweep at dm=${best_dm} ---"

    for ps in "${PS_VALUES[@]}"; do
        echo ""
        echo "    [ps=$ps] Starting server..."

        local threads="${mode/t/}"
        local csv_label="${variant_label},${mode},${threads},${best_dm},${ps}"
        local log_label="${variant_label}_${mode}"

        if start_server "$model_path" "$best_dm" "$ps" "$mode" "$log_label" "$max_wait"; then
            warmup_server
            local avg
            avg=$(run_measurement "$csv_label")
            echo "    [ps=$ps] avg=${avg} t/s"

            local is_better
            is_better=$(python3 -c "print(1 if float('${avg}') > float('${best_avg}') else 0)")
            if [ "$is_better" = "1" ]; then
                best_ps=$ps
                best_avg=$avg
            fi
        else
            echo "    [ps=$ps] FAILED to start server"
        fi

        kill_server
    done

    echo ""
    echo "    >>> Phase 2 winner: ps=${best_ps} (${best_avg} t/s)"
    echo "$best_ps"
}

# Full sweep for one variant in one mode
# Args: model_path mode variant_label max_wait
sweep_variant_mode() {
    local model_path="$1"
    local mode="$2"
    local variant_label="$3"
    local max_wait="${4:-600}"

    echo ""
    echo "  ────────────────────────────────────────────"
    echo "  ${variant_label} / ${mode}"
    echo "  ────────────────────────────────────────────"

    # Phase 1: sweep dm
    local dm_output
    dm_output=$(sweep_dm "$model_path" "$mode" "$variant_label" "$max_wait")
    local best_dm
    best_dm=$(echo "$dm_output" | tail -1)

    # Phase 2: sweep p_split at best dm
    local ps_output
    ps_output=$(sweep_ps "$model_path" "$mode" "$variant_label" "$best_dm" "$max_wait")
    local best_ps
    best_ps=$(echo "$ps_output" | tail -1)

    echo ""
    echo "  >>> RESULT: ${variant_label}/${mode} => dm=${best_dm}, ps=${best_ps}"
}

# ============================================================
# Setup
# ============================================================

mkdir -p "$DATA_DIR" "$LOG_DIR"
echo "variant,mode,threads,draft_max,p_split,request_idx,tokens_generated,wall_ms,decode_tps" > "$CSV_FILE"

TOTAL_STARTS=$((3 * 2 * (${#DM_VALUES[@]} + ${#PS_VALUES[@]})))
START_TIME=$(date +%s)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Coder-32B Full dm x p_split Sweep                        ║"
echo "║  3 quants x 2 modes x (5 dm + 5 ps) = ${TOTAL_STARTS} server starts    ║"
echo "║  Started: $(date)                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  CSV: $CSV_FILE"
echo "  Logs: $LOG_DIR"
echo ""

# ============================================================
# Phase counters
# ============================================================
VARIANT_NUM=0
TOTAL_VARIANTS=6  # 3 quants x 2 modes

next_variant() {
    VARIANT_NUM=$((VARIANT_NUM + 1))
    local elapsed=$(( $(date +%s) - START_TIME ))
    local elapsed_min=$((elapsed / 60))
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  [$VARIANT_NUM/$TOTAL_VARIANTS] $1  (elapsed: ${elapsed_min}m)"
    echo "═══════════════════════════════════════════════════════════════"
}

# ============================================================
# Variant 1: Q4_K_M
# ============================================================

next_variant "Q4_K_M / 192t (interleave)"
sweep_variant_mode "$CODER_Q4KM" "192t" "Q4KM" 600

next_variant "Q4_K_M / 48t (NUMA quarter)"
sweep_variant_mode "$CODER_Q4KM" "48t" "Q4KM" 600

# ============================================================
# Variant 2: Q8_0
# ============================================================

next_variant "Q8_0 / 192t (interleave)"
sweep_variant_mode "$CODER_Q8" "192t" "Q8_0" 600

next_variant "Q8_0 / 48t (NUMA quarter)"
sweep_variant_mode "$CODER_Q8" "48t" "Q8_0" 600

# ============================================================
# Variant 3: f16
# ============================================================

next_variant "f16 / 192t (interleave)"
sweep_variant_mode "$CODER_F16" "192t" "f16" 900

next_variant "f16 / 48t (NUMA quarter)"
sweep_variant_mode "$CODER_F16" "48t" "f16" 900

# ============================================================
# Summary
# ============================================================

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SWEEP COMPLETE — ${TOTAL_MIN} minutes                                  ║"
echo "║  CSV: $CSV_FILE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

python3 << 'PYEOF'
import csv, sys, os

# Find the CSV
csv_path = os.environ.get("CSV_FILE", "")
if not csv_path:
    import glob
    files = sorted(glob.glob("/mnt/raid0/llm/epyc-inference-research/data/coder_sweep/coder_sweep_*.csv"))
    csv_path = files[-1] if files else ""

if not csv_path:
    print("No CSV found")
    sys.exit()

# Parse CSV: group by (variant, mode, dm, ps) -> list of tps
groups = {}
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["variant"], row["mode"], row["draft_max"], row["p_split"])
        tps = float(row["decode_tps"])
        if tps > 0:
            groups.setdefault(key, []).append(tps)

# Find best config per (variant, mode)
best = {}  # (variant, mode) -> (dm, ps, avg_tps)
for (variant, mode, dm, ps), tps_list in groups.items():
    avg = sum(tps_list) / len(tps_list)
    vm_key = (variant, mode)
    if vm_key not in best or avg > best[vm_key][2]:
        best[vm_key] = (dm, ps, avg)

print("=" * 65)
print("  SUMMARY: Best (dm, p_split) per Variant per Mode")
print("=" * 65)
print()
print(f"  {'Variant':<10} {'Mode':<8} {'Best dm':<10} {'Best ps':<10} {'Avg t/s':<10}")
print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")

for (variant, mode) in sorted(best.keys()):
    dm, ps, avg = best[(variant, mode)]
    print(f"  {variant:<10} {mode:<8} {dm:<10} {ps:<10} {avg:<10.2f}")

print()

# Also print the full dm sweep results for reference
print("=" * 65)
print("  FULL RESULTS: All (dm, p_split) Measurements")
print("=" * 65)
print()
print(f"  {'Variant':<10} {'Mode':<8} {'dm':<6} {'ps':<8} {'Avg t/s':<10} {'N':<4}")
print(f"  {'-'*10} {'-'*8} {'-'*6} {'-'*8} {'-'*10} {'-'*4}")

for (variant, mode, dm, ps) in sorted(groups.keys()):
    tps_list = groups[(variant, mode, dm, ps)]
    avg = sum(tps_list) / len(tps_list)
    print(f"  {variant:<10} {mode:<8} {dm:<6} {ps:<8} {avg:<10.2f} {len(tps_list):<4}")

print()
PYEOF

echo ""
echo "Done. CSV at: $CSV_FILE"
