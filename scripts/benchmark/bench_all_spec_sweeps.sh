#!/bin/bash
# Comprehensive Spec Param Sweep — All Models, All Modes
#
# Full dm x p_split sweep for every spec-decode model at both:
#   - 192t reference (numactl --interleave=all, ub 8192)
#   - Deployment thread count (taskset, ub 512)
#
# Models (ordered fast → slow):
#   1. Worker 7B f16         — 192t + 24t
#   2. Old frontdoor 30B-A3B — 192t + 48t
#   3. Coder-32B Q4_K_M      — 192t + 48t
#   4. Coder-32B Q8_0        — 192t + 48t
#   5. Coder-32B f16         — 192t + 48t
#   6. Qwen3.5-122B MoE      — 192t + 96t
#   7. Coder-480B MoE        — 192t + 96t
#
# Sweep plan per (model, mode):
#   Phase 1: dm in [8, 16, 24, 32, 48] at p_split=0 → find best dm
#   Phase 2: p_split in [0, 0.05, 0.1, 0.2, 0.3] at best dm → find best ps
#   (480B: dm in [16, 24, 32, 48], 5 requests, 4 warmup rounds)
#
# Total: 7 models × 2 modes × 10 configs = 140 server starts
# Estimated runtime: ~10 hours (ideal overnight run)
#
# Run: bash scripts/benchmark/bench_all_spec_sweeps.sh 2>&1 | tee /tmp/bench_all_spec.log

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"
RESP_FILE="/tmp/sweep_response.json"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/all_spec_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"
CSV_FILE="${DATA_DIR}/all_spec_sweep_${TIMESTAMP}.csv"
SUMMARY_FILE="${DATA_DIR}/sweep_summary_${TIMESTAMP}.csv"

N_PREDICT=128
PORT=8180

# ============================================================
# Model paths
# ============================================================

# Worker 7B
WORKER_7B="/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf"
WORKER_DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Old frontdoor 30B-A3B (MoE)
FRONTDOOR_30B="${MODEL_BASE}/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
FRONTDOOR_30B_DRAFT="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

# Coder-32B variants
CODER_Q4KM="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
CODER_Q8="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"
CODER_F16="/mnt/raid0/llm/models/Qwen2.5-Coder-32B-Instruct-GGUF-f16/qwen2.5-coder-32b-instruct-fp16-00001-of-00009.gguf"
CODER_DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Architect 122B (MoE hybrid)
ARCH_122B="${MODEL_BASE}/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
ARCH_122B_DRAFT="${MODEL_BASE}/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

# Architect 480B (MoE)
ARCH_480B="${MODEL_BASE}/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
ARCH_480B_DRAFT="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

# CPU bindings
CPUS_QUARTER="0-23,96-119"   # Node 0 quarter A (48 HW threads)
CPUS_HALF="0-47,96-143"      # Node 0 full (96 HW threads)

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
    local n_warmup=${1:-7}
    echo -n "      warmup: " >&2
    for ((w=1; w<=n_warmup; w++)); do
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
# Args: model_path draft_path dm ps mode threads extra_args log_label max_wait
start_server() {
    local model_path="$1"
    local draft_path="$2"
    local dm="$3"
    local ps="$4"
    local mode="$5"
    local threads="$6"
    local extra_args="$7"
    local log_label="$8"
    local max_wait="${9:-600}"

    local log_file="${LOG_DIR}/${log_label}_dm${dm}_ps${ps}.log"

    if [ "$mode" = "192t" ]; then
        # shellcheck disable=SC2086
        numactl --interleave=all "$LLAMA_SERVER" -m "$model_path" -md "$draft_path" \
            --draft-max "$dm" --draft-p-split "$ps" --lookup -fa on \
            -t 192 -np 1 --port $PORT -ngl 0 -ub 8192 \
            $extra_args \
            > "$log_file" 2>&1 &
    else
        local cpus
        if [ "$threads" -le 24 ]; then
            cpus="$CPUS_QUARTER"
        elif [ "$threads" -le 48 ]; then
            cpus="$CPUS_QUARTER"
        else
            cpus="$CPUS_HALF"
        fi
        # shellcheck disable=SC2086
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$model_path" -md "$draft_path" \
            --draft-max "$dm" --draft-p-split "$ps" --lookup -fa on \
            -t "$threads" -np 1 --port $PORT -ngl 0 -ub 512 \
            $extra_args \
            > "$log_file" 2>&1 &
    fi

    wait_for_server "$max_wait"
}

# Run N requests and return avg tps; also writes CSV rows
# Args: csv_label n_requests
# Returns: avg_tps (printed to stdout)
run_measurement() {
    local csv_label="$1"
    local n_requests="${2:-10}"
    local tps_values=()

    echo -n "      requests: " >&2
    for ((r=0; r<n_requests; r++)); do
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

    python3 -c "
vals = [float(v) for v in [$(IFS=,; echo "${tps_values[*]}")] if v > 0]
if vals:
    print(f'{sum(vals)/len(vals):.2f}')
else:
    print('0.00')
"
}

# Sweep dm values, return best dm
# Args: model_path draft_path mode threads variant_label extra_args dm_grid n_requests n_warmup max_wait
sweep_dm() {
    local model_path="$1"
    local draft_path="$2"
    local mode="$3"
    local threads="$4"
    local variant_label="$5"
    local extra_args="$6"
    local dm_grid="$7"
    local n_requests="${8:-10}"
    local n_warmup="${9:-7}"
    local max_wait="${10:-600}"

    local best_dm=8
    local best_avg="0.00"

    echo ""
    echo "    --- Phase 1: dm sweep at p_split=0 ---"

    for dm in $dm_grid; do
        echo ""
        echo "    [dm=$dm] Starting server..."

        local csv_label="${variant_label},${mode},${threads},${dm},0"
        local log_label="${variant_label}_${mode}"

        if start_server "$model_path" "$draft_path" "$dm" "0" "$mode" "$threads" "$extra_args" "$log_label" "$max_wait"; then
            warmup_server "$n_warmup"
            local avg
            avg=$(run_measurement "$csv_label" "$n_requests")
            echo "    [dm=$dm] avg=${avg} t/s"

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
# Args: model_path draft_path mode threads variant_label extra_args best_dm n_requests n_warmup max_wait
sweep_ps() {
    local model_path="$1"
    local draft_path="$2"
    local mode="$3"
    local threads="$4"
    local variant_label="$5"
    local extra_args="$6"
    local best_dm="$7"
    local n_requests="${8:-10}"
    local n_warmup="${9:-7}"
    local max_wait="${10:-600}"

    local best_ps="0"
    local best_avg="0.00"

    echo ""
    echo "    --- Phase 2: p_split sweep at dm=${best_dm} ---"

    for ps in 0 0.05 0.1 0.2 0.3; do
        echo ""
        echo "    [ps=$ps] Starting server..."

        local csv_label="${variant_label},${mode},${threads},${best_dm},${ps}"
        local log_label="${variant_label}_${mode}"

        if start_server "$model_path" "$draft_path" "$best_dm" "$ps" "$mode" "$threads" "$extra_args" "$log_label" "$max_wait"; then
            warmup_server "$n_warmup"
            local avg
            avg=$(run_measurement "$csv_label" "$n_requests")
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

# Full sweep for one model in one mode
# Args: model_path draft_path mode threads variant_label extra_args dm_grid n_requests n_warmup max_wait
sweep_model_mode() {
    local model_path="$1"
    local draft_path="$2"
    local mode="$3"
    local threads="$4"
    local variant_label="$5"
    local extra_args="$6"
    local dm_grid="$7"
    local n_requests="${8:-10}"
    local n_warmup="${9:-7}"
    local max_wait="${10:-600}"

    echo ""
    echo "  ────────────────────────────────────────────"
    echo "  ${variant_label} / ${mode} (${threads} threads)"
    echo "  ────────────────────────────────────────────"

    # Phase 1: sweep dm
    local dm_output
    dm_output=$(sweep_dm "$model_path" "$draft_path" "$mode" "$threads" "$variant_label" "$extra_args" "$dm_grid" "$n_requests" "$n_warmup" "$max_wait")
    local best_dm
    best_dm=$(echo "$dm_output" | tail -1)

    # Phase 2: sweep p_split at best dm
    local ps_output
    ps_output=$(sweep_ps "$model_path" "$draft_path" "$mode" "$threads" "$variant_label" "$extra_args" "$best_dm" "$n_requests" "$n_warmup" "$max_wait")
    local best_ps
    best_ps=$(echo "$ps_output" | tail -1)

    echo ""
    echo "  >>> RESULT: ${variant_label}/${mode} => dm=${best_dm}, ps=${best_ps}"

    # Append to summary
    echo "${variant_label},${mode},${threads},${best_dm},${best_ps}" >> "$SUMMARY_FILE"
}

# ============================================================
# Setup
# ============================================================

mkdir -p "$DATA_DIR" "$LOG_DIR"
echo "model,variant,mode,threads,draft_max,p_split,request_idx,tokens_generated,wall_ms,decode_tps" > "$CSV_FILE"
echo "model,mode,threads,best_dm,best_ps" > "$SUMMARY_FILE"

START_TIME=$(date +%s)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Comprehensive Spec Param Sweep — All Models, All Modes    ║"
echo "║  7 models × 2 modes × (5 dm + 5 ps) = ~140 server starts  ║"
echo "║  Started: $(date)                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Raw CSV:     $CSV_FILE"
echo "  Summary CSV: $SUMMARY_FILE"
echo "  Logs:        $LOG_DIR"
echo ""

VARIANT_NUM=0
TOTAL_VARIANTS=14  # 7 models × 2 modes

next_variant() {
    VARIANT_NUM=$((VARIANT_NUM + 1))
    local elapsed=$(( $(date +%s) - START_TIME ))
    local elapsed_min=$((elapsed / 60))
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  [$VARIANT_NUM/$TOTAL_VARIANTS] $1  (elapsed: ${elapsed_min}m)"
    echo "═══════════════════════════════════════════════════════════════"
}

DM_GRID_STANDARD="8 16 24 32 48"
DM_GRID_480B="16 24 32 48"  # skip dm=8 for 480B (too small for 35B active params)

# ============================================================
# Model 1: Worker 7B f16 (fastest — ~2 min/config)
# Draft: Coder-0.5B Q8, deployment: 1×24t
# ============================================================

next_variant "Worker 7B f16 / 192t (interleave)"
sweep_model_mode "$WORKER_7B" "$WORKER_DRAFT" "192t" 192 "worker_7b_f16" "" "$DM_GRID_STANDARD" 10 7 300

next_variant "Worker 7B f16 / 24t (NUMA quarter)"
sweep_model_mode "$WORKER_7B" "$WORKER_DRAFT" "24t" 24 "worker_7b_f16" "" "$DM_GRID_STANDARD" 10 7 300

# ============================================================
# Model 2: Old frontdoor 30B-A3B Q4KM (MoE, ~3 min/config)
# Draft: Coder-DRAFT-0.75B Q4, deployment: 4×48t
# ============================================================

next_variant "Qwen3-Coder-30B-A3B Q4KM / 192t (interleave)"
sweep_model_mode "$FRONTDOOR_30B" "$FRONTDOOR_30B_DRAFT" "192t" 192 "frontdoor_30b_a3b" "" "$DM_GRID_STANDARD" 10 7 600

next_variant "Qwen3-Coder-30B-A3B Q4KM / 48t (NUMA quarter)"
sweep_model_mode "$FRONTDOOR_30B" "$FRONTDOOR_30B_DRAFT" "48t" 48 "frontdoor_30b_a3b" "" "$DM_GRID_STANDARD" 10 7 600

# ============================================================
# Model 3: Coder-32B Q4_K_M (~3 min/config)
# Draft: Coder-0.5B Q8, deployment: 4×48t
# ============================================================

next_variant "Coder-32B Q4_K_M / 192t (interleave)"
sweep_model_mode "$CODER_Q4KM" "$CODER_DRAFT" "192t" 192 "coder_32b_q4km" "" "$DM_GRID_STANDARD" 10 7 600

next_variant "Coder-32B Q4_K_M / 48t (NUMA quarter)"
sweep_model_mode "$CODER_Q4KM" "$CODER_DRAFT" "48t" 48 "coder_32b_q4km" "" "$DM_GRID_STANDARD" 10 7 600

# ============================================================
# Model 4: Coder-32B Q8_0 (~4 min/config)
# Draft: Coder-0.5B Q8, deployment: 4×48t
# ============================================================

next_variant "Coder-32B Q8_0 / 192t (interleave)"
sweep_model_mode "$CODER_Q8" "$CODER_DRAFT" "192t" 192 "coder_32b_q8" "" "$DM_GRID_STANDARD" 10 7 600

next_variant "Coder-32B Q8_0 / 48t (NUMA quarter)"
sweep_model_mode "$CODER_Q8" "$CODER_DRAFT" "48t" 48 "coder_32b_q8" "" "$DM_GRID_STANDARD" 10 7 600

# ============================================================
# Model 5: Coder-32B f16 (~5 min/config)
# Draft: Coder-0.5B Q8, deployment: 4×48t
# ============================================================

next_variant "Coder-32B f16 / 192t (interleave)"
sweep_model_mode "$CODER_F16" "$CODER_DRAFT" "192t" 192 "coder_32b_f16" "" "$DM_GRID_STANDARD" 10 7 900

next_variant "Coder-32B f16 / 48t (NUMA quarter)"
sweep_model_mode "$CODER_F16" "$CODER_DRAFT" "48t" 48 "coder_32b_f16" "" "$DM_GRID_STANDARD" 10 7 900

# ============================================================
# Model 6: Qwen3.5-122B-A10B Q4KM (MoE hybrid, ~5 min/config)
# Draft: Qwen3.5-0.8B Q8, deployment: 1×96t
# Extra: --override-kv for moe8 expert count
# ============================================================

EXTRA_122B="--override-kv qwen3moe.expert_used_count=int:8"

next_variant "Qwen3.5-122B-A10B Q4KM / 192t (interleave)"
sweep_model_mode "$ARCH_122B" "$ARCH_122B_DRAFT" "192t" 192 "arch_122b_q4km" "$EXTRA_122B" "$DM_GRID_STANDARD" 10 7 900

next_variant "Qwen3.5-122B-A10B Q4KM / 96t (NUMA half)"
sweep_model_mode "$ARCH_122B" "$ARCH_122B_DRAFT" "96t" 96 "arch_122b_q4km" "$EXTRA_122B" "$DM_GRID_STANDARD" 10 7 900

# ============================================================
# Model 7: Coder-480B-A35B Q4KM (MoE, ~10 min/config)
# Draft: Coder-DRAFT-0.75B Q4, deployment: 1×96t
# Optimized: 5 requests, 4 warmup, skip dm=8
# ============================================================

next_variant "Coder-480B-A35B Q4KM / 192t (interleave)"
sweep_model_mode "$ARCH_480B" "$ARCH_480B_DRAFT" "192t" 192 "arch_480b_q4km" "" "$DM_GRID_480B" 5 4 1200

next_variant "Coder-480B-A35B Q4KM / 96t (NUMA half)"
sweep_model_mode "$ARCH_480B" "$ARCH_480B_DRAFT" "96t" 96 "arch_480b_q4km" "" "$DM_GRID_480B" 5 4 1200

# ============================================================
# Summary
# ============================================================

TOTAL_ELAPSED=$(( $(date +%s) - START_TIME ))
TOTAL_MIN=$((TOTAL_ELAPSED / 60))
TOTAL_HRS=$((TOTAL_MIN / 60))
REMAIN_MIN=$((TOTAL_MIN % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SWEEP COMPLETE — ${TOTAL_HRS}h ${REMAIN_MIN}m                                      ║"
echo "║  Raw CSV:     $CSV_FILE"
echo "║  Summary CSV: $SUMMARY_FILE"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Print summary table
echo "═══════════════════════════════════════════════════════════════"
echo "  BEST PARAMS PER MODEL (from summary CSV)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
column -t -s',' "$SUMMARY_FILE"
echo ""

# Detailed analysis from raw CSV
python3 << 'PYEOF'
import csv, sys, os

csv_path = os.environ.get("CSV_FILE", "")
if not csv_path:
    import glob
    files = sorted(glob.glob("/mnt/raid0/llm/epyc-inference-research/data/all_spec_sweep/all_spec_sweep_*.csv"))
    csv_path = files[-1] if files else ""

if not csv_path:
    print("No CSV found")
    sys.exit()

# Parse: group by (model, mode, dm, ps) -> list of tps
groups = {}
with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = (row["model"], row["mode"], row["draft_max"], row["p_split"])
        tps = float(row["decode_tps"])
        if tps > 0:
            groups.setdefault(key, []).append(tps)

# Best per (model, mode)
best = {}
for (model, mode, dm, ps), tps_list in groups.items():
    avg = sum(tps_list) / len(tps_list)
    vm_key = (model, mode)
    if vm_key not in best or avg > best[vm_key][2]:
        best[vm_key] = (dm, ps, avg)

print("=" * 75)
print("  VERIFIED OPTIMAL PARAMS (from raw measurements)")
print("=" * 75)
print()
print(f"  {'Model':<22} {'Mode':<6} {'Best dm':<10} {'Best ps':<10} {'Avg t/s':<10}")
print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")

for (model, mode) in sorted(best.keys()):
    dm, ps, avg = best[(model, mode)]
    print(f"  {model:<22} {mode:<6} {dm:<10} {ps:<10} {avg:<10.2f}")

print()

# Full grid for reference
print("=" * 75)
print("  FULL MEASUREMENT GRID")
print("=" * 75)
print()
print(f"  {'Model':<22} {'Mode':<6} {'dm':<6} {'ps':<8} {'Avg t/s':<10} {'N':<4}")
print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*8} {'-'*10} {'-'*4}")

for (model, mode, dm, ps) in sorted(groups.keys()):
    tps_list = groups[(model, mode, dm, ps)]
    avg = sum(tps_list) / len(tps_list)
    print(f"  {model:<22} {mode:<6} {dm:<6} {ps:<8} {avg:<10.2f} {len(tps_list):<4}")

print()
PYEOF

echo ""
echo "Done. Raw CSV:     $CSV_FILE"
echo "      Summary CSV: $SUMMARY_FILE"
