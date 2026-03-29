#!/bin/bash
# Dual-Mode Spec Param Sweep — Production Models
#
# Sweeps (draft_max, p_split) in both deployment modes for each spec-decode model:
#   1. 192t mode (no taskset) — max single-request throughput
#   2. NUMA mode (taskset to production config) — per-instance throughput
#
# Restarts server for each (dm, p_split) combo — per-request speculative.n_max
# is unreliable (tested: inconsistent behavior across endpoints).
#
# Models:
#   - coder_escalation Q4KM (32B dense): 192t then 48t Q0A, dm sweep, p_split=0 only
#   - architect_general 122B (MoE hybrid): 192t then 96t node0, dm + p_split sweep
#   - architect_coding 480B (MoE): 192t then 96t node0, dm + p_split sweep
#   - worker 7B f16 (dense): 192t then 24t Q0A, dm + p_split sweep
#
# Output: CSV per model in data/spec_param_sweep/

set -euo pipefail

# Cleanup on exit — kill any llama-server we spawned
cleanup() {
    pkill -9 -f "llama-server.*--port 8180" 2>/dev/null || true
    lsof -ti :8180 2>/dev/null | xargs kill -9 2>/dev/null || true
}
trap cleanup EXIT

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL_BASE="/mnt/raid0/llm/lmstudio/models"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/spec_param_sweep"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=128
N_REQUESTS=10
BASE_PORT=8180

# NUMA bindings
NODE0_CPUS="0-47,96-143"
NODE0A_CPUS="0-23,96-119"

# Models
CODER_Q4KM="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
CODER_DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

ARCH_122B="${MODEL_BASE}/unsloth/Qwen3.5-122B-A10B-GGUF/Q4_K_M/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf"
ARCH_122B_DRAFT="${MODEL_BASE}/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

ARCH_480B="${MODEL_BASE}/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
ARCH_480B_DRAFT="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

WORKER_7B="/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf"
WORKER_DRAFT="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
    "Design a microservice architecture for a real-time chat application with message persistence and delivery guarantees:"
    "Write a Rust implementation of a lock-free concurrent queue using compare-and-swap operations:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "Dual-Mode Spec Param Sweep"
echo "=========================="
echo "n_predict=$N_PREDICT, n_requests=$N_REQUESTS"
echo "Results: $DATA_DIR"
echo ""

# ============================================================
# Helper Functions
# ============================================================

wait_for_server() {
    local port=$1
    local max_wait=${2:-600}
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: server on port $port did not start within ${max_wait}s"
            return 1
        fi
    done
}

warmup_server() {
    local port=$1
    curl -s "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1
}

run_completion() {
    local port=$1
    local prompt="$2"
    local n_predict=$3
    local start_ms end_ms elapsed_ms tokens tps
    start_ms=$(date +%s%N | cut -b1-13)
    local response
    response=$(curl -s --max-time 600 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"test\",
            \"messages\": [{\"role\": \"user\", \"content\": $(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],
            \"max_tokens\": ${n_predict},
            \"temperature\": 0.0,
            \"stream\": false
        }" 2>/dev/null)
    end_ms=$(date +%s%N | cut -b1-13)
    elapsed_ms=$((end_ms - start_ms))
    tokens=$(echo "$response" | python3 -c "
import json, sys
try:
    r = json.load(sys.stdin)
    print(r.get('usage', {}).get('completion_tokens', 0))
except:
    print(0)
" 2>/dev/null)
    if [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ]; then
        tps=$(python3 -c "print(f'{$tokens / ($elapsed_ms / 1000):.2f}')")
    else
        tps="0.00"
    fi
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill -9 "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    # Also kill any llama-server on our benchmark port
    local port_pids
    port_pids=$(lsof -ti :$BASE_PORT 2>/dev/null || true)
    if [ -n "$port_pids" ]; then
        echo "$port_pids" | xargs kill -9 2>/dev/null || true
        sleep 1
    fi
    sleep 2
}

# Run N_REQUESTS completions, return "avg,p50,p95"
run_sweep_requests() {
    local port=$1
    local model_name=$2
    local mode=$3
    local threads=$4
    local dm=$5
    local ps=$6
    local results_file=$7

    local tps_values=()
    for ((r=0; r<N_REQUESTS; r++)); do
        local pidx=$((r % ${#PROMPTS[@]}))
        result=$(run_completion $port "${PROMPTS[$pidx]}" "$N_PREDICT")
        local tps
        tps=$(echo "$result" | cut -d, -f3)
        tps_values+=("$tps")
        echo "$model_name,$mode,$threads,$dm,$ps,$r,$result" >> "$results_file"
    done

    python3 -c "
import sys
vals = sorted([float(v) for v in sys.argv[1:] if float(v) > 0])
if not vals:
    print('0.00,0.00,0.00')
    sys.exit()
n = len(vals)
avg = sum(vals) / n
p50 = vals[n // 2]
p95 = vals[int(n * 0.95)]
print(f'{avg:.2f},{p50:.2f},{p95:.2f}')
" "${tps_values[@]}"
}

# Sweep one model in one mode. Returns "best_dm best_ps" on last line.
sweep_model_mode() {
    local model_name=$1
    local target=$2
    local drafter=$3
    local threads=$4
    local cpus=$5
    local mode=$6
    local extra_args=$7
    local do_psplit=$8  # "yes" or "no"
    local results_file=$9

    # Coarse dm grid — skip 4 (always suboptimal for production models)
    local dm_values=(8 16 24 32 48)
    local best_dm=24
    local best_avg=0

    echo "  [${mode} — dm sweep, p_split=0]"
    for dm in "${dm_values[@]}"; do
        echo "    dm=$dm ..."

        local spec_args="--draft-max $dm --draft-p-split 0 --lookup --flash-attn on"

        if [ "$cpus" = "all" ]; then
            "$LLAMA_SERVER" -m "$target" -md "$drafter" $extra_args $spec_args \
                -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                > "$LOG_DIR/${model_name}_${mode}_dm${dm}_ps0.log" 2>&1 &
        else
            taskset -c "$cpus" "$LLAMA_SERVER" -m "$target" -md "$drafter" $extra_args $spec_args \
                -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                > "$LOG_DIR/${model_name}_${mode}_dm${dm}_ps0.log" 2>&1 &
        fi
        local PID=$!

        if ! wait_for_server $BASE_PORT; then
            echo "      FAILED to start"
            kill_servers $PID
            continue
        fi
        warmup_server $BASE_PORT

        local stats
        stats=$(run_sweep_requests $BASE_PORT "$model_name" "$mode" "$threads" "$dm" "0" "$results_file")
        local avg
        avg=$(echo "$stats" | cut -d, -f1)
        echo "      avg=${avg} t/s (p50=$(echo "$stats" | cut -d, -f2), p95=$(echo "$stats" | cut -d, -f3))"

        local is_better
        is_better=$(python3 -c "print(1 if $avg > $best_avg else 0)")
        if [ "$is_better" = "1" ]; then
            best_avg="$avg"
            best_dm=$dm
        fi

        kill_servers $PID
    done
    echo "    Best dm=$best_dm (avg=$best_avg t/s)"

    # Phase 2: p_split sweep at best dm
    local best_ps=0
    if [ "$do_psplit" = "yes" ]; then
        local ps_values=(0.05 0.1 0.3)
        echo ""
        echo "  [${mode} — p_split sweep at dm=$best_dm (baseline=$best_avg t/s)]"

        for ps in "${ps_values[@]}"; do
            echo "    p_split=$ps ..."

            local spec_args="--draft-max $best_dm --kv-unified --lookup --draft-p-split $ps --flash-attn on"

            if [ "$cpus" = "all" ]; then
                "$LLAMA_SERVER" -m "$target" -md "$drafter" $extra_args $spec_args \
                    -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                    > "$LOG_DIR/${model_name}_${mode}_dm${best_dm}_ps${ps}.log" 2>&1 &
            else
                taskset -c "$cpus" "$LLAMA_SERVER" -m "$target" -md "$drafter" $extra_args $spec_args \
                    -t "$threads" -np 1 --port $BASE_PORT -ngl 0 --metrics -ub 512 \
                    > "$LOG_DIR/${model_name}_${mode}_dm${best_dm}_ps${ps}.log" 2>&1 &
            fi
            local PID=$!

            if ! wait_for_server $BASE_PORT; then
                echo "      FAILED"
                kill_servers $PID
                continue
            fi
            warmup_server $BASE_PORT

            local stats
            stats=$(run_sweep_requests $BASE_PORT "$model_name" "$mode" "$threads" "$best_dm" "$ps" "$results_file")
            local avg
            avg=$(echo "$stats" | cut -d, -f1)
            echo "      avg=${avg} t/s (p50=$(echo "$stats" | cut -d, -f2), p95=$(echo "$stats" | cut -d, -f3))"

            local is_better
            is_better=$(python3 -c "print(1 if $avg > $best_avg else 0)")
            if [ "$is_better" = "1" ]; then
                best_avg="$avg"
                best_ps="$ps"
            fi

            kill_servers $PID
        done
        echo "    Best p_split=$best_ps (avg=$best_avg t/s)"
    fi

    echo ""
    echo "$best_dm $best_ps"
}

# ============================================================
# MODEL 1: Coder Q4_K_M (32B dense, 20 GB) — p_split=0 only
# ============================================================

echo "================================================================"
echo "=== MODEL 1: Coder Q4_K_M (32B dense, 20 GB)                ==="
echo "================================================================"

M1_RESULTS="${DATA_DIR}/coder_q4km_${TIMESTAMP}.csv"
echo "model,mode,threads,draft_max,p_split,request_idx,tokens_generated,time_ms,tokens_per_sec" > "$M1_RESULTS"

m1_192t=$(sweep_model_mode "coder_q4km" "$CODER_Q4KM" "$CODER_DRAFT" 192 "all" "192t" "" "no" "$M1_RESULTS")
m1_192t_best=$(echo "$m1_192t" | tail -1)
m1_numa=$(sweep_model_mode "coder_q4km" "$CODER_Q4KM" "$CODER_DRAFT" 48 "$NODE0A_CPUS" "numa_48t" "" "no" "$M1_RESULTS")
m1_numa_best=$(echo "$m1_numa" | tail -1)
echo ">>> Coder Q4KM: 192t=${m1_192t_best}, NUMA=${m1_numa_best}"
echo ""

# ============================================================
# MODEL 2: Architect General 122B (MoE hybrid, 69 GB)
# ============================================================

echo "================================================================"
echo "=== MODEL 2: Architect General 122B (MoE hybrid, 69 GB)     ==="
echo "================================================================"

M2_RESULTS="${DATA_DIR}/arch_122b_${TIMESTAMP}.csv"
echo "model,mode,threads,draft_max,p_split,request_idx,tokens_generated,time_ms,tokens_per_sec" > "$M2_RESULTS"
M2_EXTRA="--override-kv qwen3moe.expert_used_count=int:8"

m2_192t=$(sweep_model_mode "arch_122b" "$ARCH_122B" "$ARCH_122B_DRAFT" 192 "all" "192t" "$M2_EXTRA" "yes" "$M2_RESULTS")
m2_192t_best=$(echo "$m2_192t" | tail -1)
m2_numa=$(sweep_model_mode "arch_122b" "$ARCH_122B" "$ARCH_122B_DRAFT" 96 "$NODE0_CPUS" "numa_96t" "$M2_EXTRA" "yes" "$M2_RESULTS")
m2_numa_best=$(echo "$m2_numa" | tail -1)
echo ">>> Arch 122B: 192t=${m2_192t_best}, NUMA=${m2_numa_best}"
echo ""

# ============================================================
# MODEL 3: Architect Coding 480B (MoE, 250 GB)
# ============================================================

echo "================================================================"
echo "=== MODEL 3: Architect Coding 480B (MoE, 250 GB)            ==="
echo "================================================================"

M3_RESULTS="${DATA_DIR}/arch_480b_${TIMESTAMP}.csv"
echo "model,mode,threads,draft_max,p_split,request_idx,tokens_generated,time_ms,tokens_per_sec" > "$M3_RESULTS"

m3_192t=$(sweep_model_mode "arch_480b" "$ARCH_480B" "$ARCH_480B_DRAFT" 192 "all" "192t" "" "yes" "$M3_RESULTS")
m3_192t_best=$(echo "$m3_192t" | tail -1)
m3_numa=$(sweep_model_mode "arch_480b" "$ARCH_480B" "$ARCH_480B_DRAFT" 96 "$NODE0_CPUS" "numa_96t" "" "yes" "$M3_RESULTS")
m3_numa_best=$(echo "$m3_numa" | tail -1)
echo ">>> Arch 480B: 192t=${m3_192t_best}, NUMA=${m3_numa_best}"
echo ""

# ============================================================
# MODEL 4: Worker 7B f16 (dense, 14 GB)
# ============================================================

echo "================================================================"
echo "=== MODEL 4: Worker 7B f16 (dense, 14 GB)                   ==="
echo "================================================================"

M4_RESULTS="${DATA_DIR}/worker_7b_${TIMESTAMP}.csv"
echo "model,mode,threads,draft_max,p_split,request_idx,tokens_generated,time_ms,tokens_per_sec" > "$M4_RESULTS"

m4_192t=$(sweep_model_mode "worker_7b" "$WORKER_7B" "$WORKER_DRAFT" 192 "all" "192t" "" "yes" "$M4_RESULTS")
m4_192t_best=$(echo "$m4_192t" | tail -1)
m4_numa=$(sweep_model_mode "worker_7b" "$WORKER_7B" "$WORKER_DRAFT" 24 "$NODE0A_CPUS" "numa_24t" "" "yes" "$M4_RESULTS")
m4_numa_best=$(echo "$m4_numa" | tail -1)
echo ">>> Worker 7B: 192t=${m4_192t_best}, NUMA=${m4_numa_best}"
echo ""

# ============================================================
# Summary
# ============================================================

echo "================================================================"
echo "=== SWEEP COMPLETE ==="
echo "================================================================"
echo ""
echo "Results:"
echo "  Coder Q4KM:  $M1_RESULTS"
echo "  Arch 122B:   $M2_RESULTS"
echo "  Arch 480B:   $M3_RESULTS"
echo "  Worker 7B:   $M4_RESULTS"
echo "  Logs:        $LOG_DIR"
echo ""

export DATA_DIR TIMESTAMP
python3 << 'PYEOF'
import csv, os, glob

data_dir = os.environ.get("DATA_DIR", ".")
timestamp = os.environ.get("TIMESTAMP", "")

print("\n=== Summary: Best Configs Per Model ===\n")
print(f"{'Model':<20} {'Mode':<12} {'Threads':>7} {'Best dm':>8} {'Best ps':>8} {'Avg t/s':>10}")
print("-" * 70)

for pattern in ["coder_q4km", "arch_122b", "arch_480b", "worker_7b"]:
    files = glob.glob(f"{data_dir}/{pattern}_{timestamp}.csv")
    if not files:
        continue
    rows = []
    with open(files[0]) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    groups = {}
    for row in rows:
        key = (row["mode"], row["draft_max"], row["p_split"])
        tps = float(row["tokens_per_sec"])
        if tps > 0:
            groups.setdefault(key, []).append(tps)

    best_per_mode = {}
    for (mode, dm, ps), vals in groups.items():
        avg = sum(vals) / len(vals)
        if mode not in best_per_mode or avg > best_per_mode[mode][2]:
            threads = next((r["threads"] for r in rows if r["mode"] == mode), "?")
            best_per_mode[mode] = (dm, ps, avg, threads)

    for mode in sorted(best_per_mode.keys()):
        dm, ps, avg, threads = best_per_mode[mode]
        print(f"{pattern:<20} {mode:<12} {threads:>7} {dm:>8} {ps:>8} {avg:>10.2f}")

print()
PYEOF
