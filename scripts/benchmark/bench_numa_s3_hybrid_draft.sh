#!/bin/bash
# S3: NUMA 4-Way + External AR Draft on Hybrid Model
#
# Tests whether external draft speculation compounds with NUMA 4-way on hybrid models.
# HSD handoff showed +5.4% from freeze-recurrent + Qwen2.5-Coder-0.5B on Qwen3.5-9B.
# S2 showed 6.9x from NUMA 4-way on Qwen3.5-35B-A3B without speculation.
# Question: does the gain compound?
#
# Configs:
#   S3-A: 4×48t, no drafter (S2 baseline, expect ~49.7 t/s agg)
#   S3-B: 4×48t, Qwen3.5-0.8B Q8_0, dm=16 (same-family drafter)
#   S3-C: 4×48t, Qwen3.5-0.8B Q8_0, dm=32
#   S3-D: 4×48t, Qwen2.5-Coder-0.5B f16, dm=16 (cross-family fast drafter)
#   S3-E: 4×48t, Qwen2.5-Coder-0.5B f16, dm=32
#   S3-F: 1×96t node0, Qwen2.5-Coder-0.5B f16, dm=16 (reproduce HSD +5.4% on 35B)
#
# Note: freeze-recurrent activates automatically for hybrid models with speculation.

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
TARGET="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-Q4_K_M.gguf"
DRAFTER_08B="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
DRAFTER_05B="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_s3_hybrid_draft"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/s3_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=256
BASE_PORT=8190

NODE0A_CPUS="0-23,96-119"
NODE0B_CPUS="24-47,120-143"
NODE1A_CPUS="48-71,144-167"
NODE1B_CPUS="72-95,168-191"
NODE0_CPUS="0-47,96-143"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "S3: NUMA 4-Way + External AR Draft on Hybrid"
echo "=============================================="
echo "Target: $(basename "$TARGET") (19 GB, hybrid MoE+DeltaNet)"
echo "Drafter A: $(basename "$DRAFTER_08B") (same-family, 775 MB)"
echo "Drafter B: $(basename "$DRAFTER_05B") (cross-family fast, 949 MB)"
echo "n_predict=$N_PREDICT"
echo "Results: $RESULTS_FILE"
echo ""

echo "config,instance,threads,cpu_binding,drafter,draft_max,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1
    local max_wait=600
    local elapsed=0
    while true; do
        local health
        health=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "")
        if echo "$health" | grep -q '"status":"ok"'; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        if [ $elapsed -ge $max_wait ]; then
            echo "ERROR: port $port timeout after ${max_wait}s"
            return 1
        fi
    done
}

run_completion() {
    local port=$1 prompt="$2" n_predict=$3
    local start_ms end_ms elapsed_ms tokens tps
    start_ms=$(date +%s%N | cut -b1-13)
    local response
    response=$(curl -s --max-time 600 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":${n_predict},\"temperature\":0.0,\"stream\":false}" 2>/dev/null)
    end_ms=$(date +%s%N | cut -b1-13)
    elapsed_ms=$((end_ms - start_ms))
    tokens=$(echo "$response" | python3 -c "import json,sys;
try: print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))
except: print(0)" 2>/dev/null)
    [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ] && tps=$(python3 -c "print(f'{$tokens/($elapsed_ms/1000):.2f}')") || tps="0.00"
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 2
}

warmup_server() {
    curl -s "http://localhost:${1}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1
}

# Run a 4×48t benchmark
bench_quad() {
    local config=$1 drafter_path=$2 drafter_label=$3 dm=$4
    local extra_args=""
    [ -n "$drafter_path" ] && extra_args="-md $drafter_path --draft-max $dm"

    echo "=== $config: 4×48t, drafter=$drafter_label, dm=$dm ==="

    local P1=$BASE_PORT P2=$((BASE_PORT+1)) P3=$((BASE_PORT+2)) P4=$((BASE_PORT+3))
    local PIDS=()

    for i in 0 1 2 3; do
        local cpus port
        case $i in
            0) cpus=$NODE0A_CPUS; port=$P1;;
            1) cpus=$NODE0B_CPUS; port=$P2;;
            2) cpus=$NODE1A_CPUS; port=$P3;;
            3) cpus=$NODE1B_CPUS; port=$P4;;
        esac
        taskset -c "$cpus" "$LLAMA_SERVER" -m "$TARGET" $extra_args \
            -t 48 -np 1 --port $port -ngl 0 --metrics \
            > "$LOG_DIR/${config}_q${i}.log" 2>&1 &
        PIDS+=($!)
    done

    echo "  Loading 4 instances..."
    for port in $P1 $P2 $P3 $P4; do
        wait_for_server $port || { kill_servers "${PIDS[@]}"; return; }
    done

    # Sequential warmup (avoid wait issues)
    for port in $P1 $P2 $P3 $P4; do warmup_server $port; done
    echo "  All ready"

    for pi in "${!PROMPTS[@]}"; do
        local r1 r2 r3 r4
        r1=$(run_completion $P1 "${PROMPTS[$pi]}" "$N_PREDICT")
        r2=$(run_completion $P2 "${PROMPTS[$pi]}" "$N_PREDICT")
        r3=$(run_completion $P3 "${PROMPTS[$pi]}" "$N_PREDICT")
        r4=$(run_completion $P4 "${PROMPTS[$pi]}" "$N_PREDICT")

        echo "$config,q0a,48,node0a,$drafter_label,$dm,$pi,$r1" >> "$RESULTS_FILE"
        echo "$config,q0b,48,node0b,$drafter_label,$dm,$pi,$r2" >> "$RESULTS_FILE"
        echo "$config,q1a,48,node1a,$drafter_label,$dm,$pi,$r3" >> "$RESULTS_FILE"
        echo "$config,q1b,48,node1b,$drafter_label,$dm,$pi,$r4" >> "$RESULTS_FILE"

        local t1 t2 t3 t4 agg
        t1=$(echo "$r1" | cut -d, -f3); t2=$(echo "$r2" | cut -d, -f3)
        t3=$(echo "$r3" | cut -d, -f3); t4=$(echo "$r4" | cut -d, -f3)
        agg=$(python3 -c "print(f'{$t1+$t2+$t3+$t4:.2f}')")
        echo "  prompt $pi: q0a=$t1, q0b=$t2, q1a=$t3, q1b=$t4, agg=$agg t/s"
    done

    kill_servers "${PIDS[@]}"
    echo ""
}

# Run a 1×96t benchmark
bench_single() {
    local config=$1 drafter_path=$2 drafter_label=$3 dm=$4
    local extra_args=""
    [ -n "$drafter_path" ] && extra_args="-md $drafter_path --draft-max $dm"

    echo "=== $config: 1×96t node0, drafter=$drafter_label, dm=$dm ==="

    taskset -c "$NODE0_CPUS" "$LLAMA_SERVER" -m "$TARGET" $extra_args \
        -t 96 -np 1 --port $BASE_PORT -ngl 0 --metrics \
        > "$LOG_DIR/${config}.log" 2>&1 &
    local PID=$!

    wait_for_server $BASE_PORT || { kill_servers $PID; return; }
    warmup_server $BASE_PORT
    echo "  Server ready"

    for pi in "${!PROMPTS[@]}"; do
        local result
        result=$(run_completion $BASE_PORT "${PROMPTS[$pi]}" "$N_PREDICT")
        echo "$config,1,96,node0,$drafter_label,$dm,$pi,$result" >> "$RESULTS_FILE"
        local tps=$(echo "$result" | cut -d, -f3)
        echo "  prompt $pi: $tps t/s"
    done

    kill_servers $PID
    echo ""
}

# ============================================================
# Run all configs
# ============================================================

# S3-A: 4×48t, no drafter (baseline)
bench_quad "S3-A" "" "none" "0"

# S3-B: 4×48t, Qwen3.5-0.8B Q8_0, dm=16
bench_quad "S3-B" "$DRAFTER_08B" "q35-0.8B-Q8" "16"

# S3-C: 4×48t, Qwen3.5-0.8B Q8_0, dm=32
bench_quad "S3-C" "$DRAFTER_08B" "q35-0.8B-Q8" "32"

# S3-D: 4×48t, Qwen2.5-Coder-0.5B f16, dm=16
bench_quad "S3-D" "$DRAFTER_05B" "q25-0.5B-f16" "16"

# S3-E: 4×48t, Qwen2.5-Coder-0.5B f16, dm=32
bench_quad "S3-E" "$DRAFTER_05B" "q25-0.5B-f16" "32"

# S3-F: 1×96t node0, Qwen2.5-Coder-0.5B f16, dm=16 (reproduce HSD baseline)
bench_single "S3-F" "$DRAFTER_05B" "q25-0.5B-f16" "16"

# ============================================================
# Summary
# ============================================================
echo "=== S3 SUMMARY ==="
python3 - "$RESULTS_FILE" << 'PYEOF'
import csv, sys
from collections import defaultdict

results = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        results[row['config']].append(float(row['tokens_per_sec']))

print(f"{'Config':<8} {'Setup':<40} {'Per-inst':<12} {'Aggregate':<12}")
print("-" * 72)

labels = {
    'S3-A': '4×48t, no drafter',
    'S3-B': '4×48t, Qwen3.5-0.8B Q8_0, dm=16',
    'S3-C': '4×48t, Qwen3.5-0.8B Q8_0, dm=32',
    'S3-D': '4×48t, Qwen2.5-0.5B f16, dm=16',
    'S3-E': '4×48t, Qwen2.5-0.5B f16, dm=32',
    'S3-F': '1×96t node0, Qwen2.5-0.5B f16, dm=16',
}

baseline_agg = None
for cfg in ['S3-A', 'S3-B', 'S3-C', 'S3-D', 'S3-E', 'S3-F']:
    vals = results.get(cfg, [])
    if not vals: continue
    avg = sum(vals) / len(vals)
    if cfg == 'S3-F':
        agg = avg  # single instance
        n = 1
    else:
        # 4 instances per prompt
        per_prompt = [sum(vals[j:j+4]) for j in range(0, len(vals), 4)]
        agg = sum(per_prompt) / len(per_prompt) if per_prompt else 0
        n = 4
    if baseline_agg is None: baseline_agg = agg
    delta = f"({agg/baseline_agg:+.1%})" if baseline_agg and cfg != 'S3-A' else ""
    print(f"{cfg:<8} {labels.get(cfg,''):<40} {avg:<12.2f} {agg:<10.2f} {delta}")

PYEOF

echo ""
echo "Results: $RESULTS_FILE"
echo "Logs: $LOG_DIR"
