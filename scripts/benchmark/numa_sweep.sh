#!/bin/bash
# numa_sweep.sh — Deterministic NUMA throughput sweep for any GGUF model
#
# Flow:
#   1. Draft-max sweep at 1×96t interleave (baseline + ngram dm 16/32/48/64/96/128)
#   2. Pick best draft-max (or use --draft-max if provided)
#   3. NUMA sweep with winning draft-max: 1×96t interleave, 1×96t node0, 2×96t, 4×48t
#
# Usage:
#   numa_sweep.sh <model_path> [options]
#
# Options:
#   --name <label>          Model label for results (default: derived from filename)
#   --draft-max <n>         Skip draft-max sweep, use this value (0 = no speculation)
#   --draft-model <path>    Draft model for speculative decoding
#   --n-predict <n>         Tokens to generate per prompt (default: 256)
#   --port <port>           Base port (default: 8190)
#   --configs <list>         Comma-separated configs to run: A,B,C,D (default: all applicable)
#                           A=1×96t interleave, B=1×96t node0, C=2×96t dual, D=4×48t quad
#   --skip-quad             Skip 4×48t config (shorthand for excluding D)
#   --skip-dual             Skip 2×96t config (shorthand for excluding C)
#   --max-instances <n>     Max parallel instances (auto-detected from model size if omitted)
#   --extra-args "<args>"   Additional llama-server args
#
# Examples:
#   numa_sweep.sh /path/to/model.gguf
#   numa_sweep.sh /path/to/model.gguf --draft-max 0
#   numa_sweep.sh /path/to/model.gguf --draft-max 64 --name "M2.7-Q4XL"
#   numa_sweep.sh /path/to/model.gguf --skip-quad --extra-args "--kv-unified"

set -euo pipefail

# ============================================================
# Argument parsing
# ============================================================

MODEL_PATH=""
DRAFT_MAX=""  # empty = sweep; numeric = fixed
DRAFT_MODEL=""
MODEL_NAME=""
N_PREDICT=256
BASE_PORT=8190
CONFIGS=""  # empty = all applicable
SKIP_QUAD=false
SKIP_DUAL=false
MAX_INSTANCES=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --draft-max)     DRAFT_MAX="$2"; shift 2 ;;
        --draft-model)   DRAFT_MODEL="$2"; shift 2 ;;
        --name)          MODEL_NAME="$2"; shift 2 ;;
        --n-predict)     N_PREDICT="$2"; shift 2 ;;
        --port)          BASE_PORT="$2"; shift 2 ;;
        --configs)       CONFIGS="$2"; shift 2 ;;
        --skip-quad)     SKIP_QUAD=true; shift ;;
        --skip-dual)     SKIP_DUAL=true; shift ;;
        --max-instances) MAX_INSTANCES="$2"; shift 2 ;;
        --extra-args)    EXTRA_ARGS="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            if [ -z "$MODEL_PATH" ]; then
                MODEL_PATH="$1"
            else
                echo "ERROR: unexpected argument: $1" >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "ERROR: model path required" >&2
    echo "Usage: numa_sweep.sh <model_path> [options]" >&2
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model file not found: $MODEL_PATH" >&2
    exit 1
fi

# ============================================================
# Constants
# ============================================================

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/numa_sweeps"

NODE0_CPUS="0-47,96-143"
NODE1_CPUS="48-95,144-191"
NODE0A_CPUS="0-23,96-119"
NODE0B_CPUS="24-47,120-143"
NODE1A_CPUS="48-71,144-167"
NODE1B_CPUS="72-95,168-191"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature, gravitational waves, and black holes:"
    "Implement a concurrent hash map in C++ using fine-grained locking with reader-writer locks:"
)

DRAFT_MAX_SWEEP_VALUES=(0 16 32 48 64 96 128)

# ============================================================
# Derived values
# ============================================================

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME=$(basename "$MODEL_PATH" .gguf | sed 's/-00001-of-[0-9]*//')
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/${MODEL_NAME}_numa_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${MODEL_NAME}_${TIMESTAMP}"

# Estimate model size for auto-detecting max instances
MODEL_DIR=$(dirname "$MODEL_PATH")
MODEL_SIZE_GB=$(du -sB1 "$MODEL_DIR" 2>/dev/null | awk '{printf "%.0f", $1/1024/1024/1024}')
SINGLE_FILE_SIZE=$(stat --printf="%s" "$MODEL_PATH" 2>/dev/null || echo 0)
if [ "$SINGLE_FILE_SIZE" -gt 0 ] && [ "$MODEL_SIZE_GB" -lt 2 ]; then
    MODEL_SIZE_GB=$(awk "BEGIN{printf \"%.0f\", $SINGLE_FILE_SIZE/1024/1024/1024}")
fi

TOTAL_RAM_GB=1100

if [ -z "$MAX_INSTANCES" ]; then
    USABLE_RAM=$((TOTAL_RAM_GB - 50))
    if [ "$MODEL_SIZE_GB" -gt 0 ]; then
        MAX_INSTANCES=$((USABLE_RAM / MODEL_SIZE_GB))
        if [ "$MAX_INSTANCES" -gt 4 ]; then MAX_INSTANCES=4; fi
        if [ "$MAX_INSTANCES" -lt 1 ]; then MAX_INSTANCES=1; fi
    else
        MAX_INSTANCES=1
    fi
fi

DRAFT_ARG=""
if [ -n "$DRAFT_MODEL" ]; then
    DRAFT_ARG="-md $DRAFT_MODEL"
fi

# Apply --skip-* flags to CONFIGS
if [ "$SKIP_DUAL" = true ] && [ -z "$CONFIGS" ]; then CONFIGS="A,B,D"; fi
if [ "$SKIP_QUAD" = true ] && [ -z "$CONFIGS" ]; then CONFIGS="A,B,C"; fi
if [ "$SKIP_DUAL" = true ] && [ "$SKIP_QUAD" = true ]; then CONFIGS="A,B"; fi

should_run() {
    local config=$1
    if [ -z "$CONFIGS" ]; then return 0; fi
    echo ",$CONFIGS," | grep -q ",$config,"
}

PHASE1_RAN=false

mkdir -p "$DATA_DIR" "$LOG_DIR"

# ============================================================
# Banner
# ============================================================

echo "================================================================"
echo "  NUMA Sweep: $MODEL_NAME"
echo "================================================================"
echo "  Model:       $MODEL_PATH"
echo "  Size:        ~${MODEL_SIZE_GB} GB"
echo "  Max inst:    $MAX_INSTANCES"
echo "  Draft-max:   ${DRAFT_MAX:-sweep}"
echo "  Draft model: ${DRAFT_MODEL:-none}"
echo "  Extra args:  ${EXTRA_ARGS:-none}"
echo "  n_predict:   $N_PREDICT"
echo "  Results:     $RESULTS_FILE"
echo "  Logs:        $LOG_DIR"
echo "================================================================"
echo ""

echo "model,config,instance,threads,cpu_binding,spec,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

# ============================================================
# Helpers
# ============================================================

wait_for_server() {
    local port=$1
    local max_wait=600
    local elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 5
        elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then
            echo "    ERROR: server on port $port did not start within ${max_wait}s"
            return 1
        fi
    done
    echo "    port $port ready (${elapsed}s)"
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
        tps=$(python3 -c "print(f'{int($tokens) / (int($elapsed_ms) / 1000):.2f}')")
    else
        tps="0.00"
    fi
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 2
}

warmup_server() {
    local port=$1
    curl -s --max-time 120 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32,"temperature":0}' \
        > /dev/null 2>&1
}

run_prompts() {
    local port=$1
    local model_name=$2
    local config=$3
    local instance=$4
    local threads=$5
    local cpu_binding=$6
    local spec_label=$7

    for i in "${!PROMPTS[@]}"; do
        result=$(run_completion "$port" "${PROMPTS[$i]}" "$N_PREDICT")
        echo "$model_name,$config,$instance,$threads,$cpu_binding,$spec_label,$i,$result" >> "$RESULTS_FILE"
        tps=$(echo "$result" | cut -d, -f3)
        echo "    prompt $i: ${tps} t/s"
    done
}

# ============================================================
# Phase 1: Draft-max sweep at 1×96t interleave
# ============================================================

if [ -z "$DRAFT_MAX" ]; then
    echo "================================================================"
    echo "  Phase 1: draft-max sweep (1×96t interleave)"
    echo "================================================================"
    echo ""

    BEST_DM=0
    BEST_TPS=0

    for dm in "${DRAFT_MAX_SWEEP_VALUES[@]}"; do
        if [ "$dm" -eq 0 ]; then
            DM_SPEC_ARGS=""
            label="baseline"
        else
            DM_SPEC_ARGS="--spec-type ngram-simple --draft-max $dm"
            label="ngram_dm${dm}"
        fi

        echo "  --- draft-max=$dm ---"

        numactl --interleave=all "$LLAMA_SERVER" -m "$MODEL_PATH" $DRAFT_ARG $DM_SPEC_ARGS $EXTRA_ARGS \
            -t 96 -np 1 --port "$BASE_PORT" -ngl 0 --mlock --metrics \
            > "$LOG_DIR/dmsweep_dm${dm}.log" 2>&1 &
        PID=$!

        if ! wait_for_server "$BASE_PORT"; then
            echo "    FAILED to start"
            kill_servers $PID
            continue
        fi
        warmup_server "$BASE_PORT"

        dm_total_tps=0
        dm_count=0
        for i in "${!PROMPTS[@]}"; do
            result=$(run_completion "$BASE_PORT" "${PROMPTS[$i]}" "$N_PREDICT")
            echo "$MODEL_NAME,dmsweep_dm${dm},1,96,interleave,$label,$i,$result" >> "$RESULTS_FILE"
            tps=$(echo "$result" | cut -d, -f3)
            echo "    prompt $i: ${tps} t/s"
            dm_total_tps=$(python3 -c "print(float($dm_total_tps) + float($tps))")
            dm_count=$((dm_count + 1))
        done

        dm_avg=$(python3 -c "print(f'{float($dm_total_tps) / $dm_count:.2f}')")
        echo "    avg: ${dm_avg} t/s"

        # Track best
        is_better=$(python3 -c "print(1 if float($dm_avg) > float($BEST_TPS) else 0)")
        if [ "$is_better" -eq 1 ]; then
            BEST_DM=$dm
            BEST_TPS=$dm_avg
        fi

        kill_servers $PID
        echo ""
    done

    DRAFT_MAX=$BEST_DM
    PHASE1_RAN=true
    echo "================================================================"
    echo "  Phase 1 result: best draft-max=$DRAFT_MAX (${BEST_TPS} t/s)"
    echo "================================================================"
    echo ""
fi

# Build spec args from winning draft-max
if [ "$DRAFT_MAX" -eq 0 ]; then
    SPEC_ARGS=""
    SPEC_LABEL="baseline"
else
    SPEC_ARGS="--spec-type ngram-simple --draft-max $DRAFT_MAX"
    SPEC_LABEL="ngram_dm${DRAFT_MAX}"
fi

# ============================================================
# Phase 2: NUMA sweep with best draft-max
# ============================================================

echo "================================================================"
echo "  Phase 2: NUMA sweep (draft-max=$DRAFT_MAX)"
echo "================================================================"
echo ""

# --- Config A: 1×96t interleave ---

if should_run A; then
    echo "--- Config A: 1×96t interleave ---"

    if [ "$PHASE1_RAN" = true ]; then
        echo "    (already measured in Phase 1)"
    else
        numactl --interleave=all "$LLAMA_SERVER" -m "$MODEL_PATH" $DRAFT_ARG $SPEC_ARGS $EXTRA_ARGS \
            -t 96 -np 1 --port "$BASE_PORT" -ngl 0 --mlock --metrics \
            > "$LOG_DIR/A_1x96t_interleave.log" 2>&1 &
        PID_A=$!

        if ! wait_for_server "$BASE_PORT"; then
            echo "    Config A FAILED"
            kill_servers $PID_A
        else
            warmup_server "$BASE_PORT"
            run_prompts "$BASE_PORT" "$MODEL_NAME" "A_1x96t_interleave" "1" "96" "interleave" "$SPEC_LABEL"
            kill_servers $PID_A
        fi
    fi
    echo ""
else
    echo "--- Config A: SKIPPED ---"
    echo ""
fi

# --- Config B: 1×96t node0 ---

if should_run B; then
    echo "--- Config B: 1×96t node0 ---"

    numactl --cpunodebind=0 --membind=0 "$LLAMA_SERVER" -m "$MODEL_PATH" $DRAFT_ARG $SPEC_ARGS $EXTRA_ARGS \
        -t 96 -np 1 --port "$BASE_PORT" -ngl 0 --mlock --metrics \
        > "$LOG_DIR/B_1x96t_node0.log" 2>&1 &
    PID_B=$!

    if ! wait_for_server "$BASE_PORT"; then
        echo "    Config B FAILED"
        kill_servers $PID_B
    else
        warmup_server "$BASE_PORT"
        run_prompts "$BASE_PORT" "$MODEL_NAME" "B_1x96t_node0" "1" "96" "node0" "$SPEC_LABEL"
        kill_servers $PID_B
    fi
    echo ""
else
    echo "--- Config B: SKIPPED ---"
    echo ""
fi

# --- Config C: 2×96t dual-node ---

if should_run C && [ "$MAX_INSTANCES" -ge 2 ]; then
    echo "--- Config C: 2×96t dual-node ---"

    PORT_C1=$BASE_PORT
    PORT_C2=$((BASE_PORT + 1))

    numactl --cpunodebind=0 --membind=0 "$LLAMA_SERVER" -m "$MODEL_PATH" $DRAFT_ARG $SPEC_ARGS $EXTRA_ARGS \
        -t 96 -np 1 --port "$PORT_C1" -ngl 0 --mlock --metrics \
        > "$LOG_DIR/C_2x96t_n0.log" 2>&1 &
    PID_C1=$!
    echo "    loading instance 1 (node0)..."
    if ! wait_for_server "$PORT_C1"; then
        echo "    Config C instance 1 FAILED"
        kill_servers $PID_C1
    else
        numactl --cpunodebind=1 --membind=1 "$LLAMA_SERVER" -m "$MODEL_PATH" $DRAFT_ARG $SPEC_ARGS $EXTRA_ARGS \
            -t 96 -np 1 --port "$PORT_C2" -ngl 0 --mlock --metrics \
            > "$LOG_DIR/C_2x96t_n1.log" 2>&1 &
        PID_C2=$!
        echo "    loading instance 2 (node1)..."
        if ! wait_for_server "$PORT_C2"; then
            echo "    Config C instance 2 FAILED"
            kill_servers $PID_C1 $PID_C2
        else
            warmup_server "$PORT_C1" & warmup_server "$PORT_C2" & wait
            echo "    both ready"

            for i in "${!PROMPTS[@]}"; do
                r1=$(run_completion "$PORT_C1" "${PROMPTS[$i]}" "$N_PREDICT")
                r2=$(run_completion "$PORT_C2" "${PROMPTS[$i]}" "$N_PREDICT")

                echo "$MODEL_NAME,C_2x96t,n0,96,node0,$SPEC_LABEL,$i,$r1" >> "$RESULTS_FILE"
                echo "$MODEL_NAME,C_2x96t,n1,96,node1,$SPEC_LABEL,$i,$r2" >> "$RESULTS_FILE"

                t1=$(echo "$r1" | cut -d, -f3); t2=$(echo "$r2" | cut -d, -f3)
                agg=$(python3 -c "print(f'{float($t1) + float($t2):.2f}')")
                echo "    prompt $i: n0=${t1}, n1=${t2}, agg=${agg} t/s"
            done
            kill_servers $PID_C1 $PID_C2
        fi
    fi
    echo ""
else
    echo "--- Config C: SKIPPED ---"
    echo ""
fi

# --- Config D: 4×48t quarter-machine ---

if should_run D && [ "$MAX_INSTANCES" -ge 4 ]; then
    echo "--- Config D: 4×48t quarter-machine ---"

    PORT_D1=$BASE_PORT
    PORT_D2=$((BASE_PORT + 1))
    PORT_D3=$((BASE_PORT + 2))
    PORT_D4=$((BASE_PORT + 3))

    QUARTER_CPUS=("$NODE0A_CPUS" "$NODE0B_CPUS" "$NODE1A_CPUS" "$NODE1B_CPUS")
    QUARTER_MEMBIND=(0 0 1 1)  # n0a/n0b -> node0, n1a/n1b -> node1
    QUARTER_PORTS=($PORT_D1 $PORT_D2 $PORT_D3 $PORT_D4)
    QUARTER_NAMES=(n0a n0b n1a n1b)
    QUARTER_PIDS=()

    for q in 0 1 2 3; do
        numactl --membind="${QUARTER_MEMBIND[$q]}" taskset -c "${QUARTER_CPUS[$q]}" \
            "$LLAMA_SERVER" -m "$MODEL_PATH" $DRAFT_ARG $SPEC_ARGS $EXTRA_ARGS \
            -t 48 -np 1 --port "${QUARTER_PORTS[$q]}" -ngl 0 --mlock --metrics \
            > "$LOG_DIR/D_4x48t_${QUARTER_NAMES[$q]}.log" 2>&1 &
        QUARTER_PIDS+=($!)
        echo "    loading instance $((q+1)) (${QUARTER_NAMES[$q]})..."
        if ! wait_for_server "${QUARTER_PORTS[$q]}"; then
            echo "    Config D instance $((q+1)) FAILED"
            kill_servers "${QUARTER_PIDS[@]}"
            QUARTER_PIDS=()
            break
        fi
    done

    if [ ${#QUARTER_PIDS[@]} -eq 4 ]; then
        for p in "${QUARTER_PORTS[@]}"; do warmup_server "$p" & done; wait
        echo "    all ready"

        for i in "${!PROMPTS[@]}"; do
            TPS_PARTS=""
            for q in 0 1 2 3; do
                r=$(run_completion "${QUARTER_PORTS[$q]}" "${PROMPTS[$i]}" "$N_PREDICT")
                echo "$MODEL_NAME,D_4x48t,${QUARTER_NAMES[$q]},48,${QUARTER_NAMES[$q]},$SPEC_LABEL,$i,$r" >> "$RESULTS_FILE"
                t=$(echo "$r" | cut -d, -f3)
                TPS_PARTS="${TPS_PARTS} ${QUARTER_NAMES[$q]}=${t}"
            done
            echo "    prompt $i:${TPS_PARTS}"
        done
        kill_servers "${QUARTER_PIDS[@]}"
    fi
    echo ""
else
    echo "--- Config D: SKIPPED ---"
    echo ""
fi

# ============================================================
# Summary
# ============================================================

echo "================================================================"
echo "  SWEEP COMPLETE: $MODEL_NAME"
echo "  Best draft-max: $DRAFT_MAX"
echo "================================================================"
echo ""

python3 -c "
import csv
from collections import defaultdict

with open('$RESULTS_FILE') as f:
    rows = list(csv.DictReader(f))

# Separate dmsweep from numa configs
dm_configs = defaultdict(list)
numa_configs = defaultdict(list)

for r in rows:
    tps = float(r['tokens_per_sec'])
    if tps <= 0:
        continue
    if r['config'].startswith('dmsweep'):
        dm_configs[r['config']].append(tps)
    else:
        numa_configs[r['config']].append(tps)

if dm_configs:
    print('Draft-max sweep results:')
    for config in sorted(dm_configs):
        vals = dm_configs[config]
        avg = sum(vals) / len(vals)
        print(f'  {config:25s}  avg={avg:6.2f} t/s  (n={len(vals)})')
    print()

if numa_configs:
    print('NUMA sweep results:')
    for config in sorted(numa_configs):
        vals = numa_configs[config]
        avg = sum(vals) / len(vals)
        # For multi-instance configs, show per-instance avg and aggregate
        if any(x in config for x in ['2x96', '4x48']):
            # Group by instance
            inst_vals = defaultdict(list)
            for r in rows:
                if r['config'] == config and float(r['tokens_per_sec']) > 0:
                    inst_vals[r['instance']].append(float(r['tokens_per_sec']))
            n_inst = len(inst_vals)
            per_inst = avg
            agg = per_inst * n_inst
            print(f'  {config:25s}  per_inst={per_inst:6.2f}  agg={agg:6.2f} t/s  ({n_inst} instances)')
        else:
            print(f'  {config:25s}  avg={avg:6.2f} t/s  (n={len(vals)})')
"

echo ""
echo "Results: $RESULTS_FILE"
echo "Logs:    $LOG_DIR"
