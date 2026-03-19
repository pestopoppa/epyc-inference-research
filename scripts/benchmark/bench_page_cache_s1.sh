#!/bin/bash
# S1: Baseline Page Residency Measurement
#
# Loads 3 production models sequentially (480B, 235B, 30B frontdoor),
# measures page residency, cold vs warm latency, and page fault rates.

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/page_cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$DATA_DIR/s1_residency_${TIMESTAMP}.txt"

# Model configs (no associative arrays — bash chokes on "480B" as key)
M1_LABEL="480B-A35B"
M1_PATH="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/Qwen3-Coder-480B-A35B-Instruct-Q4_K_M-00001-of-00008.gguf"
M1_DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
M1_PORT=8089
M1_THREADS=96
M1_CPUS="0-47,96-143"
M1_EXTRA="--draft-max 48 --kv-unified"
M1_PID=""

M2_LABEL="235B-A22B"
M2_PATH="/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/Qwen3-235B-A22B-Q4_K_M-00001-of-00004.gguf"
M2_DRAFTER="/mnt/raid0/llm/models/Qwen_Qwen3-0.6B-Q8_0.gguf"
M2_PORT=8088
M2_THREADS=96
M2_CPUS="48-95,144-191"
M2_EXTRA="--draft-max 32 --kv-unified"
M2_PID=""

M3_LABEL="30B-A3B"
M3_PATH="/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
M3_DRAFTER="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"
M3_PORT=8080
M3_THREADS=48
M3_CPUS="0-23,96-119"
M3_EXTRA="--draft-max 32 --kv-unified"
M3_PID=""

log() { echo "$@" | tee -a "$RESULTS"; }

wait_for_server() {
    local port=$1 max_wait=600 elapsed=0
    while true; do
        local h; h=$(curl -s "http://localhost:${port}/health" 2>/dev/null || echo "")
        echo "$h" | grep -q '"status":"ok"' && return 0
        sleep 3; elapsed=$((elapsed + 3))
        [ $elapsed -ge $max_wait ] && return 1
    done
}

measure_rss() {
    local pid=$1 label=$2
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
        log "  $label: not running"
        return
    fi
    local rss_kb=$(grep "VmRSS:" /proc/$pid/status 2>/dev/null | awk '{print $2}')
    local rss_gb=$(python3 -c "print(f'{${rss_kb:-0}/1024/1024:.2f}')")
    log "  $label (PID $pid): RSS = ${rss_gb} GB"
}

measure_latency() {
    local port=$1 label=$2 tag=$3
    local start_ms=$(date +%s%N | cut -b1-13)
    curl -s --max-time 300 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Write hello world in Python"}],"max_tokens":32,"temperature":0.0,"stream":false}' \
        > /dev/null 2>&1
    local end_ms=$(date +%s%N | cut -b1-13)
    local elapsed_ms=$((end_ms - start_ms))
    log "  $label $tag: ${elapsed_ms} ms"
}

mkdir -p "$DATA_DIR"

log "S1: Baseline Page Residency Measurement"
log "========================================"
log "$(date)"
log ""

# ============================================================
# Try to drop caches for clean measurement
# ============================================================
log "=== Attempting to drop page cache ==="
sync
echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null 2>&1 && log "  Page cache dropped" || log "  WARNING: Cannot drop caches (not root). Measuring with existing cache."
free -h | tee -a "$RESULTS"
log ""

# ============================================================
# Load models sequentially
# ============================================================
log "=== Loading Model 1: $M1_LABEL (port $M1_PORT) ==="
taskset -c "$M1_CPUS" "$LLAMA_SERVER" \
    -m "$M1_PATH" -md "$M1_DRAFTER" $M1_EXTRA \
    -t "$M1_THREADS" -np 1 --port "$M1_PORT" -ngl 0 \
    > "$DATA_DIR/s1_${M1_LABEL}.log" 2>&1 &
M1_PID=$!
log "  PID: $M1_PID"
wait_for_server "$M1_PORT" && log "  Ready" || log "  FAILED"
log ""

log "--- Residency after $M1_LABEL loaded ---"
measure_rss "$M1_PID" "$M1_LABEL"
free -h | grep Mem | tee -a "$RESULTS"
log ""

log "=== Loading Model 2: $M2_LABEL (port $M2_PORT) ==="
taskset -c "$M2_CPUS" "$LLAMA_SERVER" \
    -m "$M2_PATH" -md "$M2_DRAFTER" $M2_EXTRA \
    -t "$M2_THREADS" -np 1 --port "$M2_PORT" -ngl 0 \
    > "$DATA_DIR/s1_${M2_LABEL}.log" 2>&1 &
M2_PID=$!
log "  PID: $M2_PID"
wait_for_server "$M2_PORT" && log "  Ready" || log "  FAILED"
log ""

log "--- Residency after $M2_LABEL loaded ---"
measure_rss "$M1_PID" "$M1_LABEL"
measure_rss "$M2_PID" "$M2_LABEL"
free -h | grep Mem | tee -a "$RESULTS"
log ""

log "=== Loading Model 3: $M3_LABEL (port $M3_PORT) ==="
taskset -c "$M3_CPUS" "$LLAMA_SERVER" \
    -m "$M3_PATH" -md "$M3_DRAFTER" $M3_EXTRA \
    -t "$M3_THREADS" -np 1 --port "$M3_PORT" -ngl 0 \
    > "$DATA_DIR/s1_${M3_LABEL}.log" 2>&1 &
M3_PID=$!
log "  PID: $M3_PID"
wait_for_server "$M3_PORT" && log "  Ready" || log "  FAILED"
log ""

log "--- Residency after all models loaded ---"
measure_rss "$M1_PID" "$M1_LABEL"
measure_rss "$M2_PID" "$M2_LABEL"
measure_rss "$M3_PID" "$M3_LABEL"
free -h | grep Mem | tee -a "$RESULTS"
log ""

# ============================================================
# Cold vs warm latency
# ============================================================
log "=== Cold Request Latency (first request after all loaded) ==="
for port_label in "$M1_PORT:$M1_LABEL" "$M2_PORT:$M2_LABEL" "$M3_PORT:$M3_LABEL"; do
    port=${port_label%%:*}; label=${port_label##*:}
    measure_latency "$port" "$label" "cold"
done
log ""

log "=== Warm Request Latency (second request) ==="
for port_label in "$M1_PORT:$M1_LABEL" "$M2_PORT:$M2_LABEL" "$M3_PORT:$M3_LABEL"; do
    port=${port_label%%:*}; label=${port_label##*:}
    measure_latency "$port" "$label" "warm"
done
log ""

log "=== Third Request (confirming warm) ==="
for port_label in "$M1_PORT:$M1_LABEL" "$M2_PORT:$M2_LABEL" "$M3_PORT:$M3_LABEL"; do
    port=${port_label%%:*}; label=${port_label##*:}
    measure_latency "$port" "$label" "warm2"
done
log ""

# ============================================================
# Page fault measurement
# ============================================================
log "=== Page Faults (10-request burst on frontdoor $M3_LABEL) ==="
if kill -0 "$M3_PID" 2>/dev/null; then
    perf stat -e page-faults,major-faults,minor-faults -p "$M3_PID" -- sleep 15 2>"$DATA_DIR/s1_perf_faults.txt" &
    PERF_PID=$!

    for i in $(seq 1 10); do
        curl -s "http://localhost:${M3_PORT}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d '{"model":"test","messages":[{"role":"user","content":"Explain binary search"}],"max_tokens":64,"temperature":0}' \
            > /dev/null 2>&1
    done

    wait $PERF_PID 2>/dev/null
    cat "$DATA_DIR/s1_perf_faults.txt" | tee -a "$RESULTS"
else
    log "  Frontdoor not running, skipping"
fi
log ""

# ============================================================
# Final residency
# ============================================================
log "=== Final Residency ==="
measure_rss "$M1_PID" "$M1_LABEL"
measure_rss "$M2_PID" "$M2_LABEL"
measure_rss "$M3_PID" "$M3_LABEL"
free -h | tee -a "$RESULTS"
log ""

# ============================================================
# Cleanup
# ============================================================
log "=== Cleanup ==="
kill "$M1_PID" "$M2_PID" "$M3_PID" 2>/dev/null || true
wait "$M1_PID" "$M2_PID" "$M3_PID" 2>/dev/null || true

log "Done. Results: $RESULTS"
