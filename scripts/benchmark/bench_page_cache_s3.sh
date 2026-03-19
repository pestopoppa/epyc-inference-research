#!/bin/bash
# S3: Page-In Verification Before Serving
#
# After loading all models, explicitly touch every page of each model's mmap'd file
# to ensure all pages are resident. Then re-measure cold-start latency.
#
# This is the same S1 setup but with a page-in step between loading and serving.

set -u

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/page_cache"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS="$DATA_DIR/s3_pagein_${TIMESTAMP}.txt"

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
        log "  $label: not running"; return
    fi
    local rss_kb=$(grep "VmRSS:" /proc/$pid/status 2>/dev/null | awk '{print $2}')
    log "  $label (PID $pid): RSS = $(python3 -c "print(f'{${rss_kb:-0}/1024/1024:.2f}')") GB"
}

measure_latency() {
    local port=$1 label=$2 tag=$3
    local start_ms=$(date +%s%N | cut -b1-13)
    curl -s --max-time 300 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"test","messages":[{"role":"user","content":"Write hello world in Python"}],"max_tokens":32,"temperature":0.0,"stream":false}' \
        > /dev/null 2>&1
    local end_ms=$(date +%s%N | cut -b1-13)
    log "  $label $tag: $((end_ms - start_ms)) ms"
}

page_in_model() {
    local model_path=$1 label=$2
    log "  Paging in $label: $(python3 -c "import os; print(f'{os.path.getsize(\"$model_path\")/1024/1024/1024:.1f}')") GB..."

    # Find all GGUF files for multi-part models (same directory, same prefix)
    local dir=$(dirname "$model_path")
    local base=$(basename "$model_path")
    # For split models: match pattern like *-00001-of-00008.gguf → *-*-of-*.gguf
    local prefix=${base%-*-of-*}

    local total_pages=0
    local start_s=$(date +%s)

    for gguf_file in "$dir"/${prefix}*.gguf; do
        if [ -f "$gguf_file" ]; then
            local pages=$(python3 -c "
import mmap, os, time
path = '$gguf_file'
fd = os.open(path, os.O_RDONLY)
size = os.fstat(fd).st_size
m = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
total = 0
for i in range(0, len(m), 4096):
    total += m[i]
m.close()
os.close(fd)
print(size // 4096)
" 2>/dev/null)
            total_pages=$((total_pages + pages))
            log "    $(basename "$gguf_file"): ${pages} pages touched"
        fi
    done

    local elapsed=$(($(date +%s) - start_s))
    log "  $label: ${total_pages} total pages in ${elapsed}s"
}

mkdir -p "$DATA_DIR"

log "S3: Page-In Verification Before Serving"
log "========================================="
log "$(date)"
log ""

# ============================================================
# Load all models (same as S1)
# ============================================================
log "=== Loading all models ==="

taskset -c "$M1_CPUS" "$LLAMA_SERVER" -m "$M1_PATH" -md "$M1_DRAFTER" $M1_EXTRA \
    -t "$M1_THREADS" -np 1 --port "$M1_PORT" -ngl 0 > "$DATA_DIR/s3_${M1_LABEL}.log" 2>&1 &
M1_PID=$!
log "  $M1_LABEL PID=$M1_PID"

taskset -c "$M2_CPUS" "$LLAMA_SERVER" -m "$M2_PATH" -md "$M2_DRAFTER" $M2_EXTRA \
    -t "$M2_THREADS" -np 1 --port "$M2_PORT" -ngl 0 > "$DATA_DIR/s3_${M2_LABEL}.log" 2>&1 &
M2_PID=$!
log "  $M2_LABEL PID=$M2_PID"

taskset -c "$M3_CPUS" "$LLAMA_SERVER" -m "$M3_PATH" -md "$M3_DRAFTER" $M3_EXTRA \
    -t "$M3_THREADS" -np 1 --port "$M3_PORT" -ngl 0 > "$DATA_DIR/s3_${M3_LABEL}.log" 2>&1 &
M3_PID=$!
log "  $M3_LABEL PID=$M3_PID"

log "  Waiting for all servers..."
wait_for_server "$M1_PORT" && log "  $M1_LABEL ready" || log "  $M1_LABEL FAILED"
wait_for_server "$M2_PORT" && log "  $M2_LABEL ready" || log "  $M2_LABEL FAILED"
wait_for_server "$M3_PORT" && log "  $M3_LABEL ready" || log "  $M3_LABEL FAILED"
log ""

# ============================================================
# Residency BEFORE page-in
# ============================================================
log "=== Residency BEFORE page-in ==="
measure_rss "$M1_PID" "$M1_LABEL"
measure_rss "$M2_PID" "$M2_LABEL"
measure_rss "$M3_PID" "$M3_LABEL"
free -h | grep Mem | tee -a "$RESULTS"
log ""

# ============================================================
# Page-in: touch every page of each model
# ============================================================
log "=== Page-In Verification ==="
page_in_model "$M1_PATH" "$M1_LABEL"
page_in_model "$M2_PATH" "$M2_LABEL"
page_in_model "$M3_PATH" "$M3_LABEL"
log ""

# ============================================================
# Residency AFTER page-in
# ============================================================
log "=== Residency AFTER page-in ==="
measure_rss "$M1_PID" "$M1_LABEL"
measure_rss "$M2_PID" "$M2_LABEL"
measure_rss "$M3_PID" "$M3_LABEL"
free -h | grep Mem | tee -a "$RESULTS"
log ""

# ============================================================
# Latency measurements (should be warm now!)
# ============================================================
log "=== First Request After Page-In (should be warm) ==="
for port_label in "$M1_PORT:$M1_LABEL" "$M2_PORT:$M2_LABEL" "$M3_PORT:$M3_LABEL"; do
    port=${port_label%%:*}; label=${port_label##*:}
    measure_latency "$port" "$label" "post-pagein-1"
done
log ""

log "=== Second Request After Page-In ==="
for port_label in "$M1_PORT:$M1_LABEL" "$M2_PORT:$M2_LABEL" "$M3_PORT:$M3_LABEL"; do
    port=${port_label%%:*}; label=${port_label##*:}
    measure_latency "$port" "$label" "post-pagein-2"
done
log ""

# ============================================================
# Summary comparison with S1
# ============================================================
log "=== COMPARISON: S1 (no page-in) vs S3 (with page-in) ==="
log "S1 cold latencies: 480B=185271ms, 235B=7276ms, 30B=7897ms"
log "S3 post-page-in latencies: see above"
log ""

# ============================================================
# Cleanup
# ============================================================
log "=== Cleanup ==="
kill "$M1_PID" "$M2_PID" "$M3_PID" 2>/dev/null || true
wait "$M1_PID" "$M2_PID" "$M3_PID" 2>/dev/null || true
log "Done. Results: $RESULTS"
