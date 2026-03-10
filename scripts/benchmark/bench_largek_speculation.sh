#!/bin/bash
set -euo pipefail

# SpecExec Phase 3 — Large-K Linear Speculation Test
#
# For each target+draft pair, starts llama-server with --draft-max K
# and runs 20 prompts from question_pool to measure throughput vs K.
#
# Test matrix: 4 pairs × 5 K values = 20 server runs
#
# Output: CSV files in data/specexec/phase3_*.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/specexec"
LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
BASE_PATH="/mnt/raid0/llm/lmstudio/models"
PORT=9090
THREADS=96
N_PROMPTS=20
MAX_TOKENS=512
WARMUP_TIMEOUT=120  # seconds to wait for server startup

mkdir -p "$DATA_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Prompt extraction ──
# Extract prompts from question_pool using Python
PROMPTS_FILE="$DATA_DIR/.phase3_prompts.json"
if [[ ! -f "$PROMPTS_FILE" ]]; then
    log "Extracting $N_PROMPTS prompts from question pool..."
    python3 -c "
import sys, json
sys.path.insert(0, '$SCRIPT_DIR')
from question_pool import load_pool, sample_from_pool
pool = load_pool()
questions = sample_from_pool(pool, suites=['coder', 'thinking'], sample_per_suite=10, seed=42)
prompts = [q['prompt'] for q in questions[:$N_PROMPTS]]
json.dump(prompts, open('$PROMPTS_FILE', 'w'), indent=2)
print(f'Extracted {len(prompts)} prompts')
"
fi

# ── Test pairs ──
declare -A TARGETS=(
    ["Qwen3.5-9B"]="$BASE_PATH/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
    ["Qwen3.5-27B"]="$BASE_PATH/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
    ["Qwen2.5-7B"]="/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf"
    ["Qwen2.5-Coder-32B"]="$BASE_PATH/lmstudio-community/Qwen2.5-Coder-32B-Instruct-GGUF/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf"
)

declare -A DRAFTS=(
    ["Qwen3.5-9B"]="$BASE_PATH/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    ["Qwen3.5-27B"]="$BASE_PATH/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"
    ["Qwen2.5-7B"]="/mnt/raid0/llm/models/Qwen2.5-0.5B-Instruct-f16.gguf"
    ["Qwen2.5-Coder-32B"]="$BASE_PATH/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"
)

K_VALUES=(16 32 64 128 256)

wait_for_server() {
    local port=$1
    local timeout=$2
    local elapsed=0
    while [[ $elapsed -lt $timeout ]]; do
        if curl -sf "http://localhost:$port/health" > /dev/null 2>&1; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
    done
    return 1
}

kill_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        log "Stopping server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    # Ensure port is free
    sleep 1
}

run_prompts() {
    local port=$1
    local outfile=$2

    python3 -c "
import json, time, sys, csv
import requests

prompts = json.load(open('$PROMPTS_FILE'))
port = $port
max_tokens = $MAX_TOKENS

# Warmup
try:
    requests.post(f'http://localhost:{port}/v1/chat/completions',
        json={'model':'test','messages':[{'role':'user','content':'Hello'}],'max_tokens':5,'temperature':0,'stream':False},
        timeout=60)
except Exception as e:
    print(f'Warmup failed: {e}', file=sys.stderr)

results = []
for i, prompt in enumerate(prompts):
    t0 = time.perf_counter()
    try:
        resp = requests.post(f'http://localhost:{port}/v1/chat/completions',
            json={'model':'test','messages':[{'role':'user','content':prompt}],'max_tokens':max_tokens,'temperature':0,'stream':False},
            timeout=300)
        wall = time.perf_counter() - t0
        resp.raise_for_status()
        data = resp.json()
        usage = data.get('usage', {})
        timings = data.get('timings', {})
        tokens = usage.get('completion_tokens', 0)
        speed = timings.get('predicted_per_second', tokens / wall if wall > 0 else 0)
        draft_n = timings.get('draft_n', 0)
        draft_accepted = timings.get('draft_n_accepted', 0)
        accept_rate = (draft_accepted / draft_n * 100) if draft_n > 0 else 0
        results.append({
            'prompt_idx': i, 'tokens': tokens, 'speed_tps': round(speed, 2),
            'wall_s': round(wall, 3), 'draft_n': draft_n, 'draft_accepted': draft_accepted,
            'acceptance_rate': round(accept_rate, 1)
        })
        print(f'  [{i+1}/{len(prompts)}] {tokens} tok, {speed:.1f} t/s, accept={accept_rate:.0f}%')
    except Exception as e:
        print(f'  [{i+1}/{len(prompts)}] ERROR: {e}', file=sys.stderr)
        results.append({'prompt_idx': i, 'tokens': 0, 'speed_tps': 0, 'wall_s': 0,
                        'draft_n': 0, 'draft_accepted': 0, 'acceptance_rate': 0, 'error': str(e)})

# Write CSV
with open('$outfile', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['prompt_idx','tokens','speed_tps','wall_s','draft_n','draft_accepted','acceptance_rate'])
    writer.writeheader()
    for r in results:
        row = {k: r.get(k, '') for k in writer.fieldnames}
        writer.writerow(row)
print(f'Results written to $outfile')

# Summary
if results:
    speeds = [r['speed_tps'] for r in results if r['speed_tps'] > 0]
    accepts = [r['acceptance_rate'] for r in results if r['acceptance_rate'] > 0]
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    avg_accept = sum(accepts) / len(accepts) if accepts else 0
    print(f'  AVG: {avg_speed:.1f} t/s, accept={avg_accept:.0f}%')
"
}

trap kill_server EXIT

# ── Main loop ──

for pair_name in "${!TARGETS[@]}"; do
    target_path="${TARGETS[$pair_name]}"
    draft_path="${DRAFTS[$pair_name]}"

    if [[ ! -f "$target_path" ]]; then
        log "SKIP pair $pair_name — target not found: $target_path"
        continue
    fi
    if [[ ! -f "$draft_path" ]]; then
        log "SKIP pair $pair_name — draft not found: $draft_path"
        continue
    fi

    for k in "${K_VALUES[@]}"; do
        outfile="$DATA_DIR/phase3_${pair_name}_k${k}.csv"
        if [[ -f "$outfile" ]]; then
            log "EXISTS: $outfile — skipping"
            continue
        fi

        kill_server

        log "Starting: $pair_name K=$k"
        log "  Target: $target_path"
        log "  Draft:  $draft_path"

        "$LLAMA_SERVER" \
            -m "$target_path" \
            -md "$draft_path" \
            --draft-max "$k" \
            --port "$PORT" \
            -t "$THREADS" \
            -np 1 \
            --numa distribute \
            -ngl 0 \
            > "$DATA_DIR/phase3_${pair_name}_k${k}_server.log" 2>&1 &
        SERVER_PID=$!

        if ! wait_for_server "$PORT" "$WARMUP_TIMEOUT"; then
            log "ERROR: Server did not start within ${WARMUP_TIMEOUT}s for $pair_name K=$k"
            kill_server
            continue
        fi

        log "Server ready. Running $N_PROMPTS prompts..."
        run_prompts "$PORT" "$outfile"
        log "Done: $outfile"

        kill_server
    done
done

log "=== PHASE 3 COMPLETE ==="
log "Results in $DATA_DIR/phase3_*.csv"

# ── Summary table ──
log "Generating summary..."
python3 -c "
import csv, glob, os
from collections import defaultdict

data_dir = '$DATA_DIR'
files = sorted(glob.glob(os.path.join(data_dir, 'phase3_*_k*.csv')))
if not files:
    print('No Phase 3 results found.')
    exit()

summary = []
for f in files:
    base = os.path.basename(f).replace('phase3_', '').replace('.csv', '')
    # Skip server logs
    if 'server' in base:
        continue
    parts = base.rsplit('_k', 1)
    if len(parts) != 2:
        continue
    pair, k = parts[0], int(parts[1])
    with open(f) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    speeds = [float(r['speed_tps']) for r in rows if float(r.get('speed_tps', 0)) > 0]
    accepts = [float(r['acceptance_rate']) for r in rows if float(r.get('acceptance_rate', 0)) > 0]
    if speeds:
        summary.append((pair, k, sum(speeds)/len(speeds), sum(accepts)/len(accepts) if accepts else 0))

summary.sort()
print(f'{'Pair':<25} {'K':>5} {'Avg t/s':>10} {'Avg Accept%':>12}')
print('-' * 55)
for pair, k, speed, accept in summary:
    print(f'{pair:<25} {k:>5} {speed:>10.1f} {accept:>11.0f}%')
"
