#!/bin/bash
# REAP-246B NUMA Concurrency Sweep
# 139 GB model — 2 instances fit in RAM (278 GB total)
# Tests: 1×96t node0 vs 2×96t (both nodes)
set -euo pipefail

LLAMA_SERVER="/mnt/raid0/llm/llama.cpp/build/bin/llama-server"
MODEL="/mnt/raid0/llm/models/Qwen3-Coder-REAP-246B-A35B-Q4_K_M.gguf"
DRAFT="/mnt/raid0/llm/models/Qwen3-Coder-Instruct-DRAFT-0.75B-32k-Q4_0.gguf"

DATA_DIR="/mnt/raid0/llm/epyc-inference-research/data/reap_246b_numa"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/reap_246b_numa_${TIMESTAMP}.csv"
LOG_DIR="${DATA_DIR}/logs_${TIMESTAMP}"

N_PREDICT=256
SPEC_FLAGS="-md $DRAFT --draft-max 32 --draft-p-split 0 --kv-unified"

NODE0="0-47,96-143"
NODE1="48-95,144-191"

PROMPTS=(
    "Write a Python function to implement a binary search tree with insert, delete, and search operations:"
    "Explain the theory of general relativity in detail, covering spacetime curvature:"
    "Implement a concurrent hash map in C++ using fine-grained locking:"
)

mkdir -p "$DATA_DIR" "$LOG_DIR"

echo "REAP-246B NUMA Concurrency Sweep"
echo "================================="
echo "Model: $(basename "$MODEL") (139 GB)"
echo "Config: dm=32, ps=0 (sweep-verified optimal)"
echo ""

echo "config,instance,threads,prompt_idx,tokens_generated,time_ms,tokens_per_sec" > "$RESULTS_FILE"

wait_for_server() {
    local port=$1 max_wait=600 elapsed=0
    while ! curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q '"status":"ok"'; do
        sleep 5; elapsed=$((elapsed + 5))
        if [ $elapsed -ge $max_wait ]; then echo "  ERROR: timeout"; return 1; fi
        if [ $((elapsed % 60)) -eq 0 ]; then echo "  ... loading ($((elapsed/60))m)"; fi
    done
    echo "  Instance on port $port ready (${elapsed}s)"
}

run_completion() {
    local port=$1 prompt="$2" n_predict=$3
    local start_ms end_ms elapsed_ms tokens tps
    start_ms=$(date +%s%N | cut -b1-13)
    local response
    response=$(curl -s --max-time 300 "http://localhost:${port}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"test\",\"messages\":[{\"role\":\"user\",\"content\":$(echo "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().strip()))')}],\"max_tokens\":${n_predict},\"temperature\":0.0,\"stream\":false}" 2>/dev/null)
    end_ms=$(date +%s%N | cut -b1-13)
    elapsed_ms=$((end_ms - start_ms))
    tokens=$(echo "$response" | python3 -c "import json,sys;print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
    if [ "$tokens" -gt 0 ] && [ "$elapsed_ms" -gt 0 ]; then
        tps=$(python3 -c "print(f'{$tokens/($elapsed_ms/1000):.2f}')")
    else tps="0.00"; fi
    echo "${tokens},${elapsed_ms},${tps}"
}

kill_servers() {
    for pid in "$@"; do kill "$pid" 2>/dev/null || true; done
    for pid in "$@"; do wait "$pid" 2>/dev/null || true; done
    sleep 3
}

# ============================================================
# Config A: 1×96t node0 (current production config)
# ============================================================
echo "=== Config A: 1×96t node0 (production) ==="
taskset -c "$NODE0" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port 8196 -ngl 0 --mlock \
    $SPEC_FLAGS > "$LOG_DIR/configA.log" 2>&1 &
PID_A=$!
wait_for_server 8196 || { kill_servers $PID_A; exit 1; }

curl -s "http://localhost:8196/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"t","messages":[{"role":"user","content":"Hi"}],"max_tokens":16,"temperature":0}' > /dev/null 2>&1

for i in "${!PROMPTS[@]}"; do
    result=$(run_completion 8196 "${PROMPTS[$i]}" "$N_PREDICT")
    echo "A_1x96t,node0,96,$i,$result" >> "$RESULTS_FILE"
    echo "  prompt $i: $(echo $result | cut -d, -f3) t/s"
done
kill_servers $PID_A
echo ""

# ============================================================
# Config B: 2×96t (both NUMA nodes, concurrent)
# ============================================================
echo "=== Config B: 2×96t (node0 + node1, concurrent) ==="
taskset -c "$NODE0" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port 8196 -ngl 0 --mlock \
    $SPEC_FLAGS > "$LOG_DIR/configB_n0.log" 2>&1 &
PID_B1=$!
taskset -c "$NODE1" "$LLAMA_SERVER" -m "$MODEL" -t 96 -np 1 --port 8197 -ngl 0 --mlock \
    $SPEC_FLAGS > "$LOG_DIR/configB_n1.log" 2>&1 &
PID_B2=$!

wait_for_server 8196 || { kill_servers $PID_B1 $PID_B2; exit 1; }
wait_for_server 8197 || { kill_servers $PID_B1 $PID_B2; exit 1; }

curl -s "http://localhost:8196/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"t","messages":[{"role":"user","content":"Hi"}],"max_tokens":16,"temperature":0}' > /dev/null 2>&1 &
curl -s "http://localhost:8197/v1/chat/completions" -H "Content-Type: application/json" \
    -d '{"model":"t","messages":[{"role":"user","content":"Hi"}],"max_tokens":16,"temperature":0}' > /dev/null 2>&1 &
wait

echo "  --- sequential ---"
for i in "${!PROMPTS[@]}"; do
    r1=$(run_completion 8196 "${PROMPTS[$i]}" "$N_PREDICT")
    r2=$(run_completion 8197 "${PROMPTS[$i]}" "$N_PREDICT")
    echo "B_2x96t_seq,node0,96,$i,$r1" >> "$RESULTS_FILE"
    echo "B_2x96t_seq,node1,96,$i,$r2" >> "$RESULTS_FILE"
    t1=$(echo $r1 | cut -d, -f3); t2=$(echo $r2 | cut -d, -f3)
    agg=$(python3 -c "print(f'{$t1+$t2:.2f}')")
    echo "  prompt $i: n0=${t1} n1=${t2} agg=${agg} t/s"
done

echo "  --- concurrent ---"
for i in "${!PROMPTS[@]}"; do
    tmpdir=$(mktemp -d -p /mnt/raid0/llm/tmp)
    (run_completion 8196 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r1") &
    P1=$!
    (run_completion 8197 "${PROMPTS[$i]}" "$N_PREDICT" > "$tmpdir/r2") &
    P2=$!
    wait $P1 $P2
    r1=$(cat "$tmpdir/r1"); r2=$(cat "$tmpdir/r2"); rm -rf "$tmpdir"
    echo "B_2x96t_conc,node0,96,$i,$r1" >> "$RESULTS_FILE"
    echo "B_2x96t_conc,node1,96,$i,$r2" >> "$RESULTS_FILE"
    t1=$(echo $r1 | cut -d, -f3); t2=$(echo $r2 | cut -d, -f3)
    agg=$(python3 -c "print(f'{$t1+$t2:.2f}')")
    echo "  concurrent prompt $i: n0=${t1} n1=${t2} agg=${agg} t/s"
done

kill_servers $PID_B1 $PID_B2
echo ""

# ============================================================
echo "=== SUMMARY ==="
python3 - "$RESULTS_FILE" << 'PYEOF'
import csv, sys
from collections import defaultdict
results = defaultdict(list)
with open(sys.argv[1]) as f:
    for row in csv.DictReader(f):
        tps = float(row['tokens_per_sec'])
        if tps > 0: results[row['config']].append(tps)

for config in sorted(results):
    vals = results[config]
    avg = sum(vals)/len(vals)
    if '2x96t' in config:
        aggs = [vals[i]+vals[i+1] for i in range(0,len(vals),2) if i+1<len(vals)]
        avg_agg = sum(aggs)/len(aggs) if aggs else 0
        print(f"{config:<25} per-inst={avg:.2f}  agg={avg_agg:.2f} t/s")
    else:
        print(f"{config:<25} {avg:.2f} t/s")

baseline = sum(results.get('A_1x96t',[0]))/max(len(results.get('A_1x96t',[])),1)
print(f"\n1x96t baseline: {baseline:.2f} t/s")
for config in ['B_2x96t_seq','B_2x96t_conc']:
    vals = results.get(config,[])
    if vals:
        aggs = [vals[i]+vals[i+1] for i in range(0,len(vals),2) if i+1<len(vals)]
        avg_agg = sum(aggs)/len(aggs) if aggs else 0
        print(f"2x96t {config.split('_')[-1]}: {avg_agg:.2f} t/s ({avg_agg/baseline:.2f}x)")
PYEOF
echo ""; echo "Results: $RESULTS_FILE"
