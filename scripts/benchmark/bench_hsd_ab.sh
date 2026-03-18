#!/bin/bash
# HSD A/B Benchmark — Measure marginal contribution of capped branch resampling
#
# Runs external draft on dense Qwen3-32B with and without --no-hsd,
# comparing draft_n_accepted counts and throughput.
#
# Runs sequentially (never concurrent) to avoid resource contention.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/env.sh"

# Configuration
LLAMA_BIN="${LLAMA_CPP_BIN}"
THREADS="${THREADS:-96}"
PORT="${PORT:-9090}"
N_PREDICT="${N_PREDICT:-256}"
N_PROMPTS="${N_PROMPTS:-20}"
DRAFT_MAX="${DRAFT_MAX:-16}"

# Models
TARGET_MODEL="${MODEL_BASE}/lmstudio-community/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
DRAFT_MODEL="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Output
DATA_DIR="${SCRIPT_DIR}/../../data/hsd"
mkdir -p "$DATA_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/hsd_ab_${TIMESTAMP}.csv"

# Prompts — same set as bench_hispec_external.sh for comparability
PROMPTS=(
  "Write a Python function to implement quicksort with detailed comments explaining each step."
  "Implement a concurrent hashmap in Python using fine-grained locking."
  "Write a Redis-like in-memory key-value store with TTL support in Python."
  "Implement the Raft consensus algorithm's leader election in Python."
  "Write a Python async web scraper that respects robots.txt and rate limits."
  "Implement a B-tree with insert and search operations in Python."
  "Write a compiler for a simple arithmetic expression language targeting a stack machine."
  "Implement a Python decorator that adds memoization with LRU eviction and TTL."
  "Write a database query optimizer that converts SQL WHERE clauses to an optimal index scan plan."
  "Implement a Python type checker for a simple type system with generics."
  "What are the trade-offs between optimistic and pessimistic concurrency control?"
  "Explain how modern CPUs use branch prediction and speculative execution."
  "Describe the CAP theorem and how different distributed databases handle partition tolerance."
  "Compare B-trees and LSM-trees for database storage engines."
  "Explain how garbage collectors work: mark-sweep vs generational vs concurrent."
  "What is the difference between cooperative and preemptive multitasking?"
  "Describe how TLS 1.3 handshake works step by step."
  "Explain the theory behind consistent hashing and virtual nodes."
  "How do modern JIT compilers like V8 optimize hot loops?"
  "Describe the memory hierarchy and how cache lines affect performance."
)

wait_for_server() {
  local port=$1
  local max_wait=${2:-300}
  local elapsed=0
  while ! curl -s "http://localhost:${port}/health" | grep -q '"status":"ok"' 2>/dev/null; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "ERROR: Server on port ${port} did not start within ${max_wait}s" >&2
      return 1
    fi
  done
  echo "  Server ready (${elapsed}s)"
}

kill_server() {
  local port=$1
  local pid
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
    sleep 1
  fi
}

run_benchmark() {
  local port=$1
  local n_prompts=$2
  local label=$3

  local count=0
  local total_draft_accepted=0
  local total_draft_generated=0

  for ((p=0; p < n_prompts && p < ${#PROMPTS[@]}; p++)); do
    local prompt="${PROMPTS[$p]}"
    local response
    response=$(curl -s --max-time 120 "http://localhost:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"test\",
        \"messages\": [{\"role\": \"user\", \"content\": $(printf '%s' "$prompt" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}],
        \"max_tokens\": ${N_PREDICT},
        \"temperature\": 0.7
      }" 2>/dev/null)

    if [ -z "$response" ] || echo "$response" | grep -q '"error"'; then
      echo "    WARN: prompt $p failed for $label" >&2
      continue
    fi

    local d_acc d_gen
    d_acc=$(echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print(t.get('draft_n_accepted',0))" 2>/dev/null || echo "0")
    d_gen=$(echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print(t.get('draft_n',0))" 2>/dev/null || echo "0")
    total_draft_accepted=$((total_draft_accepted + d_acc))
    total_draft_generated=$((total_draft_generated + d_gen))
    count=$((count + 1))
    echo "    prompt $((p+1))/${n_prompts}: accepted=${d_acc}/${d_gen}" >&2
  done

  # Aggregate metrics
  local metrics
  metrics=$(curl -s "http://localhost:${port}/metrics" 2>/dev/null || echo "")

  local gen_tokens gen_seconds tps
  gen_tokens=$(echo "$metrics" | awk '/^llamacpp:tokens_predicted_total / {print $2}' || echo "0")
  gen_seconds=$(echo "$metrics" | awk '/^llamacpp:tokens_predicted_seconds_total / {print $2}' || echo "0")

  tps="0"
  if [ -n "$gen_seconds" ] && [ "$gen_seconds" != "0" ]; then
    tps=$(echo "scale=2; $gen_tokens / $gen_seconds" | bc 2>/dev/null || echo "0")
  fi

  local accept_rate="0"
  if [ "$total_draft_generated" -gt 0 ] 2>/dev/null; then
    accept_rate=$(echo "scale=4; $total_draft_accepted / $total_draft_generated" | bc 2>/dev/null || echo "0")
  fi

  echo "${tps}|${gen_tokens}|${gen_seconds}|${total_draft_accepted}|${total_draft_generated}|${accept_rate}|${count}"
}

# ─── Main ───

echo "HSD A/B Benchmark — Capped Branch Resampling"
echo "=============================================="
echo ""
echo "Target: Qwen3-32B (dense, 64 layers)"
echo "Draft:  Qwen2.5-Coder-0.5B-Q8_0"
echo "Threads: $THREADS | N_predict: $N_PREDICT | N_prompts: $N_PROMPTS | Draft_max: $DRAFT_MAX"
echo "Results: $RESULTS_FILE"
echo ""

if [ ! -f "$TARGET_MODEL" ]; then
  echo "ERROR: Target model not found: $TARGET_MODEL" >&2
  exit 1
fi
if [ ! -f "$DRAFT_MODEL" ]; then
  echo "ERROR: Draft model not found: $DRAFT_MODEL" >&2
  exit 1
fi

# CSV header
echo "config,tps,tokens_generated,generation_sec,draft_accepted,draft_total,acceptance_rate,n_prompts" >"$RESULTS_FILE"

# ─── Run 1: WITH HSD (default) ───

echo "=== Run 1/2: WITH HSD (default) ==="
kill_server "$PORT"
numactl --interleave=all "$LLAMA_BIN/llama-server" \
  -m "$TARGET_MODEL" \
  -md "$DRAFT_MODEL" \
  --draft-max "$DRAFT_MAX" \
  -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics \
  2>/dev/null &

if ! wait_for_server "$PORT" 300; then
  echo "FAILED: server didn't start"
  kill_server "$PORT"
  exit 1
fi

result=$(run_benchmark "$PORT" "$N_PROMPTS" "with_hsd")
IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
echo ""
echo "  WITH HSD: ${tps} t/s | accepted: ${d_acc}/${d_tot} = ${a_rate} | ${n_ok} prompts"
echo "with_hsd,${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"

kill_server "$PORT"
echo ""
echo "  Waiting 10s between runs for clean state..."
sleep 10

# ─── Run 2: WITHOUT HSD (--no-hsd) ───

echo "=== Run 2/2: WITHOUT HSD (--no-hsd) ==="
kill_server "$PORT"
numactl --interleave=all "$LLAMA_BIN/llama-server" \
  -m "$TARGET_MODEL" \
  -md "$DRAFT_MODEL" \
  --draft-max "$DRAFT_MAX" \
  --no-hsd \
  -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics \
  2>/dev/null &

if ! wait_for_server "$PORT" 300; then
  echo "FAILED: server didn't start"
  kill_server "$PORT"
  exit 1
fi

result=$(run_benchmark "$PORT" "$N_PROMPTS" "without_hsd")
IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
echo ""
echo "  WITHOUT HSD: ${tps} t/s | accepted: ${d_acc}/${d_tot} = ${a_rate} | ${n_ok} prompts"
echo "without_hsd,${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"

kill_server "$PORT"

# ─── Summary ───

echo ""
echo "=========================================="
echo "Results:"
echo ""
column -t -s, "$RESULTS_FILE"
echo ""
echo "CSV: $RESULTS_FILE"
