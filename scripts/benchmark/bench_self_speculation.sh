#!/bin/bash
# Self-Speculation Benchmark Script (HSD Phase 2b)
#
# Tests self-speculative decoding with layer-exit depths:
# - Same model as target and draft, draft exits after fewer layers
# - Compares against: no speculation, external draft, prompt lookup
#
# Usage:
#   ./bench_self_speculation.sh                     # full sweep
#   MODEL=9b ./bench_self_speculation.sh            # 9B only
#   MODEL=27b ./bench_self_speculation.sh           # 27B only
#   N_PROMPTS=5 ./bench_self_speculation.sh         # fewer prompts (faster)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# shellcheck source=../lib/env.sh
source "${SCRIPT_DIR}/../lib/env.sh"

# Configuration
LLAMA_BIN="${LLAMA_CPP_BIN}"
THREADS="${THREADS:-96}"
PORT="${PORT:-9090}"
N_PREDICT="${N_PREDICT:-256}"
N_PROMPTS="${N_PROMPTS:-20}"
DRAFT_MAX="${DRAFT_MAX:-16}"
WARMUP_TOKENS="${WARMUP_TOKENS:-32}"

# Output paths
DATA_DIR="${SCRIPT_DIR}/../../data/hsd"
DOCS_DIR="${SCRIPT_DIR}/../../docs/experiments"
mkdir -p "$DATA_DIR" "$DOCS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/self_speculation_${TIMESTAMP}.csv"

# Models — Qwen3.5 dense models
QWEN35_9B="${MODEL_BASE}/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN35_27B="${MODEL_BASE}/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
DRAFT_EXT="${MODEL_BASE}/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

# Test prompts — mix of code and thinking tasks
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

# Benchmark matrix
declare -A MODEL_CONFIGS
# model_key -> "model_path|n_layers|exit_depths"
MODEL_CONFIGS["9b"]="${QWEN35_9B}|32|8,11,16"
MODEL_CONFIGS["27b"]="${QWEN35_27B}|64|16,21,32"

# ─── Helper functions ───

wait_for_server() {
  local port=$1
  local max_wait=${2:-120}
  local elapsed=0
  while ! curl -s "http://localhost:${port}/health" | grep -q '"status":"ok"' 2>/dev/null; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$max_wait" ]; then
      echo "ERROR: Server on port ${port} did not start within ${max_wait}s" >&2
      return 1
    fi
  done
}

kill_server() {
  local port=$1
  local pid
  pid=$(lsof -ti :"$port" 2>/dev/null || true)
  if [ -n "$pid" ]; then
    kill "$pid" 2>/dev/null || true
    sleep 2
    kill -9 "$pid" 2>/dev/null || true
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
    response=$(curl -s "http://localhost:${port}/v1/chat/completions" \
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

    # Extract per-request draft stats from timings (draft_n = generated, draft_n_accepted = accepted)
    local d_acc d_gen
    d_acc=$(echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print(t.get('draft_n_accepted',0))" 2>/dev/null || echo "0")
    d_gen=$(echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print(t.get('draft_n',0))" 2>/dev/null || echo "0")
    total_draft_accepted=$((total_draft_accepted + d_acc))
    total_draft_generated=$((total_draft_generated + d_gen))

    count=$((count + 1))
  done

  # Get aggregate timings from server Prometheus metrics
  local metrics
  metrics=$(curl -s "http://localhost:${port}/metrics" 2>/dev/null || echo "")

  # Parse Prometheus format: "llamacpp:metric_name value"
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

echo "Self-Speculation Benchmark (HSD Phase 2b)"
echo "=========================================="
echo ""
echo "Threads: $THREADS"
echo "N_predict: $N_PREDICT"
echo "N_prompts: $N_PROMPTS"
echo "Draft max: $DRAFT_MAX"
echo "Results: $RESULTS_FILE"
echo ""

# CSV header
echo "model,config,exit_depth,tps,tokens_generated,generation_sec,draft_accepted,draft_total,acceptance_rate,n_prompts" >"$RESULTS_FILE"

MODELS_TO_RUN="${MODEL:-9b 27b}"

for model_key in $MODELS_TO_RUN; do
  if [ -z "${MODEL_CONFIGS[$model_key]+x}" ]; then
    echo "WARN: Unknown model key '$model_key', skipping"
    continue
  fi

  IFS='|' read -r model_path n_layers exit_depths <<< "${MODEL_CONFIGS[$model_key]}"

  if [ ! -f "$model_path" ]; then
    echo "WARN: Model not found: $model_path, skipping $model_key"
    continue
  fi

  echo "=== Qwen3.5-${model_key} (${n_layers} layers) ==="
  echo ""

  # --- Config 1: Baseline (no speculation) ---
  echo -n "  [1/5+] Baseline (no spec): "
  kill_server "$PORT"

  numactl --interleave=all "$LLAMA_BIN/llama-server" \
    -m "$model_path" \
    -t "$THREADS" \
    -c 4096 \
    -np 1 \
    --port "$PORT" \
    --metrics \
    2>/dev/null &

  if ! wait_for_server "$PORT"; then
    echo "FAILED (server didn't start)"
    kill_server "$PORT"
    continue
  fi

  result=$(run_benchmark "$PORT" "$N_PROMPTS" "baseline")
  IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
  echo "${tps} t/s (${n_ok} prompts)"
  echo "${model_key},baseline,0,${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"
  kill_server "$PORT"

  # --- Config 2: External draft (0.8B) ---
  if [ -f "$DRAFT_EXT" ]; then
    echo -n "  [2/5+] External draft (0.8B): "

    numactl --interleave=all "$LLAMA_BIN/llama-server" \
      -m "$model_path" \
      -md "$DRAFT_EXT" \
      --draft-max "$DRAFT_MAX" \
      -t "$THREADS" \
      -c 4096 \
      -np 1 \
      --port "$PORT" \
      --metrics \
      2>/dev/null &

    if ! wait_for_server "$PORT" 180; then
      echo "FAILED (server didn't start)"
      kill_server "$PORT"
    else
      result=$(run_benchmark "$PORT" "$N_PROMPTS" "external_draft")
      IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
      echo "${tps} t/s (accepted: ${d_acc}/${d_tot} = ${a_rate}, ${n_ok} prompts)"
      echo "${model_key},external_draft,0,${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"
      kill_server "$PORT"
    fi
  else
    echo "  [2/5+] External draft: SKIPPED (model not found)"
  fi

  # --- Config 3: Prompt lookup ---
  echo -n "  [3/5+] Prompt lookup: "

  numactl --interleave=all "$LLAMA_BIN/llama-server" \
    -m "$model_path" \
    --lookup \
    --draft-max "$DRAFT_MAX" \
    -t "$THREADS" \
    -c 4096 \
    -np 1 \
    --port "$PORT" \
    --metrics \
    2>/dev/null &

  if ! wait_for_server "$PORT"; then
    echo "FAILED (server didn't start)"
    kill_server "$PORT"
  else
    result=$(run_benchmark "$PORT" "$N_PROMPTS" "prompt_lookup")
    IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
    echo "${tps} t/s (accepted: ${d_acc}/${d_tot} = ${a_rate}, ${n_ok} prompts)"
    echo "${model_key},prompt_lookup,0,${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"
    kill_server "$PORT"
  fi

  # --- Config 4+: Self-speculation at each exit depth ---
  IFS=',' read -ra depths <<< "$exit_depths"
  config_num=4
  for depth in "${depths[@]}"; do
    echo -n "  [${config_num}/5+] Self-spec (exit=${depth}/${n_layers}): "

    numactl --interleave=all "$LLAMA_BIN/llama-server" \
      -m "$model_path" \
      -md "$model_path" \
      --n-layer-exit-draft "$depth" \
      --draft-max "$DRAFT_MAX" \
      -t "$THREADS" \
      -c 4096 \
      -np 1 \
      --port "$PORT" \
      --metrics \
      2>/dev/null &

    if ! wait_for_server "$PORT" 180; then
      echo "FAILED (server didn't start)"
      kill_server "$PORT"
      config_num=$((config_num + 1))
      continue
    fi

    result=$(run_benchmark "$PORT" "$N_PROMPTS" "self_spec_${depth}")
    IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
    echo "${tps} t/s (accepted: ${d_acc}/${d_tot} = ${a_rate}, ${n_ok} prompts)"
    echo "${model_key},self_speculation,${depth},${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"
    kill_server "$PORT"
    config_num=$((config_num + 1))
  done

  echo ""
done

echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
column -t -s, "$RESULTS_FILE"

# Generate markdown summary
cat >"${DOCS_DIR}/self-speculation-benchmark.md" <<DOCEOF
# Self-Speculation Benchmark Results

**Date**: $(date +%Y-%m-%d)
**Branch**: production-consolidated-v2 (HSD Phase 2b)

## Configuration

- Threads: $THREADS
- Tokens predicted: $N_PREDICT per prompt
- Prompts: $N_PROMPTS (mix of code + thinking)
- Draft max: $DRAFT_MAX

## Results

\`\`\`
$(column -t -s, "$RESULTS_FILE")
\`\`\`

## Raw data

\`data/hsd/self_speculation_${TIMESTAMP}.csv\`
DOCEOF

echo "Docs saved to: ${DOCS_DIR}/self-speculation-benchmark.md"
