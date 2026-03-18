#!/bin/bash
# HiSpec + External Draft Benchmark Script
#
# Tests hierarchical speculative decoding with external drafter:
# - Baseline (no speculation)
# - External draft (standard spec decode)
# - HiSpec + external draft (intermediate verify at N/4 layers)
# - HiSpec + external draft (intermediate verify at N/2 layers)
#
# Also retests SSM hybrid model spec decode to validate checkpoint optimization.
#
# Usage:
#   ./bench_hispec_external.sh                     # full sweep
#   TARGET=dense ./bench_hispec_external.sh        # dense model only
#   TARGET=ssm ./bench_hispec_external.sh          # SSM model only
#   N_PROMPTS=5 ./bench_hispec_external.sh         # fewer prompts (faster)

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

# Output paths
DATA_DIR="${SCRIPT_DIR}/../../data/hsd"
DOCS_DIR="${SCRIPT_DIR}/../../docs/experiments"
mkdir -p "$DATA_DIR" "$DOCS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="${DATA_DIR}/hispec_external_${TIMESTAMP}.csv"

# ─── Models ───

# Dense target: Qwen3-32B (64 layers, pure attention)
QWEN3_32B="${MODEL_BASE}/lmstudio-community/Qwen3-32B-GGUF/Qwen3-32B-Q4_K_M.gguf"
QWEN3_32B_LAYERS=64

# SSM hybrid target: Qwen3.5-9B (32 layers, Mamba2 + attention)
QWEN35_9B="${MODEL_BASE}/unsloth/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf"
QWEN35_9B_LAYERS=32

# External drafters
DRAFT_QWEN3_06B="${MODEL_BASE}/unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q2_K.gguf"
DRAFT_QWEN25_CODER="${MODEL_BASE}/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"
DRAFT_QWEN35_08B="${MODEL_BASE}/unsloth/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf"

# Test prompts — mix of code and reasoning tasks
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

# ─── Helper functions ───

wait_for_server() {
  local port=$1
  local max_wait=${2:-180}
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

    local d_acc d_gen
    d_acc=$(echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print(t.get('draft_n_accepted',0))" 2>/dev/null || echo "0")
    d_gen=$(echo "$response" | python3 -c "import json,sys; r=json.load(sys.stdin); t=r.get('timings',{}); print(t.get('draft_n',0))" 2>/dev/null || echo "0")
    total_draft_accepted=$((total_draft_accepted + d_acc))
    total_draft_generated=$((total_draft_generated + d_gen))

    count=$((count + 1))
  done

  # Get aggregate timings from Prometheus metrics
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

start_server() {
  local label=$1
  shift
  echo -n "  ${label}: "
  kill_server "$PORT"
  numactl --interleave=all "$@" 2>/dev/null &
}

run_and_record() {
  local model_key=$1
  local config=$2
  local hispec_depth=$3

  if ! wait_for_server "$PORT" 300; then
    echo "FAILED (server didn't start)"
    kill_server "$PORT"
    return 1
  fi

  local result
  result=$(run_benchmark "$PORT" "$N_PROMPTS" "$config")
  IFS='|' read -r tps gen_tok gen_sec d_acc d_tot a_rate n_ok <<< "$result"
  echo "${tps} t/s (accepted: ${d_acc}/${d_tot} = ${a_rate}, ${n_ok} prompts)"
  echo "${model_key},${config},${hispec_depth},${tps},${gen_tok},${gen_sec},${d_acc},${d_tot},${a_rate},${n_ok}" >>"$RESULTS_FILE"
  kill_server "$PORT"
}

# ─── Main ───

echo "HiSpec + External Draft Benchmark"
echo "=================================="
echo ""
echo "Threads: $THREADS"
echo "N_predict: $N_PREDICT"
echo "N_prompts: $N_PROMPTS"
echo "Draft max: $DRAFT_MAX"
echo "Results: $RESULTS_FILE"
echo ""

# CSV header
echo "model,config,hispec_depth,tps,tokens_generated,generation_sec,draft_accepted,draft_total,acceptance_rate,n_prompts" >"$RESULTS_FILE"

TARGETS="${TARGET:-dense ssm}"

# ─── Dense model: Qwen3-32B ───

if echo "$TARGETS" | grep -q "dense"; then
  if [ ! -f "$QWEN3_32B" ]; then
    echo "WARN: Qwen3-32B not found at $QWEN3_32B, skipping dense tests"
  else
    echo "=== Qwen3-32B (dense, ${QWEN3_32B_LAYERS} layers) ==="
    echo ""

    # --- Baseline ---
    start_server "[1/5] Baseline" \
      "$LLAMA_BIN/llama-server" \
      -m "$QWEN3_32B" \
      -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
    run_and_record "qwen3-32b" "baseline" "0" || true

    # --- External draft (Qwen2.5-Coder-0.5B — fastest drafter from profiling) ---
    if [ -f "$DRAFT_QWEN25_CODER" ]; then
      start_server "[2/5] External draft (Qwen2.5-Coder-0.5B)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN3_32B" \
        -md "$DRAFT_QWEN25_CODER" \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen3-32b" "external_coder05b" "0" || true
    fi

    # --- External draft (Qwen3-0.6B — architecture-matched) ---
    if [ -f "$DRAFT_QWEN3_06B" ]; then
      start_server "[3/5] External draft (Qwen3-0.6B)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN3_32B" \
        -md "$DRAFT_QWEN3_06B" \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen3-32b" "external_qwen3_06b" "0" || true
    fi

    # --- HiSpec + external draft (intermediate at N/4 = 16 layers) ---
    if [ -f "$DRAFT_QWEN25_CODER" ]; then
      start_server "[4/5] HiSpec external (intermediate=${QWEN3_32B_LAYERS}/4=16)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN3_32B" \
        -md "$DRAFT_QWEN25_CODER" \
        --hierarchical-spec \
        --n-layer-exit-intermediate 16 \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen3-32b" "hispec_external_n4" "16" || true
    fi

    # --- HiSpec + external draft (intermediate at N/2 = 32 layers) ---
    if [ -f "$DRAFT_QWEN25_CODER" ]; then
      start_server "[5/5] HiSpec external (intermediate=${QWEN3_32B_LAYERS}/2=32)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN3_32B" \
        -md "$DRAFT_QWEN25_CODER" \
        --hierarchical-spec \
        --n-layer-exit-intermediate 32 \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen3-32b" "hispec_external_n2" "32" || true
    fi

    echo ""
  fi
fi

# ─── SSM hybrid model: Qwen3.5-9B (checkpoint optimization validation) ───

if echo "$TARGETS" | grep -q "ssm"; then
  if [ ! -f "$QWEN35_9B" ]; then
    echo "WARN: Qwen3.5-9B not found at $QWEN35_9B, skipping SSM tests"
  else
    echo "=== Qwen3.5-9B (SSM hybrid, ${QWEN35_9B_LAYERS} layers) — checkpoint optimization validation ==="
    echo ""

    # --- Baseline ---
    start_server "[1/4] Baseline" \
      "$LLAMA_BIN/llama-server" \
      -m "$QWEN35_9B" \
      -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
    run_and_record "qwen35-9b-ssm" "baseline" "0" || true

    # --- External draft (Qwen3.5-0.8B — same hybrid architecture) ---
    if [ -f "$DRAFT_QWEN35_08B" ]; then
      start_server "[2/4] External draft (Qwen3.5-0.8B)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN35_9B" \
        -md "$DRAFT_QWEN35_08B" \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen35-9b-ssm" "external_qwen35_08b" "0" || true
    fi

    # --- Self-speculation (exit=8/32 — best from previous benchmark) ---
    start_server "[3/4] Self-spec (exit=8/${QWEN35_9B_LAYERS})" \
      "$LLAMA_BIN/llama-server" \
      -m "$QWEN35_9B" \
      -md "$QWEN35_9B" \
      --n-layer-exit-draft 8 \
      --draft-max "$DRAFT_MAX" \
      -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
    run_and_record "qwen35-9b-ssm" "self_spec_exit8" "8" || true

    # --- External draft (Qwen2.5-Coder-0.5B — fastest drafter, cross-arch) ---
    if [ -f "$DRAFT_QWEN25_CODER" ]; then
      start_server "[4/7] External draft (Qwen2.5-Coder-0.5B)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN35_9B" \
        -md "$DRAFT_QWEN25_CODER" \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen35-9b-ssm" "external_coder05b" "0" || true
    fi

    # --- Freeze-recurrent + external draft (Qwen3.5-0.8B) ---
    if [ -f "$DRAFT_QWEN35_08B" ]; then
      start_server "[5/7] Freeze-recurrent + external (Qwen3.5-0.8B)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN35_9B" \
        -md "$DRAFT_QWEN35_08B" \
        --freeze-recurrent-draft \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen35-9b-ssm" "freeze_ext_qwen35_08b" "0" || true
    fi

    # --- Freeze-recurrent + external draft (Qwen2.5-Coder-0.5B) ---
    if [ -f "$DRAFT_QWEN25_CODER" ]; then
      start_server "[6/7] Freeze-recurrent + external (Qwen2.5-Coder-0.5B)" \
        "$LLAMA_BIN/llama-server" \
        -m "$QWEN35_9B" \
        -md "$DRAFT_QWEN25_CODER" \
        --freeze-recurrent-draft \
        --draft-max "$DRAFT_MAX" \
        -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
      run_and_record "qwen35-9b-ssm" "freeze_ext_coder05b" "0" || true
    fi

    # --- Freeze-recurrent + self-spec (exit=8) ---
    start_server "[7/7] Freeze-recurrent + self-spec (exit=8/${QWEN35_9B_LAYERS})" \
      "$LLAMA_BIN/llama-server" \
      -m "$QWEN35_9B" \
      -md "$QWEN35_9B" \
      --n-layer-exit-draft 8 \
      --freeze-recurrent-draft \
      --draft-max "$DRAFT_MAX" \
      -t "$THREADS" -c 4096 -np 1 --port "$PORT" --metrics
    run_and_record "qwen35-9b-ssm" "freeze_self_spec_exit8" "8" || true

    echo ""
  fi
fi

echo "Results saved to: $RESULTS_FILE"
echo ""
echo "Summary:"
column -t -s, "$RESULTS_FILE"

# Generate markdown summary
cat >"${DOCS_DIR}/hispec-external-draft-benchmark.md" <<DOCEOF
# HiSpec + External Draft Benchmark Results

**Date**: $(date +%Y-%m-%d)
**Branch**: feature/ssm-checkpoint-opt (from production-consolidated-v2)
**Optimization**: Double-buffer pointer swap for SSM checkpoint/restore

## Configuration

- Threads: $THREADS
- Tokens predicted: $N_PREDICT per prompt
- Prompts: $N_PROMPTS (mix of code + reasoning)
- Draft max: $DRAFT_MAX

## What's being tested

### Dense (Qwen3-32B)
HiSpec uses intermediate verification to filter bad drafts before full verification.
Intermediate logits at layer N/4 or N/2 evaluate draft tokens cheaply.

### SSM Hybrid (Qwen3.5-9B) — Checkpoint Optimization Validation
Double-buffer optimization eliminates restore memcpy (~144MB) via O(1) pointer swap.
Comparing against previous benchmark to measure improvement.

## Results

\`\`\`
$(column -t -s, "$RESULTS_FILE")
\`\`\`

## Previous SSM results (pre-optimization, 2026-03-10)

| Config | 9B t/s | Delta | Accept Rate |
|--------|--------|-------|-------------|
| baseline | 15.91 | — | — |
| external 0.8B | 10.59 | -33% | 62.5% |
| self-spec exit=8 | 8.83 | -44% | 77.1% |

## Raw data

\`data/hsd/hispec_external_${TIMESTAMP}.csv\`
DOCEOF

echo "Docs saved to: ${DOCS_DIR}/hispec-external-draft-benchmark.md"
