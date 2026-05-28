#!/usr/bin/env bash
# v4_smoke_test.sh — pre-bench verification that V4 GGUF + V4 fork binary are
# ready for the throughput + quality gates.
#
# Runs in three phases:
#   1. PREFLIGHT — canonical_recipe.py validate --v4-fork (host + binary linkage)
#   2. METADATA  — llama-gguf reads the V4 GGUF; verifies architecture=deepseek4,
#                  tensor count, expected V4-specific kv fields
#   3. SMOKE     — tiny 4-token decode via llama-cli to confirm the model loads
#                  into RAM, the deepseek4 arch graph builds without assertion,
#                  and output is non-empty (not segfault / NaN / empty)
#
# Exits non-zero on the first failure; emits a clear error per the codified-
# recipe pattern (no reconstruction-from-memory).
#
# Usage:
#   v4_smoke_test.sh -m PATH [--ctx CTX] [--no-mlock]
#
# Notes:
#   - This script does NOT need the production stack to be down (it doesn't
#     run a sustained workload). The 4-token decode is brief.
#   - But it DOES run inference (llama-cli with a real prompt). Per
#     feedback_no_concurrent_inference, the operator should be aware.

set -euo pipefail

MODEL=""
CTX=8192
USE_MLOCK=1
PORT=18173  # arbitrary; we don't use a server but reserved for future expansion

usage() {
    cat <<EOF >&2
Usage: $(basename "$0") -m MODEL_PATH [OPTIONS]
  -m, --model PATH    Path to the V4 GGUF (required)
  --ctx N             Context size for the smoke test (default: 8192)
  --no-mlock          Skip --mlock (faster but doesn't warm THP pool)
  -h, --help          Show this help

Output:
  - PREFLIGHT result (canonical_recipe.py validate --v4-fork)
  - METADATA result (llama-gguf parses + V4 kv fields present)
  - SMOKE result (4-token decode succeeds + non-empty output)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        --ctx)      CTX="$2"; shift 2 ;;
        --no-mlock) USE_MLOCK=0; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: -m MODEL required" >&2
    usage
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${EPYC_RESEARCH_REPO:-${SCRIPT_DIR%/scripts/*}}"
RECIPE_LIB="${REPO_DIR}/scripts/lib/canonical_recipe.py"
V4_FORK_DIR=/mnt/raid0/llm/llama.cpp-deepseek-v4
V4_FORK_BIN="${V4_FORK_DIR}/build/bin"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: PREFLIGHT (folds in option C of the prep work)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 1: PREFLIGHT (canonical_recipe.py validate --v4-fork) ==="
if ! python3 "$RECIPE_LIB" validate --v4-fork; then
    echo "PREFLIGHT FAILED — see error above." >&2
    exit 2
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: METADATA (llama-gguf reads the GGUF header + verifies kv fields)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 2: METADATA (llama-gguf parses + V4-specific kv fields) ==="
LLAMA_GGUF="${V4_FORK_BIN}/llama-gguf"
if [[ ! -x "$LLAMA_GGUF" ]]; then
    echo "ERROR: llama-gguf not found at $LLAMA_GGUF" >&2
    exit 3
fi

# set -e + pipefail can exit on the failing command substitution before we
# get to inspect the rc. Capture exit code via `&&...||` so the error block
# runs deterministically.
GGUF_OUT=$(LD_LIBRARY_PATH="$V4_FORK_BIN" "$LLAMA_GGUF" "$MODEL" r 2>&1) \
    && GGUF_RC=0 || GGUF_RC=$?
if [[ "$GGUF_RC" -ne 0 ]]; then
    echo "ERROR: llama-gguf failed (rc=$GGUF_RC):" >&2
    echo "$GGUF_OUT" | head -20 >&2
    exit 4
fi

# Extract key facts
N_KV=$(echo "$GGUF_OUT" | grep -E 'n_kv:' | head -1 | awk '{print $NF}')
N_TENSORS=$(echo "$GGUF_OUT" | grep -E 'n_tensors:' | head -1 | awk '{print $NF}')
echo "  n_kv:     $N_KV"
echo "  n_tensors: $N_TENSORS"

# Required V4-specific kv fields (presence check; values vary by quant)
REQUIRED_KV=(
    "deepseek4.block_count"
    "deepseek4.attention.q_lora_rank"
    "deepseek4.attention.output_lora_rank"
    "deepseek4.attention.compress_ratios"
    "deepseek4.attention.compress_rope_freq_base"
    "deepseek4.expert_count"
    "deepseek4.expert_shared_count"
)
MISSING_KV=()
for k in "${REQUIRED_KV[@]}"; do
    if ! echo "$GGUF_OUT" | grep -qF "key = $k"; then
        MISSING_KV+=("$k")
    fi
done
if [[ "${#MISSING_KV[@]}" -gt 0 ]]; then
    echo "ERROR: missing required V4 kv fields:" >&2
    for k in "${MISSING_KV[@]}"; do echo "  - $k" >&2; done
    echo "" >&2
    echo "GGUF metadata may be incomplete (download corrupted?) or this GGUF" >&2
    echo "is not actually V4-Flash. Inspect the file with:" >&2
    echo "  LD_LIBRARY_PATH=$V4_FORK_BIN $LLAMA_GGUF '$MODEL' r" >&2
    exit 5
fi
echo "  all $((${#REQUIRED_KV[@]})) required V4 kv fields present ✓"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: SMOKE (4-token decode via llama-cli; uses canonical env)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== PHASE 3: SMOKE (tiny 4-token decode) ==="
LLAMA_CLI="${V4_FORK_BIN}/llama-cli"
if [[ ! -x "$LLAMA_CLI" ]]; then
    echo "ERROR: llama-cli not found at $LLAMA_CLI" >&2
    exit 3
fi

# Build the canonical command via canonical_recipe.py emit-bench-command —
# but emit-bench-command is bench-specific. For llama-cli we compose the
# prefix + env directly here.
CLI_FLAGS=( -m "$MODEL" -t 96 -c "$CTX" --temp 0 --seed 1 -n 4 -p "Hello," --no-conversation -ngl 0 -fa 1 )
[[ "$USE_MLOCK" -eq 1 ]] && CLI_FLAGS+=( --mlock )
CLI_FLAGS+=( --no-mmap )

# Canonical env loaded from the single source of truth. The V4 fork bench dir
# also goes on LD_LIBRARY_PATH so llama-cli resolves to the fork's libllama /
# libggml (DT_RPATH from --disable-new-dtags is the primary mechanism, but
# this is belt-and-braces).
SMOKE_OUT=/tmp/v4-smoke-test.stdout
SMOKE_ERR=/tmp/v4-smoke-test.stderr
echo "Running (may take 1-2 min to mlock 153 GiB on first load):"
echo "  taskset -c 0-95 numactl --interleave=all $LLAMA_CLI ${CLI_FLAGS[*]}"
echo ""

# Build env exports from canonical_recipe.py — single source of truth includes
# OMP_* + KMP_BLOCKTIME + GGML_NUMA_WEIGHTS + LLVM-20 libomp on LD_LIBRARY_PATH.
ENV_EXPORTS=$(python3 -c "
import sys
sys.path.insert(0, '${REPO_DIR}/scripts/lib')
import canonical_recipe as r
import shlex
env = r.build_canonical_env()
keys = list(r.CANONICAL_OMP_ENV) + list(r.CANONICAL_PRODUCTION_ENV) + ['LD_LIBRARY_PATH']
out = []
for k in keys:
    v = env[k]
    if k == 'LD_LIBRARY_PATH':
        v = '${V4_FORK_BIN}:' + v
    out.append(f'{k}={shlex.quote(v)}')
print(' '.join(out))
")

# Use timeout because if the load takes too long we want to fail visibly,
# not hang. 10 minutes is generous (mlock'ing 153 GiB on cold cache + first-
# touch faulting is the long-pole; should be <5 min on this host).
eval "env $ENV_EXPORTS timeout 600 taskset -c 0-95 numactl --interleave=all \
    '$LLAMA_CLI' ${CLI_FLAGS[*]@Q} \
    > '$SMOKE_OUT' 2> '$SMOKE_ERR'" || SMOKE_RC=$?

SMOKE_RC=${SMOKE_RC:-0}

# Output analysis. `grep | tail` under pipefail returns nonzero if grep matches
# nothing (no output → empty input to tail), and set -e would then abort here
# before the custom no-output error block runs. Append `|| true` to neutralize.
GENERATED=$(grep -v "^[[:space:]]*$" "$SMOKE_OUT" 2>/dev/null | tail -5 || true)
if [[ "$SMOKE_RC" -eq 124 ]]; then
    echo "ERROR: smoke test timed out after 10 minutes (model load too slow)." >&2
    echo "  stderr tail:" >&2
    tail -10 "$SMOKE_ERR" >&2
    exit 6
elif [[ "$SMOKE_RC" -ne 0 ]]; then
    echo "ERROR: smoke test exited non-zero (rc=$SMOKE_RC):" >&2
    tail -15 "$SMOKE_ERR" >&2
    exit 7
elif [[ -z "$GENERATED" ]]; then
    echo "ERROR: smoke test produced no output (likely segfault or assert):" >&2
    tail -10 "$SMOKE_ERR" >&2
    exit 8
fi

echo "  output (last 5 non-empty lines):"
echo "$GENERATED" | sed 's/^/    /'
echo ""
echo "=== ALL PHASES PASSED ==="
echo ""
echo "Next: throughput gate via"
echo "  bench_canonical.sh --v4-fork --perf -m '$MODEL'"
echo "Then: quality gate via"
echo "  v4_quality_gate_runner.py --model '$MODEL' --output epyc-v4-logprobs.json"
