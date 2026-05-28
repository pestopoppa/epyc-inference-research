#!/usr/bin/env bash
# bench_canonical.sh — the ONLY sanctioned llama-bench entry point for EPYC.
#
# Wraps canonical_recipe.py (single source of truth for the recipe) so that no
# operator (human or agent) needs to reconstruct the bench command from memory.
# Drift bit this project at least 3 times before this script existed:
#   - 2026-05-02: launcher drifted (missing taskset, mmap=ON, AOCC libomp)
#   - 2026-05-28 multiple: wrong binary, missing OMP_DYNAMIC=false, broken
#                          ik_llama RUNPATH, THP defrag reset, perf_paranoid reset
#
# Both episodes are documented in canonical_recipe.py's module docstring. The
# fix in both cases was "use the codified recipe, don't invent the command."
#
# Usage:
#   bench_canonical.sh -m MODEL [-n N_GEN] [-p N_PROMPT] [-r REPS] [--perf]
#                       [--no-ik-llama] [-- EXTRA_BENCH_FLAGS...]
#
# Examples:
#   # gemma4-26B-A4B Q4_K_M tg512 r=2, no perf wrap
#   bench_canonical.sh -m /mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf
#
#   # Same with perf-stat wrapping (canonical event set)
#   bench_canonical.sh -m /path/to/model.gguf --perf
#
#   # Pass extra flags to llama-bench
#   bench_canonical.sh -m /path/to/model.gguf -- -ctk q8_0 -ctv q8_0
#
# All host-environment, command-shape, env-var, and binary-linkage validation
# happens BEFORE the bench runs. If anything has drifted, you get a clear
# error explaining what to fix.

set -euo pipefail

# Locate canonical_recipe.py
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${EPYC_RESEARCH_REPO:-${SCRIPT_DIR%/scripts/*}}"
RECIPE_LIB="${REPO_DIR}/scripts/lib/canonical_recipe.py"

if [[ ! -f "$RECIPE_LIB" ]]; then
    echo "ERROR: canonical_recipe.py not found at $RECIPE_LIB" >&2
    echo "Set EPYC_RESEARCH_REPO if the research repo is at a non-standard path." >&2
    exit 1
fi

# Canonical perf event set for FLOPS + DRAM BW roofline measurement.
# See cpu-decode-flops-roofline-audit.md §0.7 Phase 0 Calibration Results
# for the discovery story (this Zen 5 host exposes these events; Intel-named
# fp_arith_inst_retired.* / uncore_imc/cas_count_* do NOT exist here).
CANONICAL_PERF_EVENTS="fp_ops_retired_by_type.vector_mac,fp_ops_retired_by_type.vector_all,fp_ops_retired_by_type.scalar_all,ls_dmnd_fills_from_sys.dram_io_all,ls_hw_pf_dc_fills.dram_io_all,cycles,instructions,task-clock"

# Defaults (match canonical_recipe.py)
MODEL=""
N_GEN=512
N_PROMPT=0
REPS=2
USE_PERF=0
NO_IK_LLAMA=0
EXTRA_ARGS=()

usage() {
    cat <<EOF >&2
Usage: $(basename "$0") -m MODEL [OPTIONS] [-- EXTRA_BENCH_FLAGS...]

  -m, --model PATH    GGUF model path (required)
  -n N_GEN            Tokens to generate per rep (default: 512)
  -p N_PROMPT         Prefill tokens (default: 0 = decode-only)
  -r REPS             Repetitions (default: 2)
  --perf              Wrap in sudo perf stat with canonical event set
  --no-ik-llama       Prefer v5_clean over ik_llama (default: prefer ik_llama)
  -h, --help          Show this help

Pass any args after '--' directly to llama-bench (e.g. -ctk q8_0 -ctv q8_0).

The recipe single source of truth is:
  $RECIPE_LIB
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -n) N_GEN="$2"; shift 2 ;;
        -p) N_PROMPT="$2"; shift 2 ;;
        -r) REPS="$2"; shift 2 ;;
        --perf) USE_PERF=1; shift ;;
        --no-ik-llama) NO_IK_LLAMA=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA_ARGS=("$@"); break ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: -m MODEL is required" >&2
    usage
    exit 1
fi

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 1
fi

# Build the emit-bench-command invocation
PY_ARGS=(emit-bench-command --model "$MODEL" --n-prompt "$N_PROMPT" --n-gen "$N_GEN" --reps "$REPS")
[[ "$NO_IK_LLAMA" -eq 1 ]] && PY_ARGS+=(--no-ik-llama)
[[ "$USE_PERF" -eq 1 ]] && PY_ARGS+=(--with-perf)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    PY_ARGS+=(--extra -- "${EXTRA_ARGS[@]}")
fi

# Validate + emit the canonical command as JSON. canonical_recipe.py raises
# CanonicalRecipeViolation with a clear message if anything has drifted.
echo "=== Validating canonical recipe ===" >&2
if ! CMD_JSON=$(python3 "$RECIPE_LIB" "${PY_ARGS[@]}" 2>&1); then
    echo "$CMD_JSON" >&2
    exit 1
fi

# Parse JSON safely with Python
BINARY=$(echo "$CMD_JSON" | python3 -c "import sys, json; print(json.load(sys.stdin)['binary'])")

# Build the env-var export string and cmd-arg list
ENV_EXPORTS=$(echo "$CMD_JSON" | python3 -c "
import sys, json, shlex
env = json.load(sys.stdin)['env']
print(' '.join(f'{k}={shlex.quote(v)}' for k, v in env.items()))
")

# Read cmd into a bash array via shlex-equivalent splitting
declare -a CMD_ARGS
while IFS= read -r line; do
    CMD_ARGS+=("$line")
done < <(echo "$CMD_JSON" | python3 -c "
import sys, json
for arg in json.load(sys.stdin)['cmd']:
    print(arg)
")

echo "=== Canonical bench command ===" >&2
echo "Binary:    $BINARY" >&2
echo "Env:       $ENV_EXPORTS" >&2
echo "Cmd:       ${CMD_ARGS[*]}" >&2
if [[ "$USE_PERF" -eq 1 ]]; then
    echo "Perf wrap: $CANONICAL_PERF_EVENTS" >&2
fi
echo "=================================" >&2

# Execute
if [[ "$USE_PERF" -eq 1 ]]; then
    # sudo perf stat needs env preserved across the sudo boundary; pass via env(1)
    # AFTER perf's -- (so perf sees the env-prefix, not its own argv).
    eval "sudo perf stat -e $CANONICAL_PERF_EVENTS -- env $ENV_EXPORTS ${CMD_ARGS[*]@Q}"
else
    eval "env $ENV_EXPORTS ${CMD_ARGS[*]@Q}"
fi
