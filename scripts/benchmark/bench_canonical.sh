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
#                       [--binary PATH --source-root DIR --library-path DIR]
#                       [--ggml-iqk {0,1}] [--ggml-iqk-q8-0 1] [-- EXTRA_BENCH_FLAGS...]
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

# Suppress core dumps (feedback_no_core_dumps). Without this, a llama-bench
# assert on the V4 GGUF (153 GiB) produces a 165 GiB core that immediately
# fills the raid0 mount on a single failure.
ulimit -c 0

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
PERF_PREFLIGHT="${REPO_DIR}/scripts/benchmark/perf_counter_preflight.py"
CANONICAL_PERF_EVENTS="$(
    python3 "$PERF_PREFLIGHT" --print-event-csv 2>/dev/null || \
    echo "fp_ops_retired_by_type.vector_mac,fp_ops_retired_by_type.vector_all,fp_ops_retired_by_type.scalar_all,ls_dmnd_fills_from_sys.dram_io_all,ls_hw_pf_dc_fills.dram_io_all,cycles,instructions,task-clock"
)"

# Defaults (match canonical_recipe.py)
MODEL=""
N_GEN=512
N_PROMPT=0
REPS=2
USE_PERF=0
NO_IK_LLAMA=0
V4_FORK=0
DRY_RUN=0
BINARY_OVERRIDE=""
SOURCE_ROOT=""
LIBRARY_PATH=""
GGML_IQK=1
GGML_IQK_Q8_0=""
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
  --v4-fork           Retired; fails with mainline V4 migration guidance
  --binary PATH        Explicit llama-bench binary for a candidate A/B arm
  --source-root DIR    Git worktree root owning --binary
  --library-path DIR   Candidate llama.cpp library directory; pinned first
                      in LD_LIBRARY_PATH
                      (all three explicit identity options are required together)
  --ggml-iqk {0,1}    Set the GGML_IQK runtime gate (default: 1)
  --ggml-iqk-q8-0 1   Explicitly enable the Q8_0 IQK sub-gate
  --dry-run           Validate + print the canonical command without executing
                      llama-bench. Use this to verify the wiring without firing
                      inference (respects feedback_no_concurrent_inference).
  -h, --help          Show this help

Pass any args after '--' directly to llama-bench (e.g. -ctk q8_0 -ctv q8_0).

The recipe single source of truth is:
  $RECIPE_LIB

DeepSeek-V4 uses the normal mainline production binary. Candidate V4 arms must
pass --binary, --source-root, and --library-path together.
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
        --v4-fork) V4_FORK=1; shift ;;
        --binary) BINARY_OVERRIDE="$2"; shift 2 ;;
        --source-root) SOURCE_ROOT="$2"; shift 2 ;;
        --library-path) LIBRARY_PATH="$2"; shift 2 ;;
        --ggml-iqk) GGML_IQK="$2"; shift 2 ;;
        --ggml-iqk-q8-0) GGML_IQK_Q8_0="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
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
[[ "$V4_FORK" -eq 1 ]] && PY_ARGS+=(--v4-fork)
[[ "$USE_PERF" -eq 1 ]] && PY_ARGS+=(--with-perf)
[[ -n "$BINARY_OVERRIDE" ]] && PY_ARGS+=(--binary "$BINARY_OVERRIDE")
[[ -n "$SOURCE_ROOT" ]] && PY_ARGS+=(--source-root "$SOURCE_ROOT")
[[ -n "$LIBRARY_PATH" ]] && PY_ARGS+=(--library-path "$LIBRARY_PATH")
PY_ARGS+=(--ggml-iqk "$GGML_IQK")
[[ -n "$GGML_IQK_Q8_0" ]] && PY_ARGS+=(--ggml-iqk-q8-0 "$GGML_IQK_Q8_0")
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    # canonical_recipe.py splits sys.argv on the bare `--` before argparse;
    # no `--extra` flag needed (and would be rejected as unknown).
    PY_ARGS+=(-- "${EXTRA_ARGS[@]}")
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

# --- A0: CPU-region mutual exclusion -----------------------------------------
# Until 2026-07-27 this recipe took NO lock, while the orchestrator's dispatch
# path serializes inference through per-region flocks — so a canonical bench and
# an orchestrator placement could occupy the same physical cores with nothing
# preventing it. (The per-run operator-approval clause was the only serializer:
# a human used as a mutex.) We now acquire the SAME locks the dispatch path
# uses, for exactly the cores this run pins.
#
# The cpu list is derived from the emitted command rather than hardcoded, so it
# stays correct if the canonical prefix's width ever changes.
BENCH_CPU_LIST=$(echo "$CMD_JSON" | python3 -c "
import sys, json
cmd = json.load(sys.stdin)['cmd']
try:
    print(cmd[cmd.index('taskset') + 2])
except (ValueError, IndexError):
    print('')
")
REGION_LOCK="${REGION_LOCK_BIN:-/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock}"
LOCK_PREFIX=""
if [[ "${CANONICAL_SKIP_REGION_LOCK:-0}" == "1" ]]; then
    echo "WARNING: CANONICAL_SKIP_REGION_LOCK=1 — running WITHOUT CPU-region exclusion." >&2
    echo "         A concurrent orchestrator placement can poison this measurement." >&2
else
    # Fail closed (fabric axiom 3): an unlockable run is refused, never silently
    # downgraded to the old unprotected behaviour.
    if [[ -z "$BENCH_CPU_LIST" ]]; then
        echo "ERROR: could not derive the taskset cpu list from the canonical command." >&2
        echo "Refusing to run unlocked. Override with CANONICAL_SKIP_REGION_LOCK=1." >&2
        exit 1
    fi
    if [[ ! -x "$REGION_LOCK" ]]; then
        echo "ERROR: region-lock not found or not executable at $REGION_LOCK" >&2
        echo "Set REGION_LOCK_BIN, or override with CANONICAL_SKIP_REGION_LOCK=1." >&2
        exit 1
    fi
    LOCK_PREFIX="$REGION_LOCK run --cpu-list $BENCH_CPU_LIST --role bench-canonical --tag canonical:$(basename "$BINARY") --"
fi

echo "=== Canonical bench command ===" >&2
echo "Binary:    $BINARY" >&2
echo "Env:       $ENV_EXPORTS" >&2
echo "Cmd:       ${CMD_ARGS[*]}" >&2
if [[ -n "$LOCK_PREFIX" ]]; then
    echo "Regions:   cpu-list $BENCH_CPU_LIST (held for the run via region-lock)" >&2
else
    echo "Regions:   UNLOCKED (CANONICAL_SKIP_REGION_LOCK=1)" >&2
fi
if [[ "$USE_PERF" -eq 1 ]]; then
    echo "Perf wrap: $CANONICAL_PERF_EVENTS" >&2
fi
echo "=================================" >&2

# --dry-run: print the command but do not execute. Respects
# feedback_no_concurrent_inference for verifying wrapper wiring.
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN — skipping llama-bench execution." >&2
    exit 0
fi

# Execute
if [[ "$USE_PERF" -eq 1 ]]; then
    PERF_BIN="${PERF_BIN:-perf}"
    if ! command -v "$PERF_BIN" >/dev/null 2>&1; then
        echo "ERROR: perf binary not found: $PERF_BIN" >&2
        echo "Run: python3 $PERF_PREFLIGHT --strict" >&2
        echo "Fix: install or expose linux-tools/perf for the running kernel before --perf." >&2
        exit 1
    fi
    # sudo perf stat needs env preserved across the sudo boundary; pass via env(1)
    # AFTER perf's -- (so perf sees the env-prefix, not its own argv).
    # region-lock is OUTERMOST so the regions stay held for the whole measured
    # run, perf wrapper included.
    eval "$LOCK_PREFIX sudo $PERF_BIN stat -e $CANONICAL_PERF_EVENTS -- env $ENV_EXPORTS ${CMD_ARGS[*]@Q}"
else
    eval "$LOCK_PREFIX env $ENV_EXPORTS ${CMD_ARGS[*]@Q}"
fi
