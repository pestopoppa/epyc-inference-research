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
# INF-70/C7: NUMA pre-eviction + in-window placement proof. Defaults come from
# canonical_recipe.CANONICAL_PRE_EVICT_GIB / CANONICAL_MAX_NODE_SHARE_PCT and are
# re-read from the module below so the two cannot drift apart.
PRE_EVICT_GIB=""
MAX_NODE_SHARE=""
ALLOW_SKEW=0
LOG_DIR=""
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
  --pre-evict-gib N   Force >= N GiB free on every NUMA node before the model
                      load (default: canonical_recipe.CANONICAL_PRE_EVICT_GIB
                      = 40). 0 disables it. Without this, --interleave=all is
                      silently ignored on any full node and decode measures
                      -25% (INF-70/C7, 2026-09-02).
  --max-node-share P  Placement-proof threshold, percent (default:
                      canonical_recipe.CANONICAL_MAX_NODE_SHARE_PCT = 40).
  --allow-skew        Record the placement proof but do not fail the run when
                      it shows skew. Use ONLY for deliberately-skewed arms.
  --log-dir DIR       Where the bench log and its REQUIRED placement-proof row
                      are written (default:
                      /mnt/raid0/llm/tmp/canonical-bench/<UTC>-<model>).
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
        --pre-evict-gib) PRE_EVICT_GIB="$2"; shift 2 ;;
        --max-node-share) MAX_NODE_SHARE="$2"; shift 2 ;;
        --allow-skew) ALLOW_SKEW=1; shift ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
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

# --- INF-70/C7: NUMA placement defaults, read from the single source of truth ---
if [[ -z "$PRE_EVICT_GIB" || -z "$MAX_NODE_SHARE" ]]; then
    read -r _DEF_EVICT _DEF_SHARE < <(python3 -c "
import sys; sys.path.insert(0, '${REPO_DIR}/scripts/lib')
import canonical_recipe as r
print(r.CANONICAL_PRE_EVICT_GIB, r.CANONICAL_MAX_NODE_SHARE_PCT)")
    PRE_EVICT_GIB="${PRE_EVICT_GIB:-$_DEF_EVICT}"
    MAX_NODE_SHARE="${MAX_NODE_SHARE:-$_DEF_SHARE}"
fi
NUMA_EVICT="${REPO_DIR}/scripts/utils/numa_evict.py"
PLACEMENT_CHECK="${REPO_DIR}/scripts/utils/numa_placement_check.sh"
RUN_TAG="$(date -u +%Y%m%dT%H%M%SZ)-$(basename "$MODEL" .gguf)"
LOG_DIR="${LOG_DIR:-/mnt/raid0/llm/tmp/canonical-bench/${RUN_TAG}}"
BENCH_LOG="${LOG_DIR}/bench.log"
PLACEMENT_LOG="${LOG_DIR}/placement.log"
EVICT_LOG="${LOG_DIR}/pre-evict.log"

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
if [[ "$PRE_EVICT_GIB" -gt 0 ]]; then
    echo "Pre-evict: ${PRE_EVICT_GIB} GiB per NUMA node (INF-70/C7)" >&2
else
    echo "Pre-evict: DISABLED (--pre-evict-gib 0) — placement may skew" >&2
fi
echo "Placement: proof required, max node share ${MAX_NODE_SHARE}%" >&2
echo "Logs:      $LOG_DIR" >&2
echo "=================================" >&2

# --dry-run: print the command but do not execute. Respects
# feedback_no_concurrent_inference for verifying wrapper wiring.
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN — skipping NUMA pre-eviction and llama-bench execution." >&2
    exit 0
fi

mkdir -p "$LOG_DIR"

# --- INF-70/C7 step 1: pre-load NUMA eviction --------------------------------
# `numactl --interleave=all` is a per-allocation HINT the kernel abandons for
# any node with no free pages, silently. Measured 2026-09-02: a 98 GB artifact
# landed 57.7/10.7/8.0/17.7 GB across nodes 0-3 and decode fell 25% (10.09 ->
# 7.65 t/s) with an identical command line. Allocating and touching N GiB per
# node under --membind forces the kernel to reclaim page cache ON that node;
# freeing it leaves genuinely free pages so the interleave can be honoured.
if [[ "$PRE_EVICT_GIB" -gt 0 ]]; then
    if [[ ! -f "$NUMA_EVICT" ]]; then
        echo "ERROR: numa_evict.py not found at $NUMA_EVICT" >&2
        exit 1
    fi
    echo "=== Pre-load NUMA eviction (${PRE_EVICT_GIB} GiB/node) ===" >&2
    if ! python3 "$NUMA_EVICT" --target-gib "$PRE_EVICT_GIB" 2>&1 | tee "$EVICT_LOG" >&2; then
        echo "WARNING: pre-eviction did not reach the target on every node." >&2
        echo "         Continuing; the placement proof below is the gate." >&2
    fi
fi

# --- INF-70/C7 step 2: helpers for the in-window placement proof -------------
# Find the largest-RSS process in OUR OWN descendant tree. Never search by name
# (shared host; a name pattern is a wildcard over other sessions' processes) —
# walk /proc/<pid>/task/<pid>/children instead. The model loader is always the
# biggest thing we started.
largest_rss_descendant() {
    local root="$1" best="$1" bestrss=0 rss kids k
    local -a queue=("$root")
    local i=0
    while [[ $i -lt ${#queue[@]} ]]; do
        local cur="${queue[$i]}"; i=$((i + 1))
        [[ -r "/proc/$cur/statm" ]] || continue
        rss=$(awk '{print $2}' "/proc/$cur/statm" 2>/dev/null) || rss=0
        [[ -n "$rss" ]] || rss=0
        if [[ "$rss" -gt "$bestrss" ]]; then bestrss="$rss"; best="$cur"; fi
        for f in /proc/"$cur"/task/*/children; do
            [[ -r "$f" ]] || continue
            read -r kids < "$f" || kids=""
            for k in $kids; do queue+=("$k"); done
        done
    done
    echo "$best $bestrss"
}

# Sample placement once the resident set has stopped growing (i.e. the weights
# are loaded) and the process is still alive. A sample taken after the run is
# not evidence: the pages are gone.
sample_placement_in_window() {
    local root="$1" prev=0 stable=0 cand rss
    local floor_pages=$((1024 * 1024 * 1024 / 4096))   # 1 GiB
    while kill -0 "$root" 2>/dev/null; do
        read -r cand rss < <(largest_rss_descendant "$root")
        if [[ "$rss" -ge "$floor_pages" ]]; then
            if [[ "$rss" -le "$prev" ]]; then
                stable=$((stable + 1))
            else
                stable=0
            fi
            prev="$rss"
            if [[ "$stable" -ge 2 ]]; then
                bash "$PLACEMENT_CHECK" "$cand" \
                    --threshold "$MAX_NODE_SHARE" \
                    --label "canonical:$(basename "$MODEL")" > "$PLACEMENT_LOG" 2>&1
                echo "$?" > "${PLACEMENT_LOG}.rc"
                return 0
            fi
        fi
        sleep 3
    done
    return 1
}

# --- INF-70/C7 step 3: run, sampling placement in-window ---------------------
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
    RUN_CMD="$LOCK_PREFIX sudo $PERF_BIN stat -e $CANONICAL_PERF_EVENTS -- env $ENV_EXPORTS ${CMD_ARGS[*]@Q}"
else
    RUN_CMD="$LOCK_PREFIX env $ENV_EXPORTS ${CMD_ARGS[*]@Q}"
fi

eval "$RUN_CMD" > >(tee "$BENCH_LOG") 2>&1 &
RUN_PID=$!
sample_placement_in_window "$RUN_PID" || true
wait "$RUN_PID"
BENCH_RC=$?

# --- INF-70/C7 step 4: the placement proof is a REQUIRED row ------------------
echo "" >&2
echo "=== Placement proof (required) ===" >&2
if [[ ! -s "$PLACEMENT_LOG" ]]; then
    echo "ERROR: no placement proof was captured for this run." >&2
    echo "       $PLACEMENT_LOG is missing or empty — the run is an OBSERVATION," >&2
    echo "       not a measurement (a window that misses the phenomenon proves nothing)." >&2
    exit 1
fi
cat "$PLACEMENT_LOG" >&2
PLACEMENT_RC=$(cat "${PLACEMENT_LOG}.rc" 2>/dev/null || echo 1)
echo "Bench log:       $BENCH_LOG" >&2
echo "Placement proof: $PLACEMENT_LOG (exit $PLACEMENT_RC)" >&2

if [[ "$PLACEMENT_RC" == "3" ]]; then
    if [[ "$ALLOW_SKEW" -eq 1 ]]; then
        echo "WARNING: placement is SKEWED; --allow-skew given, reporting anyway." >&2
        echo "         Do not quote this run as a canonical number." >&2
    else
        echo "ERROR: placement is SKEWED — this timing is not a valid measurement." >&2
        echo "       Re-run after python3 $NUMA_EVICT --target-gib $PRE_EVICT_GIB," >&2
        echo "       or pass --allow-skew if the skew IS the variable under test." >&2
        exit 3
    fi
elif [[ "$PLACEMENT_RC" != "0" ]]; then
    echo "WARNING: the placement check itself failed (exit $PLACEMENT_RC); see the log." >&2
fi

exit "$BENCH_RC"
