#!/usr/bin/env bash
# v4_throughput_gate.sh — the §Throughput-gate runner for DeepSeek-V4-Flash
# under Strategy B (antirez fork).
#
# Per handoffs/active/deepseek-v4-flash-cpu-port.md §Throughput gate (amended
# 2026-05-29):
#   Tool   : V4 fork llama-completion (NOT llama-bench)
#   Mode   : raw prompt completion, -no-cnv (V4 GGUF embeds a chat template
#            and llama-completion auto-enables conv mode unless disabled)
#   Stack  : canonical NPS4 — taskset -c 0-95 numactl --interleave=all
#            -t 96 -fa 1 --mlock --no-mmap + OMP env + KMP_BLOCKTIME=10
#            + GGML_NUMA_WEIGHTS=1 + LLVM-20 libomp on LD_LIBRARY_PATH
#   Prompt : Fibonacci-iterative-and-explain (matches roofline audit prompt)
#   Decode : --temp 0 --seed 42 -n 512
#   Metric : libllama eval-time tokens-per-second (sampling.cpp:507 stderr
#            line: "eval time = X ms / N runs (... tokens per second)").
#            Excludes load time, prompt eval, total wall.
#   Floor  : Q4 ≥ 18 t/s; Q2 ≥ 35 t/s
#
# Why this script vs bench_canonical.sh:
#   llama-bench's default synthetic graph reserve (worst-case ubatch_size,
#   ubatch.pos=nullptr, last_pos = n_tokens - 1) trips V4's compressed-cache
#   assert at deepseek4.cpp:1147 GGML_ASSERT(n_comp_visible <= n_comp_cache)
#   during llama_context::sched_reserve at init time. llama-completion also
#   calls sched_reserve but with reserve shapes that fit V4's cache.
#
# Usage:
#   v4_throughput_gate.sh -m PATH [--dry-run] [--out DIR] [--n-predict N]
#                          [--label TAG]
#
# Outputs (when not --dry-run):
#   <out>/timestamp.stdout      — llama-completion stdout
#   <out>/timestamp.stderr      — llama-completion stderr (timing lines)
#   <out>/timestamp.summary.md  — parsed gate result + pass/fail vs floor
#
# Respects feedback_no_concurrent_inference: this script does inference. The
# user authorizes per-run. --dry-run prints the command WITHOUT executing.

set -euo pipefail

# Suppress core dumps (feedback_no_core_dumps). Without this, the V4 GGUF's
# 153 GiB resident size produces ~165 GiB cores on any assert — single failure
# fills the raid0 mount. Cores have no debugging value here vs stderr backtraces.
ulimit -c 0

MODEL=""
DRY_RUN=0
N_PREDICT=512
SEED=42
CTX=8192
OUT_DIR="/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization"
LABEL=""
USE_MLOCK=1
PROMPT='Write a Python function that computes the n-th Fibonacci number iteratively. Then explain it briefly.'

# Floors per §Throughput gate. These are tracked separately so the script
# can emit a verdict line; the gate decision still rests with the operator.
Q4_FLOOR_TPS=18
Q2_FLOOR_TPS=35

usage() {
    cat <<EOF >&2
Usage: $(basename "$0") -m MODEL [OPTIONS]

  -m, --model PATH    GGUF model path (required)
  -n, --n-predict N   Tokens to decode (default: 512 per §Throughput gate)
  --seed N            Sampler seed (default: 42 per §Throughput gate)
  --ctx N             Context size (default: 8192; ample for 512-token decode)
  --label TAG         Optional label (appended to output dir name)
  --no-mlock          Skip --mlock (NOT recommended; defeats THP-pool warmup)
  --out DIR           Parent output directory (default: data/cpu_optimization)
  --dry-run           Validate + print canonical command without executing
  -h, --help          Show this help

Output goes to <out>/YYYY-MM-DD-v4-throughput-gate[-LABEL]/<timestamp>.{stdout,stderr,summary.md}
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -m|--model) MODEL="$2"; shift 2 ;;
        -n|--n-predict) N_PREDICT="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        --label) LABEL="$2"; shift 2 ;;
        --no-mlock) USE_MLOCK=0; shift ;;
        --out) OUT_DIR="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    echo "ERROR: -m MODEL required" >&2; usage; exit 1
fi
if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: model file not found: $MODEL" >&2; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Locate canonical_recipe + V4 fork binaries
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${EPYC_RESEARCH_REPO:-${SCRIPT_DIR%/scripts/*}}"
RECIPE_LIB="${REPO_DIR}/scripts/lib/canonical_recipe.py"
V4_FORK_BIN=/mnt/raid0/llm/llama.cpp-deepseek-v4/build/bin
LLAMA_COMPLETION="${V4_FORK_BIN}/llama-completion"

if [[ ! -f "$RECIPE_LIB" ]]; then
    echo "ERROR: canonical_recipe.py not found at $RECIPE_LIB" >&2; exit 1
fi
if [[ ! -x "$LLAMA_COMPLETION" ]]; then
    echo "ERROR: llama-completion not found at $LLAMA_COMPLETION" >&2; exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Preflight via canonical_recipe.py (host env + V4 fork binary linkage)
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Validating canonical recipe (--v4-fork) ===" >&2
if ! python3 "$RECIPE_LIB" validate --v4-fork; then
    echo "PREFLIGHT FAILED" >&2; exit 2
fi

# ─────────────────────────────────────────────────────────────────────────────
# Build env via canonical_recipe (V4 gate-extra env: KMP_BLOCKTIME + GGML_NUMA_WEIGHTS)
# ─────────────────────────────────────────────────────────────────────────────
ENV_EXPORTS=$(python3 -c "
import sys
sys.path.insert(0, '${REPO_DIR}/scripts/lib')
import canonical_recipe as r
import shlex
env = r.build_canonical_env(use_v4_gate_extras=True)
keys = list(r.CANONICAL_OMP_ENV) + list(r.V4_GATE_EXTRA_ENV) + ['LD_LIBRARY_PATH']
out = []
for k in keys:
    v = env[k]
    if k == 'LD_LIBRARY_PATH':
        v = '${V4_FORK_BIN}:' + v
    out.append(f'{k}={shlex.quote(v)}')
print(' '.join(out))
")

# ─────────────────────────────────────────────────────────────────────────────
# Compose llama-completion command
# ─────────────────────────────────────────────────────────────────────────────
# -no-cnv: V4 GGUF has tokenizer.chat_template at kv[57]; without -no-cnv
#          completion.cpp:213 auto-enables conv mode → not bare-completion.
# -n: tokens to predict (decode count for the gate).
# -fa 1: FlashAttention on (matches gemma4 production stack).
# --mlock --no-mmap: anonymous-page resident model, matches production launch
#                    and §Throughput gate spec.
# -t 96 -c CTX: full 96 threads + 8K ctx (ample for 512-token decode).
# --temp 0 --seed: deterministic.
CLI_FLAGS=(
    -m "$MODEL"
    -t 96
    -c "$CTX"
    --temp 0
    --seed "$SEED"
    -n "$N_PREDICT"
    -p "$PROMPT"
    -no-cnv
    -ngl 0
    -fa 1
)
[[ "$USE_MLOCK" -eq 1 ]] && CLI_FLAGS+=( --mlock )
CLI_FLAGS+=( --no-mmap )

CANONICAL_PREFIX=(taskset -c 0-95 numactl --interleave=all)

echo "=== V4 Throughput Gate command ===" >&2
echo "Tool:      $LLAMA_COMPLETION" >&2
echo "Env:       $ENV_EXPORTS" >&2
echo "Cmd:       ${CANONICAL_PREFIX[*]} $LLAMA_COMPLETION ${CLI_FLAGS[*]}" >&2
echo "Floor:     Q4 ≥ ${Q4_FLOOR_TPS} t/s (eval-time decode)" >&2
echo "==================================" >&2

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY RUN — skipping llama-completion execution." >&2
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# Set up output directory
# ─────────────────────────────────────────────────────────────────────────────
DATE_TAG=$(date '+%Y-%m-%d')
TS=$(date '+%Y%m%dT%H%M%S')
RUN_DIR_NAME="${DATE_TAG}-v4-throughput-gate"
[[ -n "$LABEL" ]] && RUN_DIR_NAME="${RUN_DIR_NAME}-${LABEL}"
RUN_DIR="${OUT_DIR}/${RUN_DIR_NAME}"
mkdir -p "$RUN_DIR"
STDOUT="${RUN_DIR}/${TS}.stdout"
STDERR="${RUN_DIR}/${TS}.stderr"
SUMMARY="${RUN_DIR}/${TS}.summary.md"

echo "Output:    $RUN_DIR/${TS}.{stdout,stderr,summary.md}" >&2
echo "" >&2
echo "Running llama-completion (model load + 512-token decode; ~5-7 min wall):" >&2
echo "  expected stderr line: 'eval time = X ms / N runs (... tokens per second)'" >&2
echo "" >&2

# Execute with full canonical env. eval will respect the env-var prefix.
set +e
eval "env $ENV_EXPORTS ${CANONICAL_PREFIX[*]@Q} '$LLAMA_COMPLETION' ${CLI_FLAGS[*]@Q} > '$STDOUT' 2> '$STDERR'"
RC=$?
set -e

# ─────────────────────────────────────────────────────────────────────────────
# Parse + emit summary
# ─────────────────────────────────────────────────────────────────────────────
EVAL_LINE=$(grep -E 'eval time =.*tokens per second' "$STDERR" | grep -v 'prompt eval' | tail -1 || true)
PROMPT_LINE=$(grep -E 'prompt eval time' "$STDERR" | tail -1 || true)
LOAD_LINE=$(grep -E 'load time' "$STDERR" | tail -1 || true)
TOTAL_LINE=$(grep -E 'total time' "$STDERR" | tail -1 || true)

# Extract eval-time tokens per second value (last numeric on the line before "tokens per second").
EVAL_TPS=$(echo "$EVAL_LINE" | sed -nE 's/.*\(([0-9]+\.[0-9]+) tokens per second\)/\1/p' | tail -1)
PP_TPS=$(echo "$PROMPT_LINE" | sed -nE 's/.*\(([0-9]+\.[0-9]+) tokens per second\)/\1/p' | tail -1)

VERDICT="UNKNOWN"
VERDICT_REASON=""
if [[ -z "$EVAL_TPS" ]]; then
    VERDICT="ERROR"
    VERDICT_REASON="Could not parse eval-time tokens-per-second from stderr. llama-completion may have crashed (rc=$RC) or stderr format may have changed. Inspect $STDERR."
else
    # bash arithmetic doesn't do float; compare via awk.
    PASS=$(awk -v t="$EVAL_TPS" -v f="$Q4_FLOOR_TPS" 'BEGIN { print (t >= f) ? "PASS" : "FAIL" }')
    VERDICT="$PASS"
    VERDICT_REASON="eval-time decode ${EVAL_TPS} t/s vs floor ${Q4_FLOOR_TPS} t/s (Q4)"
fi

cat > "$SUMMARY" <<EOF
# DeepSeek-V4 §Throughput Gate Result (Strategy B, antirez fork)

**Verdict**: ${VERDICT} — ${VERDICT_REASON}

**Provisional**: this run was conducted before the quality gate. Per §Throughput gate "provisional evidence rule", the result counts toward merge ONLY if the quality gate later passes.

## Inputs

- Model: \`${MODEL}\`
- Tool: \`${LLAMA_COMPLETION}\` (V4 fork tip 2f2d44052)
- Mode: raw prompt completion (-no-cnv)
- Prompt: \`${PROMPT}\`
- Decode: --temp 0 --seed ${SEED} -n ${N_PREDICT}
- Stack: ${CANONICAL_PREFIX[*]} + canonical OMP + KMP_BLOCKTIME=10 + GGML_NUMA_WEIGHTS=1 + LLVM-20 libomp
- mlock: $([ "$USE_MLOCK" -eq 1 ] && echo "yes" || echo "no") / mmap: no

## Result (libllama timing, sampling.cpp:507)

\`\`\`
${LOAD_LINE:-<no load line>}
${PROMPT_LINE:-<no prompt eval line>}
${EVAL_LINE:-<no eval line>}
${TOTAL_LINE:-<no total line>}
\`\`\`

## Metric

- **eval time t/s**: ${EVAL_TPS:-<parse failed>} (this is the gate metric — decode-only, excludes load + prompt eval)
- prompt eval t/s: ${PP_TPS:-<parse failed>} (informational)
- llama-completion exit code: ${RC}

## Floor

- Q4 ≥ ${Q4_FLOOR_TPS} t/s eval-time decode (calibrated from V4-Flash 13B active vs gemma4-26B-A4B 4B active sustaining 76.5 t/s solo)

## Files

- stdout: \`$(basename "$STDOUT")\`
- stderr: \`$(basename "$STDERR")\`
- this summary: \`$(basename "$SUMMARY")\`
EOF

echo "" >&2
cat "$SUMMARY"
echo "" >&2

exit "$RC"
