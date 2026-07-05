#!/bin/bash
# kernel_sweep.sh — Phase 2 of the MI210 kernel-R&D loop: the inner tuning loop.
#
# Given a set of ALREADY-CHOSEN variants (a param sweep for one kernel design),
# runs each through kernel_eval.sh serially on the single GPU, ingests the
# OBSERVATION records into the strategy store, and prints the Pareto frontier.
#
# This automates the INNER (tuning) loop only. The OUTER hypothesis/design loop
# — proposing the kernel designs and the sweep points — stays planner/critic-
# interactive (per mi210-kernel-rnd-loop-proposal.md): the single-GPU serial
# cadence makes brute search too expensive, so a human/high-effort model picks
# the points using the mechanism profile. Authorize stays operator-only.
#
# Nightshift-runnable (single-GPU serial => overnight cadence). Every number is
# an OBSERVATION; nothing here gates a keep/deploy/promote decision.
#
# Usage:
#   kernel_sweep.sh --model <gguf> --specs <tsv> [--baseline-env 'VAR=0'] \
#                   [--target-kernel K] [--build] [--out f.jsonl] [--db store.sqlite]
#   where <tsv> has one variant per line:  <label>\t<variant-env>
#     e.g.   nwarps4-prefetch\tGGML_CUDA_Q8_PREFETCH=1
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

MODEL="" ; SPECS="" ; BASELINE_ENV="" ; TARGET_KERNEL="mul_mat_vec_q" ; BUILD_FLAG=""
OUT="/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_rnd_results.jsonl"
DB="${KERNEL_STORE_DB:-/mnt/raid0/llm/tmp/mi210-build/campaign/kernel_strategy_store.sqlite}"
usage(){ echo "usage: $0 --model <gguf> --specs <tsv label\\tvariant-env> [--baseline-env 'V=0'] [--target-kernel K] [--build] [--out f] [--db f]"; exit 2; }
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --specs) SPECS="$2"; shift 2;;
  --baseline-env) BASELINE_ENV="$2"; shift 2;;
  --target-kernel) TARGET_KERNEL="$2"; shift 2;;
  --build) BUILD_FLAG="--build"; shift;;
  --out) OUT="$2"; shift 2;;
  --db) DB="$2"; shift 2;;
  *) usage;;
esac; done
[ -n "$MODEL" ] && [ -n "$SPECS" ] || usage
[ -f "$SPECS" ] || { echo "FATAL: specs file not found: $SPECS"; exit 1; }

echo "=== kernel_sweep: $(grep -cvE '^\s*(#|$)' "$SPECS") variants over $(basename "$MODEL") ==="
ran=0 ; failed=0
# read label<TAB>variant-env ; ignore blank/comment lines
while IFS=$'\t' read -r label venv || [ -n "$label" ]; do
  case "$label" in ''|\#*) continue;; esac
  [ -n "$venv" ] || { echo "  skip '$label' (no variant-env)"; continue; }
  echo "--- [$((ran+1))] $label  ($venv) ---"
  if "$HERE/kernel_eval.sh" --model "$MODEL" --label "$label" \
        --variant-env "$venv" ${BASELINE_ENV:+--baseline-env "$BASELINE_ENV"} \
        --target-kernel "$TARGET_KERNEL" --out "$OUT" $BUILD_FLAG; then
    ran=$((ran+1))
  else
    # kernel_eval.sh already recorded the FAIL row (lexicographic); keep sweeping.
    failed=$((failed+1)); echo "  ($label recorded a FAIL/observation — continuing sweep)"
  fi
done < "$SPECS"

echo "=== sweep done: $ran ok, $failed failed/observation; ingesting -> store ==="
python3 "$HERE/kernel_store.py" ingest "$OUT" --db "$DB"
echo "=== current Pareto frontier (CORRECT runs only) ==="
python3 "$HERE/kernel_store.py" pareto --db "$DB" --model "$(basename "$MODEL")"
echo "=== refreshing dashboard-hub contract (kernel_dashboard.json) ==="
python3 "$HERE/kernel_store.py" export --db "$DB"
