#!/bin/bash
# PROMOTED INTO THE REPO 2026-07-29 by `auditor` from a session scratchpad, per
# epyc-root/handoffs/active/architect-model-selection-bench.md § "Follow-up tooling".
# Three corrections were required and are NOT cosmetic — see architect_bench_gpu_lib.sh:
#   1. `source` pointed at another session's /tmp scratchpad (ephemeral: dead on arrival);
#   2. `--kernel production-consolidated-v7` on a v8 binary = false provenance, now REQUIRED;
#   3. the shared lib pinned CORES=88-95, the superseded value (now 184-191).
# Architect-bench GPU arm runner. One arm, one suite, one server, cleaned up after.
# Usage: run_arm.sh <arm> <model> <specflags> <suite> <n> <max_tokens> <repeats>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/architect_bench_gpu_lib.sh"
gpu_require_kernel_label || exit 3
ART="${GPU_BENCH_ART:-/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720}"
RES="${GPU_BENCH_RES:-/mnt/raid0/llm/epyc-inference-research}"

arm="$1"; MODEL="$2"; SPEC="$3"; SUITE="$4"; N="$5"; MAXTOK="$6"; REPS="$7"
d="$ART/runs/${SUITE}/${arm}"
mkdir -p "$d"
# Resume guard: a completed arm/suite is never re-run, so the whole phase
# script is safe to restart after an interruption.
if [ -f "$d/result.json" ]; then
  echo "SKIP $arm/$SUITE (result.json already present)"; exit 0
fi
# Incomplete prior attempt (no result.json): KEEP its partial JSONL — the
# runner's idempotent (id,seed) resume skips collected draws and folds them
# into the totals, so a restart resumes instead of re-querying.
PINNED="$ART/questions_${SUITE}.json"

read -ra SPECA <<< "$SPEC"
gpu_launch "$d" "$MODEL" -np 1 -c 32768 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 "${SPECA[@]}"
st=$(gpu_wait "$d" 900)
if [ "$st" != "HEALTHY" ]; then
  echo "FAIL $arm/$SUITE: $st"; tail -5 "$d/server.stderr"; gpu_kill "$d"; exit 1
fi
rocm-smi --showmemuse --showuse > "$d/rocm_during.txt" 2>/dev/null

# Pin the item set on the first arm; every later arm (incl. the future CPU A2
# session) replays it verbatim so the comparison stays paired.
QARG=(--questions-in "$PINNED")
[ -f "$PINNED" ] || QARG=(--questions-out "$PINNED")

cd "$RES"
HF_HOME=/mnt/raid0/llm/cache/huggingface RUNNER_REQUEST_TIMEOUT_S=1800 \
uv run python scripts/benchmark/v7_quality_gate_runner.py \
  --port 18072 --host 127.0.0.1 \
  --suites "$SUITE" --n "$N" --seed 42 \
  --max-tokens "$MAXTOK" --repeats "$REPS" \
  --temperature 0.6 --top-p 0.95 --top-k 20 --no-enable-thinking \
  --endpoint chat --arm "$arm" \
  --kernel "$KERNEL_LABEL" \
  --binary "$BIN" \
  --models "$MODEL" \
  --per-question-out "$d/per_question.jsonl" \
  "${QARG[@]}" \
  --output "$d/result.json" \
  > "$d/runner.stdout" 2> "$d/runner.stderr"
rc=$?
echo "runner_exit=$rc" > "$d/runner.exit_code"
grep -E "print_timing|prompt eval time|eval time" "$d/server.stderr" | tail -6 > "$d/timings.txt" 2>/dev/null
gpu_kill "$d"
echo "DONE $arm/$SUITE rc=$rc"
[ -f "$d/result.json" ] && python3 -c "
import json;r=json.load(open('$d/result.json'))['suites'][0]
print('  %s acc=%.3f correct=%d/%d trunc=%s err=%s'%(r['suite'],r['accuracy'],r['correct'],r['n'],r.get('truncated'),r.get('errors')))"
