#!/bin/bash
# PROMOTED INTO THE REPO 2026-07-29 by `auditor` from a session scratchpad, per
# epyc-root/handoffs/active/architect-model-selection-bench.md § "Follow-up tooling".
# Three corrections were required and are NOT cosmetic — see architect_bench_gpu_lib.sh:
#   1. `source` pointed at another session's /tmp scratchpad (ephemeral: dead on arrival);
#   2. `--kernel production-consolidated-v7` on a v8 binary = false provenance, now REQUIRED;
#   3. the shared lib pinned CORES=88-95, the superseded value (now 184-191).
# E-6: budget-capped native thinking. Force-close <think> at N tokens so the
# model MUST emit an answer, converting R2d's non-termination tail into answers.
# Usage: run_budget.sh <label> <model> <specflags> <budget_N> <n> <max_tokens>
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/architect_bench_gpu_lib.sh"
gpu_require_kernel_label || exit 3
ART=/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720
RES=/mnt/raid0/llm/epyc-inference-research
label="$1"; MODEL="$2"; SPEC="$3"; BUDGET="$4"; N="$5"; MAXTOK="$6"
SUITE=gpqa_diamond_cot
d="$ART/e6_reasoning_budget/${label}_budget${BUDGET}"
mkdir -p "$d"
[ -f "$d/result.json" ] && { echo "SKIP $label budget=$BUDGET"; exit 0; }
[ -f "$d/per_question.jsonl" ] && rm -f "$d/per_question.jsonl"

read -ra SPECA <<< "$SPEC"
MSG=$'\n\n(Reasoning budget reached. I will now state my final answer.)\n'
gpu_launch "$d" "$MODEL" -np 1 -c 32768 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 "${SPECA[@]}" \
  --reasoning on --reasoning-budget "$BUDGET" --reasoning-format deepseek \
  --reasoning-budget-message "$MSG"
st=$(gpu_wait "$d" 900)
if [ "$st" != "HEALTHY" ]; then echo "FAIL $label/$BUDGET: $st"; tail -6 "$d/server.stderr"; gpu_kill "$d"; exit 1; fi

cd "$RES"
HF_HOME=/mnt/raid0/llm/cache/huggingface RUNNER_REQUEST_TIMEOUT_S=3600 \
uv run python scripts/benchmark/v7_quality_gate_runner.py \
  --port 18072 --host 127.0.0.1 --suites "$SUITE" --n "$N" --seed 42 \
  --max-tokens "$MAXTOK" --repeats 1 \
  --temperature 0.6 --top-p 0.95 --top-k 20 --enable-thinking \
  --endpoint chat --arm "${label}_budget${BUDGET}" \
  --kernel "$KERNEL_LABEL" \
  --binary "$BIN" --models "$MODEL" \
  --per-question-out "$d/per_question.jsonl" \
  --questions-in "$ART/questions_gpqa_diamond_cot.json" --limit "$N" \
  --output "$d/result.json" > "$d/runner.stdout" 2> "$d/runner.stderr"
echo "runner_exit=$?" > "$d/runner.exit_code"
gpu_kill "$d"
echo "DONE $label budget=$BUDGET"
[ -f "$d/result.json" ] && python3 -c "
import json;r=json.load(open('$d/result.json'))['suites'][0]
print('  acc=%.1f%% trunc=%s'%(r['accuracy']*100,r.get('truncated')))"
