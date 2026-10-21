#!/bin/bash
#maxjob_s=5400  (dry-run est 12m x1.5 too tight: runner ran ~3x low on the 2026-10-20 CPU run; padded to measured actual x~1.7)
set -euo pipefail
H=/mnt/raid0/llm/tmp/kv-prefix-fork/p1r-20261020/kpf18cpu
OUT=$H/rerun-20261021/item5
mkdir -p "$OUT"
cd "$H"
rc=0
nice -n 19 python3 -B "$H/kpf18_gate.py" item5 --cpu \
  --fork-bin /mnt/raid0/llm/worktrees/kv-prefix-fork-p1r-20261020/build-cpu/bin/llama-server \
  --fork-bin-prefill-budget /mnt/raid0/llm/worktrees/kv-prefix-fork-p1r-20261020/build-cpu/bin/llama-server \
  --champion-bin /mnt/raid0/llm/tmp/kv-prefix-fork/champ-e3322/build-cpu/bin/llama-server \
  --argv-file "$H/argv.frontdoor-cpu-gate.txt" --out "$OUT" --launch-prefix "taskset -c ${F8Q_CPUS:-72-95}" --port 18973 \
  --fork-min 1024 --n-predict 128 --item7-n-predict 128 --allow-shape-drift \
  --prompt-set $H/run-20261020T163128Z/prompt_set.jsonl --item5-groups 0,3 --budget-min 600 || rc=$?
echo "RESULT 42-kpf18-item5 rc=$rc"
exit $rc
