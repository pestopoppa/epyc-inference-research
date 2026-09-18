#!/bin/bash
# sub-gpu-runner chain 8: ERNIE ROCm precision-variant A/B (RUNNER_RECIPE.md §8, sd.cpp f3a7fe95),
# after chain 7 reports S5TOPUP exit. Operator-approved (relayed by the coordinator 2026-09-16).
# Never touches the prod CPU sd-server (pid 910274); the driver signals only its own captured PIDs.
set -uo pipefail
O=/tmp/claude-1000/-workspace/3f32cd05-22ee-4db4-ae10-e42cd11f4441/tasks/b7wj4z3dq.output
until grep -q "S5TOPUP exit" "$O" 2>/dev/null; do sleep 30; done
export TMPDIR=/mnt/raid0/llm/tmp
D=/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/ernie-variants
mkdir -p "$D"
# Every batched-bench the top-up launched (PIDs captured in launches.jsonl) must be dead.
L=/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/s5-topup/launches.jsonl
alive=0
if [ -f "$L" ]; then
  for p in $(python3 -c "import json,sys;[print(json.loads(l)['pid']) for l in open('$L')]"); do
    if ps -p "$p" -o args= 2>/dev/null | grep -q llama-batched-bench; then echo "top-up pid $p ALIVE"; alive=1; fi
  done
fi
echo "topup_pids_alive=$alive $(date -u +%FT%TZ)" > "$D/pre_launch_check.txt"
if [ "$alive" != 0 ]; then echo "ERNIEVAR exit 6 (top-up pid alive) $(date -u +%FT%TZ)"; exit 0; fi
cd /mnt/raid0/llm/worktrees/sub-gpu-prep-stable-diffusion.cpp
runner/ernie_rocm_variants_ab.py --dry-run > "$D/dryrun.log" 2>&1
echo "dryrun rc=$?" >> "$D/dryrun.log"
# The driver refuses in preflight when <30 GiB VRAM is free (before claiming); retry up to 60 min.
rc=1
for i in $(seq 1 60); do
  runner/ernie_rocm_variants_ab.py --out "$D" >> "$D/driver.stdout.log" 2>&1
  rc=$?
  [ -f "$D/verdict.json" ] && break
  echo "attempt $i rc=$rc no verdict; retry in 60s $(date -u +%FT%TZ)" >> "$D/driver.stdout.log"
  sleep 60
done
echo "ERNIEVAR exit $rc $(date -u +%FT%TZ)"
