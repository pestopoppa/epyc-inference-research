#!/bin/bash
# sub-gpu-runner chain 7: §5 #1 Q-B pre-registered top-up (+5 launches at B in {1,32}, Q-B arms),
# after chain 6 reports PRBT4B exit. Operator decision 2026-09-16.
set -uo pipefail
O=/tmp/claude-1000/-workspace/3f32cd05-22ee-4db4-ae10-e42cd11f4441/tasks/bz3mtref8.output
until grep -q "PRBT4B exit" "$O" 2>/dev/null; do sleep 30; done
export TMPDIR=/mnt/raid0/llm/tmp
D=/mnt/raid0/llm/tmp/sub-gpu-runner-20260916/s5-topup
mkdir -p "$D"
B=/mnt/raid0/llm/llama.cpp/build-hip/bin
{
  echo "taken $(date -u +%FT%TZ)"
  sha256sum "$B/llama-batched-bench"
  git -C /mnt/raid0/llm/llama.cpp rev-parse HEAD
  git -C /mnt/raid0/llm/llama.cpp status --porcelain --untracked-files=no | wc -l | sed 's/^/tracked_dirty_files=/'
  LD_LIBRARY_PATH="$B:/opt/rocm/lib" bash /workspace/repos/epyc-inference-research/scripts/utils/verify_ggml_linkage.sh \
    "$B/llama-batched-bench" /mnt/raid0/llm/llama.cpp
  echo "linkage_rc=$?"
} > "$D/linkage_receipt.txt" 2>&1
if ! grep -q "^linkage_rc=0$" "$D/linkage_receipt.txt"; then
  echo "S5TOPUP exit 5 (linkage receipt FAILED) $(date -u +%FT%TZ)"; exit 0
fi
cd /mnt/raid0/llm/worktrees/sub-gpu-runner-epyc-inference-research/scripts/benchmark
python3 -u /mnt/raid0/llm/tmp/sub-gpu-runner-20260916/s5_topup.py > "$D/run.log" 2>&1
echo "S5TOPUP exit $? $(date -u +%FT%TZ)"
