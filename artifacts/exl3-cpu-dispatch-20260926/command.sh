#!/bin/bash
set -euo pipefail
/workspace/repos/epyc-orchestrator/scripts/region-lock run \
  --cpu-list 95 --timeout-s 5 --role exl3-cpu-probe \
  --tag exl3-2-grouped-runtime-comparison -- \
  taskset -c 95 python3 \
  /workspace/worktrees/exl3-cpu-timing-20260926/experiments/exl3_cpu/run_evidence.py \
  --writer /tmp/exl3-research-contract/scripts/kernel_rnd/exl3/evidence.py \
  --writer-sha256 8caeb33dbb12986fadc385afe25d22bd791b036253c736f9527e67a55f85e268 \
  --binary /tmp/exl3-cpu-timing-build-hex/test_cpu \
  --blas-library /mnt/raid0/llm/epyc-inference-research/.venv/lib/python3.13/site-packages/scipy.libs/libscipy_openblas-6cdc3b4a.so \
  --canonical-bindings /workspace/worktrees/exl3-cpu-timing-20260926/experiments/exl3_cpu/fixtures/canonical_bindings.json \
  --canonical-root /tmp/exl3-research-contract \
  --region-lock-file /mnt/raid0/llm/tmp/cpu_region.exl3-cpu-probe.q3.lock \
  --output /workspace/worktrees/exl3-cpu-timing-20260926/artifacts/exl3-cpu-dispatch-20260926/run-final \
  --bench
