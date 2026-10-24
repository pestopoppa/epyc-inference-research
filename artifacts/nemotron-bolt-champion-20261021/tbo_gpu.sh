#!/bin/bash
set -uo pipefail
S=/mnt/raid0/llm/tmp/nemotron-bolt-champion-20261021; cd $S
for arm in cand:/mnt/raid0/llm/kernels/builds/gpu-20261021-59c1b915f-nemotron ctl:/mnt/raid0/llm/kernels/builds/gpu-20261020-dd3d9721c; do
  IFS=: read n b <<<"$arm"
  for op in MUL_MAT_ID ADD ARGSORT TOP_K GET_ROWS SSM_SCAN SSM_CONV; do
    LD_LIBRARY_PATH=$b/bin:/opt/rocm/lib timeout 600 $b/bin/test-backend-ops -b ROCm0 -o $op > tbo_gpu_${n}_$op.log 2>&1
    echo "$n $op rc=$? $(grep -E 'tests passed' tbo_gpu_${n}_$op.log | tail -1)"
  done
done
