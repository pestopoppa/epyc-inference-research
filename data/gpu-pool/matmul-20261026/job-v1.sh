#!/bin/bash
set -euo pipefail
cd /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-pool-matmul
[[ -z $(find /mnt/raid0/llm/tmp/compute-flags -maxdepth 1 -name "HOLD_*" -print -quit) ]]
sha256sum -c SEAL.sha256
[[ ${FILLER_CPUS:-} == 184-191 ]]
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SHIM_NPROCS=2 LD_PRELOAD=/mnt/raid0/llm/cache/shims/nprocs_shim.so
export LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909/bin:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 GGML_CUDA_DISABLE_GRAPHS=1
unset AUTOKERNEL_CORRECTNESS_CASE_SET GGML_BACKEND_PATH GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS
exec taskset -c 184-191 /usr/bin/python3 /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-pool-matmul/run-v1.py
