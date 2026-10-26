#!/bin/bash
set -euo pipefail
packet=$(cd -- "$(dirname -- "$0")" && pwd)
if ! cd "$packet"; then echo "dryrun cwd refused" >&2; exit 78; fi
if [[ ${DRY_RUN:-0} != 1 || -z ${PREFLIGHT_CPUS:-} ]]; then echo 'pinned owner preflight receipt and CPUs required' >&2; exit 78; fi
if [[ ! $PREFLIGHT_CPUS =~ ^[0-9]+,[0-9]+$ ]]; then echo 'owner preflight CPUs26,27 required' >&2; exit 78; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SHIM_NPROCS=2 LD_PRELOAD=/mnt/raid0/llm/cache/shims/nprocs_shim.so
export LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909/bin:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 GGML_CUDA_DISABLE_GRAPHS=1
unset AUTOKERNEL_CORRECTNESS_CASE_SET GGML_BACKEND_PATH GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS
if ! timeout --kill-after=5s 300s taskset -c "$PREFLIGHT_CPUS" /usr/bin/python3 "$packet/preflight-v3.py"; then echo 'real-launcher DRY_RUN refused' >&2; exit 78; fi
