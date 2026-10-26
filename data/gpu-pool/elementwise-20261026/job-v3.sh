#!/bin/bash
set -euo pipefail
packet=/mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise
approval=$packet/MAIN-APPROVED-RECOVERY-PACKET
if [[ ! -f $approval || -L $approval ]]; then echo 'MAIN bound approval absent or unsafe' >&2; exit 78; fi
if ! packet_sha=$(sha256sum "$packet/SEAL.sha256"); then echo 'packet hash refused' >&2; exit 78; fi
packet_sha=${packet_sha%% *}
if ! grep -Fxq 'proposal=approved' "$approval"; then echo 'MAIN approval refused' >&2; exit 78; fi
if ! grep -Fxq "packet-sha256=$packet_sha" "$approval"; then echo 'MAIN packet binding refused' >&2; exit 78; fi
if ! (cd "$packet" && sha256sum -c SEAL.sha256); then echo 'seal refused' >&2; exit 78; fi
cd /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise
if ! holds=$(find /mnt/raid0/llm/tmp/compute-flags -maxdepth 1 -name 'HOLD_*' -print -quit); then echo 'HOLD scan refused' >&2; exit 78; fi
if [[ -n $holds ]]; then echo 'HOLD active' >&2; exit 78; fi
if ! [[ ${FILLER_CPUS:-} == 184-191 ]]; then echo 'CPU grant refused' >&2; exit 78; fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export SHIM_NPROCS=2 LD_PRELOAD=/mnt/raid0/llm/cache/shims/nprocs_shim.so
export LD_LIBRARY_PATH=/mnt/raid0/llm/tmp/build-champion-c463f601b-hip-20260909/bin:/opt/rocm/lib
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 HSA_OVERRIDE_CPU_AFFINITY_DEBUG=0 GGML_CUDA_DISABLE_GRAPHS=1
unset AUTOKERNEL_CORRECTNESS_CASE_SET GGML_BACKEND_PATH GGML_CUDA_FORCE_MMQ GGML_CUDA_FORCE_CUBLAS
for script in job-v3.sh launch-v3.sh; do
 if ! timeout --kill-after=5s 10s taskset -c 184-191 /bin/bash -n "$script"; then echo "shell syntax refused: $script" >&2; exit 78; fi
done
if ! timeout --kill-after=5s 30s taskset -c 184-191 /usr/bin/python3 fixtures-v3.py run-v3.py > native-preflight-v3.log 2>&1; then echo "sampler preflight refused" >&2; exit 78; fi
exec taskset -c 184-191 /usr/bin/python3 /mnt/raid0/llm/tmp/codex-ni-20261026-gpu-sampler-exit-recovery/v3/elementwise/run-v3.py
