#!/usr/bin/env bash
set -euo pipefail
# Execute only in the approved P-GPU-1 quiet window using the provisional v8 promotion record.
: "${LAGUNA_PGPU1_PROVISIONAL_ATTESTATION_REF:?set provisional promotion attestation reference}"
: "${LAGUNA_PGPU1_PROMOTED_HEAD:?set promoted 40-hex production HEAD}"
: "${LAGUNA_PGPU1_PROMOTED_SERVER_SHA256:?set promoted 64-hex llama-server SHA256}"
cd /mnt/raid0/llm/epyc-inference-research
exec /usr/bin/env -i 'PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' 'LANG=C' 'LC_ALL=C' 'HIP_VISIBLE_DEVICES=0' 'ROCR_VISIBLE_DEVICES=0' '/mnt/raid0/llm/epyc-inference-research/.venv/bin/python3' '/mnt/raid0/llm/epyc-inference-research/scripts/benchmark/laguna_pgpu1_dflash_runner.py' --execute --output-dir 'data/gpu-mi210/laguna-iq2-dflash-pgpu1-v8/run-20260725T184015Z' --binary '/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server' --source-root '/mnt/raid0/llm/llama.cpp' --reps 5 --context 4096 --max-tokens 320 --min-completion-tokens 96 --seed 424242 --attestation-ref "$LAGUNA_PGPU1_PROVISIONAL_ATTESTATION_REF" --expected-production-head "$LAGUNA_PGPU1_PROMOTED_HEAD" --expected-server-sha256 "$LAGUNA_PGPU1_PROMOTED_SERVER_SHA256"
