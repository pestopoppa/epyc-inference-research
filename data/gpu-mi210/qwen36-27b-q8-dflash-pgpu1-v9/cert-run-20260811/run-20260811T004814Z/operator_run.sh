#!/usr/bin/env bash
set -euo pipefail
# Execute only in the approved P-GPU-1 quiet window using the provisional v9 promotion record.
: "${QWEN36_PGPU1_PROVISIONAL_ATTESTATION_REF:?set provisional promotion attestation reference}"
: "${QWEN36_PGPU1_PROMOTED_HEAD:?set promoted 40-hex production HEAD}"
: "${QWEN36_PGPU1_PROMOTED_SERVER_SHA256:?set promoted 64-hex llama-server SHA256}"
cd /workspace/worktrees/v9-promotion-research
exec /usr/bin/env -i 'PATH=/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' 'LANG=C' 'LC_ALL=C' 'HIP_VISIBLE_DEVICES=0' 'ROCR_VISIBLE_DEVICES=0' '/usr/bin/python3' '/workspace/worktrees/v9-promotion-research/scripts/benchmark/laguna_pgpu1_dflash_runner.py' --execute --output-dir 'data/gpu-mi210/qwen36-27b-q8-dflash-pgpu1-v9/cert-run-20260811/run-20260811T004814Z' --binary '/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server' --source-root '/mnt/raid0/llm/llama.cpp' --reps 5 --context 4096 --max-tokens 512 --min-completion-tokens 96 --seed 424242 --attestation-ref "$QWEN36_PGPU1_PROVISIONAL_ATTESTATION_REF" --expected-production-head "$QWEN36_PGPU1_PROMOTED_HEAD" --expected-server-sha256 "$QWEN36_PGPU1_PROMOTED_SERVER_SHA256"
