#!/bin/bash
set -uo pipefail
D=/workspace/tmp/wg-lfm-1
for m in /mnt/raid0/llm/models/LFM2.5-2.6B-Q4_K_M.gguf \
         /mnt/raid0/llm/models/LFM2.5-2.6B-Q8_0.gguf \
         /mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf; do
  "$D/correctness2_q0.sh" "$m" || echo "CORRECTNESS2_FAILED $m rc=$?"
done
echo "### ROUND3_DONE $(date -Is)"
