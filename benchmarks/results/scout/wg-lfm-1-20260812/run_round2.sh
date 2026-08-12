#!/bin/bash
# WG-LFM-1 round 2 — Q8_0 iqk-gated speed arm + the (fixed) correctness arms.
set -uo pipefail
D=/workspace/tmp/wg-lfm-1
LFM4=/mnt/raid0/llm/models/LFM2.5-2.6B-Q4_K_M.gguf
LFM8=/mnt/raid0/llm/models/LFM2.5-2.6B-Q8_0.gguf
GEM=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf

"$D/bench_q8_iqk.sh"
echo "### CORRECTNESS ARMS $(date -Is)"
for m in "$LFM4" "$LFM8" "$GEM"; do
  "$D/correctness_q0.sh" "$m" || echo "CORRECTNESS_FAILED $m rc=$?"
done
echo "### ROUND2_DONE $(date -Is)"
