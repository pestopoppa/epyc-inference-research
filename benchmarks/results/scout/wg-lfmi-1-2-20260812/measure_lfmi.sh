#!/bin/bash
# WG-LFMI-1: measure GENERATED TOKEN COUNT for LFM2.5-1.2B-Instruct on the five
# WG-LFM-1 reference prompts, using llama.cpp's OWN chat-template application
# (--jinja -st) so the prompt is exactly what production would send.
#
# Why --jinja and not the raw-render harness the 2.6B used: the -Instruct template
# has no enable_thinking kwarg to inject at render time, and the handoff's second
# trap ("a raw-completion harness UNDER-SCORES a model that emits structured
# output") argues for the templated path. The raw render is kept as the control:
# check_render proves the two tokenize identically.
#
# GPU lane only (taskset 184-191, orchestration/stack_topology.yaml:220) — a
# sibling agent owns 0-87,96-183 for a CPU re-embed.
set -euo pipefail
source /workspace/tmp/wg-lfmi/gpuenv.sh

for tag in Q4_K_M Q8_0; do
  case "$tag" in
    Q4_K_M) MODEL="$M4" ;;
    Q8_0)   MODEL="$M8" ;;
  esac
  for q in 1 2 3 4 5; do
    echo "### RUN $tag q${q} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    taskset -c "$GPU_LANE" "$HIPBIN/llama-cli" \
      -m "$MODEL" -ngl 99 -t 8 --jinja -st -v \
      --temp 0 -s 42 -n 512 -c 8192 --no-warmup \
      -p "$(cat "$OUT/raw_q${q}.txt")" \
      > "$OUT/tok_${tag}_q${q}.out" 2> "$OUT/tok_${tag}_q${q}.err"
    echo "    rc=$?"
  done
done
echo "MEASURE_LFMI_DONE"
