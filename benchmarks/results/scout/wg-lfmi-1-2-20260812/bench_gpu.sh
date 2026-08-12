#!/bin/bash
# WG-LFMI-2, GPU arms only. Both quants, -fa 0 and -fa 1 (the handoff records
# -fa 0 measuring FASTER decode for gemma4 on gfx90a, so FA is swept, not assumed).
#
# Model ORDER IS ROTATED across the three rounds (Q4,Q8 / Q8,Q4 / Q4,Q8) so any
# monotonic drift in GPU clock or host state cannot be mistaken for a quant effect.
#
# Pinned to the declared GPU host lane 184-191 (stack_topology.yaml:220, -t 8);
# the sibling CPU re-embed is excluded from 184-191 and their SMT siblings 88-95.
set -euo pipefail
source /workspace/tmp/wg-lfmi/gpuenv.sh

ROUND="${1:?usage: bench_gpu.sh <round> <model...>}"; shift

for spec in "$@"; do
  tag="${spec%%:*}"; model="${spec#*:}"
  echo "### $ROUND $tag $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  taskset -c "$GPU_LANE" "$HIPBIN/llama-bench" \
    -m "$model" -ngl 99 -t 8 -fa 0,1 -p 512 -n 128,512 -r 5 -o md \
    > "$OUT/bench_${ROUND}_${tag}.md" 2> "$OUT/bench_${ROUND}_${tag}.err" &
  BPID=$!
  echo "    pid=$BPID mask=$(cat /proc/$BPID/status 2>/dev/null | grep Cpus_allowed_list)"
  # VRAM / KFD residency sampled WHILE THE BENCH PID IS ALIVE — a post-exit sample
  # cannot distinguish never-resident from finished.
  : > "$OUT/vram_${ROUND}_${tag}.log"
  while kill -0 "$BPID" 2>/dev/null; do
    printf '%s pid_alive=1 mask=%s vram_used_B=%s gpu_use_pct=%s kfd_clients=%s kfd_pids=%s\n' \
      "$(date -u +%H:%M:%S)" \
      "$(awk '/Cpus_allowed_list/{print $2}' /proc/$BPID/status 2>/dev/null)" \
      "$(rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '/^card0/{print $3}')" \
      "$(rocm-smi --showuse --csv 2>/dev/null | awk -F, '/^card0/{print $2}')" \
      "$(ls /sys/class/kfd/kfd/proc 2>/dev/null | wc -l)" \
      "$(ls /sys/class/kfd/kfd/proc 2>/dev/null | tr '\n' ',')" \
      >> "$OUT/vram_${ROUND}_${tag}.log"
    sleep 2
  done
  wait "$BPID" || echo "    bench rc=$?"
  echo "    done"
done
echo "BENCH_${ROUND}_DONE"
