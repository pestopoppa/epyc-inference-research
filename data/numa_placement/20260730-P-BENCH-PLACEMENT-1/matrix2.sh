#!/bin/bash
# Placement matrix for the remaining production CPU roles, on the canonical
# instrument (llama-bench, tg128, r=5), no speculative decoding so the arms are
# apples-to-apples and isolate the NUMA placement effect alone.
#
# Arms:
#   A_prod    = exactly as the role is wired today: straddling cpuset, no numactl
#   B_halfint = same cpuset + --interleave over the two nodes it actually spans
#   C_fullint = canonical recipe: full machine 0-95 + --interleave=all
# drop_caches before EVERY arm: numactl binds at first touch only, so a warm
# cache would silently re-measure the previous arm's placement.
LB=/mnt/raid0/llm/llama.cpp/build/bin/llama-bench
OUT=/mnt/raid0/llm/tmp/matrix2_results.txt
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

MODELS='
worker_general_gemma26B_Q4KM|/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
architect_general_qwen35_122B_Q4KM|/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
'

echo "$MODELS" | while IFS='|' read -r MNAME MPATH; do
  [ -z "$MNAME" ] && continue
  [ -f "$MPATH" ] || { echo "MISSING $MPATH" | tee -a "$OUT"; continue; }
  echo "##### $MNAME #####" | tee -a "$OUT"
  for ARM in A_prod B_halfint C_fullint; do
    case "$ARM" in
      A_prod)    CPUSET="0-47,96-143"; PRE="" ;;
      B_halfint) CPUSET="0-47,96-143"; PRE="numactl --interleave=0,1" ;;
      C_fullint) CPUSET="0-95";        PRE="numactl --interleave=all" ;;
    esac
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    echo "--- $ARM (taskset $CPUSET ${PRE:-no-numactl}) ---" | tee -a "$OUT"
    taskset -c "$CPUSET" $PRE "$LB" -m "$MPATH" -t 96 -fa 1 -p 0 -n 128 -r 5 2>&1 \
      | grep -E "tg128" | tee -a "$OUT"
  done
done
echo "=== MATRIX2 DONE ===" | tee -a "$OUT"
