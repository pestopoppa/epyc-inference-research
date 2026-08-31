#!/bin/bash
set -u
SC=/tmp/claude-1000/-workspace/e5d41ab3-6589-4d87-823e-2a119375d6da/scratchpad
BIN=/mnt/raid0/llm/tmp/inf68-baseline-tree/build-cpu/bin/llama-bench
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
UD=/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
UNI=/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/IQ4_XS-uniform/Qwen3.8-Flash-Next-IQ4_XS-uniform.gguf
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1

gate() {
  local i=0
  until awk '{exit !($1<10)}' /proc/loadavg || [ "$i" -ge 40 ]; do sleep 10; i=$((i+1)); done
  echo "GATE: waited $((i*10))s load=$(cut -d' ' -f1-3 /proc/loadavg)"
}

contam() {
  # in-window sampler: frame-2 of top (instantaneous), any non-bench process >300% CPU
  while true; do
    top -bn2 -d 0.5 2>/dev/null | awk '
      /^top -/ {c++}
      c==2 && $9+0>300 && $12!~/llama-bench/ {print strftime("%H:%M:%S"), $9, $12}'
    sleep 20
  done
}

arm() {
  local label=$1 model=$2
  gate
  contam > "$SC/inf68-contam-$label.log" & local CP=$!
  "$RL" run --cpu-list 0-95 --role bench --tag "inf68-$label" -- taskset -c 0-95 numactl --interleave=all \
    "$BIN" -m "$model" -t 48,64 -r 5 -p 512 -n 128 -mmp 0 > "$SC/inf68-rerun-$label.log" 2>&1
  local rc=$?
  kill "$CP" 2>/dev/null; wait "$CP" 2>/dev/null
  echo "ARM $label: exit=$rc contam_hits=$(wc -l < "$SC/inf68-contam-$label.log")"
}

arm ud "$UD"
arm uni "$UNI"
echo "ALLDONE"
