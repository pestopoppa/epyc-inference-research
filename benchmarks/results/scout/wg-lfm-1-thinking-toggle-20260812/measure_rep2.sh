#!/bin/bash
# WG-LFM-1 follow-up: MEASURE gemma4-26B-A4B token emission with thinking ON vs OFF.
#
# Method: the enable_thinking chat-template kwarg is applied at render time, so the
# prompt is rendered here (render.py, GGUF-embedded template) and fed to
# llama-completion in RAW mode (-no-cnv). No server, no chat-template application
# inside the tool -> the exact production prompt string is what gets tokenized.
#
# Same five prompts, same sampling and same CPU region as the 2026-08-12 scout
# (correctness2_q0.sh): q0 = cores 0-23, membind 0, -t 24, temp 0, seed 42, n 512.
set -euo pipefail
ulimit -c 0

BIN=/mnt/raid0/llm/llama.cpp/build/bin/llama-completion
MODEL=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
OUT=/workspace/tmp/wg-lfm-1-thinking

export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin:/mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1

for t in on off; do
  for q in 1 2 3 4 5; do
    echo "### RUN q${q} think=${t} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    taskset -c 0-23 numactl --membind=0 -- \
      "$BIN" -m "$MODEL" -t 24 -fa 1 --no-mmap --no-warmup -no-cnv \
      --temp 0 -s 42 -n 512 -c 8192 --verbose-prompt \
      -f "${OUT}/prompt_q${q}_think${t}.pf" \
      > "${OUT}/gen2_q${q}_think${t}.out" 2> "${OUT}/gen2_q${q}_think${t}.err"
    echo "    done rc=$?"
  done
done
echo "MEASURE_DONE"
