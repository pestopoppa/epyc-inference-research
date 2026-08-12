#!/bin/bash
# WG-LFM-1 scout correctness probe — deterministic (temp 0, seed 42),
# GGUF-embedded chat template via --jinja (default on in v9).
# Paired with the speed arms so a fast wrong answer cannot pass as a result.
set -euo pipefail
ulimit -c 0

BIN=/mnt/raid0/llm/llama.cpp/build/bin/llama-cli
OUTDIR=/workspace/tmp/wg-lfm-1

export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:/opt/AMD/aocc-compiler-5.0.0/lib:/mnt/raid0/llm/llama.cpp/build/bin:/mnt/raid0/llm/llama.cpp-dflash/build/bin:/opt/rocm/lib
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1

declare -a PROMPTS=(
  "What is the capital city of Japan? Reply with only the city name."
  "Compute 17 * 23. Reply with only the number."
  "List the first five prime numbers as a comma-separated list, nothing else."
  "Return ONLY a JSON object, no prose and no code fence, with keys \"name\" and \"age\" for a person named Ada who is 36 years old."
  "A shelf holds 3 boxes. Each box holds 4 jars. Each jar holds 6 marbles. How many marbles in total? Reply with only the number."
)

m="$1"
tag="$(basename "$m" .gguf)"
: > "${OUTDIR}/correct_${tag}.txt"
for i in "${!PROMPTS[@]}"; do
  {
    echo "### Q$((i+1)): ${PROMPTS[$i]}"
  } >> "${OUTDIR}/correct_${tag}.txt"
  taskset -c 0-23 numactl --membind=0 -- \
    "$BIN" -m "$m" -t 24 -fa 1 --no-mmap -st --no-warmup \
    --temp 0 -s 42 -n 96 -c 4096 -p "${PROMPTS[$i]}" \
    > "${OUTDIR}/.raw_${tag}_$i.txt" 2> "${OUTDIR}/.err_${tag}_$i.txt"
  cat "${OUTDIR}/.raw_${tag}_$i.txt" >> "${OUTDIR}/correct_${tag}.txt"
  echo "" >> "${OUTDIR}/correct_${tag}.txt"
done
echo "CORRECTNESS_DONE ${tag}"
