#!/bin/bash
# usage: ppl.sh ARM CPULIST NTHREADS OUTDIR
set -euo pipefail
source /mnt/raid0/llm/tmp/ds41-w5-20261021/bin/env.sh
ARM=$1; CPUS=$2; NT=$3; OUT=$4; M=$(model_for $ARM)
mkdir -p $OUT
argv=(taskset -c $CPUS numactl --interleave=all $B/llama-perplexity -m $M -f /mnt/raid0/llm/data/wiki.test.raw -c 512 -b 512 --chunks 32 -t $NT -fa on --device none -ngl 0)
printf '%q ' "${argv[@]}" > $OUT/ppl-$ARM.argv
"${argv[@]}" > $OUT/ppl-$ARM.log 2>&1
