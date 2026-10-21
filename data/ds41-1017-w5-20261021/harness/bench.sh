#!/bin/bash
set -euo pipefail
source /mnt/raid0/llm/tmp/ds41-w5-20261021/bin/env.sh
ARM=$1; OUT=$2; M=$(model_for $ARM); mkdir -p $OUT
argv=(taskset -c 0-95 numactl --interleave=all $B/llama-bench -m $M -t 48 -fa 1 -mmp 0 -p 512 -n 128 -r 3 -o json)
printf '%q ' "${argv[@]}" > $OUT/bench-$ARM.argv
"${argv[@]}" > $OUT/bench-$ARM.json 2> $OUT/bench-$ARM.err
