#!/bin/bash
# usage: pair.sh <pair#> <A_cpus> <B_cpus> <seedbase> <greedymode>
# Runs arm A (1017 drafter) and arm B (Sept drafter) concurrently, 24 threads each.
set -uo pipefail
P=$1; ACPU=$2; BCPU=$3; SEED=$4; GM=$5
W=/mnt/raid0/llm/tmp/ds41-w6.1-20261021
DA=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf
DB=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-DSpark.gguf
$W/arm.sh p${P}A $DA $((18940+P*2)) $SEED $GM $ACPU 24 &
PA=$!
$W/arm.sh p${P}B $DB $((18941+P*2)) $SEED $GM $BCPU 24 &
PB=$!
trap 'kill $PA $PB 2>/dev/null; wait $PA $PB; exit 143' TERM INT
wait $PA; wait $PB
echo "pair $P done $(date -u +%T)"
