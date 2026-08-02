#!/bin/bash
set -uo pipefail
i=0
while [ ! -f /mnt/raid0/llm/tmp/gpu_coresidency/.stop_load ]; do
  t0=$(date +%s.%N)
  out=$(curl -s --max-time 300 http://127.0.0.1:9001/inference \
        -F file=@/mnt/raid0/llm/whisper.cpp/samples/jfk.wav \
        -F temperature=0.0 -F response_format=json 2>&1)
  t1=$(date +%s.%N)
  echo "whisper rep=$i wall=$(echo "$t1 - $t0" | bc) chars=${#out}"
  i=$((i+1))
done
echo "whisper_loop finished after $i transcriptions"
