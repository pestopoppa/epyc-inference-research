#!/bin/bash
# Block-A correctness driver: q0 only, one region-lock claim per step, stops at a step boundary after 11:00Z.
set -uo pipefail
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
W=/mnt/raid0/llm/tmp/ds41-w5-20261021
R=$W/runs; mkdir -p $R
step() { local name=$1; shift
  if [ -e $R/$name.done ]; then echo "skip $name"; return 0; fi
  if [ "$(date -u +%H%M)" -ge 1100 ]; then echo "deadline: stop before $name"; exit 3; fi
  echo "$(date -u +%T) start $name"
  $RL run --cpu-list 0-23 --role build --timeout-s 7200 --tag ds41-w5-$name -- "$@" > $R/$name.out 2>&1
  local rc=$?; echo "$(date -u +%T) end $name rc=$rc"
  [ $rc -eq 0 ] && touch $R/$name.done || { echo "FAILED $name"; exit 1; }
}
step build make -C /mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu llama-perplexity -j 24
step coh-1017 python3 $W/bin/coherence.py 1017 0-23 24 $R 18611
step ppl-1017 $W/bin/ppl.sh 1017 0-23 24 $R
step coh-sept python3 $W/bin/coherence.py sept-q4 0-23 24 $R 18612
step ppl-sept $W/bin/ppl.sh sept-q4 0-23 24 $R
echo ALLDONE
