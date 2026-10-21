#!/bin/bash
# D3 timing driver (18:40-19:40Z): full host, bench role, one claim per step. ABBA order.
set -uo pipefail
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
W=/mnt/raid0/llm/tmp/ds41-w5-20261021; R=$W/runs; mkdir -p $R
step() { local name=$1; shift
  if [ -e $R/$name.done ]; then echo "skip $name"; return 0; fi
  if [ "$(date -u +%H%M)" -ge 1935 ]; then echo "deadline: stop before $name"; exit 3; fi
  echo "$(date -u +%T) start $name"
  $RL run --cpu-list 0-95 --role bench --gpu-quiet shared --timeout-s 7200 --tag ds41-w5-$name -- "$@" > $R/$name.out 2>&1
  local rc=$?; echo "$(date -u +%T) end $name rc=$rc"
  [ $rc -eq 0 ] && touch $R/$name.done || { echo "FAILED $name"; exit 1; }
}
step served-1017 python3 $W/bin/served.py 1017 $R 18621
step served-sept python3 $W/bin/served.py sept-q4 $R 18622
step bench-sept $W/bin/bench.sh sept-q4 $R
step bench-1017 $W/bin/bench.sh 1017 $R
echo ALLDONE
