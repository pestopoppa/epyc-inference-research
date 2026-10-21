#!/bin/bash
# ABA timing driver per operator ruling 2026-10-21: A1 16:40-16:55 (GPU idle), B 16:55-18:55 (beside f8 G1), A2 18:55-19:05 hard stop.
set -uo pipefail
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
W=/mnt/raid0/llm/tmp/ds41-w5-20261021; R=$W/runs-aba; mkdir -p $R
G1="G1 = llama-server Qwen3.8-27B-Q8_0 + DFlash2 drafter, -np 4, 48.65 GiB VRAM peak, host 184-191 (SMT siblings of 88-95, NUMA3), bursty 60-80k prefills + 256-tok decodes; paired beside workspace-f8 G1 per operator ruling 2026-10-21"
until_utc() { while [ "$(date -u +%H%M)" -lt "$1" ]; do sleep 20; done; }
step() { local name=$1 stop=$2; shift 2
  if [ "$(date -u +%H%M)" -ge "$stop" ]; then echo "$(date -u +%T) skip $name (past $stop)"; return 0; fi
  echo "$(date -u +%T) start $name SEG=$SEG GPUJOB=$GPUJOB"
  $RL run --cpu-list 0-95 --role bench --timeout-s 900 --tag ds41-w5-$name -- "$@" > $R/$name.out 2>&1
  echo "$(date -u +%T) end $name rc=$?"
}
until_utc 1640
export SEG=A1 GPUJOB="none (GPU idle)" NREQ=4
step A1-served-1017 1650 python3 $W/bin/served.py 1017 $R 18621
step A1-served-sept 1652 python3 $W/bin/served.py sept-q4 $R 18622
until_utc 1656
export SEG=B GPUJOB="$G1" NREQ=4
step B-served-1017 1840 python3 $W/bin/served.py 1017 $R 18623
step B-served-sept 1840 python3 $W/bin/served.py sept-q4 $R 18624
step B-bench-1017 1840 $W/bin/bench.sh 1017 $R
step B-bench-sept 1840 $W/bin/bench.sh sept-q4 $R
SEG=B2 step B-served-1017b 1845 python3 $W/bin/served.py 1017 $R 18625
until_utc 1855
export SEG=A2 GPUJOB="none (GPU idle)" NREQ=3
step A2-served-1017 1857 python3 $W/bin/served.py 1017 $R 18626
echo ALLDONE
