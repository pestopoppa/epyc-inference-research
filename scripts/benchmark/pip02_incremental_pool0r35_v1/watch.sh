#!/bin/bash
set -euo pipefail
SRC=$(dirname "$(readlink -f "$0")")
DEPLOY=$(dirname "$SRC")
: "${PIP_DEPLOY_MANIFEST_SHA:?}"
W=/mnt/raid0/llm/tmp/codex3-pip02-phasec-20261029/scoring-shards-20261030/paddle-watch-r35
P=/mnt/raid0/llm/tmp/codex3-kpf-item4-control-20261030/prep
mkdir -p "$W"
exec 9>"$W/watcher.lock"
flock -n 9 || exit 1
check_source() {
 printf '%s  %s\n' "$PIP_DEPLOY_MANIFEST_SHA" "$DEPLOY/SHA256SUMS" | sha256sum --check --status
 (cd "$DEPLOY" && sha256sum --quiet -c SHA256SUMS)
}
check_source
while :; do
 check_source
 for pair in 122,123 26,27 50,51 74,75 2,3; do
  if /usr/bin/python3 -c 'import sys;sys.path.insert(0,sys.argv[1]);import prep_admission;prep_admission.live(tuple(map(int,sys.argv[2].split(","))))' "$P" "$pair" 2>/dev/null; then
   scan="$W/discovery.tmp.$$"
   if flock -n -E 99 /mnt/raid0/llm/tmp/compute-flags/prep-locks/26-27.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c "$pair" /usr/bin/python3 "$P/prep_supervisor.py" 900 -- /usr/bin/python3 -B "$SRC/epoch_watch.py" list paddle >"$scan" 2>"$W/discovery.err"; then
    mv "$scan" "$W/discovery.latest"
    while read -r n epoch; do
     [[ "$n" =~ ^[0-9]+$ && "$epoch" =~ ^[0-9a-f]{64}$ ]] || continue
     check_source
     if ! /usr/bin/python3 -c 'import sys;sys.path.insert(0,sys.argv[1]);import prep_admission;prep_admission.live(tuple(map(int,sys.argv[2].split(","))))' "$P" "$pair" 2>/dev/null; then break; fi
     flock -n -E 99 /mnt/raid0/llm/tmp/compute-flags/prep-locks/26-27.lock /bin/bash -c '
      set -euo pipefail
      src=$1; pair=$2; n=$3; epoch=$4; prep=$5
      d=$(taskset -c "$pair" /usr/bin/python3 -B "$src/epoch_watch.py" prepare paddle "$n" "$epoch")
      set +e
      env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_SCORE_OUTPUT="$d" PYTHONPATH=/mnt/raid0/llm/omnidocbench/.venv/lib/python3.11/site-packages taskset -c "$pair" /usr/bin/python3 "$prep/prep_supervisor.py" 900 -- /home/daniele/.local/share/uv/python/cpython-3.11.17-linux-x86_64-gnu/bin/python3.11 -S "$src/run_one.py" "$n" >"$d/run.log" 2>&1
      rc=$?
      /usr/bin/python3 - "$d" "$rc" <<"PY"
import json,sys,datetime
from pathlib import Path
p=Path(sys.argv[1]);x={"returncode":int(sys.argv[2]),"ended_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"receipt_present":bool(list(p.glob("*.score-receipt.json")))}
with (p/"execution-status.json").open("x") as f:json.dump(x,f,sort_keys=True);f.write("\n")
PY
      exit "$rc"
     ' bash "$SRC" "$pair" "$n" "$epoch" "$P" || true
    done < "$W/discovery.latest"
   else
    rm -f "$scan"
   fi
   break
  fi
 done
 sleep 30
done
