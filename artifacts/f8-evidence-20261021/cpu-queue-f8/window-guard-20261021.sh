#!/bin/bash
# Keep the f8 CPU queue out of the peer's full-host timing window (2026-10-21 ~11:30-14:30Z,
# stage-3-intake-action-plan: candidate C bundle gate, CS-47, CS-11). Until the window opens, any pending job
# whose #maxjob_s would carry it past 11:25Z is parked; during the window everything stays parked;
# after 14:35Z all parked jobs return to pending/. Moves files only; never signals a process.
set -euo pipefail
Q=/mnt/raid0/llm/tmp/cpu-queue-f8
P=/mnt/raid0/llm/tmp/cpu-queue-f8/parked-window-20261021
mkdir -p "$P"
cut_s=$(date -u -d '2026-10-21 11:25:00' +%s)
end_s=$(date -u -d '2026-10-21 14:35:00' +%s)
log() { echo "$(date -u +%FT%TZ) [window-guard] $*" >> "$Q/log/runner.log"; }
log "armed: park jobs that would cross 11:25Z; release after 14:35Z"
while [ "$(date -u +%s)" -lt "$end_s" ]; do
  now=$(date -u +%s)
  for j in "$Q"/pending/*.sh; do
    [ -e "$j" ] || continue
    mx=$(grep -m1 -oE '^#maxjob_s=[0-9]+' "$j" | cut -d= -f2 || true)
    mx=${mx:-3600}
    if [ $((now + mx)) -gt "$cut_s" ]; then
      mv "$j" "$P/" && log "parked $(basename "$j") (max ${mx}s would cross 11:25Z)"
    fi
  done
  sleep 30
done
for j in "$P"/*.sh; do
  [ -e "$j" ] || continue
  mv "$j" "$Q/pending/" && log "released $(basename "$j")"
done
log "done"
