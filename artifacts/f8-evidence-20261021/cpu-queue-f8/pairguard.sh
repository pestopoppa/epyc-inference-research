#!/bin/bash
# Guard for paired jobs running on quarter $1 under another session's claim (PIDs = timeout wrappers I launched).
# If the holder releases the quarter while my jobs still run: take the quarter under MY OWN claim until they end.
# If someone else claims it first: SIGTERM my jobs (captured PIDs only) so no timing run shares their cores.
set -uo pipefail
Q=/mnt/raid0/llm/tmp/cpu-queue-f8; RL=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock
QR=$1; shift; PIDS="$*"
log(){ echo "$(date -u +%FT%TZ) [pairguard $QR] $*" >> "$Q/log/runner.log"; }
alive(){ for p in $PIDS; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
holder=$("$RL" status 2>/dev/null | awk -v q="$QR" '$1==q {print $3, $4}')
log "start guarding PIDs $PIDS on $QR (host claim: $holder)"
while alive; do
  st=$("$RL" status 2>/dev/null | awk -v q="$QR" '$1==q {print $2, $3, $4}')
  case "$st" in
    free*)
      log "host released $QR — taking it under own claim"
      "$RL" run --regions "$QR" --role build --tag "f8q-pair-takeover-$QR" --timeout-s 5 -- bash -c "while $(for p in $PIDS; do printf 'kill -0 %s 2>/dev/null || ' "$p"; done) false; do sleep 15; done" >> "$Q/log/pairguard-$QR.log" 2>&1 && { log "own claim held until jobs ended"; break; }
      log "could not take $QR (claimed by someone else) — stopping my paired jobs"
      for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done; sleep 130
      for p in $PIDS; do kill -0 "$p" 2>/dev/null && { kill -KILL "$p" 2>/dev/null; log "SIGKILL $p"; }; done
      break;;
    HELD*) if [ "$st" != "HELD $holder" ]; then
             log "holder changed to '$st' — stopping my paired jobs"
             for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done; sleep 130
             for p in $PIDS; do kill -0 "$p" 2>/dev/null && kill -KILL "$p" 2>/dev/null; done; break; fi;;
  esac
  sleep 15
done
log "done"
