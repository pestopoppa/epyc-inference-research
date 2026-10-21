#!/bin/bash
# workspace-f8 CPU correctness queue: runs /mnt/raid0/llm/tmp/cpu-queue-f8/pending/*.sh one at a time per worker,
# each under its own region-lock claim on ONE free quarter (role build). Blackout 00:15Z-06:30Z (exclusive timing
# window) and stop for good at STOP_AT. Job scripts must be self-contained, foreground, and finish within MAXJOB_S.
set -euo pipefail
Q=/mnt/raid0/llm/tmp/cpu-queue-f8
mkdir -p "$Q/pairlock"
RL=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock
WORKER=${1:?worker id}
STOP_AT=$(date -u -d "${STOP_AT:-2026-10-21T20:00:00Z}" +%s)
MAXJOB_S=${MAXJOB_S:-7200}
log(){ echo "$(date -u +%FT%TZ) [w$WORKER] $*" >> "$Q/log/runner.log"; }
in_blackout(){  # 00:15Z..06:30Z, plus never start a job that would cross 00:15Z
  local now; now=$(date -u +%s); local d; d=$(date -u +%F)
  local b0; b0=$(date -u -d "$d 00:15:00" +%s); local b1; b1=$(date -u -d "$d 06:30:00" +%s)
  local nb0; nb0=$(date -u -d "$d 00:15:00 +1 day" +%s)
  if [ "$now" -ge "$b0" ] && [ "$now" -lt "$b1" ]; then return 0; fi
  return 1
}
log "start"
while [ "$(date -u +%s)" -lt "$STOP_AT" ]; do
  if in_blackout; then sleep 60; continue; fi
  job=""
  for f in "$Q"/pending/*.sh; do [ -e "$f" ] || break
    fmax=$(grep -m1 -oE '^#maxjob_s=[0-9]+' "$f" | cut -d= -f2 || true); fmax=${fmax:-$MAXJOB_S}
    tnow=$(date -u +%s); nb=$(date -u -d "$(date -u +%F) 00:15:00 +1 day" +%s)
    [ $((tnow + fmax)) -gt "$nb" ] && continue   # cannot finish before the next 00:15Z blackout: leave it, try the next job
    if mv "$f" "$Q/running/" 2>/dev/null; then job="$Q/running/$(basename "$f")"; break; fi
  done
  if [ -z "$job" ]; then sleep 60; continue; fi
  q=""
  while [ -z "$q" ]; do
    if in_blackout; then mv "$job" "$Q/pending/"; job=""; break; fi
    q=$("$RL" status 2>/dev/null | awk '$2=="free" && $1 ~ /^q[0-3]$/ {print $1; exit}')
    if [ -z "$q" ] && [ -f "$Q/PAIR_OK" ]; then
      # pairing: a quarter held by a peer's build-role (correctness) claim, never bench/kpf18/timing roles, never q3 (GPU host threads)
      for pq in $("$RL" status 2>/dev/null | awk '$2=="HELD" && $3 ~ /^build\[/ && $1 ~ /^q[0-2]$/ {print $1}'); do
        if mkdir "$Q/pairlock/$pq" 2>/dev/null; then q="pair:$pq"; break; fi   # at most ONE paired job per quarter
      done
      [ -n "$q" ] && break
    fi
    [ -n "$q" ] || sleep 60
  done
  [ -n "$job" ] || continue
  name=$(basename "$job" .sh)
  jmax=$(grep -m1 -oE '^#maxjob_s=[0-9]+' "$job" | cut -d= -f2 || true); jmax=${jmax:-$MAXJOB_S}
  nb0=$(date -u -d "$(date -u +%F) 00:15:00 +1 day" +%s); now=$(date -u +%s)
  if [ $((now + jmax)) -gt "$nb0" ] && [ "$now" -lt "$nb0" ] && [ "$now" -ge "$(date -u -d "$(date -u +%F) 06:30:00" +%s)" ]; then
    mv "$job" "$Q/pending/"; log "defer $name (needs ${jmax}s, would cross 00:15Z)"; sleep 60; continue
  fi
  log "run $name on $q (max ${jmax}s)"
  rc=0
  if [[ "$q" == pair:* ]]; then
    pq=${q#pair:}; case "$pq" in q0) cpus=0-23,96-119;; q1) cpus=24-47,120-143;; q2) cpus=48-71,144-167;; esac
    holder=$("$RL" status 2>/dev/null | awk -v q="$pq" '$1==q {print $3, $4}')
    echo "ran on $pq ($cpus) under '$holder' claim by agreement (pairing, nice 10)" > "$Q/log/$name.log"
    timeout --signal=TERM --kill-after=120 "$jmax" nice -n 10 taskset -c "$cpus" env F8Q_CPUS="$cpus" bash "$job" >> "$Q/log/$name.log" 2>&1 &
    jpid=$!
    # a paired job holds no claim of its own: guard it (take the quarter if the host releases it, else stop the job)
    setsid nohup "$Q/pairguard.sh" "$pq" "$jpid" >> "$Q/log/pairguard-$pq.out" 2>&1 < /dev/null &
    wait "$jpid" || rc=$?
  else
    case "$q" in q0) cpus=0-23,96-119;; q1) cpus=24-47,120-143;; q2) cpus=48-71,144-167;; q3) cpus=72-95,168-191;; esac
    timeout --signal=TERM --kill-after=120 "$jmax" "$RL" run --regions "$q" --role build --tag "f8q-$name" --timeout-s 120 -- env F8Q_CPUS="$cpus" taskset -c "$cpus" bash "$job" > "$Q/log/$name.log" 2>&1 || rc=$?
  fi
  [[ "$q" == pair:* ]] && rmdir "$Q/pairlock/${q#pair:}" 2>/dev/null || true
  log "end $name rc=$rc"
  if [ "$rc" = 75 ] || grep -q "could not acquire\|timed out acquiring" "$Q/log/$name.log"; then mv "$job" "$Q/pending/"; log "requeued $name (claim lost)"; else mv "$job" "$Q/done/"; echo "$rc" > "$Q/done/$name.rc"; fi
done
log "stop"
