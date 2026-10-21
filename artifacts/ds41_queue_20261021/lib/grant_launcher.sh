#!/bin/bash
# Generic grant-protocol launcher for DS41-class queue items (shell only, no LLM). Protocol:
#   /mnt/raid0/llm/tmp/compute-flags/FILLERS.md  +  docs/guides/agent-workflows/compute-fillers.md (epyc-root)
# Required env (set by the per-item wrapper): ID INNER RUNDIR MINWIN_S STOPFILE
# Optional env: MIN_PHYS (min physical cores the grant must name), FILLER_GRANTS_DIR FILLER_FLAGS_DIR FILLERS_MD FILLER_POLL
#   FILLER_STUB_CMD (test: run this instead of the claim+INNER), FILLER_NO_CLAIM=1 (test only)
# Behaviour: polls $GRANTS/$ID.grant (45 s). No grant = never runs. Runs ONLY between start and end, under its own
# build-role region-lock claim, INNER in its own setsid group. INNER gets FILLER_END (epoch), FILLER_CPUS,
# FILLER_THREADS, FILLER_RUNDIR, STOPFILE. At end: TERM -> KILL the captured group, verify dead, delete grant,
# write $GRANTS/$ID.ended (rc=<n> time=<UTC> reason=<..>; 124 hard end, 4 refused, 2 unusable grant, 3 window too small,
# 143 launcher signalled). INNER writes $RUNDIR/COMPLETE when all work is done -> .complete + row COMPLETE + exit.
set -uo pipefail
: "${ID:?}" "${INNER:?}" "${RUNDIR:?}" "${MINWIN_S:?}" "${STOPFILE:?}"
FLAGS=${FILLER_FLAGS_DIR:-/mnt/raid0/llm/tmp/compute-flags}
GRANTS=${FILLER_GRANTS_DIR:-$FLAGS/grants}
FILLERS_MD=${FILLERS_MD:-$FLAGS/FILLERS.md}
POLL=${FILLER_POLL:-45}
MIN_PHYS=${MIN_PHYS:-1}
RL=${RL:-/workspace/repos/epyc-orchestrator/scripts/region-lock}
GUARD=${GUARD:-$(dirname "$(readlink -f "$0")")/ds41_guard.sh}
GRANT=$GRANTS/$ID.grant; ENDED=$GRANTS/$ID.ended; COMPLETE=$GRANTS/$ID.complete
mkdir -p "$RUNDIR" "$GRANTS"
ts() { date -u +%FT%TZ; }
log() { echo "$(ts) [$ID] $*"; }
epoch() { date -u -d "$1" +%s 2>/dev/null; }
. "$GUARD"
expand() { local t a b; IFS=, read -ra T <<<"$1"; for t in "${T[@]}"; do case $t in *-*) a=${t%-*}; b=${t#*-}; seq "$a" "$b";; *) echo "$t";; esac; done; }
valid_list() { [[ $1 =~ ^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$ ]] && ! expand "$1" | awk '$1>191{f=1} END{exit !f}'; }
phys_count() { expand "$1" | awk '{print $1%96}' | sort -un | wc -l; }
parse_regions() { # qN quarters (24 cores + SMT siblings) or a literal CPU list; mi210/host-thread tokens are not usable here
  local tok qs="" list="" n a b s e parts=""
  IFS=, read -ra TK <<<"${1//;/,}"
  for tok in "${TK[@]}"; do case $tok in q[0-3]) qs="$qs ${tok#q}";; [0-9]*) list="${list:+$list,}$tok";; *) ;; esac; done
  if [ -n "$list" ]; then CPUS=$list
  elif [ -n "$qs" ]; then for n in $qs; do a=$((n*24)); b=$((a+23)); s=$((a+96)); e=$((s+23)); parts="${parts:+$parts,}$a-$b,$s-$e"; done; CPUS=$parts
  else return 1; fi
  valid_list "$CPUS" || return 1
  THREADS=$(phys_count "$CPUS")
}
write_ended() { # rc reason
  printf 'rc=%s time=%s reason=%s\n' "$1" "$(ts)" "$2" > "$ENDED.tmp" && mv -f "$ENDED.tmp" "$ENDED"
  rm -f "$GRANT"; log "ended rc=$1 reason=$2 (grant deleted)"
}
gone() { ! kill -0 -- "-$1" 2>/dev/null; }
kill_group() { local g=$1 i
  gone "$g" && return 0
  kill -TERM -- "-$g" 2>/dev/null
  for i in $(seq 1 60); do sleep 1; gone "$g" && break; done
  gone "$g" || { kill -KILL -- "-$g" 2>/dev/null; sleep 3; }
  if gone "$g"; then log "group $g dead (verified)"; return 0; else log "STILL ALIVE group $g"; return 1; fi
}
mark_complete() {
  printf 'complete time=%s\n' "$(ts)" > "$COMPLETE"
  [ -f "$FILLERS_MD" ] && sed -i "/^| $ID |/ s/| ARMED |/| COMPLETE |/" "$FILLERS_MD"
  write_ended 0 complete; log "all work done: $COMPLETE written, row marked COMPLETE; exiting for good"
}
exec 7>"$RUNDIR/launcher.lock"
flock -n 7 || { log "another launcher for $ID already running; exiting"; exit 0; }
echo $$ > "$RUNDIR/launcher.pid"
CUR=""; SLP=""
sleep_poll() { sleep "$1" & SLP=$!; wait $SLP 2>/dev/null; }
on_term() { log "launcher signalled"; [ -n "$CUR" ] && kill_group "$CUR"; [ -e "$GRANT" ] && write_ended 143 launcher-signalled
  date -u +%FT%TZ > "$RUNDIR/LAUNCHER_EXITED"; exit 143; }
trap on_term TERM INT
run_one() {
  local regions start end s e now rc reason live
  regions=$(sed -n 's/.*regions=\([^ ]*\).*/\1/p' "$GRANT" | tr -d '\n')
  start=$(sed -n 's/.*start=\([^ ]*\).*/\1/p' "$GRANT" | tr -d '\n')
  end=$(sed -n 's/.*end=\([^ ]*\).*/\1/p' "$GRANT" | tr -d '\n')
  s=$(epoch "$start"); e=$(epoch "$end")
  if [ -z "$regions" ] || [ -z "$s" ] || [ -z "$e" ] || [ "$s" -ge "$e" ]; then write_ended 2 "malformed-grant"; return; fi
  now=$(date -u +%s)
  if [ "$now" -ge "$e" ]; then write_ended 3 "grant-expired-unrun"; return; fi
  if [ "$now" -lt "$s" ]; then GRANT_WAIT=$(( s - now )); return 9; fi
  rm -f "$ENDED"
  if ! parse_regions "$regions"; then write_ended 2 "regions-unusable:$regions"; return; fi
  if [ "$THREADS" -lt "$MIN_PHYS" ]; then write_ended 2 "grant-names-$THREADS-physical-cores-need-$MIN_PHYS"; return; fi
  if [ $(( e - now )) -lt "$MINWIN_S" ]; then write_ended 3 "window-$(( e - now ))s-below-min-${MINWIN_S}s"; return; fi
  if [ -e "$STOPFILE" ]; then write_ended 0 "stop-file-before-start"; return; fi
  live=$(ds41_live_servers)
  if [ -n "$live" ]; then write_ended 4 "refused:ds41-class-server-live:pids=$(echo "$live" | awk '{printf "%s,", $1}')"; return; fi
  log "START regions=$regions cpus=$CPUS threads=$THREADS end=$end"
  export FILLER_END=$e FILLER_CPUS="$CPUS" FILLER_THREADS=$THREADS FILLER_RUNDIR="$RUNDIR" STOPFILE RUNDIR
  rm -f "$RUNDIR/REFUSED"
  if [ -n "${FILLER_STUB_CMD:-}" ]; then setsid bash -c "$FILLER_STUB_CMD" </dev/null &
  elif [ "${FILLER_NO_CLAIM:-0}" = 1 ]; then setsid /bin/bash "$INNER" </dev/null >>"$RUNDIR/inner.out" 2>&1 &
  else setsid "$RL" run --cpu-list "$CPUS" --role build --timeout-s 600 --tag "$ID" -- /bin/bash "$INNER" </dev/null >>"$RUNDIR/inner.out" 2>&1 &
  fi
  CUR=$!
  [ "$(ps -o pgid= -p "$CUR" | tr -d ' ')" = "$CUR" ] || log "WARN: $CUR is not a group leader"
  echo "$CUR" > "$RUNDIR/inner.pgid"
  reason=exited
  while kill -0 "$CUR" 2>/dev/null; do
    if [ "$(date -u +%s)" -ge "$e" ]; then reason=hard-end; kill_group "$CUR"; break; fi
    sleep 3
  done
  wait "$CUR" 2>/dev/null; rc=$?
  [ $reason = hard-end ] && rc=124
  kill_group "$CUR"; CUR=""
  if [ -e "$RUNDIR/COMPLETE" ]; then mark_complete; return 20; fi
  if [ "$rc" = 4 ] && [ -s "$RUNDIR/REFUSED" ]; then write_ended 4 "refused:$(tail -1 "$RUNDIR/REFUSED" | tr ' ' '_' | cut -c1-160)"; return; fi
  write_ended "$rc" "$reason"
}
log "launcher up pid=$$ poll=${POLL}s grants=$GRANTS stop=$STOPFILE minwin=${MINWIN_S}s inner=$INNER"
[ -e "$RUNDIR/COMPLETE" ] && { mark_complete; exit 0; }
while :; do
  [ -e "$COMPLETE" ] && { log "complete marker present; exiting"; exit 0; }
  if [ ! -e "$GRANT" ]; then sleep_poll "$POLL"; continue; fi
  GRANT_WAIT=""; run_one; rc=$?
  [ $rc = 20 ] && exit 0
  if [ $rc = 9 ]; then w=$GRANT_WAIT; [ "$w" -gt "$POLL" ] && w=$POLL; sleep_poll "$w"; fi
done
