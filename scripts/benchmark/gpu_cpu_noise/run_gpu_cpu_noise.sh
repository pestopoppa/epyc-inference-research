#!/bin/bash
# GPU -> CPU timing contamination experiment: grant-driven launcher (shell only, no LLM).
# Pre-registration: epyc-root docs/design/gpu-cpu-noise-contribution-prereg-20261021.md (copied to $RUN/PREREG.md).
# Queue row: /mnt/raid0/llm/tmp/compute-flags/FILLERS.md, id gpu-cpu-noise, class block-timing-quiet.
#
#   run_gpu_cpu_noise.sh dry-run     foreground: driver preflight + full plan, NO inference, NO servers
#   run_gpu_cpu_noise.sh stub-test   foreground: fake grant in a temp grants dir + stub bench/server; proves
#                                    poll, start, end, cleanup, END flag, .ended  (no inference)
#   run_gpu_cpu_noise.sh arm         DETACH the grant poller (setsid nohup ... </dev/null &), pidfile $RUN/launcher.pid
#   run_gpu_cpu_noise.sh status      show pidfile / grant / END
#
# The poller runs ONLY between start and end of grants/gpu-cpu-noise.grant (written by the calendar owner,
# workspace-58): `regions=q0,q1,q2,q3,mi210_0 start=<UTC> end=<UTC hard>`. No grant = never runs. One grant =
# one session, then it waits for the next (until .complete exists). The session runs under
# `region-lock run --gpu-quiet exclusive` (no regions: each CPU rep takes its own 0-95 claim inside
# bench_canonical.sh) and takes the mi210_0 device flock itself. At the hard end (or 3.5 h) it TERMs the
# session's process group (captured, setsid), waits, KILLs, verifies, then ALWAYS writes $RUN/END.
#
# W2-Q38FN (addendum $RUN/PREREG.ADDENDUM-W2-Q38FN.md): prefix any command with --w2, e.g.
#   run_gpu_cpu_noise.sh --w2 {dry-run|stub-test|arm|status}
# Grant id gpu-cpu-noise-w2 (grants/gpu-cpu-noise-w2.grant, regions=q0,q1,q2,q3,mi210_0). ONE persistent Qwen3.8-
# Flash-Next server on 0-95 (production :8074 recipe, v10 store binary) holds its OWN 0-95 claim for its lifetime;
# the session holds gpu-quiet EXCLUSIVE only (hold-once: nothing else claims CPU). Launcher state files carry a -w2
# suffix (launcher-w2.{lock,pid,log}, END-w2, session-w2.pid, STOP-w2) so W1 and W2 never collide. The session is
# refused (rc 2) if the addendum no longer matches its .sha256.
# Test overrides (env): GCN_RUN_DIR GCN_GRANTS_DIR GCN_POLL GCN_STUB=1 GCN_MAX_HOURS GCN_FILLERS_MD
set -uo pipefail
SELF=$(readlink -f "$0")
HERE=$(dirname "$SELF")
W2=0; [ "${1:-}" = --w2 ] && { W2=1; shift; }
RUN=${GCN_RUN_DIR:-/mnt/raid0/llm/tmp/gpu-cpu-noise-20261021}
if [ $W2 = 1 ]; then
  ID=gpu-cpu-noise-w2; SUF=-w2; PREREG=$RUN/PREREG.ADDENDUM-W2-Q38FN.md; MAX_HOURS=${GCN_MAX_HOURS:-1.5}
  WARGS=(--workload w2-q38fn --blocks 5 --blocks-min 4 --run-len 4 --est-rep-s 37 --est-switch-s 25
         --max-hours "$MAX_HOURS" --stop-file "$RUN/STOP-w2")
  STUB_PORTS=(--port 18970 --cpu-port 18971); W2FLAG=--w2
else
  ID=gpu-cpu-noise; SUF=""; PREREG=$RUN/PREREG.md; MAX_HOURS=${GCN_MAX_HOURS:-3.5}
  WARGS=(); STUB_PORTS=(--port 18969); W2FLAG=""
fi
FLAGS=/mnt/raid0/llm/tmp/compute-flags
GRANTS=${GCN_GRANTS_DIR:-$FLAGS/grants}
FILLERS_MD=${GCN_FILLERS_MD:-$FLAGS/FILLERS.md}
POLL=${GCN_POLL:-45}
STOPFILE=$RUN/STOP$SUF
RL=/workspace/repos/epyc-orchestrator/scripts/region-lock
DRIVER=$HERE/gpu_cpu_noise_driver.py
ANALYZE=$HERE/analyze_gpu_cpu_noise.py
PY=/usr/bin/python3
APY=/mnt/raid0/llm/epyc-inference-research/.venv/bin/python; [ -x "$APY" ] || APY=$PY
GRANT=$GRANTS/$ID.grant; ENDED=$GRANTS/$ID.ended; COMPLETE=$GRANTS/$ID.complete
ts() { date -u +%FT%TZ; }
log() { echo "$(ts) [$ID] $*"; }
epoch() { date -u -d "$1" +%s 2>/dev/null; }
mkdir -p "$RUN" "$GRANTS"

write_end() { # rc reason session  -- the END flag is written on EVERY exit path of a session
  printf 'rc=%s time=%s reason=%s session=%s\n' "$1" "$(ts)" "$2" "${3:-none}" > "$RUN/END$SUF.tmp" && mv -f "$RUN/END$SUF.tmp" "$RUN/END$SUF"
}
prereg_ok() { # W2: the addendum must still match the sha256 recorded when it was frozen
  [ $W2 = 0 ] && return 0
  [ -f "$PREREG.sha256" ] && (cd "$(dirname "$PREREG")" && sha256sum -c --status "$(basename "$PREREG").sha256")
}
write_ended() { # rc reason
  printf 'rc=%s time=%s reason=%s\n' "$1" "$(ts)" "$2" > "$ENDED.tmp" && mv -f "$ENDED.tmp" "$ENDED"
  rm -f "$GRANT"; log "ended rc=$1 reason=$2 (grant deleted)"
}
gone() { ! kill -0 -- "-$1" 2>/dev/null; }
kill_group() { local g=$1 i
  gone "$g" && return 0
  kill -TERM -- "-$g" 2>/dev/null
  for i in $(seq 1 90); do sleep 1; gone "$g" && break; done   # the driver cleans its own children on TERM
  gone "$g" || { kill -KILL -- "-$g" 2>/dev/null; sleep 2; }
  if gone "$g"; then log "group $g dead (verified)"; else log "STILL ALIVE group $g"; return 1; fi
}
kill_registry() { # children the driver registered (pid starttime what): kill only if start time still matches
  local f=$1 pid st what cur
  [ -f "$f" ] || return 0
  while read -r pid st what; do
    [ "$st" = None ] && continue
    cur=$(awk '{print $22}' /proc/$pid/stat 2>/dev/null) || continue
    [ "$cur" = "$st" ] || continue
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    sleep 1; [ -d /proc/$pid ] && log "STILL ALIVE registered $pid ($what)" || log "registered $pid ($what) killed (verified)"
  done < "$f"
}
regions_ok() { local r=",$1,"; for t in q0 q1 q2 q3 mi210_0; do [[ $r == *",$t,"* ]] || return 1; done; }

session() { # end_epoch -> rc ; runs ONE session to completion or hard end
  local e=$1 cap now
  S=$(date -u +%Y%m%dT%H%M%SZ); now=$(date -u +%s)
  cap=$(awk -v n="$now" -v h="$MAX_HOURS" 'BEGIN{printf "%d", n+h*3600}')
  [ "$e" -gt "$cap" ] && e=$cap
  rc=1; reason=launcher-exception; CUR=""
  log "SESSION $S start end=$(date -u -d @$e +%FT%TZ)"
  if ! prereg_ok; then
    log "REFUSED: $PREREG does not match $PREREG.sha256"; write_end 2 prereg-sha-mismatch "$S"; return 2
  fi
  local DARGS=(--run-dir "$RUN" --session "$S" --deadline-epoch "$e" --prereg "$PREREG" "${WARGS[@]}")
  if [ "${GCN_STUB:-0}" = 1 ]; then
    setsid $PY "$DRIVER" stub "${DARGS[@]}" --blocks 2 --blocks-min 1 --run-len 2 --warm-cpu 1 --warm-gpu-s 0 \
      --leadin-min-s 0 --stub-rep-s 0.2 "${STUB_PORTS[@]}" ${GCN_STUB_EXTRA:-} > "$RUN/session-$S.out" 2>&1 < /dev/null &
  else
    setsid $RL run --gpu-quiet exclusive --role bench --tag $ID-$S --timeout-s 600 -- \
      $PY "$DRIVER" run "${DARGS[@]}" > "$RUN/session-$S.out" 2>&1 < /dev/null &
  fi
  CUR=$!; echo "$CUR $S" > "$RUN/session$SUF.pid"
  reason=exited
  while kill -0 "$CUR" 2>/dev/null; do
    if [ "$(date -u +%s)" -ge "$e" ]; then reason=hard-end; kill_group "$CUR"; break; fi
    sleep 3
  done
  wait "$CUR" 2>/dev/null; rc=$?
  [ $reason = hard-end ] && rc=124
  kill_group "$CUR"; kill_registry "$RUN/runs/$S/children.pids"
  [ -f "$RUN/runs/$S/END.json" ] && reason="$reason:$(sed -n 's/.*"reason": "\([^"]*\)".*/\1/p' "$RUN/runs/$S/END.json")"
  if [ -s "$RUN/runs/$S/reps.jsonl" ]; then
    $APY "$ANALYZE" "$RUN/runs/$S" > "$RUN/runs/$S/analysis.out" 2>&1 && log "analysis written $RUN/runs/$S/ANALYSIS.md"
    $APY "$ANALYZE" "$RUN/runs/$S" --include-invalid > /dev/null 2>&1
  fi
  write_end "$rc" "$reason" "$S"; CUR=""
  log "SESSION $S done rc=$rc reason=$reason"
  return $rc
}

run_one() { # one grant -> one session
  local regions start end s e now rc
  regions=$(sed -n 's/.*regions=\([^ ]*\).*/\1/p' "$GRANT" | tr -d '\n')
  start=$(sed -n 's/.*start=\([^ ]*\).*/\1/p' "$GRANT" | tr -d '\n')
  end=$(sed -n 's/.*end=\([^ ]*\).*/\1/p' "$GRANT" | tr -d '\n')
  s=$(epoch "$start"); e=$(epoch "$end"); now=$(date -u +%s)
  if [ -z "$regions" ] || [ -z "$s" ] || [ -z "$e" ] || [ "$s" -ge "$e" ]; then write_ended 2 malformed-grant; return 0; fi
  [ "$now" -ge "$e" ] && { write_ended 3 grant-expired-unrun; return 0; }
  [ "$now" -lt "$s" ] && return 9
  rm -f "$ENDED"
  regions_ok "$regions" || { write_ended 2 "regions-must-include-q0,q1,q2,q3,mi210_0:$regions"; return 0; }
  [ -e "$STOPFILE" ] && { write_ended 0 stop-file-before-start; return 0; }
  session "$e"; rc=$?
  if [ $rc = 0 ]; then
    printf 'complete time=%s\n' "$(ts)" > "$COMPLETE"
    [ -f "$FILLERS_MD" ] && sed -i "/^| $ID |/ s/| REGISTERED |/| COMPLETE |/; /^| $ID |/ s/| ARMED |/| COMPLETE |/" "$FILLERS_MD"
  fi
  write_ended $rc "session-rc-$rc"
  return 0
}

CUR=""; S=""
on_term() { # the session in flight (if any) is ours: stop it, verify, and still write END + .ended
  log "launcher signalled"
  if [ -n "$CUR" ]; then
    kill_group "$CUR"; kill_registry "$RUN/runs/$S/children.pids"
    write_end 143 launcher-signalled "$S"; write_ended 143 launcher-signalled
  fi
  exit 143
}
poll_loop() {
  exec 7>"$RUN/launcher$SUF.lock"
  flock -n 7 || { log "another launcher already running; exiting"; exit 0; }
  trap on_term TERM INT
  log "launcher up pid=$$ poll=${POLL}s grants=$GRANTS run=$RUN"
  while :; do
    [ -e "$COMPLETE" ] && { log "complete marker present; exiting"; exit 0; }
    [ -e "$STOPFILE" ] && { log "STOP file present; exiting"; exit 0; }
    if [ -e "$GRANT" ]; then
      run_one; [ $? = 9 ] && { sleep "$POLL" & wait $!; continue; }
      [ "${GCN_ONESHOT:-0}" = 1 ] && exit 0
    else
      sleep "$POLL" & wait $!
    fi
  done
}

case "${1:-}" in
  dry-run)
    S=dry-$(date -u +%Y%m%dT%H%M%SZ)
    prereg_ok || { log "dry-run: $PREREG does not match $PREREG.sha256"; exit 2; }
    $PY "$DRIVER" dry-run --run-dir "$RUN" --session "$S" --prereg "$PREREG" "${WARGS[@]}"; rc=$?
    log "dry-run rc=$rc  (plan: $RUN/runs/$S/timeline.jsonl, record: $RUN/runs/$S/RUN_RECORD.json)"; exit $rc ;;
  stub-test)
    T=$(mktemp -d /mnt/raid0/llm/tmp/gpu-cpu-noise-stubtest.XXXXXX)
    export GCN_RUN_DIR=$T/run GCN_GRANTS_DIR=$T/grants GCN_POLL=1 GCN_STUB=1 GCN_ONESHOT=1 GCN_FILLERS_MD=$T/FILLERS.md
    mkdir -p "$T/run" "$T/grants"
    for f in "$PREREG" "$PREREG.sha256"; do [ -f "$f" ] && cp -f "$f" "$T/run/"; done
    [ -f "$T/run/$(basename "$PREREG")" ] || touch "$T/run/$(basename "$PREREG")"
    printf '| %s | x | x | x | x | x | x | REGISTERED | block-timing-quiet |\n' "$ID" > "$T/FILLERS.md"
    printf 'regions=q0,q1,q2,q3,mi210_0 start=%s end=%s\n' "$(date -u -d '-1 min' +%FT%TZ)" "$(date -u -d '+10 min' +%FT%TZ)" > "$T/grants/$ID.grant"
    bash "$SELF" $W2FLAG _poll; rc=$?
    echo "--- stub-test results in $T"; cat "$T/run/END$SUF" "$T/grants/$ID.ended" 2>&1; ls "$T/grants"; cat "$T/FILLERS.md"
    S=$(ls "$T/run/runs" | head -1)
    ok=1
    [ -f "$T/run/END$SUF" ] && grep -q '^rc=0 ' "$T/run/END$SUF" || ok=0
    [ -f "$T/grants/$ID.ended" ] && [ -f "$T/grants/$ID.complete" ] && [ ! -e "$T/grants/$ID.grant" ] || ok=0
    grep -q "| COMPLETE |" "$T/FILLERS.md" || ok=0
    [ -s "$T/run/runs/$S/reps.jsonl" ] && [ -f "$T/run/runs/$S/ANALYSIS.md" ] || ok=0
    if [ $W2 = 1 ]; then  # the W2 path really ran: W2 rows, the W2 decision block, and the CPU server lifecycle
      grep -q '"workload": "w2-q38fn"' "$T/run/runs/$S/reps.jsonl" || { echo "NO W2 ROWS"; ok=0; }
      grep -q 'W2-Q38FN predeclared decision' "$T/run/runs/$S/ANALYSIS.md" || { echo "NO W2 DECISION"; ok=0; }
      grep -q '"step": "cpu_server_up"' "$T/run/runs/$S/timeline.jsonl" && grep -q '"step": "cpu_server_down"' "$T/run/runs/$S/timeline.jsonl" || { echo "NO CPU SERVER LIFECYCLE"; ok=0; }
      grep -q '"workload": "w2-q38fn"' "$T/run/runs/$S/END.json" || { echo "END.json not W2"; ok=0; }
    fi
    while read -r pid st what; do [ "$(awk '{print $22}' /proc/$pid/stat 2>/dev/null)" = "$st" ] && { echo "LEAK $pid $what"; ok=0; }; done < "$T/run/runs/$S/children.pids"
    echo "STUB-TEST $([ $ok = 1 ] && echo PASS || echo FAIL)  dir=$T"; [ $ok = 1 ] ;;
  arm)
    prereg_ok || { log "REFUSED arm: $PREREG does not match $PREREG.sha256"; exit 2; }
    [ -s "$RUN/launcher$SUF.pid" ] && kill -0 "$(cat "$RUN/launcher$SUF.pid")" 2>/dev/null && { log "already armed pid $(cat "$RUN/launcher$SUF.pid")"; exit 0; }
    setsid nohup bash "$SELF" $W2FLAG _poll < /dev/null >> "$RUN/launcher$SUF.log" 2>&1 &
    echo $! > "$RUN/launcher$SUF.pid"; log "armed: poller pid $(cat "$RUN/launcher$SUF.pid"), log $RUN/launcher$SUF.log" ;;
  _poll) poll_loop ;;
  status)
    echo "pidfile: $(cat "$RUN/launcher$SUF.pid" 2>/dev/null || echo none)"; echo "grant: $(cat "$GRANT" 2>/dev/null || echo none)"
    echo "END: $(cat "$RUN/END$SUF" 2>/dev/null || echo none)"; echo "session: $(cat "$RUN/session$SUF.pid" 2>/dev/null || echo none)" ;;
  *) sed -n '2,28p' "$SELF"; exit 64 ;;
esac
