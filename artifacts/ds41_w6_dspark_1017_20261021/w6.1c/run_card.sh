#!/bin/bash
# DS41-W6.1c INNER (inside the launcher's build-role claim, own process group). Strictly SEQUENTIAL arms, one DS41 server
# at a time. Pair p = one plain (target-only) arm + one dspark arm on the same cpus/threads, ABBA order:
# odd p plain-then-dspark, even p dspark-then-plain. Seeds 1000*p+234+prompt (pair 1 = the W6 seeds).
# Modes per arm: card (T=1.0 top_p=1.0, the new arm) + stoch (T=0.7, the W6 reference). Pair 1 also runs greedy
# (text identity plain-vs-dspark). Resumable per arm (runs/<label>/DONE). >=5 complete pairs => COMPLETE.
# Env from the launcher: FILLER_END FILLER_CPUS FILLER_THREADS RUNDIR STOPFILE. Optional: W61_PAIRS (default 5).
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
LIB=${LIB:-$HERE/../../ds41_queue_20261021/lib}
. "$LIB/ds41_guard.sh"; . "$LIB/srv.sh"
NT=${FILLER_THREADS:?}; CPUS=${FILLER_CPUS:?}
NP=${W61_PAIRS:-5}
R=$RUNDIR/results; mkdir -p "$R/runs" "$R/discarded"
log() { echo "$(date -u +%FT%TZ) [w61c] $*"; }
armdone() { [ -e "$R/runs/$1/DONE" ]; }
armthreads() { sed -n 's/.*"threads":\([0-9]*\).*/\1/p' "$R/runs/$1/arm.json" 2>/dev/null; }
cpus_of() { sed -n 's/.*"cpus":"\([^"]*\)".*/\1/p' "$R/runs/$1/arm.json" 2>/dev/null; }
run_arm() { # pair kind  ; returns 0 done/skipped-already, 3 no time/stop, else rc
  local p=$1 kind=$2 label modes est port
  label=p${p}_$kind
  armdone "$label" && return 0
  stopped && { log "STOP file"; return 3; }
  if [ "$kind" = dspark ]; then est=1500; else est=1000; fi
  [ "$p" = 1 ] && est=$(( est + 400 ))
  if [ "$(time_left)" -lt $(( est + 90 )) ]; then log "SKIP $label: $(time_left)s left < ${est}s+90"; return 3; fi
  modes=card,stoch; [ "$p" = 1 ] && modes=card,stoch,greedy
  if [ "$kind" = plain ]; then port=18994; else port=18993; fi
  rm -rf "$R/runs/$label"
  log "ARM $label modes=$modes -t $NT cpus=$CPUS left=$(time_left)s"
  bash "$HERE/arm.sh" "$R/runs/$label" "$kind" "$port" $(( 1000 * p + 234 )) "$modes" 2,3 "$CPUS" "$NT" "$FILLER_END"
  local rc=$?
  if [ $rc = 0 ]; then date -u +%FT%TZ > "$R/runs/$label/DONE"; return 0; fi
  log "ARM $label rc=$rc"; return $rc
}
for p in $(seq 1 "$NP"); do
  if [ $(( p % 2 )) = 1 ]; then ORD="plain dspark"; else ORD="dspark plain"; fi
  # a half-finished pair on different threads/cpus is useless for a speedup: discard the stale arm
  for k in plain dspark; do
    if armdone "p${p}_$k" && { [ "$(armthreads p${p}_$k)" != "$NT" ] || [ "$(cpus_of p${p}_$k)" != "$CPUS" ]; }; then
      other=$([ $k = plain ] && echo dspark || echo plain)
      if ! armdone "p${p}_$other"; then log "discard p${p}_$k (threads/cpus differ from this grant)"; mv "$R/runs/p${p}_$k" "$R/discarded/p${p}_$k.$(date +%s)"; fi
    fi
  done
  for k in $ORD; do
    run_arm "$p" "$k"; rc=$?
    case $rc in 0) ;; 3|75) log "pausing at p$p $k (resume on the next grant)"; break 2;; 4) exit 4;; *) log "arm failed rc=$rc"; exit "$rc";; esac
  done
done
n=0; for p in $(seq 1 "$NP"); do armdone "p${p}_plain" && armdone "p${p}_dspark" && n=$((n+1)); done
log "complete pairs: $n / $NP"
PY=${PY:-/mnt/raid0/llm/epyc-inference-research/.venv/bin/python}
"$PY" -I "$HERE/analyze_card.py" "$R/runs" "$R/analysis_card.json" > "$R/analysis_card.txt" 2>&1 || log "analyze failed (see analysis_card.txt)"
[ "$n" -ge "$NP" ] && { date -u +%FT%TZ > "$RUNDIR/COMPLETE"; log "all pairs done"; }
exit 0
