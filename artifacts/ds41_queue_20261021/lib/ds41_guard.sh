#!/bin/bash
# Sourced. Read-only DS41-class host guard (ps only; NEVER pkill/pgrep, never kills anything).
# ds41_live_servers : prints "pid args" for every live llama-server whose argv names a DeepSeek-V4.1-Flash model
#                     (superset of the DeepSeek-V4.1-Flash-1017 shards: the Sept Q4 is just as large). Exit 0 always.
# ds41_guard_or_refuse <what> : exit 4 (after writing $RUNDIR/REFUSED) if another DS41-class server is live.
# The first token of argv must end in "llama-server", so this script's own grep/ps and a wrapper whose argv merely
# CONTAINS the words are never matched.
ds41_live_servers() {
  ps -eo pid=,args= 2>/dev/null | grep -E '^ *[0-9]+ +[^ ]*llama-server( |$).*DeepSeek-V4\.1-Flash' || true
}
ds41_guard_or_refuse() {
  local live; live=$(ds41_live_servers)
  if [ -n "$live" ]; then
    local pids; pids=$(echo "$live" | awk '{printf "%s ", $1}')
    echo "REFUSED $(date -u +%FT%TZ) ($1): DS41-class llama-server already live: pids $pids" | tee -a "${RUNDIR:-/tmp}/REFUSED" >&2
    exit 4
  fi
}
# ds41_memgate <label> : MemAvailable must be >= 400 GiB (276 private + headroom) before a 1017 server load.
ds41_memgate() {
  local avail; avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if [ "$avail" -lt $(( 400 * 1024 * 1024 )) ]; then
    echo "REFUSED $(date -u +%FT%TZ) ($1): MemAvailable $((avail/1024/1024)) GiB < 400 GiB" | tee -a "${RUNDIR:-/tmp}/REFUSED" >&2
    exit 4
  fi
}
