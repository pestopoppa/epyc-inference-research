#!/bin/bash
# memgate.sh <label> <max_wait_s>
# Pre-load gate for ONE DS41-class server: MemAvailable - 300 GiB must stay >= 100 GiB, and no other
# llama-server whose argv names a DS41 GGUF may be alive (read-only /proc scan; never kills anything).
# Exit 0 = go, 1 = still blocked after max_wait_s.
set -uo pipefail
LABEL=$1; MAXW=$2
NEED_KIB=$(( 400 * 1024 * 1024 ))
t0=$(date +%s)
while :; do
  avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  others=""
  for d in /proc/[0-9]*; do
    c=$( { tr '\0' ' ' < "$d/cmdline"; } 2>/dev/null ) || continue
    case "$c" in
      *llama-server*DeepSeek-V4.1-Flash*) others="$others ${d#/proc/}";;
    esac
  done
  if [ "$avail" -ge "$NEED_KIB" ] && [ -z "$others" ]; then
    echo "[memgate $LABEL] GO $(date -u +%FT%TZ) MemAvailable=$((avail/1024/1024))GiB (need>=400) other_ds41_servers=none"
    exit 0
  fi
  if [ $(( $(date +%s) - t0 )) -ge "$MAXW" ]; then
    echo "[memgate $LABEL] BLOCKED $(date -u +%FT%TZ) MemAvailable=$((avail/1024/1024))GiB other_ds41_servers=[${others# }]"
    exit 1
  fi
  sleep 15
done
