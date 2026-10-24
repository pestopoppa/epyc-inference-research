#!/bin/bash
# Live-affinity check for session.sh (sourced). Pure bash; PROC_ROOT overridable for self-tests.
#
# Why not just read /proc/$PID/status: that is the MAIN THREAD's mask. With OMP_PROC_BIND=spread +
# OMP_PLACES=cores (required by the same block) libgomp binds the master thread to place 0, so the
# main thread legitimately reads "0" under a correct `taskset -c 0-95` (false rc=68, 2026-10-24).
# Correct check: every thread's mask is within the expected set, their union is non-empty, and the
# process-level mask (/proc/PID/status) is within it too. A missing taskset still fails (mask is
# the whole machine, not a subset of CPUS).

# expand "0-3,8,10-11" -> one cpu per line
aff_expand() {
  local list=$1 part a b i
  local IFS=,
  for part in $list; do
    [ -n "$part" ] || continue
    a=${part%-*}; b=${part#*-}
    for ((i=a; i<=b; i++)); do echo "$i"; done
  done
}

# aff_check PID CPUS  -> 0 ok; 1 violation (reason on stdout); 2 unreadable/empty
aff_check() {
  local pid=$1 want=$2 root=${PROC_ROOT:-/proc} f m proc_mask cpu n=0
  declare -A allowed union
  while read -r cpu; do allowed[$cpu]=1; done < <(aff_expand "$want")
  proc_mask=$(awk '/^Cpus_allowed_list:/{print $2}' "$root/$pid/status" 2>/dev/null)
  [ -n "$proc_mask" ] || { echo "no process mask for pid $pid"; return 2; }
  while read -r cpu; do
    [ -n "${allowed[$cpu]:-}" ] || { echo "process mask $proc_mask not within $want (cpu $cpu)"; return 1; }
  done < <(aff_expand "$proc_mask")
  for f in "$root/$pid"/task/*/status; do
    [ -r "$f" ] || continue
    m=$(awk '/^Cpus_allowed_list:/{print $2}' "$f")
    [ -n "$m" ] || continue
    n=$((n+1))
    while read -r cpu; do
      [ -n "${allowed[$cpu]:-}" ] || { echo "thread $(basename "$(dirname "$f")") mask $m not within $want (cpu $cpu)"; return 1; }
      union[$cpu]=1
    done < <(aff_expand "$m")
  done
  [ "$n" -gt 0 ] && [ "${#union[@]}" -gt 0 ] || { echo "no thread masks readable for pid $pid"; return 2; }
  echo "threads=$n union_cpus=${#union[@]} process_mask=$proc_mask within $want"
  return 0
}
