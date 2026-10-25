#!/bin/bash
# Source only INSIDE official filler8's whole-body taskset. No dry-run bypass.
set -euo pipefail
: "${FILLER_ID:?}" "${FILLER_CPUS:?}" "${FILLER_END:?}" "${FILLER_DEADLINE:?}" "${FILLER_STOPFILE:?}"
case "$FILLER_ID" in codex-ni-v11-086-slow2-synthetic-20261025|codex-ni-v11-086-slow2-build-20261025) ;; *) exit 70;; esac
grant="/mnt/raid0/llm/tmp/compute-flags/grants/$FILLER_ID.grant"
[[ -f "$grant" && ! -f "${grant%.grant}.ended" && ! -e "$FILLER_STOPFILE" ]]
for hold in /mnt/raid0/llm/tmp/compute-flags/HOLD_*; do [[ ! -e "$hold" ]] || exit 70; done
record=$(cat "$grant")
[[ $record =~ ^regions=([^[:space:]]+)[[:space:]]start=([^[:space:]]+)[[:space:]]end=([^[:space:]]+)$ ]] || exit 70
canonical_regions=${BASH_REMATCH[1]}; canonical_start=${BASH_REMATCH[2]}; canonical_end=${BASH_REMATCH[3]}
now=$(date -u +%s)
[[ "$now" -ge "$(date -u -d "$canonical_start" +%s)" && "$now" -lt "$(date -u -d "$canonical_end" +%s)" ]]
[[ "$FILLER_END" -le "$(date -u -d "$canonical_end" +%s)" && "$now" -lt "$FILLER_DEADLINE" ]]
# First interpreter: evidence of actual affinity against BOTH declared and canonical CPUs.
/usr/bin/python3 -B -I - "$FILLER_CPUS" "$canonical_regions" <<'PY'
import json, os, sys

def expand(s):
    cpus = set()
    for chunk in s.split(','):
        if not chunk or not all(x.isdecimal() for x in chunk.split('-')):
            raise ValueError('non-CPU region')
        pair = chunk.split('-')
        if len(pair) not in (1, 2): raise ValueError('bad range')
        lo, hi = int(pair[0]), int(pair[-1])
        if lo > hi: raise ValueError('reversed range')
        cpus.update(range(lo, hi + 1))
    return cpus
actual = set(os.sched_getaffinity(0)); declared, canonical = map(expand, sys.argv[1:])
assert actual and declared and actual <= declared <= canonical, (actual, declared, canonical)
print(json.dumps({'affinity_pass': True, 'actual': sorted(actual), 'declared': sorted(declared), 'canonical': sorted(canonical)}), flush=True)
PY
