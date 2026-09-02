#!/usr/bin/env bash
# numa_placement_check.sh — prove (or disprove) NUMA placement of a running process.
#
# WHY: `numactl --interleave=all` is a hint the kernel silently abandons for any
# node that has no free pages, so the command line is NOT evidence of even
# placement. The 2026-09-02 instance (INF-70/C7): a 98 GB model launched under
# the canonical recipe landed 57.7/10.7/8.0/17.7 GB across nodes 0-3 and decode
# measured -25%. Nothing in the invocation, the logs, or the exit status said so.
# The only evidence is `numastat -p <pid>` sampled WHILE the process holds the
# model. Run this against every CPU model process during its measured window;
# `scripts/utils/numa_evict.py` is the prophylactic that makes it pass.
#
# Usage:
#   numa_placement_check.sh <pid> [--threshold PCT] [--min-total-mb MB]
#   numa_placement_check.sh --numastat-file FILE [--threshold PCT] ...   # offline/tests
#
# Options:
#   --threshold PCT     fail if the largest node's share exceeds PCT (default: 40)
#   --min-total-mb MB   below this total the sample is too small to judge and the
#                       check reports ADVISORY / exits 0 (default: 1024)
#   --label TEXT        free-form tag printed in the header (e.g. the bench arm)
#
# Exit codes:
#   0  placement acceptable (or sample too small to judge — reported as ADVISORY)
#   2  usage error, missing pid, or numastat unavailable/unparseable
#   3  SKEW: the largest node's share exceeds the threshold
#
# On an even 4-node interleave the expected share is 25%; the 40% default leaves
# room for the non-weight allocations (KV cache, scratch, the loader's own heap)
# while still catching the 61% failure that motivated this check.

set -uo pipefail

PID=""
NUMASTAT_FILE=""
THRESHOLD=40
MIN_TOTAL_MB=1024
LABEL=""

usage() { sed -n '2,30p' "$0" >&2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threshold)      THRESHOLD="$2"; shift 2 ;;
        --min-total-mb)   MIN_TOTAL_MB="$2"; shift 2 ;;
        --numastat-file)  NUMASTAT_FILE="$2"; shift 2 ;;
        --label)          LABEL="$2"; shift 2 ;;
        -h|--help)        usage; exit 0 ;;
        -*)               echo "ERROR: unknown option $1" >&2; usage; exit 2 ;;
        *)                PID="$1"; shift ;;
    esac
done

# --- acquire the numastat sample ---------------------------------------------
if [[ -n "$NUMASTAT_FILE" ]]; then
    [[ -f "$NUMASTAT_FILE" ]] || { echo "ERROR: no such file: $NUMASTAT_FILE" >&2; exit 2; }
    NUMASTAT_OUT="$(cat "$NUMASTAT_FILE")"
    SRC="file:$NUMASTAT_FILE"
elif [[ -n "$PID" ]]; then
    [[ "$PID" =~ ^[0-9]+$ ]] || { echo "ERROR: pid must be numeric, got '$PID'" >&2; exit 2; }
    [[ -d "/proc/$PID" ]] || { echo "ERROR: no such process: $PID" >&2; exit 2; }
    command -v numastat >/dev/null 2>&1 || { echo "ERROR: numastat not on PATH" >&2; exit 2; }
    NUMASTAT_OUT="$(numastat -p "$PID" 2>/dev/null)"
    SRC="pid:$PID"
else
    echo "ERROR: give a <pid> or --numastat-file FILE" >&2; usage; exit 2
fi

echo "=== NUMA placement check ($SRC${LABEL:+, $LABEL}) at $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "$NUMASTAT_OUT"

# --- Rss / AnonHugePages (live pid only) -------------------------------------
if [[ -z "$NUMASTAT_FILE" && -r "/proc/$PID/status" ]]; then
    echo "--- /proc/$PID/status + smaps_rollup ---"
    grep -E '^(VmRSS|RssAnon|RssFile|VmSwap):' "/proc/$PID/status" || true
    if [[ -r "/proc/$PID/smaps_rollup" ]]; then
        grep -E '^(Rss|AnonHugePages|Anonymous):' "/proc/$PID/smaps_rollup" || true
    fi
fi

# --- per-node share ----------------------------------------------------------
# The `Total` row of `numastat -p` carries per-node MB followed by the grand
# total. Parse the LAST such row (a sampler log can hold several).
SUMMARY="$(
    printf '%s\n' "$NUMASTAT_OUT" | awk -v thr="$THRESHOLD" -v mintot="$MIN_TOTAL_MB" '
        $1 == "Total" && NF >= 3 {
            n = 0
            for (i = 2; i < NF; i++) v[n++] = $i + 0
            total = $NF + 0
            found = 1
        }
        END {
            if (!found) { print "PARSE_FAIL"; exit }
            if (total <= 0) { print "PARSE_FAIL"; exit }
            sum = 0
            for (i = 0; i < n; i++) sum += v[i]
            # trust the summed nodes over the printed grand total
            if (sum > 0) total = sum
            maxshare = 0; maxnode = -1
            line = ""
            for (i = 0; i < n; i++) {
                share = 100.0 * v[i] / total
                line = line sprintf("  node %d: %10.2f MB  %5.1f%%\n", i, v[i], share)
                if (share > maxshare) { maxshare = share; maxnode = i }
            }
            even = 100.0 / n
            printf "OK %d %.2f %d %.1f %.1f\n%s", n, total, maxnode, maxshare, even, line
        }
    '
)"

if [[ "${SUMMARY%% *}" != "OK" ]]; then
    echo "ERROR: could not parse a 'Total' row out of the numastat output" >&2
    exit 2
fi

read -r _ NNODES TOTAL_MB MAXNODE MAXSHARE EVEN <<<"$(printf '%s\n' "$SUMMARY" | head -1)"
printf '%s\n' "$SUMMARY" | tail -n +2

echo "--- verdict ---"
echo "nodes=$NNODES total=${TOTAL_MB}MB even_share=${EVEN}% max=node${MAXNODE}@${MAXSHARE}% threshold=${THRESHOLD}%"

if awk -v t="$TOTAL_MB" -v m="$MIN_TOTAL_MB" 'BEGIN{exit !(t < m)}'; then
    echo "ADVISORY: total resident ${TOTAL_MB} MB is below --min-total-mb ${MIN_TOTAL_MB}; sample too small to judge placement."
    exit 0
fi

if awk -v s="$MAXSHARE" -v t="$THRESHOLD" 'BEGIN{exit !(s > t)}'; then
    echo "SKEW: node${MAXNODE} holds ${MAXSHARE}% of ${TOTAL_MB} MB (> ${THRESHOLD}%)."
    echo "      --interleave=all was not honoured. Run scripts/utils/numa_evict.py and reload;"
    echo "      any timing taken in this window is NOT a valid measurement."
    exit 3
fi

echo "PASS: placement within threshold."
exit 0
