#!/bin/bash
# run.sh — wrapper for pb_hol_1.py (PB-HOL-1 slot-order head-of-line collision probe).
#   PBHOL_GPU_WINDOW_ID=<window id> ./run.sh --deadline 2026-10-21T16:38:00Z [--dry-run] [other pb_hol_1.py args]
# Execute runs: refused without PBHOL_GPU_WINDOW_ID; wrapped in the CPU region lock for q3 (184-191) like run_kpf18.sh;
# the python runner takes the mi210_0 device claim, owns its single llama-server child by PID, hard-stops at --deadline
# and stops/verifies-dead that child on every exit path. This wrapper signals nothing.
# Opt out of the region lock ONLY with PBHOL_NO_REGION_LOCK="<reason>".
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBE="$HERE/pb_hol_1.py"
REGION_LOCK=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock
CPU_LIST="${PBHOL_CPU_LIST:-184-191}"
args=("$@"); dry=0; have_dl=0
for a in "${args[@]}"; do
    case "$a" in --dry-run) dry=1 ;; --deadline|--deadline=*) have_dl=1 ;; esac
done
[ "$have_dl" = 1 ] || { echo "REFUSING: --deadline is required (hard stop, e.g. 2026-10-21T16:40:00Z)" >&2; exit 64; }
if [ "$dry" = 1 ]; then exec python3 -B "$PROBE" "${args[@]}"; fi
if [ -z "${PBHOL_GPU_WINDOW_ID:-}" ]; then
    echo "REFUSING: PBHOL_GPU_WINDOW_ID is not set (run only inside the coordinated GPU window)" >&2; exit 64
fi
args+=(--gpu-window-id "$PBHOL_GPU_WINDOW_ID")
if [ -n "${PBHOL_NO_REGION_LOCK:-}" ]; then
    echo "[run.sh] region lock SKIPPED: $PBHOL_NO_REGION_LOCK" >&2
    exec python3 -B "$PROBE" "${args[@]}"
fi
[ -x "$REGION_LOCK" ] || { echo "REFUSING: $REGION_LOCK not found" >&2; exit 64; }
exec "$REGION_LOCK" run --cpu-list "$CPU_LIST" --role pbhol1 --tag "pbhol1:$PBHOL_GPU_WINDOW_ID" -- \
    python3 -B "$PROBE" "${args[@]}"
