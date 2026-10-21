#!/bin/bash
# run_kpf13.sh — wrapper for kpf13_restore_ab.py (KPF-13.1 / 13.1a: drafter restore vs recompute A/B at the fork).
#
# Correctness class (token identity and draft counts, no timing claim): it takes a BUILD-role region-lock claim on
# KPF13_CPU_LIST (a CPU quarter, e.g. 72-95; on a GPU run the host threads, e.g. 184-191) and needs NO quiet host.
# It refuses unless KPF13_WINDOW_ID names the grant/window it runs in, and the driver refuses to execute without a
# hard --deadline-utc. --dry-run takes no lock and starts nothing.
#
#   KPF13_WINDOW_ID=<grant id> KPF13_CPU_LIST=72-95 run_kpf13.sh --out DIR --deadline-utc 2026-10-23T06:00:00Z
#   KPF13_WINDOW_ID=x KPF13_CPU_LIST=72-95 run_kpf13.sh --out DIR --dry-run
#
# This script signals nothing: the python driver owns its one llama-server child by PID.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$HERE/kpf13_restore_ab.py"
REGION_LOCK="${KPF13_REGION_LOCK:-/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock}"

[ -n "${KPF13_WINDOW_ID:-}" ] || { echo "REFUSING: KPF13_WINDOW_ID is not set (name the grant / window id)" >&2; exit 64; }
[ -n "${KPF13_CPU_LIST:-}" ] || { echo "REFUSING: KPF13_CPU_LIST is not set (the cpu list this run claims)" >&2; exit 64; }

args=("$@")
dry=0
for a in "${args[@]}"; do
    [ "$a" = "--dry-run" ] && dry=1
done
args=(run "${args[@]}" --cpu-list "$KPF13_CPU_LIST" --window-id "$KPF13_WINDOW_ID")

if [ "$dry" = 1 ]; then
    exec python3 -B "$DRIVER" "${args[@]}"
fi
[ -x "$REGION_LOCK" ] || { echo "REFUSING: $REGION_LOCK not found" >&2; exit 64; }
exec "$REGION_LOCK" run --cpu-list "$KPF13_CPU_LIST" --role build --tag "kpf13:$KPF13_WINDOW_ID" -- \
    env KPF13_REGION="$KPF13_CPU_LIST" python3 -B "$DRIVER" "${args[@]}"
