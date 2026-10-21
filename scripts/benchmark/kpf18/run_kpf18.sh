#!/bin/bash
# run_kpf18.sh — wrapper for kpf18_gate.py (KPF-18 GPU gate, cross-slot prefix fork). See RUNBOOK.md.
#
# Refuses unless KPF18_GPU_WINDOW_ID names the coordinated GPU window. Passes every argument through to
# kpf18_gate.py and adds --gpu-window-id "$KPF18_GPU_WINDOW_ID" if the caller did not.
#
# Execute runs (no --dry-run) are wrapped in the CPU region lock for the server's host lane:
#   region-lock run --cpu-list "$KPF18_CPU_LIST" --role kpf18 --tag "$KPF18_GPU_WINDOW_ID" -- python3 kpf18_gate.py ...
# KPF18_CPU_LIST defaults to 184-191 (the :8083 launch prefix's taskset). Opt out ONLY with
# KPF18_NO_REGION_LOCK="<reason>" (the reason is passed through and recorded by the runner's run_meta.json).
# --dry-run never takes the lock and the runner starts nothing.
#
# This wrapper signals nothing: the python runner owns its one llama-server child by PID.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATE="$HERE/kpf18_gate.py"
REGION_LOCK=/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock
CPU_LIST="${KPF18_CPU_LIST:-184-191}"

if [ -z "${KPF18_GPU_WINDOW_ID:-}" ]; then
    echo "REFUSING: KPF18_GPU_WINDOW_ID is not set. Run only inside a coordinated GPU window, e.g." >&2
    echo "  KPF18_GPU_WINDOW_ID=<window id> $0 all --dry-run ..." >&2
    exit 64
fi

args=("$@")
have_wid=0
dry=0
for a in "${args[@]}"; do
    case "$a" in
        --gpu-window-id|--gpu-window-id=*) have_wid=1 ;;
        --dry-run) dry=1 ;;
    esac
done
if [ "$have_wid" = 0 ]; then
    args+=(--gpu-window-id "$KPF18_GPU_WINDOW_ID")
fi

if [ "$dry" = 1 ]; then
    exec python3 -B "$GATE" "${args[@]}"
fi

if [ -n "${KPF18_NO_REGION_LOCK:-}" ]; then
    echo "[run_kpf18] region lock SKIPPED: $KPF18_NO_REGION_LOCK" >&2
    exec python3 -B "$GATE" "${args[@]}"
fi

[ -x "$REGION_LOCK" ] || { echo "REFUSING: $REGION_LOCK not found (set KPF18_NO_REGION_LOCK=<reason> to skip)" >&2; exit 64; }
exec "$REGION_LOCK" run --cpu-list "$CPU_LIST" --role kpf18 --tag "kpf18:$KPF18_GPU_WINDOW_ID" -- \
    python3 -B "$GATE" "${args[@]}"
