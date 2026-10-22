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
# KPF-13.1 REDUCED RE-RUN under rules v2 (no-fork cache-hit controls, S=2 cell, measured near-tie band), plain MTP,
# one CPU quarter (q1 shown; q3 = 72-95,168-191), RAM admission first (ram_admit.sh 45):
#   KPF13_WINDOW_ID=<grant id> KPF13_CPU_LIST=24-47,120-143 run_kpf13.sh --reduced --threads 24 \
#       --out /mnt/raid0/llm/tmp/kpf13-reduced-v2/run --deadline-utc <grant end - 2 min> \
#       --stop-file /mnt/raid0/llm/tmp/kpf13-reduced-v2/STOP --prefill-tok-s 120 --decode-tok-s 12 --load-s 120
#   12 held-out cells (groups 0-3 x S 1,128,2), N=64, per cell W1 CA1 A1 CB1 B1 CA2 A2 CB2 B2 W2.
#   Estimate: ~44 min at the planning rates above (largest step ~205 s); ~31 min at the rates measured in the v1
#   reduced run (prefill ~180 tok/s, MTP decode ~19 tok/s), and v1 itself ran 1.36x under that model, so expect
#   ~23-25 min wall. Ask for a 45-min grant. (The default rates 30/3.5/240 give ~140 min: ~4.5x pessimistic.)
#
# This script signals nothing: the python driver owns its one llama-server child by PID.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRIVER="$HERE/kpf13_restore_ab.py"
REGION_LOCK="${KPF13_REGION_LOCK:-/mnt/raid0/llm/epyc-orchestrator/scripts/region-lock}"
# Inside region-lock the child's PATH starts with the orchestrator .venv (Python 3.11): name the interpreter explicitly.
PY="${KPF13_PYTHON:-/usr/bin/python3}"

[ -n "${KPF13_WINDOW_ID:-}" ] || { echo "REFUSING: KPF13_WINDOW_ID is not set (name the grant / window id)" >&2; exit 64; }
[ -n "${KPF13_CPU_LIST:-}" ] || { echo "REFUSING: KPF13_CPU_LIST is not set (the cpu list this run claims)" >&2; exit 64; }

args=("$@")
dry=0
for a in "${args[@]}"; do
    [ "$a" = "--dry-run" ] && dry=1
done
args=(run "${args[@]}" --cpu-list "$KPF13_CPU_LIST" --window-id "$KPF13_WINDOW_ID")

if [ "$dry" = 1 ]; then
    exec "$PY" -B "$DRIVER" "${args[@]}"
fi
[ -x "$REGION_LOCK" ] || { echo "REFUSING: $REGION_LOCK not found" >&2; exit 64; }
exec "$REGION_LOCK" run --cpu-list "$KPF13_CPU_LIST" --role build --timeout-s "${KPF13_LOCK_TIMEOUT_S:-3600}" --tag "kpf13:$KPF13_WINDOW_ID" -- \
    env KPF13_REGION="$KPF13_CPU_LIST" "$PY" -B "$DRIVER" "${args[@]}"
