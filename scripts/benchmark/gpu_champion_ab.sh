#!/bin/bash
# gpu_champion_ab.sh -- run the GPU champion A/B (ab_arms.sh + ab_probe.py) of ANY two store GPU builds.
# Written 2026-10-20 (ab-probe-restore) so the consolidation's GPU measure arms and bundle gate call one committed entry point.
# Protocol = the 2026-10-04 KVU-19 paired receipt (epyc-root artifacts/champion-fold-kvu19-20261004/): alternating P C P C ... on
# ONE spare port (8197), the live :8083 DFlash2 argv (port and slot dir swapped), host threads -t 8 on cores 72-79 membind 3,
# probe `mtp --identity-tokens 256 --probs-tokens 64 --pp-items 300 --pp-reps 2 --tg-n 256 --tg-reps 2`, 15 pairs (the first
# launch is the warm-up the writer drops), noise floor = P-arm range as ab_probe.py summarize computes it.
#
#   gpu_champion_ab.sh --p-build DIR --p-tip SHA --p-buildno N --c-build DIR --c-tip SHA --c-buildno N --run DIR
#                      [--pairs 15] [--order "P C ..."] [--args-file F | --args-pid PID] [--probe-args "..."] [--kind mtp|vision]
#                      [--llama-bench "ARGS"] [--dry-run (default) | --execute]
# --dry-run prints every command and verifies sha256 / ggml linkage of both builds (ldd only); it launches nothing.
# --execute REFUSES unless AB_INSIDE_CLAIM=1: run it under `region-lock run --cpu-list 0-95 --role bench --gpu-quiet exclusive`.
# Own PIDs only (ab_arms.sh captures and verifies them); never pkill/pgrep by name.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P_BUILD=""; P_TIP=""; P_BUILDNO=""; C_BUILD=""; C_TIP=""; C_BUILDNO=""; RUN=""; PAIRS=15; ORDER=""
ARGS_FILE=""; ARGS_PID=""; KIND=mtp; LB=""; MODE=dry
PROBE_ARGS="--identity-tokens 256 --probs-tokens 64 --pp-items 300 --pp-reps 2 --tg-n 256 --tg-reps 2"
usage() { sed -n '2,16p' "$0"; exit 64; }
while [ $# -gt 0 ]; do case "$1" in
    --p-build) P_BUILD=$2; shift 2;; --p-tip) P_TIP=$2; shift 2;; --p-buildno) P_BUILDNO=$2; shift 2;;
    --c-build) C_BUILD=$2; shift 2;; --c-tip) C_TIP=$2; shift 2;; --c-buildno) C_BUILDNO=$2; shift 2;;
    --run) RUN=$2; shift 2;; --pairs) PAIRS=$2; shift 2;; --order) ORDER=$2; shift 2;;
    --args-file) ARGS_FILE=$2; shift 2;; --args-pid) ARGS_PID=$2; shift 2;; --kind) KIND=$2; shift 2;;
    --probe-args) PROBE_ARGS=$2; shift 2;; --llama-bench) LB=$2; shift 2;;
    --dry-run) MODE=dry; shift;; --execute) MODE=exec; shift;; -h|--help) usage;; *) echo "unknown argument $1" >&2; usage;; esac; done
for v in P_BUILD P_TIP P_BUILDNO C_BUILD C_TIP C_BUILDNO RUN; do [ -n "${!v}" ] || { echo "missing --$(echo "$v" | tr 'A-Z_' 'a-z-')" >&2; usage; }; done
[ -x "$P_BUILD/bin/llama-server" ] && [ -x "$C_BUILD/bin/llama-server" ] || { echo "REFUSE: no llama-server under $P_BUILD/bin or $C_BUILD/bin" >&2; exit 1; }
if [ -z "$ORDER" ]; then ORDER=""; for _ in $(seq 1 "$PAIRS"); do ORDER+="P C "; done; ORDER=${ORDER% }; fi
if [ "$MODE" = exec ] && [ "${AB_INSIDE_CLAIM:-0}" != 1 ]; then
    echo "REFUSE: --execute needs AB_INSIDE_CLAIM=1 (run under region-lock --role bench --gpu-quiet exclusive, after the operator window)" >&2; exit 3; fi
mkdir -p "$RUN"
# arm argv: an explicit NUL-separated file, or derived from the live :8083 server (read-only /proc/<pid>/cmdline)
if [ -z "$ARGS_FILE" ]; then
    if [ -z "$ARGS_PID" ]; then ARGS_PID="$(ss -ltnpH '( sport = :8083 )' 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2 || true)"; fi
    [ -n "$ARGS_PID" ] || { echo "REFUSE: no --args-file and no listener on :8083 to derive the argv from" >&2; exit 1; }
    mkdir -p "$RUN/slots"
    python3 -I "$HERE/ab_probe.py" args --pid "$ARGS_PID" --port 8197 --slot-dir "$RUN/slots" > "$RUN/server_args.bin"
    ARGS_FILE="$RUN/server_args.bin"
fi
# an explicit --args-file is an ARM argv (no binary): point its --slot-save-path (and --port) at this run, in a private copy
mkdir -p "$RUN/slots"
python3 -I - "$ARGS_FILE" "$RUN/server_args.arm.bin" "$RUN/slots" <<'PY'
import sys
a = [x for x in open(sys.argv[1]).read().split("\0") if x]
for i, x in enumerate(a):
    if x == "--slot-save-path": a[i + 1] = sys.argv[3]
    if x == "--port": a[i + 1] = "8197"
open(sys.argv[2], "w").write("\0".join(a) + "\0")
PY
ARGS_FILE="$RUN/server_args.arm.bin"
MODEL="$(python3 -I - "$ARGS_FILE" <<'PY'
import sys
a = [x for x in open(sys.argv[1]).read().split("\0") if x]
print(a[a.index("-m") + 1] if "-m" in a else a[a.index("--model") + 1])
PY
)"
export P_BUILD P_TIP P_BUILDNO CAND_BUILD="$C_BUILD" CAND_TIP="$C_TIP" CAND_BUILDNO="$C_BUILDNO"
cmd=(bash "$HERE/ab_arms.sh" --kind "$KIND" --run "$RUN" --args-file "$ARGS_FILE" --model "$MODEL" --order "$ORDER" --est-arm-s 70
     --probe-args "$PROBE_ARGS")
[ -z "$LB" ] || cmd+=(--llama-bench "$LB")
[ "$MODE" = dry ] && cmd+=(--dry-run)
echo "[gpu_champion_ab] mode=$MODE P=$P_BUILD ($P_TIP, $P_BUILDNO) C=$C_BUILD ($C_TIP, $C_BUILDNO) run=$RUN model=$MODEL pairs=$(wc -w <<<"$ORDER")arms"
exec "${cmd[@]}"
