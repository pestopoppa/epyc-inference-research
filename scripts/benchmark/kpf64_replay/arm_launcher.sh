#!/bin/bash
# KPF-64 arm launcher: start an OWN llama-server for ONE arm, prove its linkage, (GPU) sample VRAM,
# run the replay driver against it, then kill ONLY the captured PIDs and verify they are dead.
#
#   WINDOW_CONFIRMED=1 KPF64_SERVER_BIN=<kernel-store>/cpu/llama-server \
#     arm_launcher.sh --arm B --out DIR --sessions SESSIONS_DIR [--dry-run] [-- extra driver args]
#
# Env: KPF64_SERVER_BIN (required; the kernel store, kernels/production/{cpu,gpu}; never
#      llama.cpp/build/bin), KPF64_DEVICE=cpu|gpu (default cpu), KPF64_PORT (default 18964),
#      KPF64_MODEL (default: arms.json model), KPF64_THREADS, KPF64_LAUNCH_PREFIX (e.g.
#      "region-lock run --cpu-list 0-95 --"), KPF64_LINKAGE (override verify_ggml_linkage.sh path).
# --dry-run: runs every refusal check, prints the exact commands, starts nothing.
set -euo pipefail

if [ "${WINDOW_CONFIRMED:-}" != "1" ]; then
  echo "arm_launcher: refusing: WINDOW_CONFIRMED=1 is not set (claim a KPF-64 timing window first)" >&2
  exit 3
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARMS="${KPF64_ARMS:-$HERE/arms.json}"
ARM="" OUT="" SESSIONS="" DRY=0 EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --arm) ARM="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --sessions) SESSIONS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "arm_launcher: unknown arg $1" >&2; exit 2 ;;
  esac
done
[ -n "$ARM" ] || { echo "arm_launcher: --arm required" >&2; exit 2; }
DEVICE="${KPF64_DEVICE:-cpu}"
case "$DEVICE" in cpu|gpu) ;; *) echo "arm_launcher: KPF64_DEVICE must be cpu|gpu" >&2; exit 2 ;; esac
PORT="${KPF64_PORT:-18964}"
if [ "$PORT" -ge 8000 ] && [ "$PORT" -le 8199 ]; then
  echo "arm_launcher: refusing port $PORT: 8000-8199 is the production/dashboard range" >&2
  exit 3
fi
if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.5); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
  echo "arm_launcher: refusing: port $PORT already has a listener" >&2
  exit 3
fi
[ -n "${KPF64_SERVER_BIN:-}" ] || { echo "arm_launcher: KPF64_SERVER_BIN is required (kernel store binary)" >&2; exit 2; }
case "$KPF64_SERVER_BIN" in
  */llama.cpp/build/bin/*|*/llama.cpp/build-hip/bin/*)
    echo "arm_launcher: refusing: $KPF64_SERVER_BIN is a frozen in-tree build dir, not the kernel store" >&2; exit 3 ;;
esac
[ -x "$KPF64_SERVER_BIN" ] || { echo "arm_launcher: $KPF64_SERVER_BIN is not executable" >&2; exit 2; }
BIN_DIR="$(cd "$(dirname "$KPF64_SERVER_BIN")" && pwd)"

# argv from the declared arm table (flag names verified against common/arg.cpp)
CMDOUT="$(KPF64_MODEL="${KPF64_MODEL:-}" KPF64_THREADS="${KPF64_THREADS:-}" python3 - "$ARMS" "$ARM" "$PORT" "$KPF64_SERVER_BIN" "$DEVICE" <<'PY'
import json, os, sys
arms, arm, port, binp, dev = sys.argv[1:6]
t = json.load(open(arms)); a = t["arms"].get(arm)
if a is None:
    sys.stderr.write(f"arm_launcher: unknown arm {arm!r}; have {sorted(t['arms'])}\n"); sys.exit(2)
c = [binp, "-m", os.environ.get("KPF64_MODEL") or t["model"], "--port", port,
     "-np", str(t["slots"]), "-c", str(t["ctx_total"]),
     "-t", os.environ.get("KPF64_THREADS") or str(t["threads_default"]),
     "--ctx-checkpoints", str(a["ctx_checkpoints"]), "--cache-ram", str(a["cache_ram_mib"])] + t["common_args"]
if dev == "gpu":
    c += ["-ngl", "99"]
print("\n".join(c))
PY
)" || exit 2
mapfile -t CMD <<<"$CMDOUT"
# per-arm replay-driver args (KPF-LB2 follow-up arms E/F; A-D declare none) and the sessions mode they need
DARGS_OUT="$(python3 - "$ARMS" "$ARM" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))["arms"][sys.argv[2]]
print(a.get("sessions_mode", "flat"))
print("\n".join(a.get("driver_args", [])))
PY
)"
mapfile -t DARGS <<<"$DARGS_OUT"
SESSIONS_MODE="${DARGS[0]}"; DARGS=("${DARGS[@]:1}")
[ "${DARGS[0]:-}" != "" ] || DARGS=()
LINK="${KPF64_LINKAGE:-$HERE/../../utils/verify_ggml_linkage.sh}"
PREFIX=(); [ -z "${KPF64_LAUNCH_PREFIX:-}" ] || read -r -a PREFIX <<<"$KPF64_LAUNCH_PREFIX"
OUT="${OUT:-/mnt/raid0/llm/tmp/kpf64-arm-$ARM}"
DRIVER=(python3 "$HERE/replay_driver.py" --sessions "${SESSIONS:-SESSIONS_DIR}" --out "$OUT/replay" --arm "$ARM"
        --port "$PORT" --server-log "$OUT/server.log" --slots 4 "${DARGS[@]}")

if [ "$DRY" = 1 ]; then
  echo "DRY-RUN arm=$ARM device=$DEVICE port=$PORT (nothing started)"
  echo "LD_LIBRARY_PATH=$BIN_DIR"
  echo "server : ${PREFIX[*]:-} ${CMD[*]}"
  echo "linkage: $LINK $KPF64_SERVER_BIN $BIN_DIR"
  [ "$DEVICE" = gpu ] && echo "vram   : sampler loop (rocm-smi --showmeminfo vram) every 10s -> $OUT/vram.csv during the run"
  echo "sessions: adapter.py --mode $SESSIONS_MODE (arm $ARM needs the $SESSIONS_MODE sessions dir)"
  echo "driver : ${DRIVER[*]} --own-server-pid <captured> ${EXTRA[*]:-}"
  exit 0
fi

mkdir -p "$OUT"
export LD_LIBRARY_PATH="$BIN_DIR"   # own libs only; never inherit another tree's ggml
bash "$LINK" "$KPF64_SERVER_BIN" "$BIN_DIR" | tee "$OUT/linkage.txt"

SPID="" VPID=""
cleanup() {
  for p in $VPID $SPID; do
    kill "$p" 2>/dev/null || true
  done
  for p in $VPID $SPID; do
    for _ in $(seq 1 30); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
    if kill -0 "$p" 2>/dev/null; then kill -KILL "$p" 2>/dev/null || true; sleep 1; fi
    if kill -0 "$p" 2>/dev/null; then echo "arm_launcher: PID $p STILL ALIVE" >&2; else echo "arm_launcher: PID $p dead (verified)"; fi
  done
  rm -f "$OUT/server.pid"
}
trap cleanup EXIT

"${PREFIX[@]}" "${CMD[@]}" >"$OUT/server.log" 2>&1 &
SPID=$!
echo "$SPID" >"$OUT/server.pid"
for _ in $(seq 1 1800); do
  kill -0 "$SPID" 2>/dev/null || { echo "arm_launcher: server died during load" >&2; exit 1; }
  curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null || { echo "arm_launcher: server not healthy" >&2; exit 1; }

if [ "$DEVICE" = gpu ]; then
  ( while kill -0 "$SPID" 2>/dev/null; do
      echo "$(date -u +%s),$(rocm-smi --showmeminfo vram --csv 2>&1 | tr '\n' ';')" >>"$OUT/vram.csv"; sleep 10
    done ) &
  VPID=$!
fi

"${DRIVER[@]}" --own-server-pid "$SPID" "${EXTRA[@]}"
