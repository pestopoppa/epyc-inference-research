#!/bin/bash
# DS41-T3.3c rerun INNER (inside the launcher's build-role claim, own process group). ONE 1017 target-only server, served WITH
# --jinja --chat-template-file (template sha recorded), NO drafter, NO generation: only /apply-template and /tokenize run, so the
# server is idle after load (mmap'd; a few GiB private). Then t33.py v3 (robust arguments probe + variants for golden1-4) for
# both encoding.py copies, then the offline analysis. Env from the launcher: FILLER_END FILLER_CPUS FILLER_THREADS RUNDIR STOPFILE.
set -uo pipefail
HERE=$(dirname "$(readlink -f "$0")")
LIB=${LIB:-$HERE/../lib}
. "$LIB/ds41_guard.sh"; . "$LIB/srv.sh"
BIN=${T33C_BIN:-/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin}   # same binary as T3.3 / T3.3c v2 (port build)
M1017=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf
TPL=${T33C_TEMPLATE:-/mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja}
SEPT=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding
ENC1017=${T33C_ENC1017:-/mnt/raid0/llm/tmp/ds41-parity-20261021/ds41-t33c-1017b/enc1017}
PY=${PY:-/mnt/raid0/llm/epyc-inference-research/.venv/bin/python}
PORT=${T33C_PORT:-18965}
NT=${FILLER_THREADS:?}; CPUS=${FILLER_CPUS:?}
OUT=$RUNDIR/results; mkdir -p "$OUT"
log() { echo "$(date -u +%FT%TZ) [t33c] $*"; }
trap 'kill_srv' EXIT
trap 'kill_srv; exit 143' TERM INT
[ -s "$TPL" ] || { echo "template missing: $TPL"; exit 6; }
check_bin "$BIN" || exit 6
[ "$(time_left)" -lt 600 ] && { log "SKIP: only $(time_left)s left"; exit 3; }
stopped && { log "STOP file"; exit 3; }
ds41_guard_or_refuse "t33c"; ds41_memgate "t33c"
ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "port $PORT in use"; exit 4; }
sha256sum "$TPL" | tee "$OUT/template.sha256"
mkdir -p "$OUT/enc1017"; cp "$ENC1017/encoding.py" "$OUT/enc1017/encoding.py" 2>/dev/null || rm -f "$OUT/enc1017/encoding.py"
export LD_LIBRARY_PATH=$BIN
taskset -c "$CPUS" "$BIN/llama-server" -m "$M1017" --host 127.0.0.1 --port "$PORT" -t "$NT" -tb "$NT" -np 1 -c 4096 --no-warmup --no-webui --cache-ram 0 \
   --jinja --chat-template-file "$TPL" > "$OUT/server.log" 2>&1 &
SRV_PID=$!; echo "$SRV_PID" > "$OUT/server.pid"
wait_ready "$PORT" 1500 || { log "not ready"; exit 7; }
"$PY" -I "$HERE/t33.py" "$PORT" "$SEPT" "$OUT" sept_encoding | tee "$OUT/summary.txt"
if [ -s "$OUT/enc1017/encoding.py" ] && ! cmp -s "$OUT/enc1017/encoding.py" "$SEPT/encoding.py"; then
  "$PY" -I "$HERE/t33.py" "$PORT" "$OUT/enc1017" "$OUT" remote_main_encoding | tee -a "$OUT/summary.txt"
else echo "1017 encoding.py: not available or identical to the Sept copy" | tee -a "$OUT/summary.txt"; fi
kill_srv
for f in "$OUT"/t33-*.json; do
  "$PY" -I "$HERE/analyze_t33c.py" "$f" --out "${f%.json}.analysis.json" > /dev/null || log "analysis failed for $f"
done
date -u +%FT%TZ > "$RUNDIR/COMPLETE"; log "done"
exit 0
