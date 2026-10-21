#!/bin/bash
# DS41-W6.1c arm: ONE 1017 server (kind=dspark: target + 1017 DSpark drafter; kind=plain: target only), run client.py, tear down.
# usage: arm.sh <outdir> <kind dspark|plain> <port> <seedbase> <modes csv> <nmax csv> <cpus> <threads> [deadline_epoch]
# Same binary / flags as W6.1 and W6.2 (port build build-cpu, -fa on, -np 1, -c 4096, draft-dspark, --spec-draft-n-max 3).
# Refuses (exit 4) if ANY other DS41-class llama-server is live (ps scan, read-only) or MemAvailable < 400 GiB.
# Exit: 0 ok, 4 refused, 5 server/client failure, 6 bad binary, 75 deadline.
set -uo pipefail
OUT=$1; KIND=$2; PORT=$3; SEEDBASE=$4; MODES=$5; NMAX=$6; CPUS=$7; NT=$8; DEADLINE=${9:-}
HERE=$(dirname "$(readlink -f "$0")")
LIB=${LIB:-$HERE/../../ds41_queue_20261021/lib}
. "$LIB/ds41_guard.sh"; . "$LIB/srv.sh"
BIN=${W61_BIN:-/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin}
TGT=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf
DRAFT=${W61_DRAFT:-/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-DSpark.gguf}
PY=${PY:-/mnt/raid0/llm/epyc-inference-research/.venv/bin/python}
mkdir -p "$OUT"
trap 'kill_srv; exit 143' TERM INT
trap 'kill_srv' EXIT
check_bin "$BIN" || exit 6
ds41_guard_or_refuse "w61 arm $KIND"; ds41_memgate "w61 arm $KIND"
ss -ltn "sport = :$PORT" | grep -q LISTEN && { echo "port $PORT in use; refuse"; exit 4; }
export LD_LIBRARY_PATH=$BIN OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
ARGS=(--no-webui -t "$NT" -tb "$NT" -np 1 -c 4096 --host 127.0.0.1 --port "$PORT" -m "$TGT" -fa on)
[ "$KIND" = dspark ] && ARGS+=(-md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max 3)
taskset -c "$CPUS" "$BIN/llama-server" "${ARGS[@]}" > "$OUT/server.log" 2>&1 &
SRV_PID=$!
echo "$SRV_PID" > "$OUT/pid"
printf '{"kind":"%s","drafter":"%s","target":"%s","bin":"%s","port":%s,"seedbase":%s,"modes":"%s","nmax":"%s","cpus":"%s","threads":%s,"start":"%s","host_rule":"single DS41-class server, sequential arms"}\n' \
  "$KIND" "$([ "$KIND" = dspark ] && echo "$DRAFT")" "$TGT" "$BIN" "$PORT" "$SEEDBASE" "$MODES" "$NMAX" "$CPUS" "$NT" "$(date -u +%FT%TZ)" > "$OUT/arm.json"
echo "[arm $KIND] server pid $SRV_PID cpus $CPUS -t $NT start $(date -u +%T)"
wait_ready "$PORT" 1500 || { echo "[arm $KIND] not ready"; exit 5; }
grep Cpus_allowed_list /proc/"$SRV_PID"/status > "$OUT/affinity.txt"
"$PY" -I "$HERE/client.py" "$PORT" "$OUT" "$SEEDBASE" "$KIND" "$MODES" "$NMAX" $DEADLINE > "$OUT/client.log" 2>&1
rc=$?
grep SUMMARY "$OUT/client.log"
kill_srv
echo "[arm $KIND] end rc=$rc $(date -u +%T)"
[ $rc = 0 ] && exit 0
[ $rc = 75 ] && exit 75
exit 5
