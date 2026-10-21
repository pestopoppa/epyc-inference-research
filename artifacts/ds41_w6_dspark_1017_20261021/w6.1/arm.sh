#!/bin/bash
# usage: arm.sh <label> <drafter.gguf> <port> <seedbase> <greedymode 0|1|2> <cpulist> <threads>
# (run under region-lock --cpu-list 0-47; greedymode 2 = greedy only)
set -uo pipefail
LABEL=$1; DRAFT=$2; PORT=$3; SEEDBASE=$4; GREEDY=$5; CPUS=$6; NT=$7
W=/mnt/raid0/llm/tmp/ds41-w6.1-20261021
OUT=/mnt/raid0/llm/tmp/ds41-w6.1-20261021/runs/$LABEL; mkdir -p "$OUT"
BIN=/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin
TGT=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf
export LD_LIBRARY_PATH=$BIN OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
taskset -c "$CPUS" "$BIN/llama-server" --no-webui -t "$NT" -tb "$NT" -np 1 -c 4096 --host 127.0.0.1 --port "$PORT" -m "$TGT" -fa on \
   -md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max 3 > "$OUT/server.log" 2>&1 &
SPID=$!
CPID=""
cleanup() {
  [ -n "$CPID" ] && kill "$CPID" 2>/dev/null
  kill $SPID 2>/dev/null; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && kill -9 $SPID
  for i in $(seq 1 60); do ps -p $SPID >/dev/null || break; sleep 1; done
  ps -p $SPID >/dev/null && echo "[$LABEL] STILL ALIVE $SPID $(date -u +%T)" || echo "[$LABEL] server $SPID dead $(date -u +%T)"
}
trap 'cleanup; exit 143' TERM INT
echo "server pid $SPID" > "$OUT/pid"; echo "[$LABEL] server pid $SPID cpus $CPUS start $(date -u +%T)"
ok=0
for i in $(seq 1 360); do
  if ! kill -0 $SPID 2>/dev/null; then echo "[$LABEL] server died"; break; fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 3
done
echo "[$LABEL] healthy=$ok at $(date -u +%T)"
if [ $ok = 1 ]; then
  /mnt/raid0/llm/epyc-inference-research/.venv/bin/python -I $W/client.py "$PORT" "$OUT" "$SEEDBASE" "$GREEDY" > "$OUT/client.log" 2>&1 &
  CPID=$!
  wait $CPID
  grep SUMMARY "$OUT/client.log"
fi
cleanup
