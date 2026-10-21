#!/bin/bash
# usage: arm.sh <label> <drafter.gguf> <port>   (run under region-lock; cpu list via taskset by caller)
set -uo pipefail
LABEL=$1; DRAFT=$2; PORT=$3
W=/mnt/raid0/llm/tmp/ds41-w6-20261021
OUT=$W/runs/$LABEL; mkdir -p "$OUT"
BIN=/mnt/raid0/llm/llama.cpp-experimental-ds41-20261005/build-cpu/bin
TGT=/mnt/raid0/llm/models/antirez/deepseek-v4.1-flash-gguf/DeepSeek-V4.1-Flash-1017-MXFP4-experts-00001-of-00009.gguf
export LD_LIBRARY_PATH=$BIN OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false GGML_IQK=1
"$BIN/llama-server" --no-webui -t 24 -tb 24 -np 1 -c 4096 --host 127.0.0.1 --port "$PORT" -m "$TGT" -fa on \
   -md "$DRAFT" --spec-type draft-dspark --spec-draft-n-max 3 > "$OUT/server.log" 2>&1 &
SPID=$!
echo "server pid $SPID" > "$OUT/pid"
ok=0
for i in $(seq 1 360); do
  if ! kill -0 $SPID 2>/dev/null; then echo "server died"; break; fi
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
  sleep 5
done
if [ $ok = 1 ]; then
  /mnt/raid0/llm/epyc-inference-research/.venv/bin/python -I $W/client.py "$PORT" "$OUT"
fi
kill $SPID 2>/dev/null; for i in $(seq 1 30); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
kill -0 $SPID 2>/dev/null && kill -9 $SPID
sleep 1; ps -p $SPID >/dev/null && echo "STILL ALIVE $SPID" || echo "server $SPID dead"
grep -i -E "dspark|draft|error|failed" "$OUT/server.log" | head -30
