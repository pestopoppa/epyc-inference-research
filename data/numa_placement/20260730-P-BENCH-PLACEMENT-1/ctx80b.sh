#!/bin/bash
# ingest_long_context (Qwen3-Next-80B-A3B IQ2_M) — placement A/B x context-length curve.
# This role is wired to NUMA_NODE0 "0-47,96-143" -t 96 with NO numactl policy, which
# straddles NPS4 node0+node1. Arms:
#   A_prod    = exactly as production launches it (no numactl), page cache WARM
#   B_halfint = same cpuset, + --interleave=0,1, COLD (interleave binds at first touch only)
#   C_fullint = full machine 0-95 + --interleave=all, COLD  (the canonical recipe)
# No speculative decoding: this GGUF carries no MTP head, so these are raw decode rates.
M=/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct.i1-IQ2_M.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/ctx80b_results.txt
PORT=19996
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"
python3 /mnt/raid0/llm/tmp/mkprompts.py | tee -a "$OUT"

# name|cpuset|policy|drop_caches
ARMS='
A_prod|0-47,96-143||0
B_halfint|0-47,96-143|--interleave=0,1|1
C_fullint|0-95|--interleave=all|1
'

echo "$ARMS" | while IFS='|' read -r NAME CPUSET POLICY DROP; do
  [ -z "$NAME" ] && continue
  SLOG=/mnt/raid0/llm/tmp/ctx80b_${NAME}.log
  echo "=== ARM $NAME cpuset=$CPUSET policy=${POLICY:-none} drop_caches=$DROP ===" | tee -a "$OUT"
  [ "$DROP" = "1" ] && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  : > "$SLOG"
  taskset -c "$CPUSET" numactl $POLICY "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c 40960 -t 96 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --mlock \
    --device none --log-colors off >> "$SLOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 360); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
  if ! grep -q "model loaded" "$SLOG"; then
    echo "  NEVER LOADED" | tee -a "$OUT"; kill -9 "$SRV" 2>/dev/null; sleep 5; continue
  fi
  for CTXNAME in p0k5 p8k p32k; do
    R=$(curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @/mnt/raid0/llm/tmp/req_${CTXNAME}.json)
    # Read the rates straight out of llama.cpp's own timings, never wall-clock.
    PE=$(grep "prompt eval time" "$SLOG" | tail -1 | sed -E 's/.*\/ *([0-9]+) tokens.*/\1/')
    DEC=$(grep "eval time =" "$SLOG" | grep -v "prompt eval" | tail -1 | sed -E 's/.*, *([0-9.]+) tokens per second.*/\1/')
    PEV=$(grep "prompt eval time" "$SLOG" | tail -1 | sed -E 's/.*, *([0-9.]+) tokens per second.*/\1/')
    echo "  $CTXNAME : prompt_tokens=$PE  prefill=${PEV} tok/s  DECODE=${DEC} tok/s" | tee -a "$OUT"
  done
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 3
done
echo "=== CTX80B DONE ===" | tee -a "$OUT"
