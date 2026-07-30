#!/bin/bash
# E5 re-run done correctly: production model on the CORRECTED placement
# (full machine 0-95 + numactl --interleave=all, cold-loaded so the interleave
# policy actually applies at first touch), sweeping -np.
#
# GATE: np=1 must reproduce the AutoPilot anchor (35-40 tok/s median_request_tps)
# or every downstream cell is invalid. Verified at 36-39 before this sweep.
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/npsweep_results.txt
PORT=19998
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

for NP in 1 2 4 8 16 32; do
  CTX=$(( 2048 * NP )); [ "$CTX" -lt 8192 ] && CTX=8192
  SLOG=/mnt/raid0/llm/tmp/npsweep_np${NP}.log
  echo "=== np=$NP ctx=$CTX ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  : > "$SLOG"
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np "$NP" -c "$CTX" -t 96 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --mlock \
    --device none --spec-type draft-mtp --spec-draft-n-max 4 --draft-p-split 0 --device-draft none \
    --log-colors off >> "$SLOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 240); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
  if ! grep -q "model loaded" "$SLOG"; then
    echo "  NEVER LOADED" | tee -a "$OUT"; kill -9 "$SRV" 2>/dev/null; continue
  fi
  START=$(date +%s.%N)
  PIDS=""
  for r in $(seq 1 "$NP"); do
    curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42,"chat_template_kwargs":{"enable_thinking":false}}' \
      -o /dev/null 2>/dev/null &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait "$p"; done
  END=$(date +%s.%N)
  kill -TERM "$SRV" 2>/dev/null; sleep 8; kill -9 "$SRV" 2>/dev/null; sleep 3
  python3 /mnt/raid0/llm/tmp/np_parse.py "$SLOG" "$START" "$END" >> "$OUT"
done
echo "=== SWEEP DONE ===" | tee -a "$OUT"
