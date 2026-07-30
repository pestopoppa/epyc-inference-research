#!/bin/bash
# Concurrency curve per production CPU role, all on the CORRECTED placement
# (full machine 0-95 + --interleave=all, cold). Completes the original E5
# question -- "what is the optimal -np for each role" -- for the three roles
# beyond frontdoor, which is already covered by npsweep_results.txt.
# No speculative decoding: keeps every role on the same footing, and avoids the
# gemma4 MTP ASSERT(S>0) wedge.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/npall_results.txt
PORT=19980
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

MODELS='
worker_general_gemma26B|/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf|1 2 4 8
ingest_long_context_80B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf|1 2 4 8
architect_general_122B|/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf|1 2 4
'

echo "$MODELS" | while IFS='|' read -r MNAME MPATH NPLIST; do
  [ -z "$MNAME" ] && continue
  [ -f "$MPATH" ] || { echo "MISSING $MPATH" | tee -a "$OUT"; continue; }
  echo "##### $MNAME #####" | tee -a "$OUT"
  for NP in $NPLIST; do
    CTX=$(( 2048 * NP )); [ "$CTX" -lt 8192 ] && CTX=8192
    SLOG=/mnt/raid0/llm/tmp/npall_${MNAME}_np${NP}.log; : > "$SLOG"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    taskset -c 0-95 numactl --interleave=all "$LS" -m "$MPATH" --host 127.0.0.1 --port $PORT \
      -np "$NP" -c "$CTX" -t 96 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
      --no-mmap --device none --log-colors off >> "$SLOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 400); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
    if ! grep -q "model loaded" "$SLOG"; then
      echo "  np=$NP NEVER LOADED" | tee -a "$OUT"; kill -9 "$SRV" 2>/dev/null; sleep 5; continue
    fi
    RP=""
    for r in $(seq 1 "$NP"); do
      curl -s --max-time 1200 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
        -o /dev/null 2>/dev/null &
      RP="$RP $!"
    done
    for p in $RP; do wait "$p"; done
    echo "  np=$NP" | tee -a "$OUT"
    python3 /mnt/raid0/llm/tmp/np_parse.py "$SLOG" 0 1 2>/dev/null | grep -E "per-stream|AGGREGATE" | tee -a "$OUT"
    kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  done
done
echo "=== NPALL DONE ===" | tee -a "$OUT"
