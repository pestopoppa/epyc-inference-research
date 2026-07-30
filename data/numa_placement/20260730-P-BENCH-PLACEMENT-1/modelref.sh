#!/bin/bash
# Corrected single-stream reference per production model.
# Arm A = exactly how production wires these roles today (straddling cpuset
#         0-47,96-143, no numactl policy, warm page cache).
# Arm C = canonical recipe (full machine 0-95 + --interleave=all, cold).
# No speculative decoding anywhere: gemma4 MTP has a known ASSERT(S>0) wedge, and
# leaving spec-dec off keeps every model on the same footing (raw decode).
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/modelref_results.txt
PORT=19990
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

MODELS='
gemma26B_Q4KM|/mnt/raid0/llm/models/gemma-4-26B-A4B-it-Q4_K_M-current.gguf
gemma31B_Q4KM|/mnt/raid0/llm/models/gemma-4-31B-it-Q4_K_M.gguf
qwen35B_Q8_nonMTP|/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
'

echo "$MODELS" | while IFS='|' read -r MNAME MPATH; do
  [ -z "$MNAME" ] && continue
  [ -f "$MPATH" ] || { echo "MISSING $MPATH" | tee -a "$OUT"; continue; }
  echo "##### $MNAME #####" | tee -a "$OUT"
  for ARM in A_prod C_fullint; do
    if [ "$ARM" = "A_prod" ]; then CPUSET="0-47,96-143"; POLICY=""; DROP=0
    else CPUSET="0-95"; POLICY="--interleave=all"; DROP=1; fi
    SLOG=/mnt/raid0/llm/tmp/mref_${MNAME}_${ARM}.log; : > "$SLOG"
    [ "$DROP" = "1" ] && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    taskset -c "$CPUSET" numactl $POLICY "$LS" -m "$MPATH" --host 127.0.0.1 --port $PORT \
      -np 1 -c 8192 -t 96 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --mlock \
      --no-mmap --device none --log-colors off >> "$SLOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
    if ! grep -q "model loaded" "$SLOG"; then
      echo "  $ARM : NEVER LOADED" | tee -a "$OUT"; kill -9 "$SRV" 2>/dev/null; sleep 5; continue
    fi
    for rep in 1 2; do
      curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
        -o /dev/null 2>/dev/null
    done
    D=$(grep "eval time =" "$SLOG" | grep -v "prompt eval" | sed -E 's/.*, *([0-9.]+) tokens per second.*/\1/' | tail -2 | tr '\n' ' ')
    echo "  $ARM (cpuset=$CPUSET policy=${POLICY:-none}) decode reps: $D tok/s" | tee -a "$OUT"
    kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  done
done
echo "=== MODELREF DONE ===" | tee -a "$OUT"
