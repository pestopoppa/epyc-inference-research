#!/bin/bash
# Firm up the biggest throughput opportunity found today: worker_general (gemma
# 26B-A4B) currently runs slots=1 but measured 196.48 tok/s aggregate at np=8.
# That single figure was n=1 on a SHORT prompt, so it is not yet a basis for a
# serving-policy change. Two questions:
#   1. Does it reproduce with reps?
#   2. Does the batching win survive a realistic prompt length, or is it an
#      artifact of 30-token prompts where prefill is negligible?
# Both prompt lengths are run at each np so the comparison is within-arm.
M=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/slotcheck_results.txt
PORT=19970
REPS=3
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

for NP in 1 4 8; do
  CTX=$(( 16384 * NP )); [ "$CTX" -lt 32768 ] && CTX=32768
  for PROMPT in short p8k; do
    SLOG=/mnt/raid0/llm/tmp/slot_${PROMPT}_np${NP}.log; : > "$SLOG"
    echo "=== np=$NP prompt=$PROMPT ctx=$CTX reps=$REPS ===" | tee -a "$OUT"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
      -np "$NP" -c "$CTX" -t 96 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
      --no-mmap --device none --log-colors off >> "$SLOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
    if ! grep -q "model loaded" "$SLOG"; then
      echo "  NEVER LOADED" | tee -a "$OUT"; kill -9 "$SRV" 2>/dev/null; sleep 5; continue
    fi
    for rep in $(seq 1 $REPS); do
      RP=""
      for r in $(seq 1 "$NP"); do
        if [ "$PROMPT" = "short" ]; then
          curl -s --max-time 1200 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
            -o /dev/null 2>/dev/null &
        else
          curl -s --max-time 1200 "http://127.0.0.1:$PORT/v1/chat/completions" \
            -H 'Content-Type: application/json' -d @/mnt/raid0/llm/tmp/req_p8k.json \
            -o /dev/null 2>/dev/null &
        fi
        RP="$RP $!"
      done
      for p in $RP; do wait "$p"; done
    done
    kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
    python3 - "$SLOG" <<'PY' | tee -a "$OUT"
import re, sys, statistics
lines = [l for l in open(sys.argv[1]) if "eval time =" in l and "prompt eval" not in l]
r = [float(m.group(1)) for l in lines for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
pe = [float(m.group(1)) for l in open(sys.argv[1]) if "prompt eval time" in l
      for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
if r:
    # per-stream median across all reps; aggregate = median per-stream x concurrent streams
    print(f"  per-stream tok/s: n={len(r)} median={statistics.median(r):.2f} "
          f"min={min(r):.2f} max={max(r):.2f}")
    print(f"  prefill tok/s   : median={statistics.median(pe):.2f}" if pe else "  prefill: n/a")
else:
    print("  NO TIMINGS")
PY
  done
done
echo "=== SLOTCHECK DONE ===" | tee -a "$OUT"
