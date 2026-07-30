#!/bin/bash
# Does the production 4-quarter fleet shape actually get node-local weights?
#
# Finding under test: llama.cpp mmap's the GGUF, so its pages live in the shared
# page cache and are placed ONCE, at first touch, by whichever instance loads first.
# Four quarters sharing one file therefore share one placement -> at most ONE quarter
# is node-local and the other three pay remote access for every weight read.
# --no-mmap makes each instance allocate private anonymous memory, which its own
# --membind can place locally. Cost: 4x resident model size.
#
# Arm MMAP   = production behaviour (shared mmap, per-quarter membind)
# Arm NOMMAP = --no-mmap, per-quarter membind, 4 private local copies
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/quadfleet_results.txt
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

# quarter index -> cpuset (exactly one NPS4 node each)
CPUSETS=("0-23,96-119" "24-47,120-143" "48-71,144-167" "72-95,168-191")

for ARM in MMAP NOMMAP; do
  MMAPFLAG=""
  [ "$ARM" = "NOMMAP" ] && MMAPFLAG="--no-mmap"
  echo "=== ARM $ARM (4 quarters, np=1 each, membind to own node) ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  PIDS=""
  for q in 0 1 2 3; do
    SLOG=/mnt/raid0/llm/tmp/quad_${ARM}_q${q}.log
    : > "$SLOG"
    taskset -c "${CPUSETS[$q]}" numactl --membind=$q "$LS" -m "$M" \
      --host 127.0.0.1 --port $((19900 + q)) \
      -np 1 -c 8192 -t 24 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja $MMAPFLAG \
      --device none --spec-type draft-mtp --spec-draft-n-max 4 --draft-p-split 0 --device-draft none \
      --log-colors off >> "$SLOG" 2>&1 &
    PIDS="$PIDS $!"
  done
  # all four must be up before any request, or the load itself perturbs the others
  ALLUP=1
  for i in $(seq 1 400); do
    ALLUP=1
    for q in 0 1 2 3; do
      grep -q "model loaded" /mnt/raid0/llm/tmp/quad_${ARM}_q${q}.log || ALLUP=0
    done
    [ "$ALLUP" = "1" ] && break
    sleep 5
  done
  if [ "$ALLUP" != "1" ]; then
    echo "  NOT ALL LOADED" | tee -a "$OUT"
  else
    # record measured weight locality per instance before touching them
    for q in 0 1 2 3; do
      SP=$(pgrep -f "port $((19900 + q))" | head -1)
      if [ -n "$SP" ]; then
        LOC=$(awk -v n="N$q=" '$0 ~ /huge|anon|file/ {for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/){split($i,a,"=");tot[a[1]]+=a[2]}} END{s=0;for(k in tot)s+=tot[k]; if(s>0) printf "%.1f%% on own node (%d pages total)", 100*tot["N" ENVIRON["Q"]]/s, s}' Q=$q /proc/$SP/numa_maps 2>/dev/null)
        echo "  q$q locality: ${LOC:-unavailable}" | tee -a "$OUT"
      fi
    done
    # fire one request at each quarter simultaneously
    RPIDS=""
    for q in 0 1 2 3; do
      curl -s --max-time 900 "http://127.0.0.1:$((19900 + q))/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42,"chat_template_kwargs":{"enable_thinking":false}}' \
        -o /dev/null 2>/dev/null &
      RPIDS="$RPIDS $!"
    done
    for p in $RPIDS; do wait "$p"; done
    TOT=0
    for q in 0 1 2 3; do
      D=$(grep "eval time =" /mnt/raid0/llm/tmp/quad_${ARM}_q${q}.log | grep -v "prompt eval" | tail -1 | sed -E 's/.*, *([0-9.]+) tokens per second.*/\1/')
      echo "  q$q decode: ${D:-NA} tok/s" | tee -a "$OUT"
      TOT=$(python3 -c "print(round($TOT + ${D:-0}, 2))")
    done
    echo "  FLEET AGGREGATE: $TOT tok/s" | tee -a "$OUT"
  fi
  for p in $PIDS; do kill -TERM "$p" 2>/dev/null; done
  sleep 12
  for p in $PIDS; do kill -9 "$p" 2>/dev/null; done
  sleep 5
done
echo "=== QUADFLEET DONE ===" | tee -a "$OUT"
