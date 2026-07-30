#!/bin/bash
# E5 Stage-B, done correctly: the ACTUAL production instance shapes from
# epyc-orchestrator/scripts/server/stack_numa.py, each given the NUMA policy its
# cpuset requires, cold-loaded so --interleave applies at first touch, swept over -np.
#
# Shape defs (verified against live NPS4 topology 2026-07-30):
#   node0=0-23,96-119  node1=24-47,120-143  node2=48-71,144-167  node3=72-95,168-191
#   HALF    = stack_numa NUMA_NODE0 "0-47,96-143" -> spans node0+node1 -> interleave=0,1
#   QUARTER = stack_numa NUMA_Q0A   "0-23,96-119" -> single node0    -> membind=0
# -t variants probe SMT: HALF has 48 physical cores (96 logical), QUARTER has 24 (48).
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/shapesweep_results.txt
PORT=19997
REPS=2
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

# name|cpuset|numa-policy|threads|np-list
CELLS='
HALF|0-47,96-143|--interleave=0,1|96|1 2 4 8 16
HALFphys|0-47,96-143|--interleave=0,1|48|1 4
QUARTER|0-23,96-119|--membind=0|48|1 2 4 8
QUARTERphys|0-23,96-119|--membind=0|24|1 4
'

echo "$CELLS" | while IFS='|' read -r NAME CPUSET POLICY THREADS NPLIST; do
  [ -z "$NAME" ] && continue
  for NP in $NPLIST; do
    CTX=$(( 2048 * NP )); [ "$CTX" -lt 8192 ] && CTX=8192
    TAG="${NAME}_np${NP}"
    SLOG=/mnt/raid0/llm/tmp/shape_${TAG}.log
    echo "=== $NAME np=$NP cpuset=$CPUSET policy=$POLICY -t $THREADS ctx=$CTX ===" | tee -a "$OUT"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    : > "$SLOG"
    taskset -c "$CPUSET" numactl $POLICY "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
      -np "$NP" -c "$CTX" -t "$THREADS" -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --mlock \
      --device none --spec-type draft-mtp --spec-draft-n-max 4 --draft-p-split 0 --device-draft none \
      --log-colors off >> "$SLOG" 2>&1 &
    SRV=$!
    for i in $(seq 1 300); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
    if ! grep -q "model loaded" "$SLOG"; then
      echo "  NEVER LOADED" | tee -a "$OUT"; kill -9 "$SRV" 2>/dev/null; sleep 5; continue
    fi
    START=$(date +%s.%N)
    for rep in $(seq 1 $REPS); do
      PIDS=""
      for r in $(seq 1 "$NP"); do
        curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42,"chat_template_kwargs":{"enable_thinking":false}}' \
          -o /dev/null 2>/dev/null &
        PIDS="$PIDS $!"
      done
      for p in $PIDS; do wait "$p"; done
    done
    END=$(date +%s.%N)
    kill -TERM "$SRV" 2>/dev/null; sleep 8; kill -9 "$SRV" 2>/dev/null; sleep 3
    python3 /mnt/raid0/llm/tmp/np_parse.py "$SLOG" "$START" "$END" >> "$OUT"
  done
done
echo "=== SHAPE SWEEP DONE ===" | tee -a "$OUT"
