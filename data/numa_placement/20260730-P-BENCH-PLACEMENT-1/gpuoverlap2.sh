#!/bin/bash
# Round 2. Round 1 used a bash spin loop on 184-191 and measured ZERO impact —
# but that proxy is almost pure branch prediction: no memory traffic, no vector
# units. CPU decode on this host is bandwidth-bound, so the axis that actually
# matters is whether the GPU lane's host threads steal MEMORY BANDWIDTH from the
# co-resident CPU instance. This proxy streams large buffers instead of spinning.
#
# Still a proxy, and now deliberately a PESSIMISTIC one: real HIP host threads
# spend most of their wait in synchronisation, not in a memory-streaming loop.
# Round 1 is the optimistic bound, this is the pessimistic one; the truth for a
# resident-and-queried GPU model sits between them.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
OUT=/mnt/raid0/llm/tmp/gpuoverlap2_results.txt
PORT=19915
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

cat > /mnt/raid0/llm/tmp/bwburn.py <<'PYEOF'
# Streams a buffer far larger than L3 so every pass goes to DRAM.
import sys, numpy as np
n = 64 * 1024 * 1024 // 8          # 64 MiB of float64, well past any cache
a = np.ones(n); b = np.ones(n)
while True:
    a += b                          # read a, read b, write a -> ~192 MiB/pass
PYEOF

burn_start () {
  BURN_PIDS=""
  for c in 184 185 186 187 188 189 190 191; do
    taskset -c $c python3 /mnt/raid0/llm/tmp/bwburn.py &
    BURN_PIDS="$BURN_PIDS $!"
  done
}
burn_stop () { for p in $BURN_PIDS; do kill -9 "$p" 2>/dev/null; done; BURN_PIDS=""; sleep 3; }

run () { # label cpuset policy threads burn(0|1)
  local LABEL="$1" CPUSET="$2" POL="$3" TH="$4" BURN="$5"
  local SL=/mnt/raid0/llm/tmp/gov2_${LABEL}.log; : > "$SL"
  echo "=== $LABEL | cpuset=$CPUSET | -t $TH | bw-proxy=$([ "$BURN" = 1 ] && echo ON || echo off) ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  taskset -c "$CPUSET" numactl $POL "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c 8192 -t "$TH" -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off \
    --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none >> "$SL" 2>&1 &
  local SRV=$!
  for i in $(seq 1 400); do grep -q "model loaded" "$SL" && break; sleep 5; done
  if ! grep -q "model loaded" "$SL"; then echo "    NEVER LOADED" | tee -a "$OUT"; kill -9 $SRV; return; fi
  [ "$BURN" = 1 ] && burn_start && sleep 5
  for r in 1 2 3; do
    curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
      -o /dev/null 2>/dev/null
  done
  [ "$BURN" = 1 ] && burn_stop
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SL" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
t=open(sys.argv[1]).read()
r=[float(m.group(1)) for l in t.splitlines() if "eval time =" in l and "prompt eval" not in l
   for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
print(f"    decode tok/s: n={len(r)} median={st.median(r):.2f} min={min(r):.2f} max={max(r):.2f}" if r else "    NO TIMINGS")
PY
}

run full_alone   "0-95"        "--interleave=all" 96 0
run full_withBW  "0-95"        "--interleave=all" 96 1
run half_alone   "0-47,96-143" "--interleave=0,1" 48 0
run half_withBW  "0-47,96-143" "--interleave=0,1" 48 1
echo "=== GPUOVERLAP2 DONE ===" | tee -a "$OUT"
