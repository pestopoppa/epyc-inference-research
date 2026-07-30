#!/bin/bash
# Does ALLOCATING a large -c cost decode speed, or only memory?
#
# The claim under test: attention runs over the tokens actually present, not the
# allocated capacity, so a 262k-capable instance serving a short prompt should
# decode at the same rate as an 8k-capable one. If true, context can be managed
# LAZILY — provision the maximum and decide when to compact — and the only cost
# of a large -c is resident KV memory. Policy is about to be built on this, so it
# is worth a direct check rather than an argument.
#
# Both models tested because KV size differs by 6x: the 35B allocates 10.9 GiB at
# 262k, gemma 63.8 GiB. If allocation size has any effect at all, gemma shows it.
# Identical short prompt in every cell, so occupancy is held constant and only
# capacity varies.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/ctxalloc_results.txt
PORT=19905
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf

run () { # label model ub ctx [spec...]
  local LABEL="$1" M="$2" UB="$3" CTX="$4"; shift 4
  local SL=/mnt/raid0/llm/tmp/ca_${LABEL}.log; : > "$SL"
  echo "=== $LABEL | -c $CTX ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  local T0=$(date +%s)
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c "$CTX" -t 96 -ub "$UB" -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off "$@" >> "$SL" 2>&1 &
  local SRV=$!
  for i in $(seq 1 400); do grep -q "model loaded" "$SL" && break; sleep 5; done
  if ! grep -q "model loaded" "$SL"; then
    echo "    NEVER LOADED"; tail -3 "$SL" | sed 's/^/      /' | tee -a "$OUT"
    kill -9 $SRV 2>/dev/null; sleep 5; return
  fi
  echo "    load $(( $(date +%s) - T0 ))s   RAM $(free -g | awk '/Mem:/{print $3}') GB" | tee -a "$OUT"
  for r in 1 2 3; do
    curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
      -o /dev/null 2>/dev/null
  done
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SL" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
t=open(sys.argv[1]).read()
r=[float(m.group(1)) for l in t.splitlines() if "eval time =" in l and "prompt eval" not in l
   for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
print(f"    decode tok/s: n={len(r)} median={st.median(r):.2f} min={min(r):.2f} max={max(r):.2f}" if r else "    NO TIMINGS")
PY
}

run q35_c8k    "$Q35" 8192   8192 --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none
run q35_c262k  "$Q35" 8192 262144 --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none
run gemma_c8k  "$G"    512   8192 -md "$GD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16
run gemma_c262k "$G"   512 262144 -md "$GD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16
echo "=== CTXALLOC DONE ===" | tee -a "$OUT"
