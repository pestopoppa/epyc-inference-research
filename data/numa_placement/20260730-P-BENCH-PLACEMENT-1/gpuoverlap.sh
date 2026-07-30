#!/bin/bash
# Does a CPU serving instance contend with the GPU lane's host threads?
#
# Structural answer first (gpu_shadow_lane_lease.py): the lane pins host threads
# to logical CPUs 184-191, which fold to PHYSICAL cores 88-95 = region q3. So a
# `0-95` full instance shares physical cores with the lane; a `0-47,96-143` half
# folds to physical 0-47 (q0,q1) and is disjoint. This measures that.
#
# No ROCm llama-server exists on disk (build-v8-hip is a 17 KB stub), so the lane
# is simulated by a WORST-CASE proxy: 8 busy-spinning threads pinned to 184-191,
# i.e. a host thread that never yields. Real HIP host threads submit and wait, so
# this BOUNDS the damage rather than predicting it. If the bound is small, the
# question is answered without building ROCm.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
OUT=/mnt/raid0/llm/tmp/gpuoverlap_results.txt
PORT=19925
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

spin_start () {  # 8 busy threads on 184-191 — one per logical CPU of the lane
  SPIN_PIDS=""
  for c in 184 185 186 187 188 189 190 191; do
    taskset -c $c bash -c 'while :; do :; done' &
    SPIN_PIDS="$SPIN_PIDS $!"
  done
}
spin_stop () { for p in $SPIN_PIDS; do kill -9 "$p" 2>/dev/null; done; SPIN_PIDS=""; sleep 2; }

run () { # label cpuset policy threads spin(0|1)
  local LABEL="$1" CPUSET="$2" POL="$3" TH="$4" SPIN="$5"
  local SL=/mnt/raid0/llm/tmp/gov_${LABEL}.log; : > "$SL"
  echo "=== $LABEL | cpuset=$CPUSET | -t $TH | lane-proxy=$([ "$SPIN" = 1 ] && echo ON || echo off) ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  taskset -c "$CPUSET" numactl $POL "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c 8192 -t "$TH" -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off \
    --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none >> "$SL" 2>&1 &
  local SRV=$!
  for i in $(seq 1 400); do grep -q "model loaded" "$SL" && break; sleep 5; done
  if ! grep -q "model loaded" "$SL"; then echo "    NEVER LOADED" | tee -a "$OUT"; kill -9 $SRV; return; fi
  [ "$SPIN" = 1 ] && spin_start && sleep 3
  for r in 1 2 3; do
    curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
      -o /dev/null 2>/dev/null
  done
  [ "$SPIN" = 1 ] && spin_stop
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SL" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
t=open(sys.argv[1]).read()
r=[float(m.group(1)) for l in t.splitlines() if "eval time =" in l and "prompt eval" not in l
   for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
a=[float(x) for x in re.findall(r"draft acceptance\s*=\s*([\d.]+)", t)]
print(f"    decode tok/s: n={len(r)} median={st.median(r):.2f} min={min(r):.2f} max={max(r):.2f}"
      + (f"  accept={sum(a)/len(a):.3f}" if a else "") if r else "    NO TIMINGS")
PY
}

run full_alone   "0-95"          "--interleave=all"  96 0
run full_withGPU "0-95"          "--interleave=all"  96 1
run half_alone   "0-47,96-143"   "--interleave=0,1"  48 0
run half_withGPU "0-47,96-143"   "--interleave=0,1"  48 1
echo "=== GPUOVERLAP DONE ===" | tee -a "$OUT"
