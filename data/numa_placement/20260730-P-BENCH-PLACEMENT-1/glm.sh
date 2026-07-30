#!/bin/bash
# GLM-5.2 UD-IQ2_M (238.6 GB, 6 shards) — basic full-instance np=1 reference.
# This model is structurally full-machine-only: at ~222 GiB of weights it would
# fit on a single NPS4 node (~263 GiB free) with almost no KV headroom, so a
# quarter or membind shape is not viable for it.
#
# Two arms, to check whether any prior number for it was taken on the defective
# placement:
#   A_prod    = straddling 0-47,96-143, no numactl (the mis-wired shape)
#   C_fullint = canonical 0-95 + --interleave=all
# No speculation: no draft model exists for this GGUF.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
M=/mnt/raid0/llm/models/GLM-5.2-UD-IQ2_M/UD-IQ2_M/GLM-5.2-UD-IQ2_M-00001-of-00006.gguf
OUT=/mnt/raid0/llm/tmp/glm_results.txt
PORT=19940
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"
[ -f "$M" ] || { echo "MISSING $M" | tee -a "$OUT"; exit 1; }

for ARM in C_fullint A_prod; do
  if [ "$ARM" = "C_fullint" ]; then CPUSET="0-95"; POL="--interleave=all"
  else CPUSET="0-47,96-143"; POL=""; fi
  SLOG=/mnt/raid0/llm/tmp/glm_${ARM}.log; : > "$SLOG"
  echo "=== $ARM  cpuset=$CPUSET policy=${POL:-none} ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  START=$(date +%s)
  taskset -c "$CPUSET" numactl $POL "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c 8192 -t 96 -ub 8192 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off >> "$SLOG" 2>&1 &
  SRV=$!
  # 238 GB at ~2.5 GB/s is ~95s cold; allow generously
  for i in $(seq 1 600); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
  if ! grep -q "model loaded" "$SLOG"; then
    echo "  NEVER LOADED after $(( $(date +%s) - START ))s" | tee -a "$OUT"
    tail -4 "$SLOG" | sed 's/^/    /' | tee -a "$OUT"
    kill -9 "$SRV" 2>/dev/null; sleep 8; continue
  fi
  echo "  load: $(( $(date +%s) - START ))s   RAM used: $(free -g | awk '/Mem:/{print $3}') GB" | tee -a "$OUT"
  for r in 1 2 3; do
    curl -s --max-time 2400 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
      -o /dev/null 2>/dev/null
  done
  kill -TERM "$SRV" 2>/dev/null; sleep 15; kill -9 "$SRV" 2>/dev/null; sleep 6
  python3 - "$SLOG" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
t = open(sys.argv[1]).read()
r = [float(m.group(1)) for l in t.splitlines()
     if "eval time =" in l and "prompt eval" not in l
     for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
p = [float(m.group(1)) for l in t.splitlines() if "prompt eval time" in l
     for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
print(f"  decode tok/s: n={len(r)} median={st.median(r):.2f} min={min(r):.2f} max={max(r):.2f}" if r
      else "  NO DECODE TIMINGS")
if p: print(f"  prefill tok/s: median={st.median(p):.2f}")
PY
done
echo "=== GLM DONE ===" | tee -a "$OUT"
