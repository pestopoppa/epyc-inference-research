#!/bin/bash
# Directly-measured multi-instance grid, replacing the extrapolated columns.
# Every instance gets --no-mmap so its --membind/--interleave actually owns private,
# node-local pages (with shared mmap only one instance can be local -- see locverify).
# Thread counts are PHYSICAL cores only; SMT oversubscription measured -8..-13%.
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/fleetgrid_results.txt
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

run_fleet () {
  local LABEL="$1" NINST="$2" NP="$3" THREADS="$4"; shift 4
  local -a CPUSETS=() POLICIES=()
  local i=0
  while [ $i -lt "$NINST" ]; do CPUSETS[$i]="$1"; POLICIES[$i]="$2"; shift 2; i=$((i+1)); done
  local CTX=$(( 2048 * NP )); [ "$CTX" -lt 8192 ] && CTX=8192
  echo "=== $LABEL  ${NINST}x instances  np=$NP  T=$((NINST*NP))  -t $THREADS ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  local -a SP=()
  for i in $(seq 0 $((NINST-1))); do
    local SLOG=/mnt/raid0/llm/tmp/fg_${LABEL}_np${NP}_i${i}.log; : > "$SLOG"
    taskset -c "${CPUSETS[$i]}" numactl ${POLICIES[$i]} "$LS" -m "$M" \
      --host 127.0.0.1 --port $((19920 + i)) -np "$NP" -c "$CTX" -t "$THREADS" -ub 8192 \
      -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --no-mmap --device none \
      --spec-type draft-mtp --spec-draft-n-max 4 --draft-p-split 0 --device-draft none \
      --log-colors off >> "$SLOG" 2>&1 &
    SP[$i]=$!
  done
  local UP=0
  for t in $(seq 1 400); do
    UP=1
    for i in $(seq 0 $((NINST-1))); do
      grep -q "model loaded" /mnt/raid0/llm/tmp/fg_${LABEL}_np${NP}_i${i}.log || UP=0
    done
    [ "$UP" = "1" ] && break; sleep 5
  done
  if [ "$UP" != "1" ]; then
    echo "  NOT ALL LOADED" | tee -a "$OUT"
  else
    local RP=""
    for i in $(seq 0 $((NINST-1))); do
      for r in $(seq 1 "$NP"); do
        curl -s --max-time 900 "http://127.0.0.1:$((19920 + i))/v1/chat/completions" \
          -H 'Content-Type: application/json' \
          -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42,"chat_template_kwargs":{"enable_thinking":false}}' \
          -o /dev/null 2>/dev/null &
        RP="$RP $!"
      done
    done
    for p in $RP; do wait "$p"; done
    python3 - "$LABEL" "$NP" "$NINST" <<'PY' | tee -a "$OUT"
import re, sys, glob, statistics
label, np_, ninst = sys.argv[1], sys.argv[2], int(sys.argv[3])
rates = []
for i in range(ninst):
    for line in open(f"/mnt/raid0/llm/tmp/fg_{label}_np{np_}_i{i}.log"):
        if "eval time =" in line and "prompt eval" not in line:
            m = re.search(r"([\d.]+) tokens per second", line)
            if m: rates.append(float(m.group(1)))
if rates:
    print(f"  per-stream tok/s: n={len(rates)} min={min(rates):.2f} "
          f"median={statistics.median(rates):.2f} max={max(rates):.2f}")
    print(f"  FLEET AGGREGATE : {sum(rates):.2f} tok/s")
else:
    print("  NO TIMINGS")
PY
  fi
  for i in $(seq 0 $((NINST-1))); do kill -TERM "${SP[$i]}" 2>/dev/null; done; sleep 10
  for i in $(seq 0 $((NINST-1))); do kill -9 "${SP[$i]}" 2>/dev/null; done; sleep 5
}

H0="0-47,96-143"; H1="48-95,144-191"
Q0="0-23,96-119"; Q1="24-47,120-143"; Q2="48-71,144-167"; Q3="72-95,168-191"

for NP in 1 2 4 8; do
  run_fleet HALF2 2 "$NP" 48 "$H0" "--interleave=0,1" "$H1" "--interleave=2,3"
done
for NP in 1 2 4 8; do
  run_fleet QUAD4 4 "$NP" 24 "$Q0" "--membind=0" "$Q1" "--membind=1" "$Q2" "--membind=2" "$Q3" "--membind=3"
done
echo "=== FLEETGRID DONE ===" | tee -a "$OUT"
