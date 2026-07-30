#!/bin/bash
# Production-OPTIMAL single-stream reference. Every role runs the exact
# acceleration recipe its registry entry specifies — no baselines.
#
#   frontdoor  qwen36_q8_0        draft-mtp self-draft, n_max 4
#   worker     gemma4-26B-A4B     draft-mtp + SEPARATE draft assistant-v6-Q8_0,
#                                 n_max 2, threads-draft 16, ub 512, p-min 0.0
#   architect  qwen35-122B-A10B   draft-mtp self-draft, n_max 4
#   ingest     qwen3-next-80B     acceleration {type: none} — off IS its recipe
#
# Self-draft roles pass NO -md (the orchestrator suppresses it via
# _same_real_model_path); only a genuinely different draft file gets -md.
# All on canonical placement, cold, --no-mmap.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/prodopt_results.txt
PORT=19960
REPS=3
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf
Q122=/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf

run () {
  local NAME="$1" MODEL="$2" UB="$3"; shift 3
  local SLOG=/mnt/raid0/llm/tmp/prodopt_${NAME}.log; : > "$SLOG"
  echo "=== $NAME ===" | tee -a "$OUT"
  echo "    extra: $*" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$MODEL" --host 127.0.0.1 --port $PORT \
    -np 1 -c 8192 -t 96 -ub "$UB" -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off "$@" >> "$SLOG" 2>&1 &
  local SRV=$!
  for i in $(seq 1 400); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
  if ! grep -q "model loaded" "$SLOG"; then
    echo "    NEVER LOADED — check $SLOG" | tee -a "$OUT"
    tail -3 "$SLOG" | sed 's/^/      /' | tee -a "$OUT"
    kill -9 "$SRV" 2>/dev/null; sleep 5; return
  fi
  for r in $(seq 1 $REPS); do
    curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
      -o /dev/null 2>/dev/null
  done
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SLOG" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
t = open(sys.argv[1]).read()
r = [float(m.group(1)) for l in t.splitlines()
     if "eval time =" in l and "prompt eval" not in l
     for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
acc = [float(x) for x in re.findall(r"draft acceptance\s*=\s*([\d.]+)", t)]
if r:
    print(f"    decode tok/s: n={len(r)} median={st.median(r):.2f} min={min(r):.2f} max={max(r):.2f}")
    if acc: print(f"    draft acceptance: mean={sum(acc)/len(acc):.3f}  (spec-dec ACTIVE)")
    else:   print(f"    draft acceptance: none reported (spec-dec inactive)")
else:
    print("    NO TIMINGS")
PY
}

run gemma_PRODSPEC   "$G"    512  -md "$GD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16
run gemma_nospec     "$G"    512
run q122_PRODSPEC    "$Q122" 8192 --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none
run q122_nospec      "$Q122" 8192
run q35_PRODSPEC     "$Q35"  8192 --spec-type draft-mtp --spec-draft-n-max 4 --draft-p-split 0 --device-draft none
echo "=== PRODOPT DONE ===" | tee -a "$OUT"
