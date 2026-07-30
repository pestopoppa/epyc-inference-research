#!/bin/bash
# Gap-fill: the production-recipe sweep ran `full` only to np=4, so at T=8 there
# is a half row (gemma 200.12, q35 104.53) and a quarter row (150.57 / 89.25)
# with NO full-machine row to compare against. Without it the T=8 rung cannot be
# read, and T=8 is where the shapes were converging in the baseline data.
# Adds full np=8 and np=16 for the two roles that have spec-dec and quarters.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/gapfill_results.txt
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf
Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf

cell () { # label model ub np [spec...]
  local LABEL="$1" MODEL="$2" UB="$3" NP="$4"; shift 4
  local CTX=$(( 4096 * NP ))
  local SL=/mnt/raid0/llm/tmp/gf_${LABEL}_np${NP}.log; : > "$SL"
  echo "=== $LABEL | full | inst=1 np=$NP T=$NP -t 96 ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$MODEL" --host 127.0.0.1 --port 19935 \
    -np "$NP" -c "$CTX" -t 96 -ub "$UB" -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off "$@" >> "$SL" 2>&1 &
  local SRV=$!
  for i in $(seq 1 400); do grep -q "model loaded" "$SL" && break; sleep 5; done
  if ! grep -q "model loaded" "$SL"; then echo "    NEVER LOADED" | tee -a "$OUT"; kill -9 $SRV; sleep 5; return; fi
  local RP=""
  for r in $(seq 1 "$NP"); do
    curl -s --max-time 1800 "http://127.0.0.1:19935/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
      -o /dev/null 2>/dev/null & RP="$RP $!"
  done
  for p in $RP; do wait "$p"; done
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SL" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
t = open(sys.argv[1]).read()
r = [float(m.group(1)) for l in t.splitlines()
     if "eval time =" in l and "prompt eval" not in l
     for m in [re.search(r"([\d.]+) tokens per second", l)] if m]
a = [float(x) for x in re.findall(r"draft acceptance\s*=\s*([\d.]+)", t)]
print(f"    per-stream median={st.median(r):.2f}  aggregate={sum(r):.2f} tok/s  n={len(r)}"
      + (f"  accept={sum(a)/len(a):.3f}" if a else "  spec=off") if r else "    NO TIMINGS")
PY
}

for NP in 8 16; do cell gemma "$G" 512 $NP -md "$GD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16; done
for NP in 8 16; do cell q35 "$Q35" 8192 $NP --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none; done
echo "=== GAPFILL DONE ===" | tee -a "$OUT"
