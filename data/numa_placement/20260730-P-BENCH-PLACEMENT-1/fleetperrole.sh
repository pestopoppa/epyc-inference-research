#!/bin/bash
# Does one full-machine instance still beat four quarters for the OTHER roles?
# The T-matched comparison so far exists only for the 35B. gemma is A4B and scales
# 5.4x to np=8 where the 35B saturates, so small-active models may still favour
# quartering -- that is exactly what this settles.
#
# Full-machine np curves already measured (npall_results.txt); this fills in the
# 4-quarter fleet at np=1 and np=2, i.e. T=4 and T=8, for a matched comparison.
# --no-mmap + per-node --membind so each quarter genuinely owns node-local weights
# (under shared mmap only one quarter can be local -- see locverify_results.txt).
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/fleetperrole_results.txt
CPUSETS=("0-23,96-119" "24-47,120-143" "48-71,144-167" "72-95,168-191")
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

# gemma26B and qwen3next80B ARE quarterable in production (stack_numa.py gives each
# a full instance plus 4 quarters). qwen35_122B is NOT — architect_general has a
# single instance on 0-95 and no quarters — so its rows below are an exploratory
# counterfactual, kept only to test whether quartering ever helps a large-ACTIVE
# MoE (A10B vs A3B/A4B). Do not read the 122B rows as a production comparison.
MODELS='
gemma26B|/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
qwen3next80B|/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
qwen35_122B|/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
'

echo "$MODELS" | while IFS='|' read -r MNAME MPATH; do
  [ -z "$MNAME" ] && continue
  [ -f "$MPATH" ] || { echo "MISSING $MPATH" | tee -a "$OUT"; continue; }
  echo "##### $MNAME — 4x quarter fleet, --no-mmap, membind per node #####" | tee -a "$OUT"
  for NP in 1 2; do
    CTX=8192
    echo "--- np=$NP per instance  (T=$((4*NP))) ---" | tee -a "$OUT"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    declare -a SP=()
    for q in 0 1 2 3; do
      SLOG=/mnt/raid0/llm/tmp/fpr_${MNAME}_np${NP}_q${q}.log; : > "$SLOG"
      taskset -c "${CPUSETS[$q]}" numactl --membind=$q "$LS" -m "$MPATH" \
        --host 127.0.0.1 --port $((19940 + q)) -np "$NP" -c "$CTX" -t 24 -ub 8192 \
        -ctk q8_0 -ctv q8_0 --flash-attn on --jinja --no-mmap --device none \
        --log-colors off >> "$SLOG" 2>&1 &
      SP[$q]=$!
    done
    UP=0
    for t in $(seq 1 500); do
      UP=1
      for q in 0 1 2 3; do grep -q "model loaded" /mnt/raid0/llm/tmp/fpr_${MNAME}_np${NP}_q${q}.log || UP=0; done
      [ "$UP" = "1" ] && break; sleep 5
    done
    if [ "$UP" != "1" ]; then
      echo "  NOT ALL LOADED" | tee -a "$OUT"
    else
      RP=""
      for q in 0 1 2 3; do
        for r in $(seq 1 "$NP"); do
          curl -s --max-time 1800 "http://127.0.0.1:$((19940 + q))/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"messages":[{"role":"user","content":"Write a Python function returning the first n Fibonacci numbers, then explain it in three sentences."}],"max_tokens":256,"temperature":0.3,"seed":42}' \
            -o /dev/null 2>/dev/null &
          RP="$RP $!"
        done
      done
      for p in $RP; do wait "$p"; done
      python3 - "$MNAME" "$NP" <<'PY' | tee -a "$OUT"
import re, sys, statistics
m, np_ = sys.argv[1], sys.argv[2]
rates = []
for q in range(4):
    for line in open(f"/mnt/raid0/llm/tmp/fpr_{m}_np{np_}_q{q}.log"):
        if "eval time =" in line and "prompt eval" not in line:
            g = re.search(r"([\d.]+) tokens per second", line)
            if g: rates.append(float(g.group(1)))
if rates:
    print(f"  per-stream tok/s: n={len(rates)} min={min(rates):.2f} median={statistics.median(rates):.2f} max={max(rates):.2f}")
    print(f"  FLEET AGGREGATE : {sum(rates):.2f} tok/s")
else:
    print("  NO TIMINGS")
PY
    fi
    for q in 0 1 2 3; do kill -TERM "${SP[$q]}" 2>/dev/null; done; sleep 12
    for q in 0 1 2 3; do kill -9 "${SP[$q]}" 2>/dev/null; done; sleep 5
  done
done
echo "=== FLEETPERROLE DONE ===" | tee -a "$OUT"
