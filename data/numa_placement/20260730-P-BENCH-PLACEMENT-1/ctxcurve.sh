#!/bin/bash
# Context-length degradation curve, on PRODUCTION recipes.
#
# This axis has only ever been measured with speculation OFF, which is not just
# incomplete but potentially misleading: draft acceptance can itself move with
# context length, so a spec-off curve is not a reliable proxy for the production
# one. Acceptance is therefore recorded per cell alongside the rate.
#
# One instance, canonical placement, np=1, so the only variable is how many
# tokens are actually resident. -c is set to comfortably exceed the prompt in
# every cell, so capacity is NOT the variable being swept — occupancy is.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/ctxcurve_results.txt
PORT=19895
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf
Q122=/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
Q80=/mnt/raid0/llm/models/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf
[ -f "$Q80" ] || Q80=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf

curve () { # label model ub [spec...]
  local LABEL="$1" M="$2" UB="$3"; shift 3
  echo "##### $LABEL #####" | tee -a "$OUT"
  for P in p0k5 p8k p32k p128k; do
    local SL=/mnt/raid0/llm/tmp/cc_${LABEL}_${P}.log; : > "$SL"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
      -np 1 -c 196608 -t 96 -ub "$UB" -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
      --no-mmap --device none --log-colors off "$@" >> "$SL" 2>&1 &
    local SRV=$!
    for i in $(seq 1 500); do grep -q "model loaded" "$SL" && break; sleep 5; done
    if ! grep -q "model loaded" "$SL"; then
      echo "  $P : NEVER LOADED" | tee -a "$OUT"; kill -9 $SRV 2>/dev/null; sleep 6; continue
    fi
    curl -s --max-time 3600 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @/mnt/raid0/llm/tmp/req_${P}.json \
      -o /dev/null 2>/dev/null
    kill -TERM "$SRV" 2>/dev/null; sleep 12; kill -9 "$SRV" 2>/dev/null; sleep 5
    python3 - "$SL" "$P" <<'PY' | tee -a "$OUT"
import re, sys
t = open(sys.argv[1]).read()
dec = [(int(m.group(1)), float(m.group(2))) for m in
       re.finditer(r"eval time =\s+[\d.]+ ms /\s+(\d+) tokens \([^)]*?([\d.]+) tokens per second", t)
       if "prompt eval" not in t[max(0,m.start()-40):m.start()]]
pe = re.findall(r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens \([^)]*?([\d.]+) tokens per second", t)
acc = re.findall(r"draft acceptance\s*=\s*([\d.]+)", t)
d = [r for n, r in dec]
print(f"  {sys.argv[2]:6}: prompt={pe[-1][0] if pe else '?':>7} tok  "
      f"prefill={float(pe[-1][1]) if pe else 0:7.2f}  DECODE={max(d) if d else 0:6.2f} tok/s"
      + (f"  accept={float(acc[-1]):.3f}" if acc else "  spec=off"))
PY
  done
}

curve frontdoor_q35   "$Q35"  8192 --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none
curve worker_gemma    "$G"     512 -md "$GD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16
curve ingest_q80      "$Q80"  8192
curve architect_q122  "$Q122" 8192 --spec-type draft-mtp --spec-draft-n-max 4 --device-draft none
echo "=== CTXCURVE DONE ===" | tee -a "$OUT"
