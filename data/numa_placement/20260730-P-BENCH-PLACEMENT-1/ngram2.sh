#!/bin/bash
# ngram on REALISTIC long context. Round 1 used synthetic filler whose 5-grams
# were 99.7% repeats (23 distinct in 8,736 words) — close to a best case for a
# context-matching drafter. These prompts are real repo source + docs: 10.6%
# repeat rate, 5,162 distinct 5-grams.
#
# Includes ingest_long_context, whose registry says acceleration {type: none}
# because the SSM/MoE hybrid has no working DRAFT-MODEL path. ngram-mod needs no
# draft model — it drafts from context — so it may give that role speculation for
# the first time. That is a hypothesis, not an expectation.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/ngram2_results.txt
PORT=19875
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf
Q122=/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-Q4_K_M/Qwen3.5-122B-A10B-UD-Q4_K_M-00001-of-00003.gguf
Q80=/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/Qwen3-Next-80B-A3B-Instruct-Q4_K_M.gguf

cell () { # label model ub spectype prompt [extra...]
  local LABEL="$1" M="$2" UB="$3" ST="$4" P="$5"; shift 5
  local SL=/mnt/raid0/llm/tmp/n2_${LABEL}_${P}.log; : > "$SL"
  echo "=== $LABEL | spec=$ST | $P ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  local SPECARG=(); [ "$ST" != "none" ] && SPECARG=(--spec-type "$ST")
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c 65536 -t 96 -ub "$UB" -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off "${SPECARG[@]}" "$@" >> "$SL" 2>&1 &
  local SRV=$!
  for i in $(seq 1 400); do grep -q "model loaded" "$SL" && break; sleep 5; done
  if ! grep -q "model loaded" "$SL"; then
    echo "    NEVER LOADED"; tail -3 "$SL" | sed 's/^/      /' | tee -a "$OUT"
    kill -9 $SRV 2>/dev/null; sleep 5; return
  fi
  for r in 1 2; do
    curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' -d @/mnt/raid0/llm/tmp/req_${P}.json \
      -o /dev/null 2>/dev/null
  done
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SL" <<'PY' | tee -a "$OUT"
import re, sys, statistics as st
dec, acc, ptok = [], [], None
for l in open(sys.argv[1], errors="ignore"):
    if "prompt eval time =" in l:
        m = re.search(r"/\s+(\d+) tokens", l)
        if m and ptok is None: ptok = int(m.group(1))     # FIRST = the cold prefill
    elif "eval time =" in l:
        m = re.search(r"([\d.]+) tokens per second", l)
        if m: dec.append(float(m.group(1)))
    elif "draft acceptance" in l:
        m = re.search(r"draft acceptance\s*=\s*([\d.]+)", l)
        if m: acc.append(float(m.group(1)))
print(f"    prompt={ptok} tok  DECODE={st.median(dec):6.2f} tok/s"
      + (f"  accept={st.median(acc):.3f}" if acc else "  (no draft stats)") if dec else "    NO TIMINGS")
PY
}

M35=(--spec-draft-n-max 4 --device-draft none)
MG=(-md "$GD" --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16)

cell q35_mtp        "$Q35"  8192 "draft-mtp"           r8k  "${M35[@]}"
cell q35_ngrammtp   "$Q35"  8192 "ngram-mod,draft-mtp" r8k  "${M35[@]}"
cell q35_mtp        "$Q35"  8192 "draft-mtp"           r32k "${M35[@]}"
cell q35_ngrammtp   "$Q35"  8192 "ngram-mod,draft-mtp" r32k "${M35[@]}"
cell gemma_mtp        "$G"   512 "draft-mtp"           r8k  "${MG[@]}"
cell gemma_ngrammtp   "$G"   512 "ngram-mod,draft-mtp" r8k  "${MG[@]}"
cell q80_nospec     "$Q80"  8192 "none"                r8k
cell q80_ngram      "$Q80"  8192 "ngram-mod"           r8k
cell q122_mtp       "$Q122" 8192 "draft-mtp"           r8k  "${M35[@]}"
cell q122_ngrammtp  "$Q122" 8192 "ngram-mod,draft-mtp" r8k  "${M35[@]}"
echo "=== NGRAM2 DONE ===" | tee -a "$OUT"
