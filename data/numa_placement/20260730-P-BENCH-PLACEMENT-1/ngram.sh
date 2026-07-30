#!/bin/bash
# INDICATIVE ngram test — is `ngram-mod,draft-mtp` worth a full study?
#
# Production launches `--spec-type draft-mtp` alone. The registry carries
# `ngram_candidate_spec_type: ngram-mod,draft-mtp` as a CANDIDATE, never
# deployed, and records a large GPU win for it (MI210 2026-07-19: no-spec 74.15,
# draft-mtp 113.45, ngram-mod+draft-mtp 183.50 on a repetitive fixture; 143.58 vs
# 330.57 prompt-diverse). Untested on CPU.
#
# Hypothesis worth the compute: ngram drafts by matching against the text already
# in context, so it should get STRONGER as context grows — exactly where MTP
# acceptance decays (frontdoor: 0.746 at 28 tok -> 0.429 at 35k). If so it
# offsets the long-context penalty rather than merely adding to the short-context
# rate, and every context-curve point must be re-measured with it.
#
# Deliberately SMALL: 2 roles x 2 spec modes x 2 depths, n=2. Enough to see a
# significant effect, cheap enough to discard if there is none. A full ladder
# follows only if this pays.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/ngram_results.txt
PORT=19885
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

Q35=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
G=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
GD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf

cell () { # label model ub spectype prompt [extra...]
  local LABEL="$1" M="$2" UB="$3" ST="$4" P="$5"; shift 5
  local SL=/mnt/raid0/llm/tmp/ng_${LABEL}_${P}.log; : > "$SL"
  echo "=== $LABEL | spec=$ST | $P ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 1 -c 65536 -t 96 -ub "$UB" -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off --spec-type "$ST" "$@" >> "$SL" 2>&1 &
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
dec, pre, acc, ptok = [], [], [], None
for l in open(sys.argv[1], errors="ignore"):
    if "prompt eval time =" in l:
        m = re.search(r"/\s+(\d+) tokens \([^)]*?([\d.]+) tokens per second", l)
        if m: ptok = int(m.group(1)); pre.append(float(m.group(2)))
    elif "eval time =" in l:
        m = re.search(r"([\d.]+) tokens per second", l)
        if m: dec.append(float(m.group(1)))
    elif "draft acceptance" in l:
        m = re.search(r"draft acceptance\s*=\s*([\d.]+)", l)
        if m: acc.append(float(m.group(1)))
print(f"    prompt={ptok} tok  prefill={st.median(pre):7.2f}  DECODE={st.median(dec):6.2f} tok/s"
      + (f"  accept={st.median(acc):.3f}" if acc else "") if dec else "    NO TIMINGS")
PY
}

MTP35=(--spec-draft-n-max 4 --device-draft none)
MTPG=(-md "$GD" --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16)

for P in p0k5 p8k; do
  cell q35_mtp      "$Q35" 8192 "draft-mtp"           "$P" "${MTP35[@]}"
  cell q35_ngrammtp "$Q35" 8192 "ngram-mod,draft-mtp" "$P" "${MTP35[@]}"
  cell gemma_mtp      "$G" 512 "draft-mtp"           "$P" "${MTPG[@]}"
  cell gemma_ngrammtp "$G" 512 "ngram-mod,draft-mtp" "$P" "${MTPG[@]}"
done
echo "=== NGRAM DONE ===" | tee -a "$OUT"
