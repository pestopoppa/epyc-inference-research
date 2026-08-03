#!/bin/bash
# GPU-only np × context throughput surface (TB-6). For each (np, generation-length L)
# cell: aggregate decode t/s + per-request decode + timeout/error rate. Maps where
# batching helps (small L) vs collapses (large L). Per-slot ctx = L (so -c = L*np);
# verifies n_ctx_slot >= L and VRAM headroom before each cell (no OOM->truncation).
set -uo pipefail
source /mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_20260723/driver/gpu_lib.sh
ART=/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_20260723
RES=/mnt/raid0/llm/epyc-inference-research
PIN=/mnt/raid0/llm/epyc-inference-research/artifacts/architect-bench-gpu-20260720/questions_olympiadbench_hard.json
mkdir -p "$ART"

cell() {  # cell <model-label> <model> <mtp_n> <np> <L>
  local lbl="$1" M="$2" MTPN="$3" NP="$4" L="$5"
  local c=$(( L * NP )); local d="$ART/${lbl}/np${NP}_L${L}"; mkdir -p "$d"; rm -f "$d/pq.jsonl"
  # cap requests so a cell is bounded: run 2*np items (fills every slot ~twice)
  local NREQ=$NP  # one wave fills every slot once — enough for a throughput read
  gpu_launch "$d" "$M" -np "$NP" -c "$c" -t 8 -tb 8 -b 2048 -ub 2048 -ctk f16 -ctv f16 \
    --spec-type draft-mtp --spec-draft-n-max "$MTPN"
  local st; st=$(gpu_wait "$d" 500)
  [ "$st" != "HEALTHY" ] && { echo "$lbl np=$NP L=$L: SERVER_FAIL($st)"; tail -3 "$d/server.stderr"; gpu_kill "$d"; return; }
  local nslot; nslot=$(grep -oiP 'n_ctx_per_seq\s*=\s*\K[0-9]+|n_ctx_slot\s*=\s*\K[0-9]+' "$d/server.stderr" | tail -1)
  local vram; vram=$(( $(rocm-smi --showmeminfo vram 2>/dev/null | grep -oiP 'used.*?:\s*\K[0-9]+' | head -1) / 1073741824 ))
  if [ "${nslot:-0}" -lt "$L" ] || [ "$vram" -gt 61 ]; then echo "$lbl np=$NP L=$L: SKIP (n_ctx_slot=$nslot vram=${vram}G)"; gpu_kill "$d"; return; fi
  cd "$RES"
  HF_HOME=/mnt/raid0/llm/cache/huggingface RUNNER_REQUEST_TIMEOUT_S=5400 \
  uv run python scripts/benchmark/v7_quality_gate_runner.py \
    --port 18072 --host 127.0.0.1 --suites olympiadbench_hard --n 155 --limit "$NREQ" --seed 42 \
    --max-tokens "$L" --repeats 1 --concurrency "$NP" \
    --temperature 0.6 --top-p 0.95 --top-k 20 --no-enable-thinking --endpoint chat --arm "${lbl}_np${NP}_L${L}" \
    --binary x --models "$M" --questions-in "$PIN" \
    --per-question-out "$d/pq.jsonl" --output "$d/r.json" > "$d/out" 2> "$d/err" || true
  gpu_kill "$d"
  python3 -c "
import json,statistics
try:
    r=json.load(open('$d/r.json'))['suites'][0]; tp=r.get('throughput',{})
    pq=[json.loads(l) for l in open('$d/pq.jsonl')]
    dec=[x['decode_tok_s'] for x in pq if x.get('decode_tok_s')]
    print('$lbl np=$NP L=$L: agg=%6.1f | per-req med=%5.1f | err=%d | vram=${vram}G' % (
      tp.get('aggregate_decode_tok_s',0), statistics.median(dec) if dec else 0, r.get('errors',0)))
except Exception as e: print('$lbl np=$NP L=$L: PARSE_FAIL', e)"
}

while ps -eo comm | grep -qx "llama-server" && rocm-smi --showmemuse 2>/dev/null | grep -q "VRAM%): [1-9]"; do sleep 20; done
echo "=== np × L throughput surface (first pass) — A1 122B-IQ2 ==="
echo "=== combine with existing 36864-ctx probe data: np1=58 np2=44 np4=62 (agg t/s) ==="
A1=/mnt/raid0/llm/models/Qwen3.5-122B-A10B-MTP-GGUF/UD-IQ2_M/Qwen3.5-122B-A10B-UD-IQ2_M.gguf
# small context (short reasoning budget): sweep to np=32 — AXA-1 B32~148 regime where batching should pay off
for NP in 1 2 4 8 16 32; do cell A1_122b_iq2 "$A1" 2 "$NP" 2048; done
# large context (big reasoning budget): sweep to np=16 — KV/bandwidth-bound; guard SKIPs any OOM cell
for NP in 1 2 4 8 16; do cell A1_122b_iq2 "$A1" 2 "$NP" 8192; done
echo "=== NP-CONTEXT STUDY (first pass) COMPLETE ==="
