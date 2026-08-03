#!/bin/bash
# RAG-only successor. It never runs or reloads the ordinary TB-6 grid.
set -euo pipefail

ART=/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_v8_20260727
PREFILL="$ART/driver/prefill_to_depth_rag.py"
VALIDATOR="$ART/driver/aggregate_np_context_v8.py"
WAIT_S=${QUEUE_WAIT_S:-30}
PORT=18072

fail() { printf 'RAG_QUEUE_FAIL: %s\n' "$*" >&2; exit 1; }
validate_grid() { # model directory, full|a4_bridge
  local base=$1 grid=$2 label
  label=$(basename "$base")
  [[ -f "$base/complete.txt" ]] || return 2
  # A marker on a differently-shaped surface is a completed invalid artifact,
  # not something to wait through.  The shared validator also binds canonical
  # labels to their only permitted mode/grid pair.
  grep -Eq "^mode=[^ ]+ grid=${grid}( |$)" "$base/provenance.txt" || return 1
  python3 "$VALIDATOR" --root "$ART" --label "$label" --require-terminal >/dev/null
}

while :; do
  status=0
  for spec in \
    "A3_ff_fable_non_mtp_q8 full" \
    "A3_ff_fable_mtp_q8 full" \
    "Laguna_ud_iq2_gpu_dflash_off full" \
    "A4_35b_a3b_v8_bridge a4_bridge"; do
    read -r label grid <<<"$spec"
    validate_grid "$ART/$label" "$grid" || { rc=$?; [[ $rc -eq 1 ]] && status=1; [[ $rc -eq 2 && $status -eq 0 ]] && status=2; }
  done
  [[ $status -eq 0 ]] && break
  [[ $status -eq 1 ]] && fail 'a completed grid is invalid; refusing RAG successor'
  sleep "$WAIT_S"
done
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null &&
  fail "port $PORT remained occupied after grid completion"

FABLE_NM=/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf
FABLE_MTP=/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf
LAGUNA=/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ2_M.gguf
A4=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
PREPARED="$ART/prefill_to_depth_rag.prepared.json"
# Precompute this manifest while the ordinary grid owns the GPU. If it is
# absent at transition time, preparing it here is correct but visibly slower.
if [[ ! -f "$PREPARED" ]]; then
  python3 "$PREFILL" --prepare --prepare-out "$PREPARED" \
    --model A3_ff_fable_non_mtp_q8 "$FABLE_NM" 0 \
    --model A3_ff_fable_mtp_q8 "$FABLE_MTP" 1 \
    --model Laguna_ud_iq2_gpu_dflash_off "$LAGUNA" 0 \
    --model A4_35b_a3b_v8_bridge "$A4" 4
fi
python3 "$PREFILL" --execute --prepared "$PREPARED" --label A3_ff_fable_non_mtp_q8 --model-path "$FABLE_NM"
python3 "$PREFILL" --execute --prepared "$PREPARED" --label A3_ff_fable_mtp_q8 --model-path "$FABLE_MTP"
python3 "$PREFILL" --execute --prepared "$PREPARED" --label Laguna_ud_iq2_gpu_dflash_off --model-path "$LAGUNA"
python3 "$PREFILL" --execute --prepared "$PREPARED" --label A4_35b_a3b_v8_bridge --model-path "$A4"
