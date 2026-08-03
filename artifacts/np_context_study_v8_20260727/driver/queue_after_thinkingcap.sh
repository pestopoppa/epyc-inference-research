#!/bin/bash
# Detached successor queue for the v8 TB-6 campaign.  It never manipulates the
# current ThinkingCap session or port; it begins only after that session proves
# complete.  A static -np/grid cell reload is mandated by llama-server startup.
set -euo pipefail

ART=/mnt/raid0/llm/epyc-inference-research/artifacts/np_context_study_v8_20260727
RES=/mnt/raid0/llm/epyc-inference-research
DRV="$ART/driver/run_model_block.sh"
VALIDATOR="$ART/driver/aggregate_np_context_v8.py"
TC="$ART/A3_tc_thinkingcap_q8"
IDENTITY_MANIFEST="$ART/prefill_to_depth_rag.prepared.json"
IDENTITY_MANIFEST_SHA=e29fd6a0536c21ada180fcc929ba5b10da6fe498693abb9bf97459f374a2e7da
PORT=18072
WAIT_S=${QUEUE_WAIT_S:-30}

fail() { printf 'QUEUE_FAIL: %s\n' "$*" >&2; exit 1; }
require_size_and_tensors() {
  local p=$1 bytes=$2 tensors=$3 got
  [[ $(stat -c %s "$p") == "$bytes" ]] || fail "byte mismatch $p"
  # Read the producer to EOF: an early-closing grep under pipefail sends
  # llama-gguf SIGPIPE and terminates the queue before it reaches the wait loop.
  got=$(/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-gguf "$p" r n 2>/dev/null |
    awk '/n_tensors:/ { value=$NF } END { print value }')
  [[ "$got" == "$tensors" ]] || fail "tensor mismatch $p: $got"
}
validate_cached_identities() {
  [[ -f "$IDENTITY_MANIFEST" ]] || fail "missing prepared identity manifest"
  [[ $(sha256sum "$IDENTITY_MANIFEST" | awk '{print $1}') == "$IDENTITY_MANIFEST_SHA" ]] ||
    fail "prepared identity manifest hash mismatch"
  python3 - "$IDENTITY_MANIFEST" "$FABLE_NM" "$FABLE_MTP" "$LAGUNA" "$A4" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
expected = {
    sys.argv[2]: "2fff409d4a22e0cb11fb0ecfafed1c669b9808f7e6bc499036c6e85297f14f4d",
    sys.argv[3]: "041c175f03b76adb70077ba470258f6b916ec4f5f066077377ef96396c3dd1d0",
    sys.argv[4]: "1a0d44795f71044de1a9671bf70def4655f4ab7294b002263dfc8046820bfd2c",
    sys.argv[5]: "c1283d8b80c3e38b2735ddbc9766d3b3126f44d6c484be419d4e101d09a76131",
}
rows = {row["path"]: row for row in manifest.get("models", [])}
if set(rows) != set(expected):
    raise SystemExit("prepared model path set differs from successor queue")
for raw_path, digest in expected.items():
    path, row = pathlib.Path(raw_path), rows[raw_path]
    stat = path.stat()
    current = {"path": str(path), "inode": stat.st_ino, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if row.get("sha256") != digest or any(row.get(key) != value for key, value in current.items()):
        raise SystemExit(f"prepared model identity drift: {path}")
PY
}
validate_thinkingcap() {
  python3 "$VALIDATOR" --root "$ART" --label A3_tc_thinkingcap_q8 --require-terminal >/dev/null &&
    python3 "$VALIDATOR" --validate-quality "$TC/quality_swebench_oracle" \
      --quality-label A3_tc_thinkingcap_q8 --quality-model "$THINKINGCAP" --quality-suite swebench_oracle \
      --quality-n 40 --quality-max-tokens 3072 --quality-questions "$RES/artifacts/architect-code-eval-20260724/questions_swebench_oracle.json" \
      --quality-thinking true >/dev/null &&
    python3 "$VALIDATOR" --validate-quality "$TC/quality_livecodebench_hard" \
      --quality-label A3_tc_thinkingcap_q8 --quality-model "$THINKINGCAP" --quality-suite livecodebench_hard \
      --quality-n 53 --quality-max-tokens 4096 --quality-questions "$RES/artifacts/architect-code-eval-20260724/questions_livecodebench_hard.json" \
      --quality-thinking true >/dev/null
}

FABLE_NM=/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-Q8_0.gguf
FABLE_MTP=/mnt/raid0/llm/models/Qwen3.6-27B-Fable-Fusion-711-GGUF/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q8_0.gguf
LAGUNA=/mnt/raid0/llm/models/Laguna-S-2.1-GGUF/Laguna-S-2.1-UD-IQ2_M.gguf
A4=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
THINKINGCAP=/mnt/raid0/llm/models/ThinkingCap-Qwen3.6-27B-GGUF/ThinkingCap-Qwen3.6-27B-Q8_0.gguf
# The full files were already hashed into the immutable prepared manifest while
# ThinkingCap owned the GPU. Revalidate inode/size/mtime plus those cached
# digests so supervisor restarts do not reread roughly 127 GB.
validate_cached_identities
require_size_and_tensors "$A4" 37801097504 753
require_size_and_tensors "$FABLE_NM" 29787701792 851
require_size_and_tensors "$FABLE_MTP" 30239022560 866
while ! validate_thinkingcap; do sleep "$WAIT_S"; done
for _ in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || break
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null &&
  fail "port $PORT remained occupied after ThinkingCap completion"

"$DRV" A3_ff_fable_non_mtp_q8 "$FABLE_NM" 0 false full full
"$DRV" A3_ff_fable_mtp_q8 "$FABLE_MTP" 1 false full full
printf 'paired_with=A3_ff_fable_non_mtp_q8\nmodel_tensor_contract=851-base-plus-15-mtp\n' > "$ART/A3_ff_fable_mtp_q8/paired_provenance.txt"
"$DRV" Laguna_ud_iq2_gpu_dflash_off "$LAGUNA" 0 false throughput_only full
"$DRV" A4_35b_a3b_v8_bridge "$A4" 4 false throughput_only a4_bridge
