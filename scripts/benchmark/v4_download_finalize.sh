#!/usr/bin/env bash
# v4_download_finalize.sh — watch for V4 Q4 download completion, verify integrity,
# and rename .tmp → final .gguf so the bench can pick it up.
#
# Usage:
#   v4_download_finalize.sh            # watches indefinitely
#   v4_download_finalize.sh --once     # checks once and exits
#
# Idempotent: safe to re-run. If the final .gguf already exists, exits success
# immediately. If only .tmp exists and is full size, renames + exits. If .tmp
# is partial, prints progress and continues watching (unless --once).

set -euo pipefail

MODEL_DIR=/mnt/raid0/llm/models/deepseek-v4-flash
TARGET_BASENAME=DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf
TMP_FILE="${MODEL_DIR}/${TARGET_BASENAME}.tmp"
FINAL_FILE="${MODEL_DIR}/${TARGET_BASENAME}"
HF_URL="https://huggingface.co/antirez/deepseek-v4-gguf/resolve/main/${TARGET_BASENAME}"

# Resolve the actual target size dynamically by following the HF redirect to
# the signed S3-style URL and reading its Content-Length. This avoids hardcoding
# the byte count (which also gets flagged by the PII precommit hook as a long
# digit run) and tracks the canonical file size from the source of truth.
fetch_expected_bytes() {
    local resolved=$(curl -sI "$HF_URL" | grep -i '^location:' | awk '{print $2}' | tr -d '\r\n')
    if [[ -z "$resolved" ]]; then
        echo "ERROR: could not resolve HF redirect for $HF_URL" >&2
        return 1
    fi
    local size=$(curl -sI "$resolved" | grep -i '^content-length:' | awk '{print $2}' | tr -d '\r\n')
    if [[ -z "$size" ]]; then
        echo "ERROR: could not read Content-Length from $resolved" >&2
        return 1
    fi
    echo "$size"
}

EXPECTED_BYTES=$(fetch_expected_bytes) || exit 1
ONCE=0
[[ "${1:-}" == "--once" ]] && ONCE=1

finalize_if_complete() {
    if [[ -f "$FINAL_FILE" ]]; then
        echo "OK: final file already present"
        echo "  $FINAL_FILE"
        echo "  size: $(stat -c %s "$FINAL_FILE") bytes"
        return 0
    fi
    if [[ ! -f "$TMP_FILE" ]]; then
        echo "ERROR: neither $FINAL_FILE nor $TMP_FILE exists"
        return 2
    fi
    local actual=$(stat -c %s "$TMP_FILE")
    if [[ "$actual" -eq "$EXPECTED_BYTES" ]]; then
        echo "DONE: size matches; renaming .tmp -> final"
        mv "$TMP_FILE" "$FINAL_FILE"
        echo "  $FINAL_FILE"
        echo "  size: $(stat -c %s "$FINAL_FILE") bytes"
        return 0
    fi
    if [[ "$actual" -gt "$EXPECTED_BYTES" ]]; then
        echo "ERROR: $TMP_FILE is LARGER than expected ($actual vs $EXPECTED_BYTES) — investigate"
        return 3
    fi
    # Still downloading
    local pct=$(( actual * 100 / EXPECTED_BYTES ))
    local gb_done=$(( actual / 1024 / 1024 / 1024 ))
    local gb_target=$(( EXPECTED_BYTES / 1024 / 1024 / 1024 ))
    echo "PROGRESS: ${gb_done} GB / ${gb_target} GB (${pct}%)"
    return 1
}

if [[ "$ONCE" -eq 1 ]]; then
    finalize_if_complete
    exit $?
fi

# Watching mode
echo "Watching for V4 Q4 download completion..."
echo "  tmp:   $TMP_FILE"
echo "  final: $FINAL_FILE"
echo "  expected size: $EXPECTED_BYTES bytes"
echo ""

while true; do
    if finalize_if_complete; then
        echo ""
        echo "Next: run throughput gate with"
        echo "  bench_canonical.sh --v4-fork --perf -m '$FINAL_FILE'"
        exit 0
    fi
    rc=$?
    if [[ "$rc" -ge 2 ]]; then
        # Error condition — bail
        exit "$rc"
    fi
    # rc == 1 means still downloading; sleep and retry
    sleep 60
done
