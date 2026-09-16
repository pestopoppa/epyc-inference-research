#!/bin/bash
# M-12 B5 prefix-reuse smoke. Run it against an ALREADY-RUNNING long-context server;
# it starts, stops and kills nothing.
#
#   scripts/benchmark/smoke_prefix_reuse.sh <port> [35b|27b] [extra smoke_prefix_reuse.py args]
#
# 35b runs the Tulving and BEAM legs (~6 min at the estimated champion rates).
# 27b runs the Tulving leg only (~4.5 min).
# The last line is "SMOKE_PREFIX_REUSE: PASS" or "SMOKE_PREFIX_REUSE: FAIL (...)",
# and the exit code matches (0 = PASS). Procedure and thresholds:
# docs/m12-long-context-gpu-recipe.md.
set -euo pipefail

PORT="${1:?usage: smoke_prefix_reuse.sh <port> [35b|27b] [args...]}"
READER="${2:-35b}"
shift $(( $# >= 2 ? 2 : 1 ))

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

# The Tulving leg needs no pandas. The BEAM leg needs pyarrow: prefer the repo venv
# (it has pyarrow since 2026-09-16), else the delta-Mem venv.
PY=""
for cand in "$REPO/.venv/bin/python" /mnt/raid0/llm/epyc-inference-research/.venv/bin/python \
            /mnt/raid0/llm/delta-Mem/.venv/bin/python; do
    if [ -x "$cand" ] && "$cand" -c "import pyarrow" >/dev/null 2>&1; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
    echo "SMOKE_PREFIX_REUSE: FAIL (no python with pyarrow found)"
    exit 1
fi

case "$READER" in
    35b) ARGS=(--expect-model "Qwen3.6-35B-A3B-MTP-Q8_0" --legs tulving,beam) ;;
    27b) ARGS=(--expect-model "Qwen3.8-27B-Q8_0" --legs tulving) ;;
    *)   echo "SMOKE_PREFIX_REUSE: FAIL (reader must be 35b or 27b, got '$READER')"; exit 1 ;;
esac

exec "$PY" "$HERE/smoke_prefix_reuse.py" --port "$PORT" --expect-slot-ctx 196608 \
    --ubatch 2048 "${ARGS[@]}" "$@"
