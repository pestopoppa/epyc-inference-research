#!/bin/bash
# YARN-JUDGE-1: ready-to-run judging window script. Refuses unless WINDOW_CONFIRMED=1.
# It does NOT start a server: the operator/main serves the judge model (the worker model) on
# localhost:$JUDGE_PORT first, OUTSIDE any YaRN timing window, and sets JUDGE_MODEL to the id from /v1/models.
# Steps: (1) the 50-item hand-validation judge pass + agreement + pin, (2) one judged HELMET cell.
set -euo pipefail
if [ "${WINDOW_CONFIRMED:-}" != "1" ]; then
  echo "REFUSED: set WINDOW_CONFIRMED=1 (a judge is served, no YaRN timing window is active)" >&2; exit 64
fi
if [ "${YARN_TIMING_WINDOW_ACTIVE:-0}" = "1" ]; then
  echo "REFUSED: YARN_TIMING_WINDOW_ACTIVE=1; judging must run outside the timing window" >&2; exit 65
fi
: "${JUDGE_PORT:?}"; : "${JUDGE_MODEL:?}"; : "${KIT_DIR:?dir holding items.jsonl and labels_sheet_filled.csv}"
OUT="${OUT_DIR:-$KIT_DIR/out}"; mkdir -p "$OUT"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/../../.." && pwd)"
cd "$ROOT"
# the served model must be the pinned one
SERVED=$(curl -s "http://localhost:${JUDGE_PORT}/v1/models" | python3 -c "import sys,json;print(json.load(sys.stdin)[\"data\"][0][\"id\"])")
[ "$SERVED" = "$JUDGE_MODEL" ] || { echo "REFUSED: served model $SERVED != JUDGE_MODEL $JUDGE_MODEL" >&2; exit 66; }
K=scripts/benchmark/yarn_gate/judge_validation_kit.py
python3 $K judge --items "$KIT_DIR/items.jsonl" --port "$JUDGE_PORT" --judge-model "$JUDGE_MODEL" --out "$OUT/judged.jsonl"
python3 $K agree --sheet "$KIT_DIR/labels_sheet_filled.csv" --judged "$OUT/judged.jsonl" --out "$OUT/agreement.json"
python3 $K pin --agreement "$OUT/agreement.json" --sheet "$KIT_DIR/labels_sheet_filled.csv" --judge-model "$JUDGE_MODEL" --out "$OUT/judge_pin.json"
if [ -n "${HELMET_CELL:-}" ]; then
  python3 scripts/benchmark/yarn_gate/helmet_judged_cell.py --cell "$HELMET_CELL" --task "${HELMET_TASK:?}" \
    --nominal-length "${HELMET_LEN:?}" --config-file "${HELMET_CONFIG:?}" ${HELMET_MAX_SAMPLES:+--max-test-samples $HELMET_MAX_SAMPLES} \
    --port "$JUDGE_PORT" --judge-model "$JUDGE_MODEL" --out "$OUT/helmet_cell_judged.json"
fi
echo "done: $OUT"
