#!/bin/bash
set -u
BIN=/mnt/raid0/llm/tmp/build-champ-tip-clean/bin
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
MODEL=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf
DRAFT=/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
OUT=/mnt/raid0/llm/tmp/mtp-submission-20260901
PORT=18099
mkdir -p "$OUT"

arm() {
  local name="$1"; shift
  echo "=== ARM $name ==="
  "$BIN/llama-server" -m "$MODEL" -np 1 -c 32768 -t 8 -tb 8 -b 2048 -ub 2048 \
    -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on \
    --host 127.0.0.1 --port "$PORT" --metrics --slots "$@" \
    > "$OUT/server-$name.log" 2>&1 &
  local PID=$!
  local i=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 3; i=$((i+1))
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "FAILED $name: server died"; tail -3 "$OUT/server-$name.log"; return 1
    fi
    if [ "$i" -gt 120 ]; then echo "FAILED $name: ready timeout"; kill -TERM "$PID"; return 1; fi
  done
  echo "$name ready in $((i*3))s"
  curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-$name-before.txt" 2>/dev/null
  python3 /mnt/raid0/llm/tmp/probe.py "http://127.0.0.1:$PORT" > "$OUT/probe-$name.txt" 2>&1
  curl -s "http://127.0.0.1:$PORT/metrics" > "$OUT/metrics-$name-after.txt" 2>/dev/null
  echo "RESULT $name: $(grep OVERALL "$OUT/probe-$name.txt" || echo 'no OVERALL line')"
  kill -TERM "$PID" 2>/dev/null
  local j=0
  while kill -0 "$PID" 2>/dev/null && [ "$j" -lt 40 ]; do sleep 1; j=$((j+1)); done
  if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null; sleep 3; fi
  if kill -0 "$PID" 2>/dev/null; then echo "WARN $name: pid $PID STILL ALIVE"; else echo "$name server $PID dead"; fi
}

arm baseline
arm mtp-n2   --spec-type draft-mtp --spec-draft-n-max 2
arm mtp-n8   --spec-type draft-mtp --spec-draft-n-max 8
arm dflash2-n8 -md "$DRAFT" -ngld 99 --spec-type draft-simple --spec-draft-n-max 8
echo "ALLDONE"
