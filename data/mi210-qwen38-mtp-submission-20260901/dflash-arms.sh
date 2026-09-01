#!/bin/bash
set -u
BIN=/mnt/raid0/llm/tmp/build-champ-tip-clean/bin
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"
MODEL=/mnt/raid0/llm/models/Qwen3.8-27B-Q8_0.gguf
DRAFT=/mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
OUT=/mnt/raid0/llm/tmp/mtp-submission-20260901
PORT=18099
mkdir -p "$OUT"

run_arm() {
  local name="$1"; shift
  echo "=== ARM $name ==="
  "$@" > "$OUT/server-$name.log" 2>&1 &
  local PID=$!
  local i=0
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    sleep 3; i=$((i+1))
    if ! kill -0 "$PID" 2>/dev/null; then
      echo "FAILED $name: server died"; tail -4 "$OUT/server-$name.log"; return 1
    fi
    if [ "$i" -gt 120 ]; then echo "FAILED $name: ready timeout"; kill -TERM "$PID"; return 1; fi
  done
  echo "$name ready in $((i*3))s"
  rocm-smi --showmeminfo vram 2>/dev/null | grep -i 'Used Memory' > "$OUT/vram-$name.txt"
  python3 /mnt/raid0/llm/tmp/probe.py "http://127.0.0.1:$PORT" > "$OUT/probe-$name.txt" 2>&1
  echo "RESULT $name: $(grep OVERALL "$OUT/probe-$name.txt" || echo none)"
  echo "ACCEPT $name: $(grep -o 'draft acceptance = [0-9.]*' "$OUT/server-$name.log" | tail -1)"
  kill -TERM "$PID" 2>/dev/null
  local j=0
  while kill -0 "$PID" 2>/dev/null && [ "$j" -lt 40 ]; do sleep 1; j=$((j+1)); done
  if kill -0 "$PID" 2>/dev/null; then kill -KILL "$PID" 2>/dev/null; sleep 3; fi
  if kill -0 "$PID" 2>/dev/null; then echo "WARN $name: $PID STILL ALIVE"; else echo "$name server $PID dead"; fi
}

# Arm 1: MY config (-c 32768, unpinned) -- like-for-like vs my baseline/MTP rows
run_arm dflash-c32k \
  "$BIN/llama-server" -m "$MODEL" -np 1 -c 32768 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
  --metrics --slots -md "$DRAFT" -ngld 99 --spec-type draft-dflash --spec-draft-n-max 8

# Arm 2: AK's healthy 70.4 t/s cell VERBATIM (-c 4096, --no-kv-unified, taskset 184-191)
run_arm dflash-c4k \
  taskset -c 184-191 "$BIN/llama-server" -m "$MODEL" -np 1 -c 4096 -t 8 -tb 8 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 --device ROCm0 -ngl 99 -fa on --host 127.0.0.1 --port "$PORT" \
  --metrics --slots -md "$DRAFT" -ngld 99 --spec-type draft-dflash --spec-draft-n-max 8 --no-kv-unified

echo "ALLDONE"
