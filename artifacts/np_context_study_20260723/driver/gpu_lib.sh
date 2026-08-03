#!/bin/bash
# Shared launch/kill helpers for the architect-bench GPU arms (v7 production kernel).
set -uo pipefail
BIN=/mnt/raid0/llm/llama.cpp/build-hip/bin/llama-server
PORT=18072
CORES=184-191   # node-3 cores; keeps the GPU server off the CPU inference stack

gpu_launch() {  # gpu_launch <logdir> <model> <extra flags...>
  local logdir="$1"; shift
  local model="$1"; shift
  mkdir -p "$logdir"
  nohup taskset -c "$CORES" env \
    LD_LIBRARY_PATH=/mnt/raid0/llm/llama.cpp/build-hip/bin GGML_IQK=1 \
    ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 \
    "$BIN" -m "$model" --host 127.0.0.1 --port "$PORT" \
    --metrics --slots --jinja --reasoning off --device ROCm0 -ngl all -fa on \
    "$@" > "$logdir/server.stdout" 2> "$logdir/server.stderr" &
  echo $! > "$logdir/server.pid"
  printf '%s ' "$BIN" -m "$model" "$@" > "$logdir/server_command.txt"
}

gpu_wait() {  # gpu_wait <logdir> <timeout_s>
  local logdir="$1"; local timeout="${2:-600}"; local pid; pid=$(cat "$logdir/server.pid")
  local deadline=$(( $(date +%s) + timeout ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! ps -p "$pid" >/dev/null 2>&1; then echo "SERVER_DIED"; return 1; fi
    if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -qi ok; then echo "HEALTHY"; return 0; fi
    sleep 3
  done
  echo "TIMEOUT"; return 1
}

gpu_kill() {  # gpu_kill <logdir>
  local logdir="$1"; local pid; pid=$(cat "$logdir/server.pid" 2>/dev/null || echo "")
  [ -z "$pid" ] && return 0
  kill -TERM "$pid" 2>/dev/null; sleep 8
  if ps -p "$pid" >/dev/null 2>&1; then kill -9 "$pid" 2>/dev/null; sleep 5; fi
  if ps -p "$pid" >/dev/null 2>&1; then echo "KILL_FAILED $pid"; return 1; fi
  echo "dead $pid"; return 0
}
