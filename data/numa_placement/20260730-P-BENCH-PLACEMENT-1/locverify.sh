#!/bin/bash
# Mechanism proof for the quarter-fleet result: measure where each instance's
# weight pages actually live, mmap (production) vs --no-mmap. No requests issued.
M=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-MTP-Q8_0.gguf
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
OUT=/mnt/raid0/llm/tmp/locverify_results.txt
CPUSETS=("0-23,96-119" "24-47,120-143" "48-71,144-167" "72-95,168-191")
: > "$OUT"

for ARM in MMAP NOMMAP; do
  MMAPFLAG=""; [ "$ARM" = "NOMMAP" ] && MMAPFLAG="--no-mmap"
  echo "=== $ARM ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  declare -a SRVPIDS=()
  for q in 0 1 2 3; do
    SLOG=/mnt/raid0/llm/tmp/loc_${ARM}_q${q}.log; : > "$SLOG"
    taskset -c "${CPUSETS[$q]}" numactl --membind=$q "$LS" -m "$M" \
      --host 127.0.0.1 --port $((19910 + q)) -np 1 -c 4096 -t 24 \
      --flash-attn on --jinja $MMAPFLAG --device none --log-colors off >> "$SLOG" 2>&1 &
    SRVPIDS[$q]=$!
  done
  for i in $(seq 1 400); do
    UP=1; for q in 0 1 2 3; do grep -q "model loaded" /mnt/raid0/llm/tmp/loc_${ARM}_q${q}.log || UP=0; done
    [ "$UP" = "1" ] && break; sleep 5
  done
  for q in 0 1 2 3; do
    echo "q$q:" | tee -a "$OUT"
    python3 /mnt/raid0/llm/tmp/numaloc.py "${SRVPIDS[$q]}" "$q" 2>&1 | tee -a "$OUT"
  done
  echo "  host free: $(free -g | awk '/Mem:/{print $3" GB used"}')" | tee -a "$OUT"
  for q in 0 1 2 3; do kill -TERM "${SRVPIDS[$q]}" 2>/dev/null; done; sleep 12
  for q in 0 1 2 3; do kill -9 "${SRVPIDS[$q]}" 2>/dev/null; done; sleep 5
done
echo "=== LOCVERIFY DONE ===" | tee -a "$OUT"
