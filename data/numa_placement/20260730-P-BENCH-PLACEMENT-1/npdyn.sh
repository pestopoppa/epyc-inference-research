#!/bin/bash
# Is -np dynamic? Operator's question: "If I'm running 4 batched decodes and two
# end while the remaining two need to keep on running, does np transition to 2
# automatically, or do the remaining two still run at np=4 speeds?"
#
# Method: launch with -np 4, fire 4 requests simultaneously — two capped at 48
# tokens (finish early) and two at 640 (keep running). llama-server logs a
# 3-second windowed decode rate (`tg_3s`) per slot as it goes, so if the batch
# shrinks dynamically the survivors' tg_3s should CLIMB the moment the short
# ones release their slots. If it stays flat, np is a fixed cost.
# Control arm: same two long requests with nothing else running.
LS=/mnt/raid0/llm/llama.cpp/build/bin/llama-server
M=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-ORIG-Q4_K_M.gguf
MD=/mnt/raid0/llm/models/gemma-4-26B-A4B-it-assistant-v6-Q8_0.gguf
OUT=/mnt/raid0/llm/tmp/npdyn_results.txt
PORT=19950
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

ask () { # $1 = max_tokens
  curl -s --max-time 1800 "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain how a CPU cache hierarchy works, in detail.\"}],\"max_tokens\":$1,\"temperature\":0.3,\"seed\":42}" \
    -o /dev/null 2>/dev/null
}

for ARM in mixed control; do
  SLOG=/mnt/raid0/llm/tmp/npdyn_${ARM}.log; : > "$SLOG"
  echo "=== ARM $ARM ===" | tee -a "$OUT"
  sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
  taskset -c 0-95 numactl --interleave=all "$LS" -m "$M" --host 127.0.0.1 --port $PORT \
    -np 4 -c 32768 -t 96 -ub 512 -ctk q8_0 -ctv q8_0 --flash-attn on --jinja \
    --no-mmap --device none --log-colors off \
    -md "$MD" --spec-type draft-mtp --spec-draft-n-max 2 --draft-p-min 0.0 --threads-draft 16 \
    >> "$SLOG" 2>&1 &
  SRV=$!
  for i in $(seq 1 300); do grep -q "model loaded" "$SLOG" && break; sleep 5; done
  grep -q "model loaded" "$SLOG" || { echo "  NEVER LOADED" | tee -a "$OUT"; kill -9 $SRV; continue; }
  if [ "$ARM" = "mixed" ]; then
    ask 48 & P1=$!; ask 48 & P2=$!; ask 640 & P3=$!; ask 640 & P4=$!
    wait $P1 $P2 $P3 $P4
  else
    ask 640 & P3=$!; ask 640 & P4=$!
    wait $P3 $P4
  fi
  kill -TERM "$SRV" 2>/dev/null; sleep 10; kill -9 "$SRV" 2>/dev/null; sleep 4
  python3 - "$SLOG" "$ARM" <<'PY' | tee -a "$OUT"
import re, sys
log, arm = sys.argv[1], sys.argv[2]
# per-slot windowed rate samples, in order, as decoding progresses
rows = []
for l in open(log):
    m = re.search(r"id\s+(\d+) \| task (\d+) \| n_decoded =\s+(\d+), tg =\s+([\d.]+) t/s, tg_3s =\s+([\d.]+) t/s", l)
    if m: rows.append((int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)), float(m.group(5))))
if not rows:
    print("  no windowed samples"); raise SystemExit
by = {}
for slot, task, n, tg, tg3 in rows:
    by.setdefault(task, []).append((n, tg3))
print(f"  {arm}: windowed decode rate (tg_3s) as each task progresses")
for task in sorted(by):
    s = by[task]
    if len(s) < 2:
        print(f"    task {task:3}: {s[0][1]:6.2f} t/s @ {s[0][0]} tok  (single sample)")
        continue
    trace = " -> ".join(f"{r:.1f}@{n}" for n, r in s)
    print(f"    task {task:3}: {trace}")
PY
done
echo "=== NPDYN DONE ===" | tee -a "$OUT"
