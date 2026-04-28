#!/bin/bash
# Phase 1.1 dispatcher v1 smoke test — K=4 with parallel aux decode + winner-state-sync.
# Confirms the new dispatch path runs without crashes on Qwen3.6-35B-A3B-Q8_0 hybrid Delta Net + Qwen3-1.7B-Q8_0 drafter, and that we get plausible output.
set -uo pipefail

OUT=/mnt/raid0/llm/epyc-inference-research/data/cpu_optimization/2026-04-30-state-sync-cost-probe
BIN=/mnt/raid0/llm/llama.cpp-experimental/build_v5_pgo_use/bin
TGT=/mnt/raid0/llm/models/Qwen3.6-35B-A3B-Q8_0.gguf
DFT=/mnt/raid0/llm/models/Qwen3-1.7B-Q8_0.gguf

export LD_LIBRARY_PATH=$BIN

cd "$OUT"
ulimit -c 0

echo "=== boot K=4 with dispatcher v1 ==="
date

numactl --interleave=all $BIN/llama-server \
    -m "$TGT" -md "$DFT" \
    -t 96 -c 4096 -fa 1 \
    --spec-numa-quarters 4 \
    --draft-max 24 --draft-min 4 \
    --port 18099 \
    > srv_smoke_v1.log 2>&1 &
SRV_PID=$!
echo "  server pid=$SRV_PID"

for i in $(seq 1 180); do
    if curl -s http://localhost:18099/health 2>/dev/null | grep -q ok; then
        echo "  server ready after ${i}s"
        sleep 30
        break
    fi
    sleep 1
done

# Single representative request
echo "--- request lru_cache n=64 (p_split=0.05) ---"
curl -s http://localhost:18099/completion \
    -H 'Content-Type: application/json' \
    -d '{"prompt": "Implement a simple LRU cache in Python with O(1) get and put operations using OrderedDict.", "n_predict": 64, "temperature": 0.0, "p_split": 0.05}' \
    > smoke_v1_comp.json 2>&1

echo "=== shutdown ==="
kill -INT $SRV_PID 2>/dev/null
sleep 3
kill -KILL $SRV_PID 2>/dev/null
wait $SRV_PID 2>/dev/null

echo ""
echo "=== completion summary ==="
jq -r '"  predicted_per_second: \(.timings.predicted_per_second // "n/a")\n  draft_n / draft_n_accepted: \(.timings.draft_n // "n/a") / \(.timings.draft_n_accepted // "n/a")\n  content head: \(.content // "(empty)")[0:120]"' smoke_v1_comp.json 2>&1 | head -3

echo ""
echo "=== one-shot sync lines ==="
grep "numa one-shot state sync" srv_smoke_v1.log

echo ""
echo "=== K-parallel verify lines ==="
grep "numa K-parallel verify" srv_smoke_v1.log | head -10

echo ""
echo "=== aux thread warnings ==="
grep -E "numa aux thread|numa: winner-state" srv_smoke_v1.log | head -5

echo ""
echo "=== completed at $(date) ==="
