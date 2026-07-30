#!/bin/bash
# Raise the headline comparison from n=1-2 toward decision-grade using the canonical
# instrument (llama-bench), 10 reps per arm, so the delta carries a real error bar.
# Arms are the two placements that matter: production-as-wired vs the canonical recipe.
LB=/mnt/raid0/llm/llama.cpp/build/bin/llama-bench
M=/mnt/raid0/llm/models/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf
OUT=/mnt/raid0/llm/tmp/highn_results.txt
export GGML_IQK=1 OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=active KMP_BLOCKTIME=10
: > "$OUT"

echo "### ARM A: production as wired (straddle 0-47,96-143, no numactl), n=10" | tee -a "$OUT"
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
taskset -c 0-47,96-143 "$LB" -m "$M" -t 96 -fa 1 -p 0 -n 128 -r 10 2>&1 | grep -E "tg128|model" | tee -a "$OUT"

echo "### ARM C: canonical recipe (0-95 + --interleave=all), n=10" | tee -a "$OUT"
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
taskset -c 0-95 numactl --interleave=all "$LB" -m "$M" -t 96 -fa 1 -p 0 -n 128 -r 10 2>&1 | grep -E "tg128|model" | tee -a "$OUT"

echo "=== HIGHN DONE ===" | tee -a "$OUT"
