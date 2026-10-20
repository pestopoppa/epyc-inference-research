#!/bin/bash
# Q38FN thread x depth sweep: llama-bench depth-0 anchor (no MTP), then served arms B0,A,C,B1.
# Run under: region-lock run --cpu-list 0-95 --role bench --gpu-quiet shared -- run_all.sh
set -euo pipefail
R=/mnt/raid0/llm/tmp/q38fn-threads-depth-20261020
BIN=/mnt/raid0/llm/kernels/builds/cpu-20261004-b0ba1d427/bin
MODEL=/mnt/raid0/llm/models/unsloth/Qwen3.8-Flash-Next-GGUF/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf
export OMP_PROC_BIND=spread OMP_PLACES=cores OMP_WAIT_POLICY=active OMP_DYNAMIC=false
export GGML_IQK=1 GGML_FA_SPLIT_KV=0 GGML_NOHUGEPAGE_PROCESS=1 GGML_FUSED_DECODE_OFF=1
export LD_LIBRARY_PATH="$BIN:${LD_LIBRARY_PATH:-}"

echo "[run_all] llama-bench anchor start $(date -u +%FT%TZ)"
taskset -c 0-95 numactl --interleave=all "$BIN/llama-bench" -m "$MODEL" -t 48,96 -p 512 -n 128 \
  -fa 1 -mmp 0 -r 3 -o jsonl > "$R/llama-bench-d0.jsonl" 2> "$R/llama-bench-d0.stderr"
echo "[run_all] llama-bench anchor done $(date -u +%FT%TZ)"

python3 "$R/driver.py"
echo "[run_all] served sweep done $(date -u +%FT%TZ)"
touch "$R/DONE"
