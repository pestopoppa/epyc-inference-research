#!/bin/bash
#maxjob_s=3600
# DS41-T3.3 token-id parity: encoding.py (Sept on disk; 1017 copy fetched best-effort) vs served /apply-template+/tokenize.
# Resources: llama-server on Sept Q4 (483 GiB mmap, only vocab/metadata + a few pages touched; no inference run).
#  ~2 GiB anon RAM, cores=$(nproc) but idle after load. needs_full_host=0. Port 18960.
# Not covered (blocker): deepseek-recipe @935817c oracle is Rust and no cargo toolchain exists on host -> recipe oracle NOT run.
# Rerun on 1017 vocab/template: 60-ds41-t33-token-parity-1017.sh (guarded).
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t33-token-parity; OUT=$PAR/$J; mkdir -p $OUT
trap kill_srv EXIT
ds41_check_bin
SEPT=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding
mkdir -p $OUT/enc1017
curl -sf -m 30 -o $OUT/enc1017/encoding.py https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py || rm -f $OUT/enc1017/encoding.py
"$BIN/llama-server" -m "${MODEL:-$Q4}" --host 127.0.0.1 --port 18960 -t "$(nproc)" -tb "$(nproc)" -np 1 -c 4096 --no-warmup --no-webui --cache-ram 0 > $OUT/server.log 2>&1 &
SRV_PID=$!
wait_ready 18960 2400
$PY -I $PAR/lib/t33.py 18960 $SEPT $OUT sept_encoding | tee $OUT/summary.txt
if [ -s $OUT/enc1017/encoding.py ] && ! cmp -s $OUT/enc1017/encoding.py $SEPT/encoding.py; then
  $PY -I $PAR/lib/t33.py 18960 $OUT/enc1017 $OUT remote_main_encoding | tee -a $OUT/summary.txt
else echo "1017 encoding.py: not fetched or identical to Sept copy" | tee -a $OUT/summary.txt; fi
echo "RESULT $J $(tail -n +1 $OUT/summary.txt | tr '\n' ' ')"
