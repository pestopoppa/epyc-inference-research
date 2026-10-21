#!/bin/bash
#maxjob_s=3600
# DS41-T3.3c (1017-mxfp4): same as 62-ds41-t33c-sept.sh but on the 1017 shards, served WITH --jinja --chat-template-file (v1 template, unchanged).
# GUARDED: exit 3 DEP_MISSING until the 1017 download is verified (a pending job exiting 3 is consumed, not requeued).
# Resources: llama-server on 1017 shard 1 (+ siblings found by name; mmap, vocab/metadata + few pages; no inference). needs_full_host=0. Port 18965.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t33c-1017; OUT=$PAR/$J; mkdir -p $OUT
trap kill_srv EXIT
ds41_need_1017; ds41_check_bin
TPL=/mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja
[ -s "$TPL" ] || { echo "template missing: $TPL"; exit 6; }
sha256sum "$TPL" | tee $OUT/template.sha256
SEPT=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding
mkdir -p $OUT/enc1017
cp $PAR/ds41-t33-template-parity/enc1017/encoding.py $OUT/enc1017/encoding.py 2>/dev/null || curl -sf -m 30 -o $OUT/enc1017/encoding.py https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py || rm -f $OUT/enc1017/encoding.py
"$BIN/llama-server" -m "$M1017" --host 127.0.0.1 --port 18965 -t "$(nproc)" -tb "$(nproc)" -np 1 -c 4096 --no-warmup --no-webui --cache-ram 0 --jinja --chat-template-file "$TPL" > $OUT/server.log 2>&1 &
SRV_PID=$!
wait_ready 18965 2400
grep -i -E "chat template|chat_template|jinja" $OUT/server.log | head -5 | tee $OUT/template_load.txt || true
$PY -I $PAR/lib/t33.py 18965 $SEPT $OUT sept_encoding | tee $OUT/summary.txt
if [ -s $OUT/enc1017/encoding.py ] && ! cmp -s $OUT/enc1017/encoding.py $SEPT/encoding.py; then
  $PY -I $PAR/lib/t33.py 18965 $OUT/enc1017 $OUT remote_main_encoding | tee -a $OUT/summary.txt
else echo "1017 encoding.py: not fetched or identical to Sept copy" | tee -a $OUT/summary.txt; fi
echo "RESULT $J era=1017-mxfp4 $(tail -n +1 $OUT/summary.txt | tr '\n' ' ')"
