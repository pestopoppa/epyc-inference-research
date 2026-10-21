#!/bin/bash
#maxjob_s=3600
# DS41-T3.3c (sept-q4): encoding.py oracle vs llama-server /apply-template (+/tokenize) with the EPYC template (--jinja --chat-template-file, v1 UNCHANGED).
# Changes vs job 61 (lib/t33.py v2): enable_thinking sent EXPLICITLY on every case (server default is thinking ON when the key is absent);
#  served prompt text + oracle text saved per case in the result JSON; object-arguments capability probe (string vs object) and tool-call
#  arguments sent the way the server delivers them; golden5 (image_url) = SKIP(no mmproj), kept in the denominator as skipped; add_special=false AND true
#  both reported, BOS NOT decided here (the server strips the template's leading BOS from the rendered prompt, so add_special=true is the chat-path view).
# Template check (read 2026-10-21): v1 sets thinking only when enable_thinking is defined AND true => defined-but-false is chat mode; no v2 template needed.
# Resources: llama-server on Sept Q4 (mmap, vocab/metadata + a few pages touched; no inference). ~2 GiB anon, needs_full_host=0. Port 18964.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t33c-sept; OUT=$PAR/$J; mkdir -p $OUT
trap kill_srv EXIT
ds41_check_bin
TPL=/mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja
[ -s "$TPL" ] || { echo "template missing: $TPL"; exit 6; }
sha256sum "$TPL" | tee $OUT/template.sha256
SEPT=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding
mkdir -p $OUT/enc1017
cp $PAR/ds41-t33-template-parity/enc1017/encoding.py $OUT/enc1017/encoding.py 2>/dev/null || curl -sf -m 30 -o $OUT/enc1017/encoding.py https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py || rm -f $OUT/enc1017/encoding.py
"$BIN/llama-server" -m "${MODEL:-$Q4}" --host 127.0.0.1 --port 18964 -t "$(nproc)" -tb "$(nproc)" -np 1 -c 4096 --no-warmup --no-webui --cache-ram 0 --jinja --chat-template-file "$TPL" > $OUT/server.log 2>&1 &
SRV_PID=$!
wait_ready 18964 2400
grep -i -E "chat template|chat_template|jinja" $OUT/server.log | head -5 | tee $OUT/template_load.txt || true
$PY -I $PAR/lib/t33.py 18964 $SEPT $OUT sept_encoding | tee $OUT/summary.txt
if [ -s $OUT/enc1017/encoding.py ] && ! cmp -s $OUT/enc1017/encoding.py $SEPT/encoding.py; then
  $PY -I $PAR/lib/t33.py 18964 $OUT/enc1017 $OUT remote_main_encoding | tee -a $OUT/summary.txt
else echo "1017 encoding.py: not fetched or identical to Sept copy" | tee -a $OUT/summary.txt; fi
echo "RESULT $J era=sept-q4 $(tail -n +1 $OUT/summary.txt | tr '\n' ' ')"
