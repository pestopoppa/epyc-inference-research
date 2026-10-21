#!/bin/bash
#maxjob_s=3600
# DS41-T3.3 (template arm) token-id parity: encoding.py vs llama-server /apply-template (+/tokenize) rendering the EPYC
# DeepSeek-V4.1 jinja template. Job 60 compared against a ChatML fallback because neither GGUF carries tokenizer.chat_template.
# Template: /mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja (targets the 1017 encoding.py, sha256 13231c59...99e2).
# Resources: llama-server on Sept Q4 (483 GiB mmap, only vocab/metadata + a few pages touched; no inference run).
#  ~2 GiB anon RAM, idle after load. needs_full_host=0. Port 18961 (job 60 used 18960).
# Oracles: Sept encoding.py (differs from 1017 only in find_last_user_index for a trailing tool-result-only user turn in
# thinking mode, so sept_encoding may legitimately diverge on tool-result-trailing thinking cases) and the 1017 copy.
# Offline proof (python jinja2 proxy, 14/14 text-identical against both encodings) was done before queuing.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-t33-template-parity; OUT=$PAR/$J; mkdir -p $OUT
trap kill_srv EXIT
ds41_check_bin
TPL=/mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja
[ -s "$TPL" ] || { echo "template missing: $TPL"; exit 6; }
sha256sum "$TPL" | tee $OUT/template.sha256
SEPT=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/encoding
mkdir -p $OUT/enc1017
cp $PAR/ds41-t33-token-parity/enc1017/encoding.py $OUT/enc1017/encoding.py 2>/dev/null || curl -sf -m 30 -o $OUT/enc1017/encoding.py https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/encoding/encoding.py || rm -f $OUT/enc1017/encoding.py
"$BIN/llama-server" -m "${MODEL:-$Q4}" --host 127.0.0.1 --port 18961 -t "$(nproc)" -tb "$(nproc)" -np 1 -c 4096 --no-warmup --no-webui --cache-ram 0 --jinja --chat-template-file /mnt/raid0/llm/models/chat-templates/deepseek-v41-flash-epyc-v1.jinja > $OUT/server.log 2>&1 &
SRV_PID=$!
wait_ready 18961 2400
grep -i -E "chat template|chat_template|jinja" $OUT/server.log | head -5 | tee $OUT/template_load.txt || true
$PY -I $PAR/lib/t33.py 18961 $SEPT $OUT sept_encoding | tee $OUT/summary.txt
if [ -s $OUT/enc1017/encoding.py ] && ! cmp -s $OUT/enc1017/encoding.py $SEPT/encoding.py; then
  $PY -I $PAR/lib/t33.py 18961 $OUT/enc1017 $OUT remote_main_encoding | tee -a $OUT/summary.txt
else echo "1017 encoding.py: not fetched or identical to Sept copy" | tee -a $OUT/summary.txt; fi
echo "RESULT $J $(tail -n +1 $OUT/summary.txt | tr '\n' ' ')"
