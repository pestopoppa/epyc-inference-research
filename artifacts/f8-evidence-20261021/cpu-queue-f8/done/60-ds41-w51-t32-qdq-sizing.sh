#!/bin/bash
#maxjob_s=900
# DS41-W5.1 (1)(2) + DS41-T3.2 (1) -- the part that needs NO model weights: (a) read-only evidence capture for the index-key owner path
# (reference model.py L540-556 vs port deepseek41.cpp dsv41_source_layer/publish), (b) best-effort fetch of the 1017 inference/model.py to
# confirm the pre-wo_a act_quant (L787/L1073), (c) a-priori NMSE of each reference fake-quant (FP8 ue8m0/32, FP4 E8M0/32, FP4 e4m3/16)
# vs Q8_0, on synthetic distributions (PROXY grade; qdq_size.py).
# NOT done here (blocker, stated in RESULT): real-weight QDQ-on/off reference run (T3.2 step 1) and the real-activation per-op NMSE (W5.1 step 2) --
# the official FP8/FP4 safetensors are not on disk (only DSpark shards 44-46 under models/deepseek-ai/DeepSeek-V4.1-Flash-mtp) and kernel.py needs
# tilelang (absent). Needs a torch-CPU shim of kernel.py + a layer fixture from the mtp shards; steps 2-3 (port QDQ op) are code work.
# Resources: ~1 GiB RAM, 1 core effective (torch CPU, tiny tensors). needs_full_host=0. No server.
set -euo pipefail
. /mnt/raid0/llm/tmp/ds41-parity-20261021/lib/common.sh
J=ds41-w51-t32-qdq-sizing; OUT=$PAR/$J; mkdir -p $OUT
REF=/mnt/raid0/llm/models/deepseek-ai/DeepSeek-V4.1-Flash/inference/model.py
{ echo "== reference (on-disk Sept copy) model.py index-key path L540-556"; sed -n 540,556p $REF
  echo "== reference wo_a call sites (Sept copy: expect NO act_quant before wo_a)"; grep -n "wo_a" $REF
  echo "== port: index-key owner path"; sed -n 620,640p $SRC/src/models/deepseek41.cpp; grep -n "wo_a" $SRC/src/models/deepseek41.cpp
  echo "== port branch/commit"; git -C $SRC log -1 --format='%h %ci %s'
} > $OUT/evidence-readonly.txt 2>&1
mkdir -p $OUT/remote1017
if curl -sf -m 30 -o $OUT/remote1017/model.py https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/inference/model.py; then
  { echo "== remote main model.py: sha $(sha256sum $OUT/remote1017/model.py | cut -c1-16)"; grep -n "act_quant\|wo_a\|index_k" $OUT/remote1017/model.py; } > $OUT/evidence-remote.txt
else echo "remote model.py fetch failed" > $OUT/evidence-remote.txt; fi
$PY -I $PAR/lib/qdq_size.py $OUT/qdq_nmse.json > $OUT/qdq.log 2>&1
echo "RESULT $J proxy-grade fp8_qdq/q8_0 NMSE ratio: $($PY -I -c "import json;d=json.load(open('$OUT/qdq_nmse.json'));print({k:round(v['ratio_fp8qdq_over_q8_0'],1) for k,v in d.items()})") ; evidence=$OUT/evidence-readonly.txt,$OUT/evidence-remote.txt ; real-weight steps NOT run (no reference weights/tilelang)"
