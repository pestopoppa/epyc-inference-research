#!/bin/bash
# job4: quantize self decoder, then parity for one chunk size (arg1). One claim.
D=/mnt/raid0/llm/tmp/cs-46-20261020
W=/mnt/raid0/llm/worktrees/cs-46-parakeet-eou-research-20261020
S=$W/scripts/speech
export OMP_NUM_THREADS=1 PYTHONPATH=$S
cd $S
CM=$1
echo "[job4/$CM] start $(date -u +%FT%TZ)"
if [ "$CM" = "160" ]; then
  uv pip install --python $D/venv-arm/bin/python onnx -q > $D/logs/onnxinstall.log 2>&1
  $D/venv-arm/bin/python -I parakeet_eou/quantize_decoder.py $D/export/decoder_joint_self_fp32.onnx $D/export/decoder_joint_self_int8.onnx > $D/logs/quant.log 2>&1; echo "[job4/$CM] quant rc=$?"
else
  while [ ! -s $D/export/decoder_joint_self_int8.onnx ]; do sleep 5; done
fi
$D/venv-arm/bin/python -I -m pytest -q -p no:cacheprovider parakeet_eou/test_frontend.py > $D/logs/unit_$CM.log 2>&1; echo "[job4/$CM] unit rc=$?"
$D/venv-arm/bin/python -I parakeet_eou/parity.py --clips $D/clips.jsonl --nemo-root $D/nemo_ref --model-dir $D/dl --export-dir $D/export --out $D/out/parity2_c$CM --chunks $CM --pairings fp32_fp32,fp32_int8dec,fp32_selfdec,fp32_selfdec_int8 > $D/logs/parity2_c$CM.log 2>&1; echo "[job4/$CM] parity rc=$?"
echo "[job4/$CM] done $(date -u +%FT%TZ)"
