#!/bin/bash
# Export a trained DFlash2 HF dir to the serving GGUF, with the converter the z-lab GGUFs used:
# convert_hf_to_gguf.py -> DFlash2DraftModel (conversion/qwen.py DFlashModel @ C14 eee3da838), then llama-quantize Q8_0.
#
#   export_gguf.sh <hf_dir> <out_prefix> [expected_block_size]
#
# Converter source: a `git archive` of the C14 commit into a scratch dir (never the frozen tree, never a checkout).
# Quantizer: the C14 kernel store's own llama-quantize (CPU work; run inside a granted CPU window).
set -euo pipefail
HF_DIR=$1
OUT=$2
BS=${3:-16}
C14_SHA=eee3da838
SRC_REPO=/mnt/raid0/llm/llama.cpp-experimental             # read-only: git archive only
STORE=/mnt/raid0/llm/kernels/builds/gpu-20261118-eee3da838-c14
TARGET_DIR=/mnt/raid0/llm/models/turboderp/Qwen3.8-27B-exl3-4.00bpw   # config.json + tokenizer only
PY=/mnt/raid0/llm/venvs/llama-gguf-convert/bin/python
HERE=$(cd "$(dirname "$0")" && pwd)
CONV=$(dirname "$OUT")/c14-convert-$C14_SHA

if [ ! -f "$CONV/convert_hf_to_gguf.py" ]; then
  mkdir -p "$CONV"
  git -C "$SRC_REPO" archive "$C14_SHA" convert_hf_to_gguf.py conversion gguf-py | tar -x -C "$CONV"
fi
PYTHONPATH="$CONV/gguf-py" "$PY" "$CONV/convert_hf_to_gguf.py" "$HF_DIR" \
  --target-model-dir "$TARGET_DIR" --outtype bf16 --outfile "$OUT-BF16.gguf"
LD_LIBRARY_PATH="$STORE/bin" "$STORE/bin/llama-quantize" "$OUT-BF16.gguf" "$OUT-Q8_0.gguf" Q8_0
PYTHONPATH="$CONV/gguf-py" "$PY" "$HERE/verify_drafter_gguf.py" "$OUT-Q8_0.gguf" --block-size "$BS" \
  --reference /mnt/raid0/llm/models/Qwen3.8-27B-DFlash2-Q8_0.gguf
sha256sum "$OUT-BF16.gguf" "$OUT-Q8_0.gguf" > "$OUT.sha256"
