#!/bin/bash
set -euo pipefail
cd /mnt/raid0/llm/tmp/cs-46-20261020/dl
B=https://huggingface.co/ysdede/parakeet-realtime-eou-120m-v1-onnx/resolve/6d6be8e
for f in README.md config.json vocab.txt decoder_joint-model.onnx decoder_joint-model.int8.onnx encoder-model.onnx; do
  [ -s "$f" ] || curl -sSL --fail -m 900 -o "$f.part" "$B/$f" && mv "$f.part" "$f"
  echo "done $f"
done
sha256sum *.onnx
