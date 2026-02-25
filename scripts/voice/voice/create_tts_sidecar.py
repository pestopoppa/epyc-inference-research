#!/usr/bin/env python3
"""Create binary sidecar weights file for llama-tts-qwen3.

Reads the Qwen3-TTS safetensors and extracts non-Talker-body weights
into a compact binary format that the C++ binary can load directly.

Format:
  header (32 bytes): magic "QWTTS02\0" + dimensions
  text_embedding: [151936, 2048] float32
  text_proj_fc1_w: [2048, 2048] float32
  text_proj_fc1_b: [2048] float32
  text_proj_fc2_w: [1024, 2048] float32
  text_proj_fc2_b: [1024] float32
  codec_embedding: [3072, 1024] float32
  cp_embeddings: 15 × [cp_vocab, 1024] float32
  cp_lm_heads: 15 × [cp_vocab, 1024] float32

Usage:
  python create_tts_sidecar.py \
    --model /mnt/raid0/llm/hf/Qwen3-TTS-12Hz-0.6B-Base \
    --output /mnt/raid0/llm/models/qwen3-tts-sidecar.bin
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file


def main():
    parser = argparse.ArgumentParser(description="Create TTS sidecar weights")
    parser.add_argument("--model", type=str, required=True, help="Qwen3-TTS model path")
    parser.add_argument("--output", type=str, required=True, help="Output binary path")
    args = parser.parse_args()

    print(f"Loading model from {args.model}...")
    tensors = load_file(str(Path(args.model) / "model.safetensors"))

    n_embd_text = 2048
    n_embd = 1024
    codec_vocab = 3072  # includes special tokens (Talker codec_embedding)
    n_code_groups = 16

    # Detect CP vocab size from first CP embedding table
    cp_embd_0 = tensors["talker.code_predictor.model.codec_embedding.0.weight"]
    cp_vocab = cp_embd_0.shape[0]
    print(f"Creating sidecar: n_embd_text={n_embd_text}, n_embd={n_embd}, "
          f"codec_vocab={codec_vocab}, cp_vocab={cp_vocab}, n_code_groups={n_code_groups}")

    with open(args.output, "wb") as f:
        # Write header (32 bytes): magic(8) + 6 int32
        header = struct.pack(
            "8s6i",
            b"QWTTS02\x00",
            n_embd_text,
            n_embd,
            codec_vocab,
            cp_vocab,
            n_code_groups,
            0,  # reserved
        )
        assert len(header) == 32, f"Header size {len(header)} != 32"
        f.write(header)

        def write_tensor(name, key, expected_shape=None):
            t = tensors[key].float().contiguous().numpy()
            if expected_shape:
                assert t.shape == tuple(expected_shape), f"{key}: {t.shape} != {expected_shape}"
            f.write(t.tobytes())
            print(f"  {name}: {key} {list(t.shape)} ({t.nbytes / 1024 / 1024:.1f} MB)")

        # Text embedding
        write_tensor("text_embedding", "talker.model.text_embedding.weight", [151936, 2048])

        # Text projection MLP
        write_tensor("text_proj_fc1_w", "talker.text_projection.linear_fc1.weight", [2048, 2048])
        write_tensor("text_proj_fc1_b", "talker.text_projection.linear_fc1.bias", [2048])
        write_tensor("text_proj_fc2_w", "talker.text_projection.linear_fc2.weight", [1024, 2048])
        write_tensor("text_proj_fc2_b", "talker.text_projection.linear_fc2.bias", [1024])

        # Codec embedding
        write_tensor("codec_embedding", "talker.model.codec_embedding.weight", [3072, 1024])

        # CP embeddings (15 tables)
        for i in range(15):
            write_tensor(f"cp_embd.{i}", f"talker.code_predictor.model.codec_embedding.{i}.weight")

        # CP lm_heads (15 heads)
        for i in range(15):
            write_tensor(f"cp_head.{i}", f"talker.code_predictor.lm_head.{i}.weight")

    size_mb = Path(args.output).stat().st_size / 1024 / 1024
    print(f"\nSidecar written: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
