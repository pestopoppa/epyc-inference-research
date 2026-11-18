#!/usr/bin/env python3
"""Invert a DFlash2 drafter GGUF back to the HF layout convert_hf_to_gguf.py (DFlash2DraftModel) reads.

Purpose: warm-start without a download. The z-lab bf16 safetensors are not on disk, but the BF16 GGUF of the
same weights is (/mnt/raid0/llm/models/dflash/df2-11-20261101/Qwen3.8-27B-DFlash2-BF16.gguf). The converter maps
names 1:1 and does not permute DFlash/Qwen3 tensors (conversion/qwen.py DFlashModel @ eee3da838), so the
inverse is a rename plus a dtype cast. The round-trip gate (smoke_test.py --roundtrip, and phase G0 of the pilot)
proves it: HF -> GGUF -> HF -> GGUF must be byte-identical tensors.

  gguf_to_hf.py <drafter.gguf> <out_dir> [--block-size 16]

--block-size rewrites dflash_config.block_size only; all weights are reused unchanged (see df2_model.py).
Quantized inputs (Q8_0 ...) are dequantized to f32 then stored bf16 -- use the BF16 GGUF for warm-start.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np

GLOBAL = {
    "fc.weight": "model.fc.weight",
    "enc.output_norm.weight": "model.hidden_norm.weight",
    "output_norm.weight": "model.norm.weight",
    "selector_hidden.weight": "model.candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "model.candidate_selector.predecessor_codebook",
    "selector_successor.weight": "model.candidate_selector.successor_codebook",
}
PER_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_conv_base": "attention_conv.base_kernel",
    "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
}


def hf_name(gguf_name: str) -> str:
    if gguf_name in GLOBAL:
        return GLOBAL[gguf_name]
    m = re.fullmatch(r"blk\.(\d+)\.(.+)", gguf_name)
    if m and m.group(2) in PER_LAYER:
        return f"model.layers.{m.group(1)}.{PER_LAYER[m.group(2)]}"
    raise KeyError(f"unmapped DFlash2 GGUF tensor {gguf_name!r}")


def field(r, key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    v = f.contents()
    return v


def tensor_f32(t) -> np.ndarray:
    import gguf
    shape = tuple(int(x) for x in reversed(t.shape.tolist()))   # numpy order = reversed ggml ne
    qt = t.tensor_type
    if qt == gguf.GGMLQuantizationType.F32:
        a = np.asarray(t.data, dtype=np.float32)
    elif qt == gguf.GGMLQuantizationType.F16:
        a = np.asarray(t.data, dtype=np.float16).astype(np.float32)
    elif qt == gguf.GGMLQuantizationType.BF16:
        a = (np.asarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else:
        a = gguf.quants.dequantize(t.data, qt)
    return a.reshape(shape)


def config_from_gguf(r, block_size: int | None) -> dict:
    a = "dflash"
    n_layer = int(field(r, f"{a}.block_count"))
    n_embd = int(field(r, f"{a}.embedding_length"))
    taps = [int(x) for x in field(r, f"{a}.target_layers")]
    swa = int(field(r, f"{a}.attention.sliding_window", 0) or 0)
    pattern = field(r, f"{a}.attention.sliding_window_pattern")
    fc = next(t for t in r.tensors if t.name == "fc.weight")
    sel = next(t for t in r.tensors if t.name == "selector_predecessor.weight")
    n_fc_in = int(fc.shape.tolist()[0])
    cfg = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "qwen3",
        "hidden_size": n_embd,
        "intermediate_size": int(field(r, f"{a}.feed_forward_length")),
        "num_hidden_layers": n_layer,
        "num_attention_heads": int(field(r, f"{a}.attention.head_count")),
        "num_key_value_heads": int(field(r, f"{a}.attention.head_count_kv")),
        "head_dim": int(field(r, f"{a}.attention.key_length")),
        "rms_norm_eps": float(field(r, f"{a}.attention.layer_norm_rms_epsilon")),
        "rope_theta": float(field(r, f"{a}.rope.freq_base")),
        "max_position_embeddings": int(field(r, f"{a}.context_length")),
        "vocab_size": int(sel.shape.tolist()[1]),
        "target_hidden_size": n_fc_in // len(taps),
        "has_embed_tokens": False,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "dflash_config": {
            "block_size": int(block_size or field(r, f"{a}.block_size")),
            "mask_token_id": int(field(r, "tokenizer.ggml.mask_token_id")),
            "target_layer_ids": [t - 1 for t in taps],           # converter writes HF id + 1
            "conv_kernel_size": int(field(r, f"{a}.conv_kernel_size")),
            "conv_group_size": int(field(r, f"{a}.conv_group_size")),
            "selector_rank": int(field(r, f"{a}.selector_rank")),
            "selector_top_k": int(field(r, f"{a}.selector_top_k")),
        },
    }
    if swa and pattern is not None:
        cfg.update(use_sliding_window=True, sliding_window=swa,
                   layer_types=["sliding_attention" if p else "full_attention" for p in pattern])
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gguf")
    ap.add_argument("out_dir")
    ap.add_argument("--block-size", type=int, default=None)
    a = ap.parse_args(argv)
    import gguf
    import torch
    from safetensors.torch import save_file

    r = gguf.GGUFReader(a.gguf)
    if field(r, "general.architecture") != "dflash":
        print("not a dflash GGUF", file=sys.stderr)
        return 2
    os.makedirs(a.out_dir, exist_ok=True)
    cfg = config_from_gguf(r, a.block_size)
    sd = {hf_name(t.name): torch.from_numpy(tensor_f32(t).copy()).to(torch.bfloat16) for t in r.tensors}
    save_file(sd, os.path.join(a.out_dir, "model.safetensors"), metadata={"format": "pt", "source_gguf": os.path.basename(a.gguf)})
    json.dump(cfg, open(os.path.join(a.out_dir, "config.json"), "w"), indent=1)
    print(f"wrote {len(sd)} tensors, block_size={cfg['dflash_config']['block_size']} -> {a.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
