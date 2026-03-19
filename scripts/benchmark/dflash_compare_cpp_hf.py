#!/usr/bin/env python3
"""Compare C++ DFlash drafter output with HF reference using EXACT same inputs.

Loads conditioning data, embeddings, and logits dumped from C++ diagnostic,
then runs the HF DFlash drafter with identical inputs and compares.
"""

import sys
import os
import json
import struct
import numpy as np

DFLASH_DIR = "/mnt/raid0/llm/cache/dflash/Qwen3-Coder-30B-A3B-DFlash"

import torch
from safetensors.torch import load_file


def load_binary_matrix(path):
    """Load [n_rows, n_cols] float32 matrix from binary dump with int32 header."""
    with open(path, "rb") as f:
        n_rows, n_cols = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(n_rows * n_cols * 4), dtype=np.float32)
        return data.reshape(n_cols, n_rows), n_rows, n_cols  # ggml is column-major: [n_rows, n_cols] → read as [n_cols][n_rows]


def load_dflash_module():
    import importlib, types
    pkg = types.ModuleType("dflash_pkg")
    pkg.__path__ = [DFLASH_DIR]
    pkg.__package__ = "dflash_pkg"
    sys.modules["dflash_pkg"] = pkg

    utils_spec = importlib.util.spec_from_file_location("dflash_pkg.utils", os.path.join(DFLASH_DIR, "utils.py"))
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules["dflash_pkg.utils"] = utils_mod
    utils_spec.loader.exec_module(utils_mod)
    pkg.utils = utils_mod

    dflash_path = os.path.join(DFLASH_DIR, "dflash.py")
    with open(dflash_path) as f:
        source = f.read().replace("from .utils import", "from dflash_pkg.utils import")
    dflash_spec = importlib.util.spec_from_file_location("dflash_pkg.dflash", dflash_path)
    dflash_mod = importlib.util.module_from_spec(dflash_spec)
    sys.modules["dflash_pkg.dflash"] = dflash_mod
    exec(compile(source, dflash_path, "exec"), dflash_mod.__dict__)
    return dflash_mod


def main():
    # Load metadata
    with open("/tmp/dflash_diag_meta.json") as f:
        meta = json.load(f)
    print(f"Metadata: {json.dumps(meta)}")

    n_ctx = meta["n_ctx"]
    blk_size = meta["blk_size"]
    n_embd = meta["n_embd"]
    n_taps = meta["n_taps"]
    pos_start = meta["pos_start"]
    block_tokens = meta["block_tokens"]

    # Load C++ conditioning data (cross_inp)
    # Layout: [n_cross_embd=10240, n_ctx] in ggml column-major → read as [n_ctx, 10240]
    with open("/tmp/dflash_diag_cross.bin", "rb") as f:
        h_rows, h_cols = struct.unpack("ii", f.read(8))
        cross_data = np.frombuffer(f.read(), dtype=np.float32).reshape(h_cols, h_rows)
    print(f"Cross data: header={h_rows}x{h_cols}, loaded shape={cross_data.shape}")
    # cross_data is [n_ctx, 10240] = [n_ctx, n_taps * n_embd]

    # Load C++ embeddings
    with open("/tmp/dflash_diag_embd.bin", "rb") as f:
        e_rows, e_cols = struct.unpack("ii", f.read(8))
        embd_data = np.frombuffer(f.read(), dtype=np.float32).reshape(e_cols, e_rows)
    print(f"Embeddings: header={e_rows}x{e_cols}, loaded shape={embd_data.shape}")
    # embd_data is [blk_size, n_embd]

    # Load C++ logits
    with open("/tmp/dflash_diag_logits.bin", "rb") as f:
        l_rows, l_cols = struct.unpack("ii", f.read(8))
        logits_data = np.frombuffer(f.read(), dtype=np.float32).reshape(l_cols, l_rows)
    print(f"Logits: header={l_rows}x{l_cols}, loaded shape={logits_data.shape}")
    # logits_data is [blk_size, n_vocab]

    # Load HF DFlash drafter
    print("\nLoading HF DFlash drafter...")
    dflash_mod = load_dflash_module()
    DFlashDraftModel = dflash_mod.DFlashDraftModel
    from transformers import Qwen3Config

    config_path = os.path.join(DFLASH_DIR, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)
    config = Qwen3Config(**config_dict)
    model = DFlashDraftModel(config)

    import glob
    state_dict = {}
    for sf_path in sorted(glob.glob(os.path.join(DFLASH_DIR, "*.safetensors"))):
        state_dict.update(load_file(sf_path))
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"  Loaded: {config.num_hidden_layers} layers, hidden={config.hidden_size}")

    # Convert C++ data to PyTorch tensors
    # target_hidden: [1, n_ctx, n_taps * n_embd] — this is the RAW concatenated hidden states
    # (before fc projection — the C++ dumps the cross_inp which is the raw data, NOT the fc output)
    target_hidden = torch.from_numpy(cross_data.copy()).unsqueeze(0).float()
    print(f"  target_hidden shape: {target_hidden.shape} (expected [1, {n_ctx}, {n_taps * n_embd}])")

    # noise_embedding: [1, blk_size, n_embd]
    noise_embedding = torch.from_numpy(embd_data.copy()).unsqueeze(0).float()
    print(f"  noise_embedding shape: {noise_embedding.shape} (expected [1, {blk_size}, {n_embd}])")

    # Position IDs: [1, n_ctx + blk_size] spanning [0, pos_start + blk_size)
    position_ids = torch.arange(n_ctx + blk_size).unsqueeze(0)
    print(f"  position_ids: [{position_ids[0,0].item()}, ..., {position_ids[0,-1].item()}]")

    # Run HF DFlash forward with same inputs
    with torch.no_grad():
        hf_output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
    print(f"  HF output shape: {hf_output.shape}")

    # To get logits, we need the target model's lm_head
    # Since we don't have it, let's compare the hidden states before lm_head
    # But we DO have the C++ logits which already went through lm_head
    # We can still compare the hidden state norms and patterns

    print("\n=== Hidden State Comparison (before lm_head) ===")
    hf_norms = hf_output[0].norm(dim=-1)
    print("  HF per-position norms:")
    for i in range(blk_size):
        print(f"    pos {i:2d}: norm={hf_norms[i]:.4f}")

    # Check if HF positions are differentiated
    print("\n  HF pairwise diffs (max abs):")
    for i in range(min(4, blk_size)):
        for j in range(i+1, min(i+3, blk_size)):
            diff = (hf_output[0, i] - hf_output[0, j]).abs().max().item()
            print(f"    pos {i} vs {j}: {diff:.6f}")

    # Compare C++ logits analysis
    print("\n=== C++ vs HF Per-Position Analysis ===")
    cpp_logits = logits_data  # [blk_size, n_vocab]
    for i in range(blk_size):
        cpp_argmax = cpp_logits[i].argmax()
        cpp_max = cpp_logits[i].max()
        cpp_norm = np.sqrt((cpp_logits[i][:1000] ** 2).sum())
        print(f"  pos {i:2d}: C++ argmax={cpp_argmax:6d} logit={cpp_max:.4f} norm={cpp_norm:.4f}"
              f"  |  HF hidden norm={hf_norms[i]:.4f}")

    # Extract and compare the fc + hidden_norm outputs
    # The C++ cross_inp has the RAW concatenated hidden states.
    # We can manually apply fc + hidden_norm and compare with HF's internal computation.
    print("\n=== fc+hidden_norm comparison ===")
    with torch.no_grad():
        # Manually apply fc projection
        fc_out = model.fc(target_hidden)  # [1, n_ctx, n_embd]
        fc_norm_out = model.hidden_norm(fc_out)  # [1, n_ctx, n_embd]

    print(f"  fc output shape: {fc_out.shape}")
    print(f"  fc output norms per token: {fc_out[0].norm(dim=-1)[:5].tolist()}")
    print(f"  hidden_norm output norms per token: {fc_norm_out[0].norm(dim=-1)[:5].tolist()}")

    # Save HF hidden output for external comparison
    hf_hidden = hf_output[0].numpy()
    np.save("/tmp/dflash_diag_hf_hidden.npy", hf_hidden)
    print(f"\n  Saved HF hidden states to /tmp/dflash_diag_hf_hidden.npy ({hf_hidden.shape})")

    # Now let's also test: what happens if we run HF with blk_size=2 (only id_last + 1 mask)?
    print("\n=== draft_max=2 comparison (position 1 only) ===")
    noise_emb_2 = noise_embedding[:, :2, :]  # just id_last + first mask
    pos_ids_2 = torch.arange(n_ctx + 2).unsqueeze(0)
    with torch.no_grad():
        hf_out_2 = model(
            position_ids=pos_ids_2,
            noise_embedding=noise_emb_2,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )
    diff_pos1 = (hf_out_2[0, 1] - hf_output[0, 1]).abs().max().item()
    print(f"  Position 1 hidden diff (2-token vs 16-token): {diff_pos1:.6f}")
    if diff_pos1 < 1e-4:
        print("  SAME: position 1 hidden identical regardless of block size")
    else:
        print(f"  DIFFERENT: non-causal attention makes position 1 change (diff={diff_pos1:.6f})")
        print("  This is EXPECTED in HF — noise tokens affect each other via non-causal attention")

    # Check if the HF drafter with REAL conditioning produces meaningful predictions
    # by looking at whether different positions predict different tokens
    print("\n=== HF position diversity (with real conditioning) ===")
    # We need lm_head to get actual tokens — use a proxy: project hidden states to get top features
    hf_h = hf_output[0]  # [16, 2048]
    # Just check diversity of hidden state directions
    for i in range(blk_size):
        # Cosine similarity with position 1
        cos_sim = torch.nn.functional.cosine_similarity(
            hf_h[i:i+1], hf_h[1:2], dim=-1).item()
        if i <= 3 or i == blk_size - 1:
            print(f"  pos {i} cosine sim with pos 1: {cos_sim:.6f}")


if __name__ == "__main__":
    main()
