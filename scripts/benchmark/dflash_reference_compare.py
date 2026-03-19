#!/usr/bin/env python3
"""DFlash Reference Comparison — Multi-Position Diagnostic

Loads the HF DFlash drafter model and tests whether multi-position
prediction works correctly. This diagnoses the block-mode failure
where positions 2-15 produce wrong predictions in the C++ implementation.

Tests:
1. Synthetic conditioning: random target_hidden → verify all positions produce different outputs
2. Synthetic conditioning: verify position 1 vs position 2+ have different attention patterns
3. If target model available: real conditioning → compare per-position logits

Usage:
    python3 dflash_reference_compare.py [--with-target]
"""

import sys
import os
import json
import argparse
import numpy as np

# DFlash model dir
DFLASH_DIR = "/mnt/raid0/llm/cache/dflash/Qwen3-Coder-30B-A3B-DFlash"
TARGET_DIR = "/mnt/raid0/llm/lmstudio/models/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"

import torch
from safetensors.torch import load_file

# Patch the dflash module to handle relative imports
import importlib
import types

def _load_dflash_module():
    """Load dflash.py and utils.py as a proper package."""
    # Create a fake package
    pkg = types.ModuleType("dflash_pkg")
    pkg.__path__ = [DFLASH_DIR]
    pkg.__package__ = "dflash_pkg"
    sys.modules["dflash_pkg"] = pkg

    # Load utils first
    utils_spec = importlib.util.spec_from_file_location(
        "dflash_pkg.utils", os.path.join(DFLASH_DIR, "utils.py"))
    utils_mod = importlib.util.module_from_spec(utils_spec)
    sys.modules["dflash_pkg.utils"] = utils_mod
    utils_spec.loader.exec_module(utils_mod)
    pkg.utils = utils_mod

    # Patch dflash.py to use absolute import
    dflash_path = os.path.join(DFLASH_DIR, "dflash.py")
    with open(dflash_path) as f:
        source = f.read()
    source = source.replace("from .utils import", "from dflash_pkg.utils import")

    dflash_spec = importlib.util.spec_from_file_location(
        "dflash_pkg.dflash", dflash_path)
    dflash_mod = importlib.util.module_from_spec(dflash_spec)
    sys.modules["dflash_pkg.dflash"] = dflash_mod
    exec(compile(source, dflash_path, "exec"), dflash_mod.__dict__)
    pkg.dflash = dflash_mod

    return dflash_mod


def load_dflash_drafter():
    """Load DFlash drafter model from safetensors."""
    dflash_mod = _load_dflash_module()
    DFlashDraftModel = dflash_mod.DFlashDraftModel
    from transformers import Qwen3Config

    config_path = os.path.join(DFLASH_DIR, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config = Qwen3Config(**config_dict)
    model = DFlashDraftModel(config)

    # Load weights from safetensors
    import glob
    state_dict = {}
    for sf_path in sorted(glob.glob(os.path.join(DFLASH_DIR, "*.safetensors"))):
        state_dict.update(load_file(sf_path))

    # The state dict keys may have "model." prefix or not
    # Try loading directly first
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model.eval()
    return model, config


def test_synthetic_multiposition(model, config):
    """Test 1: Verify multi-position produces different outputs with synthetic conditioning."""
    print("\n=== Test 1: Synthetic Multi-Position ===")

    block_size = config.block_size  # 16
    hidden_size = config.hidden_size  # 2048
    n_taps = len(model.target_layer_ids)  # 5
    n_ctx = 10  # synthetic context length

    # Create synthetic inputs
    torch.manual_seed(42)

    # target_hidden: [1, n_ctx, n_taps * hidden_size]
    target_hidden = torch.randn(1, n_ctx, n_taps * hidden_size, dtype=torch.float32)

    # noise_embedding: [1, block_size, hidden_size]
    # Position 0 = "real" token embedding, positions 1-15 = mask token embedding
    noise_embedding = torch.randn(1, 1, hidden_size)  # id_last embedding
    mask_embedding = torch.randn(1, 1, hidden_size)    # mask token embedding
    noise_embedding = torch.cat([noise_embedding, mask_embedding.expand(1, block_size - 1, hidden_size)], dim=1)

    # position_ids: [1, n_ctx + block_size] (all positions for RoPE)
    position_ids = torch.arange(n_ctx + block_size).unsqueeze(0)

    with torch.no_grad():
        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )

    # output shape: [1, block_size, hidden_size]
    print(f"  Output shape: {output.shape}")
    print(f"  Expected: [1, {block_size}, {hidden_size}]")

    # Check: are all positions different?
    for i in range(block_size):
        for j in range(i + 1, block_size):
            diff = (output[0, i] - output[0, j]).abs().max().item()
            if diff < 1e-6:
                print(f"  WARNING: positions {i} and {j} are IDENTICAL (max diff={diff:.2e})")
            elif i < 3 and j < 3:
                print(f"  Positions {i} vs {j}: max diff = {diff:.4f}")

    # Check norms (are outputs reasonable, not degenerate?)
    norms = output[0].norm(dim=-1)
    print(f"  Output norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
    print(f"  Position 0 norm: {norms[0]:.4f}")
    print(f"  Position 1 norm: {norms[1]:.4f}")
    print(f"  Position 15 norm: {norms[-1]:.4f}")

    return output


def test_identical_mask_positions(model, config):
    """Test 2: With mask tokens at all positions (no id_last), check differentiation via RoPE."""
    print("\n=== Test 2: All-Mask Tokens (RoPE-only differentiation) ===")

    block_size = config.block_size
    hidden_size = config.hidden_size
    n_taps = len(model.target_layer_ids)
    n_ctx = 10

    torch.manual_seed(42)
    target_hidden = torch.randn(1, n_ctx, n_taps * hidden_size, dtype=torch.float32)

    # All positions use the SAME embedding (like mask_token_id)
    mask_emb = torch.randn(1, 1, hidden_size)
    noise_embedding = mask_emb.expand(1, block_size, hidden_size).clone()

    position_ids = torch.arange(n_ctx + block_size).unsqueeze(0)

    with torch.no_grad():
        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )

    print(f"  Output shape: {output.shape}")

    # Even with identical input embeddings, RoPE should differentiate positions
    diffs = []
    for i in range(block_size):
        for j in range(i + 1, min(i + 3, block_size)):
            diff = (output[0, i] - output[0, j]).abs().max().item()
            diffs.append(diff)
            if i < 3:
                print(f"  Positions {i} vs {j}: max diff = {diff:.6f}")

    avg_diff = sum(diffs) / len(diffs) if diffs else 0
    print(f"  Average pairwise max diff: {avg_diff:.6f}")
    if avg_diff < 1e-4:
        print("  PROBLEM: positions barely differentiated — RoPE may not be working")
    else:
        print("  OK: positions are differentiated (RoPE working)")


def test_with_lm_head(model, config):
    """Test 3: Apply a synthetic lm_head and check per-position argmax tokens."""
    print("\n=== Test 3: Per-Position Argmax (synthetic lm_head) ===")

    block_size = config.block_size
    hidden_size = config.hidden_size
    vocab_size = config.vocab_size
    n_taps = len(model.target_layer_ids)
    n_ctx = 10

    torch.manual_seed(42)
    target_hidden = torch.randn(1, n_ctx, n_taps * hidden_size, dtype=torch.float32)

    # Realistic block: id_last=100, rest=mask
    noise_embedding = torch.randn(1, block_size, hidden_size)

    position_ids = torch.arange(n_ctx + block_size).unsqueeze(0)

    with torch.no_grad():
        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
            is_causal=False,
        )

    # Create a random lm_head (just to verify per-position token differentiation)
    torch.manual_seed(123)
    lm_head = torch.randn(vocab_size, hidden_size) * 0.01  # [vocab, hidden]

    logits = output @ lm_head.T  # [1, block_size, vocab]
    tokens = logits.argmax(dim=-1)[0]  # [block_size]

    print(f"  Logits shape: {logits.shape}")
    print(f"  Per-position argmax tokens:")
    for i in range(block_size):
        top_val = logits[0, i].max().item()
        print(f"    pos {i:2d}: token {tokens[i]:6d}  (logit={top_val:.4f})")

    n_unique = len(set(tokens.tolist()))
    print(f"  Unique tokens: {n_unique}/{block_size}")
    if n_unique < block_size // 2:
        print("  WARNING: low diversity — many positions predict the same token")


def test_kv_cache_round1(model, config):
    """Test 4: Round 1 with KV cache (matches HF spec_generate setup)."""
    print("\n=== Test 4: Round 1 with KV Cache (matching spec_generate) ===")

    from transformers import DynamicCache

    block_size = config.block_size
    hidden_size = config.hidden_size
    n_taps = len(model.target_layer_ids)
    n_ctx = 10

    torch.manual_seed(42)
    target_hidden = torch.randn(1, n_ctx, n_taps * hidden_size, dtype=torch.float32)

    # Same as spec_generate round 1
    mask_token_id = model.mask_token_id or 151669
    noise_embedding = torch.randn(1, block_size, hidden_size)  # pretend these are embed(block_tokens)

    # Round 1: KV cache empty, position_ids span [0, n_ctx+block_size)
    past_key_values = DynamicCache()
    position_ids = torch.arange(n_ctx + block_size).unsqueeze(0)

    with torch.no_grad():
        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            past_key_values=past_key_values,
            use_cache=True,
            is_causal=False,
        )

    print(f"  Output shape: {output.shape}")
    print(f"  KV cache length after round 1: {past_key_values.get_seq_length()}")

    # Check per-position differentiation
    for i in range(min(4, block_size)):
        for j in range(i + 1, min(i + 2, block_size)):
            diff = (output[0, i] - output[0, j]).abs().max().item()
            print(f"  Positions {i} vs {j}: max diff = {diff:.6f}")

    # Skip position 0 (id_last) — draft logits come from positions 1..15
    draft_output = output[:, 1:, :]  # [1, 15, hidden_size]
    print(f"  Draft output shape: {draft_output.shape}")

    # Verify draft positions are all different
    unique_check = True
    for i in range(14):
        diff = (draft_output[0, i] - draft_output[0, i + 1]).abs().max().item()
        if diff < 1e-6:
            print(f"  WARNING: draft positions {i+1} and {i+2} are identical!")
            unique_check = False
    if unique_check:
        print("  OK: all draft positions produce different outputs")


def test_draft_max_2_vs_16(model, config):
    """Test 5: Compare draft_max=2 (1 draft token) vs draft_max=16 (15 draft tokens).

    This directly mirrors the C++ diagnostic where draft_max=2 gives 27%
    but draft_max=16 gives 1.4%. If HF shows the same pattern, the issue
    is in the DFlash model/architecture, not our C++ code.
    """
    print("\n=== Test 5: draft_max=2 vs draft_max=16 (Key Diagnostic) ===")

    block_size = config.block_size
    hidden_size = config.hidden_size
    n_taps = len(model.target_layer_ids)
    n_ctx = 10

    torch.manual_seed(42)
    target_hidden = torch.randn(1, n_ctx, n_taps * hidden_size, dtype=torch.float32)

    # Fixed noise embedding (to compare between 2-token and 16-token modes)
    torch.manual_seed(99)
    id_last_emb = torch.randn(1, 1, hidden_size)
    mask_emb = torch.randn(1, 1, hidden_size)

    # --- Mode A: 2 tokens (id_last + 1 mask) ---
    noise_2 = torch.cat([id_last_emb, mask_emb], dim=1)  # [1, 2, hidden_size]
    pos_ids_2 = torch.arange(n_ctx + 2).unsqueeze(0)

    with torch.no_grad():
        out_2 = model(
            position_ids=pos_ids_2,
            noise_embedding=noise_2,
            target_hidden=target_hidden,
            past_key_values=None, use_cache=False, is_causal=False,
        )

    # --- Mode B: 16 tokens (id_last + 15 masks) ---
    noise_16 = torch.cat([id_last_emb, mask_emb.expand(1, 15, hidden_size)], dim=1)
    pos_ids_16 = torch.arange(n_ctx + 16).unsqueeze(0)

    with torch.no_grad():
        out_16 = model(
            position_ids=pos_ids_16,
            noise_embedding=noise_16,
            target_hidden=target_hidden,
            past_key_values=None, use_cache=False, is_causal=False,
        )

    # Compare position 1 output between 2-token and 16-token modes
    diff_pos1 = (out_2[0, 1] - out_16[0, 1]).abs().max().item()
    print(f"  Position 1 output diff (2-token vs 16-token): {diff_pos1:.6f}")

    if diff_pos1 < 1e-4:
        print("  Position 1 is IDENTICAL between modes — matches C++ observation")
        print("  (adding noise tokens 2-15 doesn't corrupt position 1)")
    else:
        print(f"  Position 1 DIFFERS between modes (diff={diff_pos1:.6f})")
        print("  (noise tokens 2-15 DO affect position 1 via non-causal attention)")

    # Check position 0 (id_last)
    diff_pos0 = (out_2[0, 0] - out_16[0, 0]).abs().max().item()
    print(f"  Position 0 output diff (2-token vs 16-token): {diff_pos0:.6f}")

    # Show per-position norms for 16-token mode
    norms_16 = out_16[0].norm(dim=-1)
    print(f"\n  16-token mode per-position norms:")
    for i in range(16):
        print(f"    pos {i:2d}: norm={norms_16[i]:.4f}")


def main():
    parser = argparse.ArgumentParser(description="DFlash Reference Comparison")
    parser.add_argument("--with-target", action="store_true",
                        help="Also load target model (requires ~60GB RAM)")
    args = parser.parse_args()

    print("Loading DFlash drafter model...")
    model, config = load_dflash_drafter()
    print(f"  Loaded: {config.num_hidden_layers} layers, hidden={config.hidden_size}, "
          f"heads={config.num_attention_heads}/{config.num_key_value_heads}, "
          f"block_size={config.block_size}")
    print(f"  Target layer IDs: {model.target_layer_ids}")
    print(f"  fc weight shape: {model.fc.weight.shape}")
    print(f"  hidden_norm weight shape: {model.hidden_norm.weight.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params / 1e6:.1f}M")

    test_synthetic_multiposition(model, config)
    test_identical_mask_positions(model, config)
    test_with_lm_head(model, config)
    test_kv_cache_round1(model, config)
    test_draft_max_2_vs_16(model, config)

    print("\n=== All tests complete ===")


if __name__ == "__main__":
    main()
