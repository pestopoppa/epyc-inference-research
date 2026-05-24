"""Shape correctness — forward pass returns the residual-delta shape the
host backbone expects.

This is the lowest-bar sanity test: if shapes are wrong, every later test
fails for the wrong reason.
"""
import torch


def test_engram_output_matches_hidden_states_shape(tiny_engram, tiny_backbone_config, tiny_input_ids):
    B, L = tiny_input_ids.shape
    D = tiny_backbone_config.hidden_size
    G = tiny_backbone_config.hc_mult

    hidden_states = torch.randn(B, L, G, D)
    output = tiny_engram(hidden_states=hidden_states, input_ids=tiny_input_ids)

    assert output.shape == (B, L, G, D), (
        f"Engram output shape {tuple(output.shape)} != expected (B={B}, L={L}, G={G}, D={D})"
    )
    assert output.dtype == hidden_states.dtype


def test_hash_output_shape(tiny_hash_mapping, tiny_input_ids):
    ids_np = tiny_input_ids.numpy()
    hashes = tiny_hash_mapping.hash(ids_np)
    B, L = tiny_input_ids.shape
    num_heads_total = tiny_hash_mapping.num_heads_total()
    for lid, h in hashes.items():
        assert h.shape == (B, L, num_heads_total), (
            f"layer {lid}: hash shape {h.shape} != ({B}, {L}, {num_heads_total})"
        )
        assert h.dtype.kind == "i", f"hash dtype must be integer, got {h.dtype}"


def test_engram_runs_for_each_layer(tiny_engram_config, tiny_backbone_config, tiny_hash_mapping, tiny_input_ids):
    """Both layers in the toy config can independently construct + forward."""
    from engram.modules import Engram

    B, L = tiny_input_ids.shape
    hidden_states = torch.randn(B, L, tiny_backbone_config.hc_mult, tiny_backbone_config.hidden_size)
    for layer_id in tiny_engram_config.layer_ids:
        e = Engram(
            engram_config=tiny_engram_config,
            backbone_config=tiny_backbone_config,
            layer_id=layer_id,
            hash_mapping=tiny_hash_mapping,
        )
        out = e(hidden_states=hidden_states, input_ids=tiny_input_ids)
        assert out.shape == hidden_states.shape
