"""HF-style backbone splicing via EngramLayerWrapper.

We mock the HF layer interface (a simple nn.Module that consumes a hidden
states tensor and returns a tuple containing it) so the tests have no
network/disk dependency. A separate integration test against a real HF
model belongs in a follow-on, not here.
"""
import torch
import torch.nn as nn

from engram.hooks import EngramLayerWrapper, collect_wrappers, splice_engram_into
from engram.init import apply_identity_init


class MockHFLayer(nn.Module):
    """HF DecoderLayer stand-in.

    Real HF layers return a tuple `(hidden_states, ...)`; ours just returns
    `hidden_states + 0.1 * identity` so we can tell whether the layer was
    called at all. Accepts arbitrary kwargs and ignores them.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, *args, **kwargs):
        return (self.proj(hidden_states),)


class MockHFModel(nn.Module):
    """Minimal HF-causal-LM-shaped model with .model.layers list."""

    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)

        class Inner(nn.Module):
            def __init__(self_inner, layers: nn.ModuleList):
                super().__init__()
                self_inner.layers = layers

        self.model = Inner(nn.ModuleList([MockHFLayer(hidden_size) for _ in range(num_layers)]))
        self.lm_head = nn.Linear(hidden_size, vocab_size)


def test_wrapper_passes_through_hf_layer(tiny_engram, tiny_input_ids, tiny_backbone_config):
    """When identity-initialized, the wrapper should produce the same output
    as calling the underlying HF layer directly."""
    apply_identity_init(tiny_engram)

    hf_layer = MockHFLayer(tiny_backbone_config.hidden_size)
    wrapper = EngramLayerWrapper(hf_layer, tiny_engram, hc_mult=tiny_backbone_config.hc_mult)
    wrapper.set_input_ids(tiny_input_ids)

    B, L = tiny_input_ids.shape
    h = torch.randn(B, L, tiny_backbone_config.hidden_size)

    direct = hf_layer(h)
    wrapped = wrapper(h)

    # Both are tuples
    assert isinstance(direct, tuple) and isinstance(wrapped, tuple)
    assert torch.allclose(direct[0], wrapped[0], atol=1e-6), (
        "Identity-init wrapper changed the HF layer's output. "
        f"max diff = {(direct[0] - wrapped[0]).abs().max().item():.3e}"
    )


def test_wrapper_raises_when_input_ids_missing(tiny_engram, tiny_backbone_config):
    """Forgetting set_input_ids before forward must fail loudly, not produce
    silently wrong outputs."""
    hf_layer = MockHFLayer(tiny_backbone_config.hidden_size)
    wrapper = EngramLayerWrapper(hf_layer, tiny_engram, hc_mult=tiny_backbone_config.hc_mult)
    h = torch.randn(2, 8, tiny_backbone_config.hidden_size)

    try:
        wrapper(h)
    except RuntimeError as e:
        assert "input_ids" in str(e).lower()
        return
    raise AssertionError("EngramLayerWrapper.forward should raise without cached input_ids")


def test_wrapper_validates_batch_shape(tiny_engram, tiny_backbone_config):
    """A stale set_input_ids that doesn't match this forward's [B, L] must
    raise, not silently mis-hash."""
    hf_layer = MockHFLayer(tiny_backbone_config.hidden_size)
    wrapper = EngramLayerWrapper(hf_layer, tiny_engram, hc_mult=tiny_backbone_config.hc_mult)
    wrapper.set_input_ids(torch.randint(0, 100, (2, 8), dtype=torch.long))

    # Mismatched batch dimension
    h = torch.randn(4, 8, tiny_backbone_config.hidden_size)
    try:
        wrapper(h)
    except RuntimeError as e:
        assert "shape" in str(e).lower() or "input_ids" in str(e).lower()
        return
    raise AssertionError("Should have raised on B-mismatch")


def test_splice_engram_into_replaces_layer(
    tiny_engram_config, tiny_backbone_config, tiny_engram, tiny_input_ids
):
    model = MockHFModel(
        vocab_size=tiny_backbone_config.vocab_size,
        hidden_size=tiny_backbone_config.hidden_size,
        num_layers=4,
    )

    layer_before = model.model.layers[1]
    wrapper = splice_engram_into(model, layer_index=1, engram=tiny_engram, hc_mult=tiny_backbone_config.hc_mult)

    assert model.model.layers[1] is wrapper
    assert wrapper.hf_layer is layer_before


def test_collect_wrappers_finds_all_spliced_layers(
    tiny_engram_config, tiny_backbone_config, tiny_hash_mapping
):
    from engram.modules import Engram

    model = MockHFModel(
        vocab_size=tiny_backbone_config.vocab_size,
        hidden_size=tiny_backbone_config.hidden_size,
        num_layers=4,
    )
    # Splice Engrams at two of the four layers.
    e1 = Engram(tiny_engram_config, tiny_backbone_config, layer_id=0, hash_mapping=tiny_hash_mapping)
    e2 = Engram(tiny_engram_config, tiny_backbone_config, layer_id=1, hash_mapping=tiny_hash_mapping)
    splice_engram_into(model, 0, e1, hc_mult=tiny_backbone_config.hc_mult)
    splice_engram_into(model, 2, e2, hc_mult=tiny_backbone_config.hc_mult)

    wrappers = collect_wrappers(model)
    assert len(wrappers) == 2


def test_spliced_model_forward_works_end_to_end(
    tiny_engram_config, tiny_backbone_config, tiny_hash_mapping, tiny_input_ids
):
    from engram.modules import Engram

    model = MockHFModel(
        vocab_size=tiny_backbone_config.vocab_size,
        hidden_size=tiny_backbone_config.hidden_size,
        num_layers=2,
    )
    engram = Engram(tiny_engram_config, tiny_backbone_config, layer_id=0, hash_mapping=tiny_hash_mapping)
    apply_identity_init(engram)
    wrapper = splice_engram_into(model, 0, engram, hc_mult=tiny_backbone_config.hc_mult)

    # Run a single forward across the spliced model.
    wrapper.set_input_ids(tiny_input_ids)
    h0 = model.embed(tiny_input_ids)
    h1 = model.model.layers[0](h0)[0]
    h2 = model.model.layers[1](h1)[0]
    logits = model.lm_head(h2)
    assert logits.shape == (tiny_input_ids.shape[0], tiny_input_ids.shape[1], tiny_backbone_config.vocab_size)


def test_two_engram_layers_share_hash_mapping(
    tiny_engram_config, tiny_backbone_config, tiny_hash_mapping
):
    """When wiring multiple Engram layers, they should share one hash mapping
    so prime-search runs once and per-layer multipliers are consistent."""
    from engram.modules import Engram

    e1 = Engram(tiny_engram_config, tiny_backbone_config, layer_id=0, hash_mapping=tiny_hash_mapping)
    e2 = Engram(tiny_engram_config, tiny_backbone_config, layer_id=1, hash_mapping=tiny_hash_mapping)
    assert e1.hash_mapping is e2.hash_mapping
