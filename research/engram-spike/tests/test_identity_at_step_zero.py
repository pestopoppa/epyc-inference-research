"""Load-bearing invariant: with identity init, Engram contributes exactly 0
at step 0, so adding it to a frozen backbone is a no-op until training.

If this test ever fails, the frozen-backbone retrofit thesis (Track B) is
broken at the architecture level — there is no point spending GPU-hours on
proxy training, because the backbone is being perturbed at step 0.

The paper requires this property for identity preservation; the upstream
demo OMITS the zero-init lines. Our `engram.init.apply_identity_init`
adds them back. This test pins that contract.
"""
import torch

from engram.init import apply_identity_init


def test_identity_init_yields_zero_engram_output(tiny_engram, tiny_input_ids, tiny_backbone_config):
    """After apply_identity_init, engram.forward(any hidden, any tokens) == 0."""
    apply_identity_init(tiny_engram)

    B, L = tiny_input_ids.shape
    G = tiny_backbone_config.hc_mult
    D = tiny_backbone_config.hidden_size

    # Use a *non-zero* hidden_states so we know the zero output is from the
    # zero-inits, not from a zero input cascading through.
    hidden_states = torch.randn(B, L, G, D)
    output = tiny_engram(hidden_states=hidden_states, input_ids=tiny_input_ids)

    assert output.shape == hidden_states.shape
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6), (
        f"Engram output not zero after identity init. "
        f"max abs = {output.abs().max().item():.3e}, "
        f"mean abs = {output.abs().mean().item():.3e}"
    )


def test_identity_init_preserves_residual_stream(tiny_engram, tiny_input_ids, tiny_backbone_config):
    """Simulate `hidden = engram(hidden, x) + hidden` and assert hidden is
    bit-identical to its input."""
    apply_identity_init(tiny_engram)

    B, L = tiny_input_ids.shape
    G = tiny_backbone_config.hc_mult
    D = tiny_backbone_config.hidden_size

    hidden_before = torch.randn(B, L, G, D)
    delta = tiny_engram(hidden_states=hidden_before, input_ids=tiny_input_ids)
    hidden_after = hidden_before + delta

    assert torch.allclose(hidden_before, hidden_after, atol=1e-6), (
        f"Residual stream perturbed by identity-initialized Engram. "
        f"max diff = {(hidden_after - hidden_before).abs().max().item():.3e}"
    )


def test_default_init_does_NOT_preserve_identity(tiny_engram, tiny_input_ids, tiny_backbone_config):
    """Sanity guard: without apply_identity_init, Engram DOES perturb the
    stream. If this stops being true, our test is no longer measuring what
    we think it is (e.g. someone reordered ops to make Engram identity by
    default, which would be a bug we'd want to catch)."""
    B, L = tiny_input_ids.shape
    G = tiny_backbone_config.hc_mult
    D = tiny_backbone_config.hidden_size

    hidden_states = torch.randn(B, L, G, D)
    output = tiny_engram(hidden_states=hidden_states, input_ids=tiny_input_ids)

    # Use a generous tolerance — we just want to confirm "not identity".
    assert output.abs().max().item() > 1e-4, (
        "Default-init Engram unexpectedly produced near-zero output. "
        "Either init was changed (intentional?) or something is masking activation."
    )


def test_identity_init_does_not_zero_unrelated_params(tiny_engram):
    """apply_identity_init should ONLY zero value_proj.weight, value_proj.bias,
    and short_conv.conv.weight. Everything else stays at its default init —
    important because we want gradient signal to flow once training starts.
    """
    apply_identity_init(tiny_engram)

    # Parameters that SHOULD be zero
    assert torch.all(tiny_engram.value_proj.weight == 0)
    assert torch.all(tiny_engram.value_proj.bias == 0)
    assert torch.all(tiny_engram.short_conv.conv.weight == 0)

    # Parameters that should NOT be zero (default init is non-zero with prob ~1)
    assert tiny_engram.multi_head_embedding.embedding.weight.abs().sum().item() > 0, (
        "Embedding table should not be zero-initialized"
    )
    for kp in tiny_engram.key_projs:
        assert kp.weight.abs().sum().item() > 0, "key_projs should retain default init"
    # RMSNorm weight defaults to all-ones, so its sum is non-zero by definition.
    for norm in tiny_engram.norm1:
        assert torch.all(norm.weight == 1)
