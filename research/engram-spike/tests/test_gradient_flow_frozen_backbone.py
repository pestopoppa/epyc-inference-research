"""Frozen-backbone gradient-flow test.

When we plug Engram on top of a frozen backbone and run a training step,
gradients must flow into the Engram parameters and NOT into the backbone
parameters. If this fails, the entire Track B retrofit plan is broken at
the optimizer-plumbing level.
"""
import torch
import torch.nn as nn

from engram.init import apply_identity_init, freeze, trainable_parameters


class TinyBackbone(nn.Module):
    """Stand-in for a real HF backbone — just enough to give Engram somewhere
    to live and to produce a loss."""

    def __init__(self, vocab_size: int, hidden_size: int, hc_mult: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.hc_mult = hc_mult
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward_pre_engram(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token IDs → hidden states shaped for Engram input."""
        h = self.embed(input_ids)  # [B, L, D]
        # Broadcast across hc_mult to match Engram's expected shape.
        return h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).contiguous()

    def forward_post_engram(self, h_hc: torch.Tensor) -> torch.Tensor:
        """Collapse hc dim and project to logits."""
        h = h_hc[:, :, 0, :]  # take first hc stream (matches upstream demo)
        return self.lm_head(h)


def test_freeze_removes_requires_grad(tiny_engram_config, tiny_backbone_config):
    backbone = TinyBackbone(
        vocab_size=tiny_backbone_config.vocab_size,
        hidden_size=tiny_backbone_config.hidden_size,
        hc_mult=tiny_backbone_config.hc_mult,
    )
    # Before freeze: all backbone params require grad
    assert all(p.requires_grad for p in backbone.parameters())
    freeze(backbone)
    assert not any(p.requires_grad for p in backbone.parameters())


def test_gradients_flow_only_to_engram_when_backbone_frozen(
    tiny_engram, tiny_backbone_config, tiny_input_ids
):
    """End-to-end: freeze backbone, Engram is trainable, do one backward,
    verify .grad is None on backbone params and non-None on Engram params."""
    backbone = TinyBackbone(
        vocab_size=tiny_backbone_config.vocab_size,
        hidden_size=tiny_backbone_config.hidden_size,
        hc_mult=tiny_backbone_config.hc_mult,
    )
    freeze(backbone)

    # Engram is NOT identity-initialized here — we want non-trivial gradients.
    h_before = backbone.forward_pre_engram(tiny_input_ids)
    delta = tiny_engram(hidden_states=h_before, input_ids=tiny_input_ids)
    h_after = h_before + delta
    logits = backbone.forward_post_engram(h_after)

    # Trivial loss: encourage logits sum to zero (proxy for "any loss function").
    loss = logits.pow(2).mean()
    loss.backward()

    # No backbone param should have a gradient.
    backbone_grads = [(n, p.grad) for n, p in backbone.named_parameters()]
    bad_backbone = [n for n, g in backbone_grads if g is not None]
    assert not bad_backbone, (
        f"Frozen backbone params received gradients: {bad_backbone}"
    )

    # Engram params should have gradients (at least some non-zero).
    engram_grads = [(n, p.grad) for n, p in tiny_engram.named_parameters() if p.requires_grad]
    assert engram_grads, "No trainable Engram parameters — config error"
    has_signal = any(g is not None and g.abs().sum().item() > 0 for _, g in engram_grads)
    assert has_signal, (
        "No Engram param received a non-zero gradient. "
        "Optimizer plumbing or autograd graph likely broken."
    )


def test_two_group_optimizer_factory(tiny_engram, tiny_backbone_config):
    """Sanity check that we can build the two-group AdamW the paper specifies:
    backbone params (lr=1×, wd=0.1) + Engram params (lr=5×, wd=0).

    Here we test the typing/wiring, not the convergence."""
    backbone = TinyBackbone(
        vocab_size=tiny_backbone_config.vocab_size,
        hidden_size=tiny_backbone_config.hidden_size,
        hc_mult=tiny_backbone_config.hc_mult,
    )
    # Frozen backbone path: only the Engram params reach the optimizer.
    freeze(backbone)
    backbone_train = [p for p in backbone.parameters() if p.requires_grad]
    engram_train = [p for p in tiny_engram.parameters() if p.requires_grad]

    assert len(backbone_train) == 0
    assert len(engram_train) > 0

    base_lr = 1e-4
    optim = torch.optim.AdamW(
        [
            {"params": backbone_train, "lr": base_lr, "weight_decay": 0.1},
            {"params": engram_train, "lr": base_lr * 5, "weight_decay": 0.0},
        ]
    )
    # AdamW silently ignores empty param groups, which is what we want here.
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["lr"] == base_lr
    assert optim.param_groups[0]["weight_decay"] == 0.1
    assert optim.param_groups[1]["lr"] == base_lr * 5
    assert optim.param_groups[1]["weight_decay"] == 0.0


def test_trainable_parameters_helper(tiny_engram):
    """trainable_parameters() should yield only requires_grad=True entries."""
    # Default: everything trainable.
    names_all = [n for n, _ in trainable_parameters(tiny_engram)]
    assert len(names_all) > 0
    # Freeze a specific param and verify it drops out.
    tiny_engram.value_proj.weight.requires_grad = False
    names_partial = [n for n, _ in trainable_parameters(tiny_engram)]
    assert "value_proj.weight" not in names_partial
    assert any("multi_head_embedding" in n for n in names_partial)
