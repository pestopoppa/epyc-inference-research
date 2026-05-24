"""Identity-preserving initialization for retrofit onto a frozen backbone.

The paper (§3) requires Engram's contribution to be exactly zero at step 0,
so that `H ← H + Engram(H, x)` reduces to `H ← H` and the backbone's
behavior is unperturbed. Gradients then flow into Engram parameters as the
task loss demands, and the model "learns to use the table" without ever
shocking the frozen backbone.

Upstream `engram_demo_v1.py` ships default PyTorch init (Kaiming for
Linear, normal for Embedding) — i.e. Engram has non-zero output at step 0.
The paper specifies zero-init but the demo omits the lines. Below is the
minimum sufficient set to actually produce zero output:

  value_proj.weight = 0    → value_proj(embeddings) = bias
  value_proj.bias   = 0    → value_proj(embeddings) = 0
  short_conv.conv.weight = 0  → short_conv(0) = 0

With those three tensors zeroed: `output = value + short_conv(value)`
              = (gate * 0) + short_conv(gate * 0)
              = 0 + short_conv(0)
              = 0 + 0
              = 0   ✓

Note: nn.RMSNorm(0) = 0 / sqrt(eps) = 0, so the RMSNorm chain inside
ShortConv handles all-zero inputs gracefully (no NaN). Verified by
tests/test_identity_at_step_zero.py.

The embedding table, key_projs, and RMSNorm weights are LEFT at their
default initialization — none of them is on the output path when value is
zero, so they do not need zero-init. Leaving them at their default keeps
the gradient signal informative the moment the first training step nudges
value_proj off zero.
"""
import torch
import torch.nn as nn

from engram.modules import Engram


def apply_identity_init(engram: Engram) -> None:
    """Zero the parameters required for `engram.forward(...) == 0` at step 0.

    Modifies the module in-place. Safe to call multiple times.
    """
    with torch.no_grad():
        # 1. value_proj: zero weight AND bias → value_proj(emb) = 0
        nn.init.zeros_(engram.value_proj.weight)
        if engram.value_proj.bias is not None:
            nn.init.zeros_(engram.value_proj.bias)

        # 2. short_conv.conv: zero weight (bias is already False in ShortConv)
        nn.init.zeros_(engram.short_conv.conv.weight)


def freeze(module: nn.Module) -> None:
    """Set requires_grad=False on every parameter of `module`.

    Used to freeze a pretrained backbone before plugging Engram on top.
    """
    for p in module.parameters():
        p.requires_grad = False


def trainable_parameters(module: nn.Module):
    """Yield (name, param) pairs where requires_grad=True."""
    for name, p in module.named_parameters():
        if p.requires_grad:
            yield name, p


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Total parameter count of `module`.

    Useful for sanity-logging "frozen backbone = 1.5B params, trainable
    Engram = 80M params" at the start of a training run.
    """
    return sum(
        p.numel()
        for p in module.parameters()
        if not trainable_only or p.requires_grad
    )


def make_two_group_adamw(
    backbone: nn.Module,
    engram: nn.Module,
    base_lr: float = 1e-4,
    engram_lr_mult: float = 5.0,
    backbone_weight_decay: float = 0.1,
    engram_weight_decay: float = 0.0,
    betas=(0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build the paper's two-group AdamW.

    Engram embedding / projection params: LR = base_lr × engram_lr_mult, WD = 0
    Backbone params: LR = base_lr, WD = backbone_weight_decay

    If the backbone is fully frozen (no requires_grad=True params), AdamW
    silently ignores the empty group; this is the intended frozen-retrofit
    path. Keeping both groups in the API lets the same call site handle
    frozen and co-trained ablations symmetrically.
    """
    backbone_train = [p for p in backbone.parameters() if p.requires_grad]
    engram_train = [p for p in engram.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        [
            {
                "params": backbone_train,
                "lr": base_lr,
                "weight_decay": backbone_weight_decay,
            },
            {
                "params": engram_train,
                "lr": base_lr * engram_lr_mult,
                "weight_decay": engram_weight_decay,
            },
        ],
        betas=betas,
        eps=eps,
    )
