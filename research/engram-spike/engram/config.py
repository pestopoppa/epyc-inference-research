"""Dataclass configs for Engram and the host backbone.

Both upstream globals (`engram_cfg`, `backbone_config`) are replaced by
explicit configs passed at construction time, so a single process can host
multiple Engram modules with different settings (e.g. for ablations).
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EngramConfig:
    """Configuration for one Engram module family.

    Defaults match the upstream demo. For unit tests, override with the tiny
    values in tests/conftest.py.
    """

    # Vocab size per n-gram order (length = max_ngram_size - 1, since n starts at 2).
    # Each entry is the *target* slot count per hash head; the actual count
    # is the next prime ≥ this value, picked once per (layer, n, head).
    engram_vocab_size: List[int] = field(default_factory=lambda: [129280 * 5, 129280 * 5])

    # Max n-gram order. n iterates over {2, ..., max_ngram_size}.
    max_ngram_size: int = 3

    # Per-head embedding dim is n_embed_per_ngram // n_head_per_ngram.
    # Concatenated across (n, k) heads → (max_ngram_size - 1) * n_embed_per_ngram.
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8

    # Which layer indices in the host backbone get an Engram module.
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])

    # Token id used for left-padding the n-gram shift window.
    pad_id: int = 2

    # Seed for the multiplicative-XOR hash multiplier generator.
    seed: int = 0

    # Depthwise causal Conv1D kernel size. Dilation is set to max_ngram_size.
    kernel_size: int = 4


@dataclass
class BackboneConfig:
    """Configuration of the host Transformer the Engram module is spliced into.

    Only the few quantities Engram needs to size itself. The actual backbone
    can be any nn.Module that exposes hidden states of shape
    [B, L, HC_MULT, hidden_size] at the splice point.
    """

    hidden_size: int = 1024
    hc_mult: int = 4  # hyper-connection multiplicity (1 for a vanilla backbone)
    vocab_size: int = 129280
    num_layers: int = 30
