"""Shared fixtures and tiny configs for the test suite.

All configs here are intentionally small enough to make every test finish in
well under a second on CPU. The goal is to exercise the data flow and
invariants, not to produce meaningful learned representations.
"""
import sys
from pathlib import Path

import pytest
import torch

# Make the package importable without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.config import BackboneConfig, EngramConfig
from engram.hash import NgramHashMapping
from engram.modules import Engram


@pytest.fixture
def tiny_engram_config() -> EngramConfig:
    """Toy Engram config: 2 heads, vocab~50, max_ngram=3."""
    return EngramConfig(
        engram_vocab_size=[50, 50],  # one per n-gram order (n=2, n=3)
        max_ngram_size=3,
        n_embed_per_ngram=32,  # 32 // n_head_per_ngram = 16 per head
        n_head_per_ngram=2,
        layer_ids=[0, 1],
        pad_id=0,
        seed=42,
        kernel_size=4,
    )


@pytest.fixture
def tiny_backbone_config() -> BackboneConfig:
    """Toy backbone: hidden=32, hc_mult=2, vocab=100."""
    return BackboneConfig(
        hidden_size=32,
        hc_mult=2,
        vocab_size=100,
        num_layers=2,
    )


@pytest.fixture
def tiny_hash_mapping(tiny_engram_config, tiny_backbone_config) -> NgramHashMapping:
    return NgramHashMapping(
        engram_vocab_size=tiny_engram_config.engram_vocab_size,
        max_ngram_size=tiny_engram_config.max_ngram_size,
        n_head_per_ngram=tiny_engram_config.n_head_per_ngram,
        layer_ids=tiny_engram_config.layer_ids,
        tokenizer_vocab_size=tiny_backbone_config.vocab_size,
        pad_id=tiny_engram_config.pad_id,
        seed=tiny_engram_config.seed,
    )


@pytest.fixture
def tiny_engram(tiny_engram_config, tiny_backbone_config, tiny_hash_mapping) -> Engram:
    """A freshly-constructed Engram module on the toy configs, default init."""
    torch.manual_seed(0)
    return Engram(
        engram_config=tiny_engram_config,
        backbone_config=tiny_backbone_config,
        layer_id=0,
        hash_mapping=tiny_hash_mapping,
    )


@pytest.fixture
def tiny_input_ids() -> torch.Tensor:
    """A small [B=2, L=8] batch of token ids inside the toy vocab."""
    torch.manual_seed(0)
    return torch.randint(low=0, high=100, size=(2, 8), dtype=torch.long)
