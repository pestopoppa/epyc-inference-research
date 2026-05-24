"""Engram — reference reimplementation of DeepSeek's conditional-memory module.

Vendored and refactored from github.com/deepseek-ai/Engram (Apache-2.0) for use
in the epyc-root engram-conditional-memory handoff (Track B Phase 0).
"""
from engram.config import BackboneConfig, EngramConfig
from engram.hash import NgramHashMapping
from engram.hooks import EngramLayerWrapper, splice_engram_into
from engram.init import (
    apply_identity_init,
    count_parameters,
    freeze,
    make_two_group_adamw,
    trainable_parameters,
)
from engram.modules import Engram, MultiHeadEmbedding, ShortConv

__all__ = [
    "BackboneConfig",
    "EngramConfig",
    "Engram",
    "EngramLayerWrapper",
    "MultiHeadEmbedding",
    "NgramHashMapping",
    "ShortConv",
    "apply_identity_init",
    "count_parameters",
    "freeze",
    "make_two_group_adamw",
    "splice_engram_into",
    "trainable_parameters",
]
