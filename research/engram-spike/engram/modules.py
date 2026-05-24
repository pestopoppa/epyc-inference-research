"""Engram nn.Modules: ShortConv, MultiHeadEmbedding, Engram.

Forward-pass math matches the upstream demo verbatim. The two material
changes vs upstream are:

  1. No global state — `Engram(engram_config, backbone_config, layer_id)`
     takes explicit configs.
  2. `apply_identity_init(engram)` (see engram/init.py) zeros the parameters
     required to make `engram.forward(...)` return all-zeros at step 0,
     which the upstream demo omits despite the paper requiring it for
     identity preservation.
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from engram.config import BackboneConfig, EngramConfig
from engram.hash import NgramHashMapping


class ShortConv(nn.Module):
    """Depthwise causal Conv1D with per-hyper-connection RMSNorm + SiLU.

    Verbatim port of upstream. Causal via left-padded `padding=(k-1)*dilation`
    then truncation back to T after conv.
    """

    def __init__(
        self,
        hidden_size: int,
        kernel_size: int = 4,
        dilation: int = 1,
        norm_eps: float = 1e-5,
        hc_mult: int = 4,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        self.norms = nn.ModuleList(
            [nn.RMSNorm(hidden_size, eps=norm_eps) for _ in range(hc_mult)]
        )
        if self.activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, HC_MULT, D] → [B, T, HC_MULT, D]."""
        B, T, G, C = x.shape
        if G != self.hc_mult:
            raise ValueError(f"Input groups {G} != hc_mult {self.hc_mult}")

        normed_chunks = [self.norms[i](x[:, :, i, :]) for i in range(G)]
        x_norm = torch.cat(normed_chunks, dim=-1)  # [B, T, G*C]
        x_bct = x_norm.transpose(1, 2)  # [B, G*C, T]
        y_bct = self.conv(x_bct)
        y_bct = y_bct[..., :T]  # truncate causal pad
        if self.activation:
            y_bct = self.act_fn(y_bct)
        return y_bct.transpose(1, 2).view(B, T, G, C).contiguous()


class MultiHeadEmbedding(nn.Module):
    """One concatenated embedding table covering K heads of differing sizes.

    Each head is offset into a single contiguous `nn.Embedding(sum(N_k), D)`.
    Storing them in one tensor lets a single gather kernel serve all heads,
    which is the per-token BW saving in the paper's offload story.
    """

    def __init__(self, list_of_N: List[int], D: int):
        super().__init__()
        self.num_heads = len(list_of_N)
        self.embedding_dim = D

        offsets = [0]
        for n in list_of_N[:-1]:
            offsets.append(offsets[-1] + n)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))

        total_N = sum(list_of_N)
        self.embedding = nn.Embedding(num_embeddings=total_N, embedding_dim=D)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """[..., num_heads] int64 → [..., num_heads, D] float."""
        shifted = input_ids + self.offsets
        return self.embedding(shifted)


class Engram(nn.Module):
    """One Engram module for one Transformer layer.

    Forward signature: `engram(hidden_states, input_ids) -> residual_delta`.
    The host backbone is responsible for the residual add:
        hidden_states = engram(hidden_states, input_ids) + hidden_states

    Args:
        engram_config: EngramConfig (see config.py)
        backbone_config: BackboneConfig (sizes the projections)
        layer_id: which layer index this module belongs to (must be in
            engram_config.layer_ids)
        hash_mapping: optional pre-built NgramHashMapping (sharable across
            layers; saves the prime-search cost). If None, builds a fresh one.
        tokenizer_vocab_size: required if hash_mapping is None
    """

    def __init__(
        self,
        engram_config: EngramConfig,
        backbone_config: BackboneConfig,
        layer_id: int,
        hash_mapping: Optional[NgramHashMapping] = None,
        tokenizer_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        if layer_id not in engram_config.layer_ids:
            raise ValueError(
                f"layer_id={layer_id} not in engram_config.layer_ids={engram_config.layer_ids}"
            )
        self.engram_config = engram_config
        self.backbone_config = backbone_config
        self.layer_id = layer_id

        if hash_mapping is None:
            if tokenizer_vocab_size is None:
                raise ValueError(
                    "Either hash_mapping or tokenizer_vocab_size must be provided"
                )
            hash_mapping = NgramHashMapping(
                engram_vocab_size=engram_config.engram_vocab_size,
                max_ngram_size=engram_config.max_ngram_size,
                n_head_per_ngram=engram_config.n_head_per_ngram,
                layer_ids=engram_config.layer_ids,
                tokenizer_vocab_size=tokenizer_vocab_size,
                pad_id=engram_config.pad_id,
                seed=engram_config.seed,
            )
        self.hash_mapping = hash_mapping

        # Per-head embedding table for this layer.
        flat_head_sizes = self.hash_mapping.flat_head_primes(layer_id)
        per_head_dim = engram_config.n_embed_per_ngram // engram_config.n_head_per_ngram
        self.multi_head_embedding = MultiHeadEmbedding(
            list_of_N=flat_head_sizes,
            D=per_head_dim,
        )

        # Depthwise causal conv mixing across the residual stream.
        self.short_conv = ShortConv(
            hidden_size=backbone_config.hidden_size,
            kernel_size=engram_config.kernel_size,
            dilation=engram_config.max_ngram_size,
            hc_mult=backbone_config.hc_mult,
        )

        # Projections from concatenated embeddings → residual width.
        engram_hidden_size = (engram_config.max_ngram_size - 1) * engram_config.n_embed_per_ngram
        self.value_proj = nn.Linear(engram_hidden_size, backbone_config.hidden_size)
        self.key_projs = nn.ModuleList(
            [
                nn.Linear(engram_hidden_size, backbone_config.hidden_size)
                for _ in range(backbone_config.hc_mult)
            ]
        )
        self.norm1 = nn.ModuleList(
            [nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )
        self.norm2 = nn.ModuleList(
            [nn.RMSNorm(backbone_config.hidden_size) for _ in range(backbone_config.hc_mult)]
        )

    def _hash_to_device(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the numpy hash for this layer and return a same-device LongTensor.

        Hashing is CPU-side (numpy); we then transfer the int64 indices to
        wherever the embedding table lives. This matches upstream's behavior
        and avoids the device-mismatch bug in upstream PR #15.
        """
        ids_np = input_ids.detach().cpu().numpy()
        hashes_np = self.hash_mapping.hash(ids_np, layer_ids=[self.layer_id])[self.layer_id]
        hashes = torch.from_numpy(hashes_np).to(input_ids.device)
        return hashes

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, HC_MULT, D]
            input_ids: [B, L] int64

        Returns:
            residual delta [B, L, HC_MULT, D] — host backbone adds this to
            hidden_states.
        """
        hash_ids = self._hash_to_device(input_ids)  # [B, L, num_heads_total]
        embeddings = self.multi_head_embedding(hash_ids).flatten(start_dim=-2)
        # → [B, L, (max_ngram_size-1) * n_embed_per_ngram]

        hc_mult = self.backbone_config.hc_mult
        hidden_size = self.backbone_config.hidden_size

        gates = []
        for hc_idx in range(hc_mult):
            key = self.key_projs[hc_idx](embeddings)
            normed_key = self.norm1[hc_idx](key)
            query = hidden_states[:, :, hc_idx, :]
            normed_query = self.norm2[hc_idx](query)
            gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(hidden_size)
            # Sqrt-signed-magnitude squash before sigmoid (upstream).
            gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
            gate = gate.sigmoid().unsqueeze(-1)
            gates.append(gate)
        gates_t = torch.stack(gates, dim=2)  # [B, L, HC_MULT, 1]

        value = gates_t * self.value_proj(embeddings).unsqueeze(2)
        # → [B, L, HC_MULT, hidden_size]

        output = value + self.short_conv(value)
        return output
