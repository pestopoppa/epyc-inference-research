"""Multiplicative-XOR n-gram hashing.

Per the paper (arXiv:2601.07372 §3): for each n-gram order n and each of K
hash heads, an index is computed as

    z_{t,n,k} = ((Σ XOR_j t_j · m_{n,k,j}) mod P_{n,k})

where:
  - t_j is the (canonicalized) token id at position t-n+1+j,
  - m_{n,k,j} are odd 64-bit multipliers drawn from a per-layer RNG,
  - P_{n,k} is a distinct prime ≥ engram_vocab_size[n-2] per (n, k).

Collisions within a single head are unresolved by design; the K-head
ensemble + content-aware gate is the mitigation.

This module is numpy-only. The torch wrapper just converts the int64 output
to a torch.LongTensor; the math is identical to upstream `engram_demo_v1.py`.
"""
from typing import Dict, List, Optional, Set

import numpy as np
from sympy import isprime


def find_next_prime(start: int, seen_primes: Set[int]) -> int:
    """Return the smallest prime > start that is not already in seen_primes."""
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen_primes:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Deterministic n-gram → slot-index hasher.

    The mapping is fully determined by (engram_vocab_size, max_ngram_size,
    n_head_per_ngram, layer_ids, tokenizer_vocab_size, pad_id, seed). Two
    instances built with the same args produce bit-identical outputs.

    The numpy-only impl is the reference; do not introduce a torch-native
    fast path without a parity test that asserts identical outputs.
    """

    PRIME_1 = 10007  # per-layer RNG seed offset, from upstream

    def __init__(
        self,
        engram_vocab_size: List[int],
        max_ngram_size: int,
        n_head_per_ngram: int,
        layer_ids: List[int],
        tokenizer_vocab_size: int,
        pad_id: int,
        seed: int = 0,
    ):
        assert len(engram_vocab_size) == max_ngram_size - 1, (
            f"engram_vocab_size needs one entry per n-gram order n in [2, {max_ngram_size}]; "
            f"got {len(engram_vocab_size)} entries for max_ngram_size={max_ngram_size}"
        )
        self.engram_vocab_size = engram_vocab_size
        self.max_ngram_size = max_ngram_size
        self.n_head_per_ngram = n_head_per_ngram
        self.layer_ids = list(layer_ids)
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.pad_id = int(pad_id)
        self.seed = int(seed)

        # Bound the multiplier so that (multiplier * vocab_size) fits in int64.
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // max(1, tokenizer_vocab_size))
        half_bound = max(1, M_max // 2)

        self.layer_multipliers: Dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            rng = np.random.default_rng(self.seed + self.PRIME_1 * int(layer_id))
            r = rng.integers(low=0, high=half_bound, size=(max_ngram_size,), dtype=np.int64)
            # Force odd → coprime to 2^64 (improves mixing).
            self.layer_multipliers[layer_id] = (r * 2 + 1).astype(np.int64)

        self.vocab_size_across_layers = self._calculate_head_primes()

    def _calculate_head_primes(self) -> Dict[int, List[List[int]]]:
        """Per (layer, n, head), pick distinct primes ≥ engram_vocab_size[n-2].

        Globally distinct (across all heads of all layers/orders) so that
        flattened slot ids never collide between heads.
        """
        seen_primes: Set[int] = set()
        out: Dict[int, List[List[int]]] = {}
        for layer_id in self.layer_ids:
            per_ngram: List[List[int]] = []
            for ngram in range(2, self.max_ngram_size + 1):
                target = self.engram_vocab_size[ngram - 2]
                current_start = target - 1
                primes_for_this_ngram: List[int] = []
                for _ in range(self.n_head_per_ngram):
                    p = find_next_prime(current_start, seen_primes)
                    seen_primes.add(p)
                    primes_for_this_ngram.append(p)
                    current_start = p
                per_ngram.append(primes_for_this_ngram)
            out[layer_id] = per_ngram
        return out

    def num_heads_total(self) -> int:
        """Total hash heads across all n-gram orders, for one layer."""
        return (self.max_ngram_size - 1) * self.n_head_per_ngram

    def flat_head_primes(self, layer_id: int) -> List[int]:
        """Per-head prime moduli for one layer, in (n, k) order, flattened."""
        per_ngram = self.vocab_size_across_layers[layer_id]
        return [p for heads in per_ngram for p in heads]

    def _get_ngram_hashes(self, input_ids: np.ndarray, layer_id: int) -> np.ndarray:
        """Compute hashes for one layer.

        Args:
            input_ids: int array of shape [B, T]
            layer_id: which layer's multipliers to use

        Returns:
            int64 array of shape [B, T, num_heads_total], where the last axis
            is ordered as: (n=2, k=0..K-1, n=3, k=0..K-1, ..., n=N, k=0..K-1).
        """
        x = np.asarray(input_ids, dtype=np.int64)
        if x.ndim != 2:
            raise ValueError(f"input_ids must be 2D [B, T]; got shape {x.shape}")
        B, T = x.shape

        multipliers = self.layer_multipliers[layer_id]

        def shift_k(k: int) -> np.ndarray:
            if k == 0:
                return x
            shifted = np.pad(
                x, ((0, 0), (k, 0)), mode="constant", constant_values=self.pad_id
            )[:, :T]
            return shifted

        base_shifts = [shift_k(k) for k in range(self.max_ngram_size)]

        all_hashes: List[np.ndarray] = []
        for n in range(2, self.max_ngram_size + 1):
            n_gram_index = n - 2
            tokens = base_shifts[:n]
            mix = tokens[0] * multipliers[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, tokens[k] * multipliers[k])
            head_primes = self.vocab_size_across_layers[layer_id][n_gram_index]
            for j in range(self.n_head_per_ngram):
                mod = int(head_primes[j])
                all_hashes.append((mix % mod).astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)

    def hash(self, input_ids: np.ndarray, layer_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """Compute hashes for one or more layers.

        Args:
            input_ids: int array [B, T] (already canonicalized — see tokenizer.py)
            layer_ids: subset of self.layer_ids; defaults to all

        Returns: dict {layer_id: [B, T, num_heads_total] int64}
        """
        if layer_ids is None:
            layer_ids = self.layer_ids
        return {lid: self._get_ngram_hashes(input_ids, lid) for lid in layer_ids}
