"""Hash uniformity sanity — on a random input drawn from a uniform token
distribution, no head should have any single slot getting catastrophically
more hits than expected.

This isn't a cryptographic test. The goal is to catch:
  - a broken multiplier (e.g. all-zero) that maps everything to slot 0
  - a wrong modulus (e.g. modulo 1 always returning 0)
  - off-by-one bugs in the shift logic

We tolerate fairly loose bounds. Below uses chi-square style intuition with
a generous K = 10x expected slot.
"""
import numpy as np
import pytest

from engram.hash import NgramHashMapping


@pytest.fixture
def big_hash_mapping():
    """Larger config to give meaningful uniformity stats."""
    return NgramHashMapping(
        engram_vocab_size=[256, 256],  # ~256 slots per head
        max_ngram_size=3,
        n_head_per_ngram=4,
        layer_ids=[0],
        tokenizer_vocab_size=1024,
        pad_id=0,
        seed=12345,
    )


def test_no_slot_grossly_overrepresented(big_hash_mapping):
    """On 10k random tokens, no slot should get more than 10× its expected hits."""
    rng = np.random.default_rng(0)
    # 1 batch, length 10000, uniform in [0, vocab)
    ids = rng.integers(low=0, high=1024, size=(1, 10000), dtype=np.int64)
    hashes = big_hash_mapping.hash(ids)[0]  # [1, 10000, num_heads_total]
    num_heads_total = big_hash_mapping.num_heads_total()
    flat_primes = big_hash_mapping.flat_head_primes(0)

    for head_idx in range(num_heads_total):
        head_hashes = hashes[0, :, head_idx]
        head_size = flat_primes[head_idx]
        expected_per_slot = ids.shape[1] / head_size
        bincount = np.bincount(head_hashes, minlength=head_size)
        max_count = bincount.max()
        ratio = max_count / max(1, expected_per_slot)
        assert ratio < 10.0, (
            f"Head {head_idx}: most-loaded slot got {max_count} hits "
            f"(expected ~{expected_per_slot:.1f}, ratio {ratio:.1f}× — possible hash collapse)"
        )


def test_at_least_half_the_slots_are_used(big_hash_mapping):
    """A working hash should hit at least ~half the available slots given
    10k inputs across ~256-slot heads. (Birthday paradox bounds the empty
    fraction below e^(-10000/256) ≈ 1e-17, so 'at least half used' is loose.)"""
    rng = np.random.default_rng(1)
    ids = rng.integers(low=0, high=1024, size=(1, 10000), dtype=np.int64)
    hashes = big_hash_mapping.hash(ids)[0]
    num_heads_total = big_hash_mapping.num_heads_total()
    flat_primes = big_hash_mapping.flat_head_primes(0)

    for head_idx in range(num_heads_total):
        head_hashes = hashes[0, :, head_idx]
        head_size = flat_primes[head_idx]
        used = np.unique(head_hashes).size
        assert used >= head_size // 2, (
            f"Head {head_idx}: only {used}/{head_size} slots used (≥{head_size//2} expected). "
            "Possible degeneracy in multiplier or modulus."
        )


def test_hash_indices_stay_within_bounds(big_hash_mapping):
    """Every emitted index must be < its head's prime modulus, to keep
    embedding lookups safe."""
    rng = np.random.default_rng(2)
    ids = rng.integers(low=0, high=1024, size=(4, 200), dtype=np.int64)
    hashes = big_hash_mapping.hash(ids)[0]  # [4, 200, num_heads_total]
    flat_primes = big_hash_mapping.flat_head_primes(0)
    num_heads_total = big_hash_mapping.num_heads_total()
    for head_idx in range(num_heads_total):
        head_max = int(hashes[..., head_idx].max())
        head_min = int(hashes[..., head_idx].min())
        head_size = flat_primes[head_idx]
        assert 0 <= head_min, f"Head {head_idx}: negative index {head_min}"
        assert head_max < head_size, (
            f"Head {head_idx}: index {head_max} ≥ head_size {head_size} (would OOB the embedding)"
        )
