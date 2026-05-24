"""Hash determinism + reproducibility properties.

The paper relies on the hash being a pure function of (canonical token ids,
config seed) — same inputs → same buckets across runs, across processes,
and (critically for the offload story) speculatively prefetchable from
upcoming tokens without needing model output.
"""
import numpy as np
import torch

from engram.hash import NgramHashMapping


def test_hash_is_deterministic_across_calls(tiny_hash_mapping, tiny_input_ids):
    ids_np = tiny_input_ids.numpy()
    h1 = tiny_hash_mapping.hash(ids_np)
    h2 = tiny_hash_mapping.hash(ids_np)
    for lid in h1:
        np.testing.assert_array_equal(h1[lid], h2[lid])


def test_hash_is_deterministic_across_instances(tiny_engram_config, tiny_backbone_config, tiny_input_ids):
    """Two NgramHashMapping instances built with the same args must produce
    bit-identical hashes (otherwise a saved Engram can't be reloaded)."""
    m1 = NgramHashMapping(
        engram_vocab_size=tiny_engram_config.engram_vocab_size,
        max_ngram_size=tiny_engram_config.max_ngram_size,
        n_head_per_ngram=tiny_engram_config.n_head_per_ngram,
        layer_ids=tiny_engram_config.layer_ids,
        tokenizer_vocab_size=tiny_backbone_config.vocab_size,
        pad_id=tiny_engram_config.pad_id,
        seed=tiny_engram_config.seed,
    )
    m2 = NgramHashMapping(
        engram_vocab_size=tiny_engram_config.engram_vocab_size,
        max_ngram_size=tiny_engram_config.max_ngram_size,
        n_head_per_ngram=tiny_engram_config.n_head_per_ngram,
        layer_ids=tiny_engram_config.layer_ids,
        tokenizer_vocab_size=tiny_backbone_config.vocab_size,
        pad_id=tiny_engram_config.pad_id,
        seed=tiny_engram_config.seed,
    )

    ids = tiny_input_ids.numpy()
    h1 = m1.hash(ids)
    h2 = m2.hash(ids)

    # Same multipliers
    for lid in tiny_engram_config.layer_ids:
        np.testing.assert_array_equal(m1.layer_multipliers[lid], m2.layer_multipliers[lid])
        np.testing.assert_array_equal(h1[lid], h2[lid])

    # Same prime moduli (the prime search is deterministic for given starts +
    # seen-set ordering).
    assert m1.vocab_size_across_layers == m2.vocab_size_across_layers


def test_distinct_seeds_yield_distinct_multipliers(tiny_engram_config, tiny_backbone_config):
    kwargs = dict(
        engram_vocab_size=tiny_engram_config.engram_vocab_size,
        max_ngram_size=tiny_engram_config.max_ngram_size,
        n_head_per_ngram=tiny_engram_config.n_head_per_ngram,
        layer_ids=tiny_engram_config.layer_ids,
        tokenizer_vocab_size=tiny_backbone_config.vocab_size,
        pad_id=tiny_engram_config.pad_id,
    )
    m_a = NgramHashMapping(seed=0, **kwargs)
    m_b = NgramHashMapping(seed=1, **kwargs)

    # At least one layer's multiplier should differ.
    any_diff = any(
        not np.array_equal(m_a.layer_multipliers[lid], m_b.layer_multipliers[lid])
        for lid in tiny_engram_config.layer_ids
    )
    assert any_diff, "Different seeds produced identical multipliers — RNG plumbing broken"


def test_distinct_layer_ids_yield_distinct_multipliers(tiny_engram_config, tiny_backbone_config):
    m = NgramHashMapping(
        engram_vocab_size=tiny_engram_config.engram_vocab_size,
        max_ngram_size=tiny_engram_config.max_ngram_size,
        n_head_per_ngram=tiny_engram_config.n_head_per_ngram,
        layer_ids=[0, 1, 5],
        tokenizer_vocab_size=tiny_backbone_config.vocab_size,
        pad_id=tiny_engram_config.pad_id,
        seed=0,
    )
    # All three layers should have distinct multiplier vectors.
    mults = [m.layer_multipliers[lid] for lid in [0, 1, 5]]
    for i in range(len(mults)):
        for j in range(i + 1, len(mults)):
            assert not np.array_equal(mults[i], mults[j]), (
                f"Layers {i} and {j} share multipliers — per-layer seeding broken"
            )


def test_head_primes_are_globally_distinct(tiny_hash_mapping):
    """The paper's design depends on each (layer, n, head) prime being
    globally distinct so that slot ids across heads never alias."""
    all_primes = []
    for lid, per_ngram in tiny_hash_mapping.vocab_size_across_layers.items():
        for heads in per_ngram:
            all_primes.extend(heads)
    assert len(all_primes) == len(set(all_primes)), (
        f"Prime moduli are not globally distinct: {len(all_primes)} total, "
        f"{len(set(all_primes))} unique"
    )


def test_pad_token_does_not_change_within_window_hash(tiny_hash_mapping):
    """Sanity on padding behavior: a sequence shorter than max_ngram_size has
    its earliest positions hashed against pad_id-filled shifts, so two batches
    that differ only in elements past the n-gram window should still produce
    identical hashes at the early positions."""
    # Two batches that share their first 4 tokens but differ at position 4+.
    a = np.array([[1, 2, 3, 4, 99, 99, 99, 99]], dtype=np.int64)
    b = np.array([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=np.int64)
    h_a = tiny_hash_mapping.hash(a)
    h_b = tiny_hash_mapping.hash(b)
    for lid in h_a:
        np.testing.assert_array_equal(
            h_a[lid][:, :4, :], h_b[lid][:, :4, :],
            err_msg=f"Layer {lid}: first 4 positions should match between batches that share their first 4 tokens",
        )
