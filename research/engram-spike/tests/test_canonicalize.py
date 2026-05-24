"""Canonicalization rule correctness + lookup-table build.

No network — uses a synthetic decode_fn so we don't pull HF tokenizers.
The Qwen3.6 / gemma4 real-tokenizer canonicalizer build is a separate
offline step (scripts/build_canonicalizer.py); these tests only pin the
logic.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from engram.canonicalize import (
    build_lookup_table,
    canonicalize,
    load_lookup_table,
    save_lookup_table,
)


def test_canonicalize_basic_normalization():
    # Accents stripped + lowercased
    assert canonicalize("Café") == "cafe"
    assert canonicalize("RÉSUMÉ") == "resume"
    # Whitespace collapsed
    assert canonicalize("  hello   world  ") == "hello world"
    # Tabs and newlines become single space
    assert canonicalize("a\tb\nc") == "a b c"
    # NFKC: fullwidth → halfwidth (e.g. Japanese fullwidth ASCII)
    assert canonicalize("ＡＢＣ") == "abc"


def test_canonicalize_preserves_pure_space():
    assert canonicalize(" ") == " "
    assert canonicalize("\t") == " "  # collapses to space, not empty
    assert canonicalize("\n\n") == " "


def test_canonicalize_empty_and_none_handled():
    assert canonicalize("") == ""


def test_lookup_table_collapses_case():
    """A toy 'tokenizer' with both 'Hello' and 'hello' should produce a
    smaller canonical vocab."""
    vocab = ["Hello", "hello", "HELLO", "world", "World"]

    def decode_fn(tid: int) -> str:
        return vocab[tid]

    lookup, num_canonical = build_lookup_table(
        vocab_size=len(vocab),
        decode_fn=decode_fn,
    )
    # hello/Hello/HELLO collapse to one slot; world/World collapse to another.
    assert num_canonical == 2
    # Same canonical id for the 3 case variants.
    assert lookup[0] == lookup[1] == lookup[2]
    # Same canonical id for the 2 case variants of 'world'.
    assert lookup[3] == lookup[4]
    # The two groups have distinct canonical ids.
    assert lookup[0] != lookup[3]


def test_lookup_table_collapses_accents_and_whitespace():
    vocab = ["café", "CAFE", "Cafe", "  cafe  ", "résumé", "resume"]

    def decode_fn(tid: int) -> str:
        return vocab[tid]

    lookup, num_canonical = build_lookup_table(
        vocab_size=len(vocab),
        decode_fn=decode_fn,
    )
    # All four 'cafe' variants → one slot; both 'resume' variants → another.
    assert num_canonical == 2
    assert lookup[0] == lookup[1] == lookup[2] == lookup[3]
    assert lookup[4] == lookup[5]


def test_lookup_table_special_tokens_exempted():
    """Special token ids must NOT be collapsed with regular tokens."""
    vocab = ["<pad>", "<pad>", "regular"]

    def decode_fn(tid: int) -> str:
        return vocab[tid]

    lookup, num_canonical = build_lookup_table(
        vocab_size=len(vocab),
        decode_fn=decode_fn,
        special_token_ids=[0, 1],  # both id=0 and id=1 are exempt
    )
    # Specials stay separate even though decoded text matches: 0, 1 each get own slot.
    assert lookup[0] != lookup[1]
    assert num_canonical == 3


def test_lookup_table_uses_raw_token_for_replacement_char():
    """Partial-byte tokens (decode → \\ufffd) should NOT all collapse into
    one slot — they should key on their raw BPE strings instead."""
    vocab = ["�", "�", "�"]
    raw_tokens = ["<0xC2>", "<0xC3>", "<0xC2>"]  # 0 and 2 share raw token

    def decode_fn(tid: int) -> str:
        return vocab[tid]

    def convert_fn(tid: int) -> str:
        return raw_tokens[tid]

    lookup, num_canonical = build_lookup_table(
        vocab_size=len(vocab),
        decode_fn=decode_fn,
        convert_id_to_token_fn=convert_fn,
    )
    # 0 and 2 → same slot (same raw); 1 → different slot.
    assert lookup[0] == lookup[2]
    assert lookup[0] != lookup[1]
    assert num_canonical == 2


def test_save_and_load_roundtrip(tmp_path: Path):
    lookup = np.array([0, 0, 1, 2, 2, 3], dtype=np.int64)
    num_canonical = 4
    meta = {"tokenizer": "test/toy", "trust_remote_code": False}
    out = tmp_path / "canon.json"
    save_lookup_table(lookup, num_canonical, out, meta=meta)

    loaded, n, m = load_lookup_table(out)
    np.testing.assert_array_equal(loaded, lookup)
    assert n == num_canonical
    assert m == meta

    # Schema is human-readable
    body = json.loads(out.read_text())
    assert body["schema_version"] == 1
    assert body["num_raw"] == 6
    assert body["num_canonical"] == 4
    assert body["reduction_pct"] == pytest.approx(100 * (1 - 4 / 6))


def test_reduction_percentage_is_reasonable_on_synthetic():
    """A 'tokenizer' with lots of duplicates after case-folding should show
    a meaningful reduction percentage."""
    vocab = [c for c in "AaBbCcDdEeFfGgHhIiJj"]  # 20 tokens, 10 distinct after lowercase

    def decode_fn(tid: int) -> str:
        return vocab[tid]

    lookup, num_canonical = build_lookup_table(
        vocab_size=len(vocab),
        decode_fn=decode_fn,
    )
    assert num_canonical == 10
    reduction = 1 - num_canonical / len(vocab)
    assert 0.45 < reduction < 0.55  # exactly 50%, allow float wiggle
