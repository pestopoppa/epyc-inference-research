"""Vocab canonicalization map P.

The paper (§3) collapses raw tokenizer ids that "decode to the same
canonical text" into a single id before hashing. Quoting their result:
~23% vocab reduction for DeepSeek-V3's 128k vocab. The exact P map is not
released — they describe the canonicalizer in prose only.

What we know from the upstream demo's `CompressedTokenizer.normalizer`:
    NFKC → NFD → StripAccents → Lowercase → whitespace-collapse → strip

Replicating that recipe in pure Python (no HF tokenizers dependency for
the canonicalization step itself) is straightforward, and the result is
deterministic + portable across tokenizers. The mapping function is:

    canonicalize(token_text) -> normalized_string

then `lookup_table[raw_id] = unique_id_of(canonicalize(decode(raw_id)))`.

Two complications worth noting up front:

  1. Special tokens (<|begin_of_text|>, <pad>, etc.) decode to readable
     strings that would collapse with regular text — we exempt them by
     keying on the tokenizer's `convert_ids_to_tokens` for ids inside the
     known special-tokens set.

  2. Bytes that don't decode cleanly (e.g. UTF-8 fragments inside BPE
     merges) appear as `"�"` in `decode(...)`; we key on the raw BPE
     token string for those, not on the malformed decode.

`build_lookup_table(tokenizer)` returns (lookup, num_canonical_tokens),
suitable for use by NgramHashMapping. For unit-test isolation, this module
also accepts a *callable* tokenizer (`Callable[[int], str]`) so we can
test the canonicalization logic without an HF dependency.
"""
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

# Sentinel used to detect whitespace-collapsed-to-empty cases. Same trick as
# the upstream HF normalizer chain, just inlined.
_SENTINEL = ""

# Unicode "replacement character" — appears in tokenizer.decode() when a
# token id corresponds to a partial UTF-8 sequence (typical in BPE merges).
_REPLACEMENT = "�"


def canonicalize(text: str) -> str:
    """Apply the upstream normalizer chain in pure Python.

    Steps (in order):
      1. NFKC: compatibility decomposition + canonical composition
      2. NFD: canonical decomposition (separates accents from base chars)
      3. Strip accents: drop combining marks (category Mn)
      4. Lowercase
      5. Whitespace collapse: any run of [ \\t\\r\\n] → single space
      6. Strip leading/trailing whitespace
      7. Empty-string preservation: if step 5+6 reduced everything to "",
         restore a single space (so two distinct whitespace tokens still
         collide into one canonical "space" token rather than into the
         empty string with other no-content tokens)
    """
    if not text:
        return text

    # Step 1: NFKC
    s = unicodedata.normalize("NFKC", text)
    # Step 2: NFD
    s = unicodedata.normalize("NFD", s)
    # Step 3: strip accents (drop combining marks)
    s = "".join(c for c in s if not unicodedata.combining(c))
    # Step 4: lowercase
    s = s.lower()
    # Step 5: whitespace collapse
    s = re.sub(r"[ \t\r\n]+", " ", s)
    # Step 6/7: empty-after-strip preservation
    if s == " ":
        return " "  # explicit single-space canonical
    s = s.strip()
    return s


def build_lookup_table(
    vocab_size: int,
    decode_fn: Callable[[int], str],
    convert_id_to_token_fn: Optional[Callable[[int], str]] = None,
    special_token_ids: Optional[Iterable[int]] = None,
) -> Tuple[np.ndarray, int]:
    """Build the raw_id → canonical_id lookup table.

    Args:
        vocab_size: number of raw tokenizer ids to scan
        decode_fn: maps int id → decoded text (e.g. tokenizer.decode([id]))
        convert_id_to_token_fn: optional, maps int id → raw BPE token string
            (used when decode_fn returns � for partial-byte tokens).
            If None, ids decoding to the replacement char share one slot.
        special_token_ids: optional iterable of ids to exempt from
            canonicalization (each kept in its own canonical slot).

    Returns:
        lookup: int64 np.ndarray of shape [vocab_size], where lookup[i] is
            the canonical id for raw id i
        num_canonical: number of distinct canonical ids in the output range
    """
    specials = set(special_token_ids) if special_token_ids is not None else set()
    key_to_new: Dict[str, int] = {}
    next_new = 0
    lookup = np.empty(vocab_size, dtype=np.int64)

    for tid in range(vocab_size):
        if tid in specials:
            # Each special gets its own slot, keyed by a sentinel-prefixed id.
            key = f"__special__{tid}"
        else:
            try:
                text = decode_fn(tid)
            except Exception:
                text = ""
            if _REPLACEMENT in text and convert_id_to_token_fn is not None:
                # Decode failed (partial byte); use the raw BPE token string instead.
                try:
                    key = "__raw__" + convert_id_to_token_fn(tid)
                except Exception:
                    key = "__raw__" + str(tid)
            else:
                normed = canonicalize(text)
                key = normed if normed else text  # preserve uniqueness for unusual empties

        nid = key_to_new.get(key)
        if nid is None:
            nid = next_new
            key_to_new[key] = nid
            next_new += 1
        lookup[tid] = nid

    return lookup, next_new


def save_lookup_table(lookup: np.ndarray, num_canonical: int, path: Path, meta: Optional[Dict[str, Any]] = None) -> None:
    """Persist a canonicalization map to disk as JSON.

    Format:
        {
          "schema_version": 1,
          "num_raw": <int>,
          "num_canonical": <int>,
          "reduction_pct": <float>,
          "lookup": [<int>, <int>, ...],  # length num_raw
          "meta": {...}                    # caller-provided context
        }

    Layout chosen for human-readability + small enough to git-commit even
    for 128k-vocab tokenizers (~1 MB).
    """
    obj = {
        "schema_version": 1,
        "num_raw": int(lookup.shape[0]),
        "num_canonical": int(num_canonical),
        "reduction_pct": 100.0 * (1 - num_canonical / float(lookup.shape[0])),
        "lookup": lookup.astype(int).tolist(),
        "meta": meta or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def load_lookup_table(path: Path) -> Tuple[np.ndarray, int, Dict[str, Any]]:
    """Inverse of save_lookup_table."""
    obj = json.loads(Path(path).read_text())
    assert obj.get("schema_version", 1) == 1, "Unknown lookup-table schema"
    lookup = np.asarray(obj["lookup"], dtype=np.int64)
    return lookup, int(obj["num_canonical"]), obj.get("meta", {})
