#!/usr/bin/env python3
"""Build a canonicalization lookup table for a given HF tokenizer.

Usage:
    python3 scripts/build_canonicalizer.py \\
        --tokenizer Qwen/Qwen3-30B-A3B-Instruct \\
        --output canonicalizers/qwen3-30b-a3b.json

Reports vocab reduction percentage and prints a sample of collapsed groups
for sanity checking.

Network/disk required (loads the tokenizer). Not run by the test suite —
this is the offline data-prep step that produces an artifact unit tests
can later load.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

# Make the local package importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engram.canonicalize import build_lookup_table, save_lookup_table


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, help="HF tokenizer name or path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Pass trust_remote_code=True to AutoTokenizer (needed for some custom tokenizers)",
    )
    parser.add_argument(
        "--sample-collisions", type=int, default=10,
        help="Print this many sample slots where ≥2 raw tokens collapsed (for sanity)",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers not installed. `pip install transformers`", file=sys.stderr)
        return 1

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=args.trust_remote_code)
    vocab_size = len(tokenizer)
    print(f"  raw vocab size: {vocab_size}")

    special_ids = set(tokenizer.all_special_ids)
    print(f"  special tokens exempted: {len(special_ids)}")

    def decode_fn(tid: int) -> str:
        return tokenizer.decode([tid], skip_special_tokens=False)

    def convert_fn(tid: int) -> str:
        return tokenizer.convert_ids_to_tokens(tid)

    lookup, num_canonical = build_lookup_table(
        vocab_size=vocab_size,
        decode_fn=decode_fn,
        convert_id_to_token_fn=convert_fn,
        special_token_ids=special_ids,
    )

    reduction_pct = 100.0 * (1 - num_canonical / vocab_size)
    print(f"  canonical vocab size: {num_canonical}")
    print(f"  reduction: {reduction_pct:.2f}%")

    if args.sample_collisions > 0:
        groups = defaultdict(list)
        for raw_id, canon_id in enumerate(lookup):
            groups[int(canon_id)].append(raw_id)
        collided = [(cid, ids) for cid, ids in groups.items() if len(ids) >= 2]
        collided.sort(key=lambda x: -len(x[1]))
        print(f"\n  Top-{args.sample_collisions} collision groups (canon_id → raw_ids → decoded text):")
        for cid, ids in collided[: args.sample_collisions]:
            samples = [(raw, decode_fn(raw)) for raw in ids[:4]]
            print(f"    canon {cid:>6} ({len(ids):>3} raws): " + ", ".join(repr(s) for _, s in samples))

    out_path = Path(args.output)
    save_lookup_table(
        lookup, num_canonical, out_path,
        meta={
            "tokenizer": args.tokenizer,
            "trust_remote_code": args.trust_remote_code,
        },
    )
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
