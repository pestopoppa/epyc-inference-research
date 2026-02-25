#!/usr/bin/env python3
"""Q5 Step 2: Build SoftMatcha test index and compare exact vs soft matching.

Exports ~10K snippets from V3 corpus to plaintext, builds SoftMatcha HDF5 index
with GloVe embeddings, then compares exact (threshold=1.0) vs soft (threshold=0.5)
matching on quality gate test queries.

Usage:
  python3 scripts/benchmark/softmatcha_test_index.py [--snippets 10000]
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Force gensim data to RAID
os.environ["GENSIM_DATA_DIR"] = "/mnt/raid0/llm/cache/gensim-data"

SNIPPETS_DB = "/mnt/raid0/llm/cache/corpus/v3_sharded/snippets.db"
CORPUS_TXT = "/mnt/raid0/llm/tmp/softmatcha_test_corpus.txt"
INDEX_PATH = "/mnt/raid0/llm/tmp/softmatcha_test.h5"
OUTPUT_DIR = "/mnt/raid0/llm/claude/benchmarks/results/runs/q5_softmatcha"

# Test queries — same as quality gate prompts
TEST_QUERIES = [
    ("calculate loss predictions", "ML code"),
    ("async retry exponential backoff", "async patterns"),
    ("binary search tree insert delete", "data structures"),
    ("thread safe lru cache", "concurrency"),
    ("recursive descent parser", "parsing"),
    ("shortest path graph dijkstra", "graph algorithms"),
]

THRESHOLDS_TO_TEST = [1.0, 0.8, 0.7, 0.6, 0.5]


def export_snippets(db_path: str, output_path: str, n_snippets: int) -> int:
    """Export snippets from V3 corpus to plaintext file (one per line)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # Sample proportionally from each language
    counts = dict(conn.execute(
        "SELECT language, COUNT(*) FROM snippets GROUP BY language"
    ).fetchall())
    total = sum(counts.values())

    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for lang, count in counts.items():
            lang_samples = max(100, int(n_snippets * count / total))
            step = max(1, count // lang_samples)
            rows = conn.execute(
                "SELECT code FROM snippets WHERE language = ? AND ROWID % ? = 0 LIMIT ?",
                (lang, step, lang_samples),
            ).fetchall()

            for (code,) in rows:
                # SoftMatcha expects one "document" per line
                # Replace newlines with spaces to flatten code into single line
                flat = code.replace("\n", " ").replace("\r", " ").strip()
                if flat:
                    f.write(flat + "\n")
                    written += 1

            print(f"  Exported {len(rows)} {lang} snippets")

    conn.close()
    print(f"  Total: {written} snippets written to {output_path}")
    return written


def build_index(corpus_path: str, index_path: str) -> None:
    """Build SoftMatcha HDF5 inverted file index using GloVe embeddings."""
    import h5py
    from softmatcha.tokenizers import get_tokenizer
    from softmatcha.struct import IndexInvertedFileCollection

    print(f"\n  Building index at {index_path}...")
    print(f"  Backend: gensim, Model: glove-wiki-gigaword-300")

    # Get the gensim tokenizer (which internally uses Moses)
    tokenizer_cls = get_tokenizer("gensim")
    tokenizer = tokenizer_cls.build(
        tokenizer_cls.Config(name_or_path="glove-wiki-gigaword-300")
    )

    t0 = time.monotonic()
    IndexInvertedFileCollection.build(
        index_path,
        [corpus_path],
        tokenizer,
        num_workers=8,
        buffer_size=5000,
    )
    elapsed = time.monotonic() - t0
    print(f"  Index built in {elapsed:.1f}s")

    # Print index stats
    with h5py.File(index_path, "r") as f:
        idx = f["indexes"]["0"]
        n_tokens = len(idx["tokens"])
        n_vocab = len(idx["vocabulary"])
        n_lines = len(idx["line_offsets"])
        print(f"  Tokens: {n_tokens:,}")
        print(f"  Vocabulary: {n_vocab:,}")
        print(f"  Lines: {n_lines:,}")

    index_size = os.path.getsize(index_path)
    corpus_size = os.path.getsize(corpus_path)
    print(f"  Index size: {index_size / 1024 / 1024:.1f} MB")
    print(f"  Corpus size: {corpus_size / 1024 / 1024:.1f} MB")
    print(f"  Index/corpus ratio: {index_size / corpus_size:.1f}x")


def run_queries(index_path: str, corpus_path: str) -> list[dict]:
    """Run test queries at different thresholds and compare results."""
    import h5py
    import numpy as np
    from softmatcha.tokenizers import get_tokenizer
    from softmatcha.embeddings import get_embedding
    from softmatcha.struct import IndexInvertedFile, Pattern
    from softmatcha.search.index_inverted import SearchIndexInvertedFile

    # Load components
    print(f"\n  Loading index and embeddings...")
    tokenizer_cls = get_tokenizer("gensim")
    tokenizer = tokenizer_cls.build(
        tokenizer_cls.Config(name_or_path="glove-wiki-gigaword-300")
    )

    embedding_cls = get_embedding("gensim")
    embedding = embedding_cls.build(
        embedding_cls.Config(name_or_path="glove-wiki-gigaword-300")
    )

    index_file = h5py.File(index_path, "r")
    index = IndexInvertedFile.load(index_file["indexes"]["0"])

    searcher = SearchIndexInvertedFile(index, tokenizer, embedding)

    # Read corpus for context extraction
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_lines = f.readlines()

    results = []

    for query_text, query_category in TEST_QUERIES:
        print(f"\n  Query: \"{query_text}\" [{query_category}]")

        # Tokenize and encode query
        query_tokens_str = tokenizer.tokenize(query_text)
        query_token_ids = tokenizer.encode(query_tokens_str)
        query_embeddings = embedding(query_token_ids)

        # Check which tokens are OOV
        oov_tokens = [t for t, tid in zip(query_tokens_str, query_token_ids)
                      if tid == tokenizer.unk_idx]
        if oov_tokens:
            print(f"    OOV tokens: {oov_tokens}")
        else:
            print(f"    All tokens in vocabulary")

        query_result = {
            "query": query_text,
            "category": query_category,
            "tokens": query_tokens_str,
            "oov_tokens": oov_tokens,
            "thresholds": {},
        }

        for threshold in THRESHOLDS_TO_TEST:
            # Build pattern with uniform threshold
            thresholds = [threshold] * len(query_token_ids)
            pattern = Pattern.build(query_token_ids, query_embeddings, thresholds)

            t0 = time.monotonic()
            matches = list(searcher.search(pattern))
            elapsed_ms = (time.monotonic() - t0) * 1000

            n_matches = len(matches)
            threshold_label = f"{threshold:.1f}"

            # Get sample matches with context
            sample_matches = []
            for m in matches[:5]:
                line_num = index.get_line_number(m.begin)
                if 0 <= line_num < len(corpus_lines):
                    context = corpus_lines[line_num].strip()
                    # Truncate long lines
                    if len(context) > 200:
                        # Find the match position and show surrounding context
                        context = context[:200] + "..."

                    # Get matched tokens
                    matched_tokens = tokenizer.decode(m.tokens)
                    match_scores = m.scores.tolist()

                    sample_matches.append({
                        "line": line_num,
                        "matched_tokens": matched_tokens,
                        "scores": [round(s, 3) for s in match_scores],
                        "avg_score": round(float(np.mean(m.scores)), 3),
                        "context_preview": context[:150],
                    })

            query_result["thresholds"][threshold_label] = {
                "n_matches": n_matches,
                "latency_ms": round(elapsed_ms, 2),
                "sample_matches": sample_matches,
            }

            # Print summary
            tag = "EXACT" if threshold == 1.0 else f"SOFT({threshold})"
            print(f"    {tag:12s}: {n_matches:>6} matches, {elapsed_ms:>7.1f}ms")
            if n_matches > 0 and sample_matches:
                best = sample_matches[0]
                print(f"      Best match: {best['matched_tokens']}")
                print(f"      Scores: {best['scores']}")
                print(f"      Context: {best['context_preview'][:100]}")

        results.append(query_result)

    index_file.close()
    return results


def print_summary(results: list[dict]):
    """Print comparison summary table."""
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY: Exact vs Soft Matching")
    print(f"{'=' * 80}")

    # Table header
    threshold_labels = [f"{t:.1f}" for t in THRESHOLDS_TO_TEST]
    header = f"  {'Query':<35}"
    for t in threshold_labels:
        label = "EXACT" if t == "1.0" else f"t={t}"
        header += f" {label:>10}"
    print(header)
    print(f"  {'-' * (35 + 11 * len(threshold_labels))}")

    for r in results:
        row = f"  {r['query']:<35}"
        for t in threshold_labels:
            n = r['thresholds'][t]['n_matches']
            row += f" {n:>10,}"
        oov_tag = f" [OOV: {','.join(r['oov_tokens'])}]" if r['oov_tokens'] else ""
        print(row + oov_tag)

    # Latency table
    print(f"\n  Latency (ms):")
    header = f"  {'Query':<35}"
    for t in threshold_labels:
        label = "EXACT" if t == "1.0" else f"t={t}"
        header += f" {label:>10}"
    print(header)
    print(f"  {'-' * (35 + 11 * len(threshold_labels))}")

    for r in results:
        row = f"  {r['query']:<35}"
        for t in threshold_labels:
            ms = r['thresholds'][t]['latency_ms']
            row += f" {ms:>9.1f}"
        print(row)

    # Soft matching value assessment
    print(f"\n  QUALITATIVE ASSESSMENT:")
    for r in results:
        exact_n = r['thresholds']['1.0']['n_matches']
        soft_n = r['thresholds']['0.5']['n_matches']
        ratio = soft_n / exact_n if exact_n > 0 else float('inf') if soft_n > 0 else 0
        if exact_n == 0 and soft_n == 0:
            verdict = "NO MATCHES — query too specific or all OOV"
        elif exact_n == 0 and soft_n > 0:
            verdict = f"SOFT-ONLY: {soft_n} fuzzy matches where exact finds 0"
        elif ratio > 10:
            verdict = f"SOFT MUCH WIDER: {soft_n:,} vs {exact_n:,} ({ratio:.0f}x more)"
        elif ratio > 2:
            verdict = f"SOFT WIDER: {soft_n:,} vs {exact_n:,} ({ratio:.1f}x more)"
        elif ratio > 1.1:
            verdict = f"SOFT SLIGHTLY WIDER: {soft_n:,} vs {exact_n:,} ({ratio:.1f}x more)"
        else:
            verdict = f"NO DIFFERENCE: {exact_n:,} matches at both thresholds"
        print(f"  {r['query']:<35}: {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Q5 Step 2: SoftMatcha test index")
    parser.add_argument("--snippets", type=int, default=10000,
                        help="Number of snippets to export")
    parser.add_argument("--db", default=SNIPPETS_DB, help="Path to snippets.db")
    parser.add_argument("--corpus", default=CORPUS_TXT, help="Output corpus path")
    parser.add_argument("--index", default=INDEX_PATH, help="Output index path")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export if corpus already exists")
    parser.add_argument("--skip-build", action="store_true",
                        help="Skip index build if index already exists")
    args = parser.parse_args()

    print("Q5 Step 2: SoftMatcha Test Index + Query Comparison")
    print("=" * 60)

    # 1. Export snippets
    if args.skip_export and os.path.exists(args.corpus):
        print(f"\n[1/3] Using existing corpus: {args.corpus}")
    else:
        print(f"\n[1/3] Exporting {args.snippets} snippets to {args.corpus}...")
        export_snippets(args.db, args.corpus, args.snippets)

    # 2. Build index
    if args.skip_build and os.path.exists(args.index):
        print(f"\n[2/3] Using existing index: {args.index}")
    else:
        print(f"\n[2/3] Building SoftMatcha index...")
        # Remove old index if exists
        if os.path.exists(args.index):
            os.remove(args.index)
        build_index(args.corpus, args.index)

    # 3. Run queries
    print(f"\n[3/3] Running test queries...")
    results = run_queries(args.index, args.corpus)

    # Print summary
    print_summary(results)

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
