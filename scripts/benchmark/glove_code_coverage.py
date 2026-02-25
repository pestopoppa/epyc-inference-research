#!/usr/bin/env python3
"""Q5: Measure GloVe + FastText vocabulary coverage on V3 code corpus.

Samples ~1000 snippets from snippets.db, tokenizes with sacremoses Moses,
checks each token against GloVe and FastText vocabularies.

Outputs:
  - Coverage % for each embedding model
  - Per-language breakdown (Python vs Rust vs C++)
  - Token category breakdown (keywords, identifiers, operators, types, code-specific)
  - Top-50 most frequent OOV tokens
  - Top-50 most frequent covered tokens

Usage:
  python3 scripts/benchmark/glove_code_coverage.py [--samples 1000] [--skip-fasttext]
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Force gensim data to RAID (NOT root FS)
os.environ["GENSIM_DATA_DIR"] = "/mnt/raid0/llm/cache/gensim-data"
os.environ["FASTTEXT_HOME"] = "/mnt/raid0/llm/cache/fasttext"

SNIPPETS_DB = "/mnt/raid0/llm/cache/corpus/v3_sharded/snippets.db"
OUTPUT_DIR = "/mnt/raid0/llm/claude/benchmarks/results/runs"

# Token categories for analysis
PYTHON_KEYWORDS = {
    "def", "class", "if", "elif", "else", "for", "while", "return", "import",
    "from", "as", "with", "try", "except", "finally", "raise", "yield", "lambda",
    "pass", "break", "continue", "and", "or", "not", "in", "is", "None", "True",
    "False", "assert", "del", "global", "nonlocal", "async", "await",
}
RUST_KEYWORDS = {
    "fn", "let", "mut", "pub", "struct", "enum", "impl", "trait", "use", "mod",
    "crate", "self", "super", "match", "if", "else", "for", "while", "loop",
    "return", "break", "continue", "where", "type", "const", "static", "unsafe",
    "async", "await", "move", "ref", "dyn", "box",
}
CPP_KEYWORDS = {
    "int", "void", "char", "float", "double", "bool", "auto", "const", "static",
    "class", "struct", "enum", "union", "namespace", "using", "template",
    "typename", "virtual", "override", "public", "private", "protected",
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "throw", "try", "catch", "new", "delete", "nullptr", "sizeof",
    "typedef", "inline", "extern", "volatile", "constexpr", "noexcept",
}
ALL_KEYWORDS = PYTHON_KEYWORDS | RUST_KEYWORDS | CPP_KEYWORDS

COMMON_IDENTIFIERS = {
    "data", "model", "result", "error", "value", "name", "key", "index", "size",
    "count", "list", "item", "node", "path", "file", "line", "text", "code",
    "type", "args", "kwargs", "config", "output", "input", "state", "cache",
    "buffer", "queue", "stack", "tree", "graph", "map", "set", "array",
    "string", "message", "response", "request", "client", "server", "handler",
    "parser", "builder", "factory", "manager", "context", "logger", "reader",
    "writer", "stream", "token", "hash", "length", "offset", "limit",
}

TYPE_NAMES = {
    "int", "str", "float", "bool", "list", "dict", "tuple", "set", "bytes",
    "Vec", "String", "Option", "Result", "Box", "Arc", "Rc", "HashMap",
    "HashSet", "Iterator", "i32", "i64", "u32", "u64", "f32", "f64", "usize",
    "size_t", "uint32_t", "int64_t", "string", "vector", "map", "pair",
}

CODE_SPECIFIC = {
    "self", "None", "True", "False", "nullptr", "impl", "pub", "async", "await",
    "fn", "mut", "println", "print", "len", "append", "extend", "insert",
    "remove", "pop", "push", "iter", "map", "filter", "reduce", "zip",
    "enumerate", "range", "format", "join", "split", "strip", "replace",
    "unwrap", "expect", "clone", "collect", "into", "from", "new",
    "sizeof", "static_cast", "dynamic_cast", "reinterpret_cast",
    "std", "fmt", "io", "os", "sys", "re", "json", "math", "collections",
}

OPERATORS_PUNCT = set(".,;:()[]{}=+-*/<>!&|^~%@#?\\\"'`")


def categorize_token(token: str) -> str:
    """Categorize a token into one of the analysis categories."""
    if len(token) == 1 and token in OPERATORS_PUNCT:
        return "operators_punct"
    if token in ALL_KEYWORDS:
        return "keywords"
    if token in TYPE_NAMES:
        return "type_names"
    if token in CODE_SPECIFIC:
        return "code_specific"
    if token in COMMON_IDENTIFIERS:
        return "common_identifiers"
    # Heuristic: camelCase or snake_case identifiers
    if "_" in token or (any(c.isupper() for c in token[1:]) and any(c.islower() for c in token)):
        return "compound_identifiers"
    if token.isdigit() or re.match(r"^0[xX][0-9a-fA-F]+$", token):
        return "numeric_literals"
    return "other"


def sample_snippets(db_path: str, n_samples: int, seed: int = 42) -> list[dict]:
    """Sample snippets from each language proportionally."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # Get language counts
    counts = dict(conn.execute(
        "SELECT language, COUNT(*) FROM snippets GROUP BY language"
    ).fetchall())
    total = sum(counts.values())

    snippets = []
    for lang, count in counts.items():
        lang_samples = max(10, int(n_samples * count / total))
        # Deterministic sampling via ROWID modular arithmetic
        step = max(1, count // lang_samples)
        rows = conn.execute(
            "SELECT id, code, language FROM snippets WHERE language = ? AND ROWID % ? = 0 LIMIT ?",
            (lang, step, lang_samples),
        ).fetchall()
        snippets.extend({"id": r["id"], "code": r["code"], "language": r["language"]} for r in rows)
        print(f"  Sampled {len(rows)} {lang} snippets (of {count:,})")

    conn.close()
    return snippets


def tokenize_code(code: str, tokenizer) -> list[str]:
    """Tokenize code using sacremoses Moses tokenizer."""
    # Moses tokenizer works line-by-line
    tokens = []
    for line in code.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        line_tokens = tokenizer.tokenize(line, return_str=False)
        tokens.extend(line_tokens)
    return tokens


def load_glove(model_name: str = "glove-wiki-gigaword-300"):
    """Load GloVe vectors via gensim."""
    import gensim.downloader as api

    print(f"\nLoading {model_name}...")
    print(f"  (gensim data dir: {os.environ.get('GENSIM_DATA_DIR', 'default')})")
    model = api.load(model_name)
    print(f"  Loaded: {len(model.key_to_index):,} vocabulary entries")
    return model


def load_fasttext(model_path: str = None):
    """Load FastText cc.en.300 model."""
    import fasttext
    import fasttext.util

    if model_path is None:
        model_path = "/mnt/raid0/llm/cache/fasttext/cc.en.300.bin"

    if not os.path.exists(model_path):
        print(f"\nDownloading FastText cc.en.300 to {model_path}...")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Download to the target directory
        old_cwd = os.getcwd()
        os.chdir(os.path.dirname(model_path))
        fasttext.util.download_model("en", if_exists="ignore")
        os.chdir(old_cwd)

    print(f"\nLoading FastText from {model_path}...")
    model = fasttext.load_model(model_path)
    # FastText has a fixed vocabulary but can generate vectors for ANY word
    # via character n-grams. We check the "known" vocabulary for coverage stats.
    vocab = set(model.get_words())
    print(f"  Loaded: {len(vocab):,} known vocabulary entries")
    print(f"  (FastText can also generate vectors for OOV words via subword n-grams)")
    return model, vocab


def check_coverage(
    tokens_by_lang: dict[str, Counter],
    glove_model,
    fasttext_model=None,
    fasttext_vocab: set = None,
) -> dict:
    """Check vocabulary coverage for each embedding model."""
    results = {
        "glove": {"total": 0, "covered": 0, "by_lang": {}, "by_category": defaultdict(lambda: {"total": 0, "covered": 0}),
                  "oov_tokens": Counter(), "covered_tokens": Counter()},
    }
    if fasttext_model is not None:
        results["fasttext_known"] = {
            "total": 0, "covered": 0, "by_lang": {}, "by_category": defaultdict(lambda: {"total": 0, "covered": 0}),
            "oov_tokens": Counter(), "covered_tokens": Counter(),
        }
        results["fasttext_subword"] = {
            "total": 0, "covered": 0, "by_lang": {}, "by_category": defaultdict(lambda: {"total": 0, "covered": 0}),
            "oov_tokens": Counter(), "covered_tokens": Counter(),
        }

    glove_vocab = set(glove_model.key_to_index.keys())

    for lang, token_counts in tokens_by_lang.items():
        lang_total = sum(token_counts.values())
        glove_covered = 0
        ft_known_covered = 0
        ft_subword_covered = 0

        for token, count in token_counts.items():
            category = categorize_token(token)
            token_lower = token.lower()

            # GloVe check
            in_glove = token_lower in glove_vocab or token in glove_vocab
            if in_glove:
                glove_covered += count
                results["glove"]["covered_tokens"][token] += count
            else:
                results["glove"]["oov_tokens"][token] += count
            results["glove"]["by_category"][category]["total"] += count
            if in_glove:
                results["glove"]["by_category"][category]["covered"] += count

            # FastText check (known vocabulary)
            if fasttext_vocab is not None:
                in_ft_known = token_lower in fasttext_vocab or token in fasttext_vocab
                if in_ft_known:
                    ft_known_covered += count
                    results["fasttext_known"]["covered_tokens"][token] += count
                else:
                    results["fasttext_known"]["oov_tokens"][token] += count
                results["fasttext_known"]["by_category"][category]["total"] += count
                if in_ft_known:
                    results["fasttext_known"]["by_category"][category]["covered"] += count

                # FastText subword check — always "covered" since it generates vectors
                # But we check if the vector has meaningful magnitude (not near-zero)
                try:
                    vec = fasttext_model.get_word_vector(token_lower)
                    import numpy as np
                    magnitude = float(np.linalg.norm(vec))
                    # Subword vectors with very low magnitude are essentially noise
                    in_ft_subword = magnitude > 0.5
                except Exception:
                    in_ft_subword = False

                if in_ft_subword:
                    ft_subword_covered += count
                    results["fasttext_subword"]["covered_tokens"][token] += count
                else:
                    results["fasttext_subword"]["oov_tokens"][token] += count
                results["fasttext_subword"]["by_category"][category]["total"] += count
                if in_ft_subword:
                    results["fasttext_subword"]["by_category"][category]["covered"] += count

        results["glove"]["total"] += lang_total
        results["glove"]["covered"] += glove_covered
        results["glove"]["by_lang"][lang] = {
            "total": lang_total,
            "covered": glove_covered,
            "pct": round(100 * glove_covered / lang_total, 1) if lang_total > 0 else 0,
        }

        if fasttext_vocab is not None:
            results["fasttext_known"]["total"] += lang_total
            results["fasttext_known"]["covered"] += ft_known_covered
            results["fasttext_known"]["by_lang"][lang] = {
                "total": lang_total,
                "covered": ft_known_covered,
                "pct": round(100 * ft_known_covered / lang_total, 1) if lang_total > 0 else 0,
            }
            results["fasttext_subword"]["total"] += lang_total
            results["fasttext_subword"]["covered"] += ft_subword_covered
            results["fasttext_subword"]["by_lang"][lang] = {
                "total": lang_total,
                "covered": ft_subword_covered,
                "pct": round(100 * ft_subword_covered / lang_total, 1) if lang_total > 0 else 0,
            }

    return results


def print_results(results: dict):
    """Print formatted coverage results."""
    for model_name, data in results.items():
        total = data["total"]
        covered = data["covered"]
        pct = round(100 * covered / total, 1) if total > 0 else 0

        print(f"\n{'=' * 70}")
        print(f"  {model_name.upper()} COVERAGE: {pct}% ({covered:,} / {total:,} token occurrences)")
        print(f"{'=' * 70}")

        # Per-language
        print(f"\n  Per-language breakdown:")
        print(f"  {'Language':<12} {'Covered':>10} {'Total':>10} {'Coverage':>10}")
        print(f"  {'-' * 44}")
        for lang, stats in sorted(data["by_lang"].items()):
            print(f"  {lang:<12} {stats['covered']:>10,} {stats['total']:>10,} {stats['pct']:>9.1f}%")

        # Per-category
        print(f"\n  Per-category breakdown:")
        print(f"  {'Category':<25} {'Covered':>10} {'Total':>10} {'Coverage':>10}")
        print(f"  {'-' * 57}")
        cats = data["by_category"]
        for cat in ["keywords", "common_identifiers", "type_names", "code_specific",
                     "compound_identifiers", "operators_punct", "numeric_literals", "other"]:
            if cat in cats:
                c = cats[cat]
                cat_pct = round(100 * c["covered"] / c["total"], 1) if c["total"] > 0 else 0
                print(f"  {cat:<25} {c['covered']:>10,} {c['total']:>10,} {cat_pct:>9.1f}%")

        # Top-50 OOV
        print(f"\n  Top-50 most frequent OOV tokens:")
        for i, (token, count) in enumerate(data["oov_tokens"].most_common(50)):
            cat = categorize_token(token)
            print(f"    {i + 1:>3}. {token!r:<30} {count:>8,}  [{cat}]")

        # Top-50 covered
        print(f"\n  Top-50 most frequent COVERED tokens:")
        for i, (token, count) in enumerate(data["covered_tokens"].most_common(50)):
            cat = categorize_token(token)
            print(f"    {i + 1:>3}. {token!r:<30} {count:>8,}  [{cat}]")


def main():
    parser = argparse.ArgumentParser(description="Q5: GloVe/FastText code coverage evaluation")
    parser.add_argument("--samples", type=int, default=1000, help="Number of snippets to sample")
    parser.add_argument("--skip-fasttext", action="store_true", help="Skip FastText evaluation (saves ~7GB download)")
    parser.add_argument("--db", default=SNIPPETS_DB, help="Path to snippets.db")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    print("Q5: SoftMatcha v2 GloVe/FastText Code Vocabulary Coverage Evaluation")
    print("=" * 70)

    # 1. Sample snippets
    print(f"\n[1/4] Sampling {args.samples} snippets from {args.db}...")
    snippets = sample_snippets(args.db, args.samples)
    print(f"  Total: {len(snippets)} snippets sampled")

    # 2. Tokenize with Moses (sacremoses)
    print(f"\n[2/4] Tokenizing with Moses (sacremoses)...")
    from sacremoses import MosesTokenizer
    mt = MosesTokenizer(lang="en")

    tokens_by_lang: dict[str, Counter] = defaultdict(Counter)
    total_tokens = 0
    unique_tokens: set[str] = set()

    for i, snippet in enumerate(snippets):
        tokens = tokenize_code(snippet["code"], mt)
        for t in tokens:
            tokens_by_lang[snippet["language"]][t] += 1
            unique_tokens.add(t)
        total_tokens += len(tokens)
        if (i + 1) % 200 == 0:
            print(f"  Tokenized {i + 1}/{len(snippets)} snippets ({total_tokens:,} tokens so far)")

    print(f"  Total: {total_tokens:,} token occurrences, {len(unique_tokens):,} unique tokens")
    for lang, counts in sorted(tokens_by_lang.items()):
        print(f"    {lang}: {sum(counts.values()):,} tokens, {len(counts):,} unique")

    # 3. Load embedding models
    print(f"\n[3/4] Loading embedding models...")
    glove_model = load_glove()

    fasttext_model = None
    fasttext_vocab = None
    if not args.skip_fasttext:
        fasttext_model, fasttext_vocab = load_fasttext()

    # 4. Check coverage
    print(f"\n[4/4] Checking vocabulary coverage...")
    results = check_coverage(tokens_by_lang, glove_model, fasttext_model, fasttext_vocab)

    # Print results
    print_results(results)

    # Summary decision
    print(f"\n{'=' * 70}")
    print("  DECISION SUMMARY")
    print(f"{'=' * 70}")
    glove_pct = round(100 * results["glove"]["covered"] / results["glove"]["total"], 1)
    print(f"  GloVe coverage: {glove_pct}%")
    if "fasttext_known" in results:
        ft_known_pct = round(100 * results["fasttext_known"]["covered"] / results["fasttext_known"]["total"], 1)
        ft_subword_pct = round(100 * results["fasttext_subword"]["covered"] / results["fasttext_subword"]["total"], 1)
        print(f"  FastText known vocab coverage: {ft_known_pct}%")
        print(f"  FastText subword coverage (magnitude > 0.5): {ft_subword_pct}%")

        if glove_pct < 10 and ft_known_pct < 10:
            print("\n  RECOMMENDATION: CLOSE Q5 — neither embedding meaningful for code")
        elif glove_pct < 10 and 10 <= ft_known_pct <= 30:
            print("\n  RECOMMENDATION: Evaluate FastText results qualitatively")
            print("  Check if covered tokens are useful identifiers or just common English words")
        elif glove_pct > 30 or ft_known_pct > 30:
            print("\n  RECOMMENDATION: Proceed to Step 2 — build test index")
        else:
            print(f"\n  RECOMMENDATION: Review results manually")
    else:
        if glove_pct < 10:
            print("\n  RECOMMENDATION: GloVe insufficient. Run with FastText to compare.")
        elif glove_pct > 30:
            print("\n  RECOMMENDATION: GloVe promising. Proceed to Step 2.")

    # Save JSON output
    if args.output:
        output_path = args.output
    else:
        os.makedirs(f"{OUTPUT_DIR}/q5_coverage", exist_ok=True)
        output_path = f"{OUTPUT_DIR}/q5_coverage/results.json"

    # Serialize results (convert defaultdict + Counter to plain dict)
    serializable = {}
    for model_name, data in results.items():
        serializable[model_name] = {
            "total": data["total"],
            "covered": data["covered"],
            "coverage_pct": round(100 * data["covered"] / data["total"], 1) if data["total"] > 0 else 0,
            "by_lang": data["by_lang"],
            "by_category": {k: dict(v) for k, v in data["by_category"].items()},
            "top50_oov": data["oov_tokens"].most_common(50),
            "top50_covered": data["covered_tokens"].most_common(50),
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\n  Results saved to {output_path}")


if __name__ == "__main__":
    main()
