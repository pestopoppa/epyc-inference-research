#!/usr/bin/env python3
"""Build scaled n-gram corpus index from The Stack (HuggingFace) for prompt-lookup acceleration.

Streams code from bigcode/the-stack, extracts snippets, builds a SQLite-backed
n-gram index. Supports multiple languages and handles 100GB+ corpus sizes.

Usage:
    # Full build (Python + JS + Rust + Go + C++)
    python scripts/corpus/build_index_v2.py --output /mnt/raid0/llm/cache/corpus/full_index

    # Python only
    python scripts/corpus/build_index_v2.py --languages python

    # Resume interrupted build
    python scripts/corpus/build_index_v2.py --resume

    # Add local sources on top of HF data
    python scripts/corpus/build_index_v2.py --local-roots /mnt/raid0/llm/claude/src

Index format: SQLite database with:
  - snippets(id, code, source, hash, language)
  - ngrams(gram, snippet_id) with index on gram
  - meta(key, value) for build metadata
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/mnt/raid0/llm/cache/corpus/full_index"

# Languages and their HF data_dir names in bigcode/the-stack
LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
    "c++": "c++",
}

# Snippet extraction config
MIN_SNIPPET_LINES = 5
MAX_SNIPPET_LINES = 50
MAX_SNIPPET_CHARS = 2000
NGRAM_SIZE = 4
MIN_FILE_BYTES = 100
MAX_FILE_BYTES = 100_000  # Skip huge files (minified, generated)
BATCH_SIZE = 5000  # SQLite insert batch size
PROGRESS_INTERVAL = 10_000  # Log progress every N files

# Language-specific top-level definition patterns
SPLIT_PATTERNS = {
    "python": re.compile(r"^(def |class |async def )", re.MULTILINE),
    "javascript": re.compile(
        r"^(function |const |let |var |class |export |module\.exports)", re.MULTILINE
    ),
    "typescript": re.compile(
        r"^(function |const |let |var |class |export |interface |type )", re.MULTILINE
    ),
    "rust": re.compile(r"^(fn |pub fn |impl |struct |enum |trait |mod )", re.MULTILINE),
    "go": re.compile(r"^(func |type |var |const )", re.MULTILINE),
    "c++": re.compile(
        r"^(class |struct |void |int |bool |auto |template |namespace )", re.MULTILINE
    ),
}


def create_db(db_path: str) -> sqlite3.Connection:
    """Create or open the SQLite index database."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-256000")  # 256MB cache
    conn.execute("PRAGMA mmap_size=1073741824")  # 1GB mmap
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            source TEXT DEFAULT '',
            hash TEXT NOT NULL,
            language TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS ngrams (
            gram TEXT NOT NULL,
            snippet_id INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
    """)
    return conn


def finalize_db(conn: sqlite3.Connection) -> None:
    """Create the n-gram index after all inserts are done."""
    log.info("Creating n-gram index (this may take a few minutes)...")
    t0 = time.perf_counter()
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram)"
    )
    conn.commit()
    elapsed = time.perf_counter() - t0
    log.info("N-gram index created in %.1fs", elapsed)


def extract_snippets_from_content(
    content: str,
    source: str,
    language: str,
) -> list[dict]:
    """Extract code snippets from file content.

    Splits on language-appropriate top-level definitions.
    """
    lines = content.split("\n")
    if len(lines) < MIN_SNIPPET_LINES:
        return []

    split_re = SPLIT_PATTERNS.get(language)
    snippets = []
    current: list[str] = []
    current_start = 0

    def flush():
        if len(current) >= MIN_SNIPPET_LINES:
            body = "\n".join(current)[:MAX_SNIPPET_CHARS]
            snippets.append({
                "code": body,
                "source": source,
                "hash": hashlib.sha256(body.encode()).hexdigest()[:12],
                "language": language,
            })

    for i, line in enumerate(lines):
        # Split on top-level definitions (no leading whitespace)
        if split_re and split_re.match(line) and not line.startswith((" ", "\t")):
            flush()
            current = [line]
            current_start = i
        else:
            current.append(line)
            if len(current) >= MAX_SNIPPET_LINES:
                flush()
                current = []
                current_start = i + 1

    flush()
    return snippets


def _normalize_token(token: str) -> str:
    """Strip non-alphanumeric chars (except underscore) from a token."""
    return re.sub(r"[^a-z0-9_]", "", token.lower())


def extract_ngrams(text: str, n: int = NGRAM_SIZE) -> list[str]:
    """Extract word-level n-grams from text with normalized tokens."""
    raw_words = text.lower().split()
    words = [_normalize_token(w) for w in raw_words]
    words = [w for w in words if w]  # drop empty after normalization
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def insert_batch(
    conn: sqlite3.Connection,
    snippets: list[dict],
    seen_hashes: set[str],
) -> int:
    """Insert a batch of snippets and their n-grams. Returns count inserted."""
    inserted = 0
    ngram_rows = []

    for snip in snippets:
        h = snip["hash"]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        cursor = conn.execute(
            "INSERT INTO snippets (code, source, hash, language) VALUES (?, ?, ?, ?)",
            (snip["code"], snip["source"], h, snip["language"]),
        )
        sid = cursor.lastrowid

        grams = extract_ngrams(snip["code"])
        for gram in grams:
            ngram_rows.append((gram, sid))

        inserted += 1

    if ngram_rows:
        conn.executemany(
            "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
            ngram_rows,
        )

    return inserted


def process_hf_language(
    conn: sqlite3.Connection,
    language: str,
    seen_hashes: set[str],
    max_files: int = 0,
) -> dict:
    """Stream and process a language from bigcode/the-stack."""
    from datasets import load_dataset

    data_dir = LANGUAGE_MAP.get(language, language)
    log.info("Streaming bigcode/the-stack data_dir=%s ...", data_dir)

    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{data_dir}",
            split="train",
            streaming=True,
        )
    except Exception as e:
        log.error("Failed to load %s: %s", data_dir, e)
        return {"files": 0, "snippets": 0, "skipped": 0}

    stats = {"files": 0, "snippets": 0, "skipped": 0}
    batch: list[dict] = []

    for item in ds:
        content = item.get("content", "")
        size = len(content.encode("utf-8", errors="replace"))

        if size < MIN_FILE_BYTES or size > MAX_FILE_BYTES:
            stats["skipped"] += 1
            continue

        # Skip vendored/generated files
        if item.get("is_vendor") or item.get("is_generated"):
            stats["skipped"] += 1
            continue

        path = item.get("max_stars_repo_path", item.get("path", "unknown"))
        repo = item.get("max_stars_repo_name", "")
        source = f"{repo}:{path}" if repo else path

        file_snippets = extract_snippets_from_content(content, source, language)
        batch.extend(file_snippets)

        stats["files"] += 1

        # Batch insert
        if len(batch) >= BATCH_SIZE:
            inserted = insert_batch(conn, batch, seen_hashes)
            stats["snippets"] += inserted
            conn.commit()
            batch = []

        if stats["files"] % PROGRESS_INTERVAL == 0:
            log.info(
                "  [%s] %d files processed, %d snippets indexed, %d skipped",
                language,
                stats["files"],
                stats["snippets"],
                stats["skipped"],
            )

        if max_files and stats["files"] >= max_files:
            log.info("  [%s] Reached max_files=%d, stopping", language, max_files)
            break

    # Final batch
    if batch:
        inserted = insert_batch(conn, batch, seen_hashes)
        stats["snippets"] += inserted
        conn.commit()

    log.info(
        "  [%s] Done: %d files, %d snippets, %d skipped",
        language,
        stats["files"],
        stats["snippets"],
        stats["skipped"],
    )
    return stats


def process_local_roots(
    conn: sqlite3.Connection,
    roots: list[str],
    seen_hashes: set[str],
) -> dict:
    """Index local source directories."""
    stats = {"files": 0, "snippets": 0}

    # Extension to language mapping
    ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".rs": "rust",
        ".go": "go",
        ".cpp": "c++",
        ".cc": "c++",
        ".h": "c++",
        ".hpp": "c++",
    }

    batch: list[dict] = []

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            log.warning("Skipping non-existent root: %s", root)
            continue

        log.info("Indexing local: %s", root)
        for p in root_path.rglob("*"):
            if not p.is_file():
                continue
            lang = ext_map.get(p.suffix.lower())
            if not lang:
                continue
            if p.stat().st_size < MIN_FILE_BYTES or p.stat().st_size > MAX_FILE_BYTES:
                continue

            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            file_snippets = extract_snippets_from_content(
                content, str(p), lang
            )
            batch.extend(file_snippets)
            stats["files"] += 1

            if len(batch) >= BATCH_SIZE:
                inserted = insert_batch(conn, batch, seen_hashes)
                stats["snippets"] += inserted
                conn.commit()
                batch = []

    if batch:
        inserted = insert_batch(conn, batch, seen_hashes)
        stats["snippets"] += inserted
        conn.commit()

    log.info("Local: %d files, %d snippets", stats["files"], stats["snippets"])
    return stats


def write_json_compat(conn: sqlite3.Connection, output_dir: str) -> None:
    """Write JSON-compatible files for backward compatibility with v1 retriever."""
    import json

    log.info("Writing JSON compatibility layer...")
    t0 = time.perf_counter()

    # Export snippets
    rows = conn.execute(
        "SELECT id, code, source, hash, language FROM snippets ORDER BY id"
    ).fetchall()
    snippets = []
    id_to_idx = {}
    for i, (sid, code, source, h, lang) in enumerate(rows):
        snippets.append({
            "code": code,
            "file": source,
            "start_line": 0,
            "hash": h,
            "language": lang,
        })
        id_to_idx[sid] = i

    snippets_path = os.path.join(output_dir, "snippets.json")
    with open(snippets_path, "w") as f:
        json.dump(snippets, f, separators=(",", ":"))

    # Export n-gram index (this may be large)
    ngram_index: dict[str, list[int]] = {}
    cursor = conn.execute("SELECT gram, snippet_id FROM ngrams")
    for gram, sid in cursor:
        idx = id_to_idx.get(sid)
        if idx is not None:
            ngram_index.setdefault(gram, []).append(idx)

    index_path = os.path.join(output_dir, "ngram_index.json")
    with open(index_path, "w") as f:
        json.dump(ngram_index, f, separators=(",", ":"))

    elapsed = time.perf_counter() - t0
    log.info("JSON compat written in %.1fs", elapsed)


def main():
    parser = argparse.ArgumentParser(
        description="Build scaled corpus n-gram index from The Stack"
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--languages",
        default="python,javascript,rust,go,c++",
        help="Comma-separated languages (default: %(default)s)",
    )
    parser.add_argument(
        "--local-roots",
        nargs="*",
        default=[],
        help="Additional local directories to index",
    )
    parser.add_argument(
        "--max-files-per-lang",
        type=int,
        default=0,
        help="Max files per language (0=unlimited, for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted build (keep existing data)",
    )
    parser.add_argument(
        "--json-compat",
        action="store_true",
        help="Also write JSON files for v1 retriever compatibility",
    )
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Skip index creation (for incremental builds)",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    os.makedirs(args.output, exist_ok=True)
    db_path = os.path.join(args.output, "corpus.db")

    if not args.resume and os.path.exists(db_path):
        log.info("Removing existing database (use --resume to keep)")
        os.remove(db_path)

    conn = create_db(db_path)

    # Load existing hashes for dedup
    seen_hashes: set[str] = set()
    if args.resume:
        rows = conn.execute("SELECT hash FROM snippets").fetchall()
        seen_hashes = {r[0] for r in rows}
        log.info("Resuming: %d existing snippets loaded", len(seen_hashes))

    # Process languages
    languages = [l.strip() for l in args.languages.split(",")]
    all_stats: dict[str, dict] = {}

    for lang in languages:
        if lang not in LANGUAGE_MAP:
            log.warning("Unknown language %s, skipping", lang)
            continue

        # Check if already processed (for resume)
        if args.resume:
            existing = conn.execute(
                "SELECT COUNT(*) FROM snippets WHERE language=?", (lang,)
            ).fetchone()[0]
            if existing > 0:
                log.info("Skipping %s (already has %d snippets)", lang, existing)
                all_stats[lang] = {"files": 0, "snippets": existing, "skipped": 0}
                continue

        stats = process_hf_language(conn, lang, seen_hashes, args.max_files_per_lang)
        all_stats[lang] = stats

    # Process local roots
    if args.local_roots:
        local_stats = process_local_roots(conn, args.local_roots, seen_hashes)
        all_stats["local"] = local_stats

    # Finalize
    if not args.skip_finalize:
        finalize_db(conn)

    # Write metadata
    total_snippets = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
    total_ngrams = conn.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
    elapsed = time.perf_counter() - t0

    import json

    meta = {
        "version": 2,
        "format": "sqlite",
        "ngram_size": NGRAM_SIZE,
        "num_snippets": total_snippets,
        "num_ngrams": total_ngrams,
        "languages": languages,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_time_s": round(elapsed, 1),
        "stats": all_stats,
    }
    conn.execute(
        "INSERT OR REPLACE INTO meta VALUES (?, ?)",
        ("build_info", json.dumps(meta)),
    )
    conn.commit()

    meta_path = os.path.join(args.output, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    db_size_mb = os.path.getsize(db_path) / 1024 / 1024

    log.info("=" * 60)
    log.info("Build complete in %.1fs", elapsed)
    log.info("  Snippets:  %d", total_snippets)
    log.info("  N-grams:   %d", total_ngrams)
    log.info("  DB size:   %.1f MB", db_size_mb)
    log.info("  Languages: %s", ", ".join(languages))
    log.info("  Output:    %s", args.output)
    log.info("=" * 60)

    # Write JSON compat if requested
    if args.json_compat:
        write_json_compat(conn, args.output)

    conn.close()

    # Print per-language stats
    for lang, stats in all_stats.items():
        log.info(
            "  %-12s: %d files, %d snippets",
            lang,
            stats.get("files", 0),
            stats.get("snippets", 0),
        )


if __name__ == "__main__":
    main()
