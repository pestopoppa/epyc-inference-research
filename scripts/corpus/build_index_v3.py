#!/usr/bin/env python3
"""Parallel pipelined sharded corpus index builder (v3).

Downloads languages in parallel, merges+shards each one AS IT COMPLETES
(pipelining download with merge), then builds indexes in parallel across
all shards. Aggressively uses available CPU cores and RAM.

Architecture (pipelined):
  6 download workers stream HF → temp/{lang}.db
  Main thread merges+shards each lang AS IT COMPLETES (overlapping with ongoing downloads)
  16 index workers CREATE INDEX in parallel on each shard
  Cleanup + meta.json

Usage:
    # Full build (all 6 languages, 16 shards)
    python scripts/corpus/build_index_v3.py --output /mnt/raid0/llm/cache/corpus/v3_sharded

    # Python only (fast validation)
    python scripts/corpus/build_index_v3.py --languages python --max-files-per-lang 10000

    # Resume after interruption
    python scripts/corpus/build_index_v3.py --resume

    # Custom shard count
    python scripts/corpus/build_index_v3.py --shards 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/mnt/raid0/llm/cache/corpus/v3_sharded"

LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
    "c++": "c++",
}

# Snippet extraction config (same as v2 for compatibility)
MIN_SNIPPET_LINES = 5
MAX_SNIPPET_LINES = 50
MAX_SNIPPET_CHARS = 2000
NGRAM_SIZE = 4
MIN_FILE_BYTES = 100
MAX_FILE_BYTES = 100_000
BATCH_SIZE = 10_000  # Larger batches — we have 1.13TB RAM
PROGRESS_INTERVAL = 10_000

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


# ── Snippet extraction (shared with v2) ─────────────────────────────────


def extract_snippets_from_content(
    content: str, source: str, language: str,
) -> list[dict]:
    """Extract code snippets from file content."""
    lines = content.split("\n")
    if len(lines) < MIN_SNIPPET_LINES:
        return []

    split_re = SPLIT_PATTERNS.get(language)
    snippets = []
    current: list[str] = []

    def flush():
        if len(current) >= MIN_SNIPPET_LINES:
            body = "\n".join(current)[:MAX_SNIPPET_CHARS]
            snippets.append({
                "code": body,
                "source": source,
                "hash": hashlib.md5(body.encode()).hexdigest()[:12],
                "language": language,
            })

    for line in lines:
        if split_re and split_re.match(line) and not line.startswith((" ", "\t")):
            flush()
            current = [line]
        else:
            current.append(line)
            if len(current) >= MAX_SNIPPET_LINES:
                flush()
                current = []

    flush()
    return snippets


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", token.lower())


def extract_ngrams(text: str, n: int = NGRAM_SIZE) -> list[str]:
    raw_words = text.lower().split()
    words = [_normalize_token(w) for w in raw_words]
    words = [w for w in words if w]
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def gram_to_shard(gram: str, num_shards: int) -> int:
    """Deterministic shard assignment via hash."""
    h = hashlib.md5(gram.encode()).digest()
    return int.from_bytes(h[:4], "little") % num_shards


# ── Download one language to temp DB ─────────────────────────────────────


def _create_temp_db(db_path: str) -> sqlite3.Connection:
    """Create temp per-language DB with aggressive performance settings."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")  # Temp data — can rebuild if crash
    conn.execute("PRAGMA cache_size=-512000")  # 512MB cache per lang
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA mmap_size=2147483648")  # 2GB mmap
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
        CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
    """)
    return conn


def download_language(
    language: str,
    output_dir: str,
    max_files: int = 0,
) -> dict:
    """Download and process one language into a temp DB. Runs in subprocess."""
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    db_path = os.path.join(temp_dir, f"{language}.db")

    # Check if already complete (resume support)
    done_marker = db_path + ".done"
    if os.path.exists(done_marker):
        log.info("[%s] Already downloaded (found .done marker), skipping", language)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        conn.close()
        return {"language": language, "files": 0, "snippets": count, "skipped": 0, "resumed": True}

    # Remove partial DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        for wal_file in [db_path + "-wal", db_path + "-shm"]:
            if os.path.exists(wal_file):
                os.remove(wal_file)

    conn = _create_temp_db(db_path)
    seen_hashes: set[str] = set()

    from datasets import load_dataset

    data_dir = LANGUAGE_MAP.get(language, language)
    log.info("[%s] Streaming bigcode/the-stack data_dir=%s ...", language, data_dir)

    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{data_dir}",
            split="train",
            streaming=True,
        )
    except Exception as e:
        log.error("[%s] Failed to load: %s", language, e)
        conn.close()
        return {"language": language, "files": 0, "snippets": 0, "skipped": 0, "error": str(e)}

    stats = {"language": language, "files": 0, "snippets": 0, "skipped": 0}
    ngram_rows: list[tuple[str, int]] = []

    for item in ds:
        content = item.get("content", "")
        size = len(content.encode("utf-8", errors="replace"))

        if size < MIN_FILE_BYTES or size > MAX_FILE_BYTES:
            stats["skipped"] += 1
            continue

        if item.get("is_vendor") or item.get("is_generated"):
            stats["skipped"] += 1
            continue

        path = item.get("max_stars_repo_path", item.get("path", "unknown"))
        repo = item.get("max_stars_repo_name", "")
        source = f"{repo}:{path}" if repo else path

        file_snippets = extract_snippets_from_content(content, source, language)

        for snip in file_snippets:
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

            stats["snippets"] += 1

        stats["files"] += 1

        # Batch commit ngrams — large batches for throughput
        if len(ngram_rows) >= BATCH_SIZE * 20:
            conn.executemany("INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)", ngram_rows)
            conn.commit()
            ngram_rows = []

        if stats["files"] % PROGRESS_INTERVAL == 0:
            log.info(
                "[%s] %d files, %d snippets, %d skipped",
                language, stats["files"], stats["snippets"], stats["skipped"],
            )

        if max_files and stats["files"] >= max_files:
            log.info("[%s] Reached max_files=%d", language, max_files)
            break

    # Final flush
    if ngram_rows:
        conn.executemany("INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)", ngram_rows)

    conn.commit()
    # Checkpoint WAL before closing so merge reads clean data
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    # Mark as complete
    Path(done_marker).write_text(json.dumps(stats))

    log.info(
        "[%s] Done: %d files, %d snippets, %d skipped",
        language, stats["files"], stats["snippets"], stats["skipped"],
    )
    return stats


# ── Merge one language into snippets.db + shard DBs ─────────────────────


def merge_one_language(
    language: str,
    temp_dir: str,
    snippets_conn: sqlite3.Connection,
    shard_conns: list[sqlite3.Connection],
    num_shards: int,
    global_id_start: int,
) -> tuple[int, dict]:
    """Merge a single language's temp DB into the shared output DBs.

    Returns (next_global_id, stats_dict).
    Called from main thread as each download completes.
    """
    temp_db = os.path.join(temp_dir, f"{language}.db")
    if not os.path.exists(temp_db):
        log.warning("[merge] Temp DB for %s not found, skipping", language)
        return global_id_start, {"snippets": 0, "ngrams": 0}

    t0 = time.perf_counter()
    src_conn = sqlite3.connect(temp_db)
    src_conn.execute("PRAGMA mmap_size=2147483648")
    local_count = src_conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
    log.info("[merge] %s: %d snippets, global offset=%d", language, local_count, global_id_start)

    global_id = global_id_start
    id_map: dict[int, int] = {}
    batch_snippets = []

    for local_id, code, source, h, lang in src_conn.execute(
        "SELECT id, code, source, hash, language FROM snippets ORDER BY id"
    ):
        new_id = global_id
        id_map[local_id] = new_id
        batch_snippets.append((new_id, code, source, h, lang))
        global_id += 1

        if len(batch_snippets) >= BATCH_SIZE:
            snippets_conn.executemany(
                "INSERT INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
                batch_snippets,
            )
            batch_snippets = []

    if batch_snippets:
        snippets_conn.executemany(
            "INSERT INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
            batch_snippets,
        )
    snippets_conn.commit()

    # Shard ngrams with remapped snippet IDs
    shard_buffers: list[list[tuple[str, int]]] = [[] for _ in range(num_shards)]
    ngram_count = 0
    flush_threshold = BATCH_SIZE * 10

    for gram, local_sid in src_conn.execute("SELECT gram, snippet_id FROM ngrams"):
        new_sid = id_map.get(local_sid)
        if new_sid is None:
            continue
        shard_id = gram_to_shard(gram, num_shards)
        shard_buffers[shard_id].append((gram, new_sid))
        ngram_count += 1

        if len(shard_buffers[shard_id]) >= flush_threshold:
            shard_conns[shard_id].executemany(
                "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                shard_buffers[shard_id],
            )
            shard_buffers[shard_id] = []

    # Flush remaining
    for sid in range(num_shards):
        if shard_buffers[sid]:
            shard_conns[sid].executemany(
                "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                shard_buffers[sid],
            )

    # Commit all shards
    for conn in shard_conns:
        conn.commit()

    src_conn.close()
    elapsed = time.perf_counter() - t0
    log.info("[merge] %s: %d snippets + %d ngrams merged in %.1fs", language, local_count, ngram_count, elapsed)

    return global_id, {"snippets": local_count, "ngrams": ngram_count, "merge_time_s": round(elapsed, 1)}


# ── Parallel index creation ──────────────────────────────────────────────


def create_shard_index(shard_path: str) -> dict:
    """Create index on a single shard DB. Runs in subprocess."""
    shard_name = os.path.basename(shard_path)
    t0 = time.perf_counter()
    log.info("[index] Creating index on %s ...", shard_name)

    conn = sqlite3.connect(shard_path)
    conn.execute("PRAGMA cache_size=-512000")  # 512MB
    conn.execute("PRAGMA mmap_size=4294967296")  # 4GB mmap
    conn.execute("PRAGMA temp_store=MEMORY")

    ngram_count = conn.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram)")
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    elapsed = time.perf_counter() - t0
    log.info("[index] %s: %d ngrams indexed in %.1fs", shard_name, ngram_count, elapsed)
    return {"shard": shard_name, "ngrams": ngram_count, "elapsed_s": round(elapsed, 1)}


# ── Metadata + cleanup ──────────────────────────────────────────────────


def write_metadata(
    output_dir: str,
    languages: list[str],
    num_shards: int,
    total_snippets: int,
    total_ngrams: int,
    lang_stats: dict,
    index_stats: list[dict],
    total_elapsed: float,
) -> None:
    """Write meta.json for the retriever to detect v3 format."""
    meta = {
        "version": 3,
        "format": "sharded_sqlite",
        "ngram_size": NGRAM_SIZE,
        "num_shards": num_shards,
        "num_snippets": total_snippets,
        "num_ngrams": total_ngrams,
        "languages": languages,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_time_s": round(total_elapsed, 1),
        "lang_stats": lang_stats,
        "index_stats": index_stats,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written to %s", meta_path)


def cleanup_temp(output_dir: str) -> None:
    """Remove temp language DBs."""
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        return
    for f in Path(temp_dir).iterdir():
        f.unlink()
        log.info("[cleanup] Removed %s", f.name)
    os.rmdir(temp_dir)
    log.info("[cleanup] Temp directory removed")


# ── Main: Pipelined execution ────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Parallel pipelined sharded corpus index builder (v3)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--languages", default="python,javascript,typescript,rust,go,c++",
        help="Comma-separated languages (default: %(default)s)",
    )
    parser.add_argument(
        "--shards", type=int, default=16,
        help="Number of ngram shards (default: %(default)s)",
    )
    parser.add_argument(
        "--max-files-per-lang", type=int, default=0,
        help="Max files per language (0=unlimited, for testing)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip languages with .done markers",
    )
    parser.add_argument(
        "--download-workers", type=int, default=6,
        help="Max parallel language downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--index-workers", type=int, default=16,
        help="Max parallel index creation workers (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloads (use existing temp DBs)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip merge (use existing shard DBs)",
    )
    parser.add_argument(
        "--cleanup-temp", action="store_true",
        help="Remove temp language DBs after successful build",
    )
    args = parser.parse_args()

    t_total = time.perf_counter()
    languages = [l.strip() for l in args.languages.split(",")]
    languages = [l for l in languages if l in LANGUAGE_MAP]
    output_dir = args.output
    num_shards = args.shards
    temp_dir = os.path.join(output_dir, "temp")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    log.info("=" * 60)
    log.info("Parallel Pipelined Sharded Corpus Build v3")
    log.info("  Output:    %s", output_dir)
    log.info("  Languages: %s", ", ".join(languages))
    log.info("  Shards:    %d", num_shards)
    log.info("  Pipeline:  download → merge (overlapped) → parallel index")
    log.info("=" * 60)

    # ── Prepare output DBs for merge ─────────────────────────────────
    if not args.skip_merge:
        snippets_db_path = os.path.join(output_dir, "snippets.db")
        if os.path.exists(snippets_db_path) and not args.resume:
            os.remove(snippets_db_path)

        snippets_conn = sqlite3.connect(snippets_db_path)
        snippets_conn.execute("PRAGMA journal_mode=WAL")
        snippets_conn.execute("PRAGMA synchronous=OFF")
        snippets_conn.execute("PRAGMA cache_size=-512000")
        snippets_conn.execute("PRAGMA temp_store=MEMORY")
        snippets_conn.executescript("""
            CREATE TABLE IF NOT EXISTS snippets (
                id INTEGER PRIMARY KEY,
                code TEXT NOT NULL,
                source TEXT DEFAULT '',
                hash TEXT NOT NULL,
                language TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
        """)

        shard_conns: list[sqlite3.Connection] = []
        for i in range(num_shards):
            shard_path = os.path.join(output_dir, f"shard_{i:02d}.db")
            if os.path.exists(shard_path) and not args.resume:
                os.remove(shard_path)
            conn = sqlite3.connect(shard_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=OFF")
            conn.execute("PRAGMA cache_size=-128000")
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ngrams (
                    gram TEXT NOT NULL,
                    snippet_id INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram);
            """)
            shard_conns.append(conn)
        log.info("Shard DBs created with live indexes (queryable during build)")

    # ── Pipelined: download + merge as each completes ────────────────
    global_id = 0
    total_snippets = 0
    total_ngrams = 0
    lang_stats: dict[str, dict] = {}
    download_stats: dict[str, dict] = {}
    langs_merged = 0

    if not args.skip_download:
        log.info("=== Pipelined: download + merge (%d languages) ===", len(languages))
        t_pipeline = time.perf_counter()

        with ProcessPoolExecutor(max_workers=args.download_workers) as pool:
            futures = {
                pool.submit(download_language, lang, output_dir, args.max_files_per_lang): lang
                for lang in languages
            }

            for future in as_completed(futures):
                lang = futures[future]
                try:
                    stats = future.result()
                    download_stats[lang] = stats
                    log.info(
                        "[pipeline] %s downloaded: %d snippets",
                        lang, stats.get("snippets", 0),
                    )
                except Exception as e:
                    log.error("[pipeline] %s download FAILED: %s", lang, e)
                    download_stats[lang] = {"error": str(e)}
                    continue

                # Immediately merge this language while others still download
                if not args.skip_merge and "error" not in stats:
                    global_id, m_stats = merge_one_language(
                        lang, temp_dir, snippets_conn, shard_conns, num_shards, global_id,
                    )
                    lang_stats[lang] = m_stats
                    total_snippets += m_stats["snippets"]
                    total_ngrams += m_stats["ngrams"]
                    langs_merged += 1
                    log.info(
                        "[pipeline] %s merged (%d/%d). Running total: %d snippets, %d ngrams",
                        lang, langs_merged, len(languages), total_snippets, total_ngrams,
                    )

                    # Write incremental progress
                    progress = {
                        "downloads": download_stats,
                        "merged": lang_stats,
                        "total_snippets": total_snippets,
                        "total_ngrams": total_ngrams,
                        "elapsed_s": round(time.perf_counter() - t_pipeline, 1),
                    }
                    with open(os.path.join(output_dir, "build_progress.json"), "w") as f:
                        json.dump(progress, f, indent=2)

        elapsed_pipeline = time.perf_counter() - t_pipeline
        log.info("Pipeline (download+merge) done in %.1fs", elapsed_pipeline)

    elif not args.skip_merge:
        # Skip download but still merge existing temp DBs
        log.info("=== Merge only (--skip-download) ===")
        for lang in languages:
            global_id, m_stats = merge_one_language(
                lang, temp_dir, snippets_conn, shard_conns, num_shards, global_id,
            )
            lang_stats[lang] = m_stats
            total_snippets += m_stats["snippets"]
            total_ngrams += m_stats["ngrams"]

    # Close merge DBs
    if not args.skip_merge:
        for conn in shard_conns:
            conn.commit()
            conn.close()
        snippets_conn.commit()
        snippets_conn.close()

    # If skip_merge, read existing meta for stats
    if args.skip_merge:
        meta_path = os.path.join(output_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing = json.load(f)
            total_snippets = existing.get("num_snippets", 0)
            total_ngrams = existing.get("num_ngrams", 0)
            lang_stats = existing.get("lang_stats", {})

    # ── Parallel index creation ──────────────────────────────────────
    log.info("=== Parallel index creation (%d shards, %d workers) ===", num_shards, args.index_workers)
    t_index = time.perf_counter()

    shard_paths = [
        os.path.join(output_dir, f"shard_{i:02d}.db")
        for i in range(num_shards)
    ]
    missing = [p for p in shard_paths if not os.path.exists(p)]
    if missing:
        log.error("Missing shard DBs: %s", missing)
        sys.exit(1)

    index_stats: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.index_workers) as pool:
        futures = {
            pool.submit(create_shard_index, path): path
            for path in shard_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                stats = future.result()
                index_stats.append(stats)
                log.info("[index] %s done (%.1fs)", stats["shard"], stats["elapsed_s"])
            except Exception as e:
                log.error("[index] %s FAILED: %s", os.path.basename(path), e)

    elapsed_index = time.perf_counter() - t_index
    log.info("Parallel indexing done in %.1fs", elapsed_index)

    # Checkpoint snippets.db WAL
    snippets_db = os.path.join(output_dir, "snippets.db")
    if os.path.exists(snippets_db):
        conn = sqlite3.connect(snippets_db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

    # ── Metadata + cleanup ───────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    write_metadata(
        output_dir, languages, num_shards,
        total_snippets, total_ngrams, lang_stats, index_stats, total_elapsed,
    )

    if args.cleanup_temp:
        cleanup_temp(output_dir)

    # Summary
    total_shard_size = sum(os.path.getsize(p) for p in shard_paths if os.path.exists(p))
    snippets_size = os.path.getsize(snippets_db) if os.path.exists(snippets_db) else 0

    log.info("=" * 60)
    log.info("Build complete in %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)
    log.info("  Snippets:    %d", total_snippets)
    log.info("  N-grams:     %d", total_ngrams)
    log.info("  Shards:      %d x ~%.1f MB = %.1f MB total",
             num_shards,
             total_shard_size / max(num_shards, 1) / 1024 / 1024,
             total_shard_size / 1024 / 1024)
    log.info("  Snippets DB: %.1f MB", snippets_size / 1024 / 1024)
    log.info("  Output:      %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
