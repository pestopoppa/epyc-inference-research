#!/usr/bin/env python3
"""Build n-gram corpus index for prompt-lookup acceleration.

Collects Python source files, extracts code snippets, and builds a simple
n-gram index stored as a JSON + mmap file for sub-millisecond lookup.

Usage:
    python scripts/corpus/build_index.py [--output /mnt/raid0/llm/cache/corpus/mvp_index]

The index enables corpus-augmented prompt stuffing: retrieved code snippets
are injected into prompts so llama-server's --lookup has richer n-gram
material to match against for draft token proposals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Default paths
DEFAULT_OUTPUT = "/mnt/raid0/llm/cache/corpus/mvp_index"
PROJECT_SRC = Path("/mnt/raid0/llm/claude/src")

# Snippet extraction config
MIN_SNIPPET_LINES = 5
MAX_SNIPPET_LINES = 50
MAX_SNIPPET_CHARS = 2000
NGRAM_SIZE = 4  # 4-gram for reasonable specificity


def collect_py_files(roots: list[Path], max_files: int = 5000) -> list[Path]:
    """Collect .py files from given roots, sorted by size (largest first)."""
    files = []
    for root in roots:
        if not root.exists():
            log.warning("Skipping non-existent root: %s", root)
            continue
        for p in root.rglob("*.py"):
            if p.is_file() and p.stat().st_size > 100:
                files.append(p)
            if len(files) >= max_files:
                break
    files.sort(key=lambda p: p.stat().st_size, reverse=True)
    return files[:max_files]


def extract_snippets(path: Path) -> list[dict]:
    """Extract meaningful code snippets from a Python file.

    Splits on class/function definitions to get self-contained snippets.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    lines = text.split("\n")
    snippets = []
    current: list[str] = []
    current_start = 0

    def flush():
        if len(current) >= MIN_SNIPPET_LINES:
            body = "\n".join(current)[:MAX_SNIPPET_CHARS]
            snippets.append({
                "file": str(path),
                "start_line": current_start + 1,
                "code": body,
                "hash": hashlib.sha256(body.encode()).hexdigest()[:12],
            })

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Split on top-level definitions
        if stripped.startswith(("def ", "class ", "async def ")) and not line.startswith(" "):
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


def build_ngram_index(snippets: list[dict], n: int = NGRAM_SIZE) -> dict:
    """Build character n-gram to snippet index.

    Returns mapping from n-gram -> list of (snippet_idx, position) pairs.
    Each entry is deduplicated by snippet hash.
    """
    index: dict[str, list[int]] = defaultdict(list)
    seen_hashes: set[str] = set()
    deduped_snippets = []

    for snip in snippets:
        h = snip["hash"]
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        idx = len(deduped_snippets)
        deduped_snippets.append(snip)

        code = snip["code"].lower()
        # Extract word-level n-grams with normalized tokens
        raw_words = code.split()
        words = [re.sub(r"[^a-z0-9_]", "", w) for w in raw_words]
        words = [w for w in words if w]
        for i in range(len(words) - n + 1):
            gram = " ".join(words[i:i + n])
            if idx not in index[gram]:
                index[gram].append(idx)

    return {"snippets": deduped_snippets, "ngram_index": dict(index)}


def save_index(data: dict, output_dir: str) -> None:
    """Save index to disk as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    snippets_path = os.path.join(output_dir, "snippets.json")
    index_path = os.path.join(output_dir, "ngram_index.json")
    meta_path = os.path.join(output_dir, "meta.json")

    with open(snippets_path, "w") as f:
        json.dump(data["snippets"], f, separators=(",", ":"))

    with open(index_path, "w") as f:
        json.dump(data["ngram_index"], f, separators=(",", ":"))

    meta = {
        "version": 1,
        "ngram_size": NGRAM_SIZE,
        "num_snippets": len(data["snippets"]),
        "num_ngrams": len(data["ngram_index"]),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    total_bytes = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
    )
    log.info(
        "Index saved to %s: %d snippets, %d n-grams, %.1f MB",
        output_dir, meta["num_snippets"], meta["num_ngrams"],
        total_bytes / 1024 / 1024,
    )


def get_stdlib_path() -> Path | None:
    """Get CPython stdlib source path."""
    import sysconfig
    stdlib = sysconfig.get_paths()["stdlib"]
    p = Path(stdlib)
    return p if p.exists() else None


def get_site_packages_sources() -> list[Path]:
    """Get source directories for key packages."""
    paths = []
    try:
        import numpy
        p = Path(numpy.__file__).parent
        if p.exists():
            paths.append(p)
    except ImportError:
        pass
    try:
        import torch
        p = Path(torch.__file__).parent
        if p.exists():
            paths.append(p)
    except ImportError:
        pass
    return paths


def main():
    parser = argparse.ArgumentParser(description="Build corpus n-gram index")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--project-only", action="store_true", help="Only index project src/")
    args = parser.parse_args()

    t0 = time.perf_counter()

    # Collect source roots
    roots = [PROJECT_SRC]
    if not args.project_only:
        stdlib = get_stdlib_path()
        if stdlib:
            roots.append(stdlib)
            log.info("Including stdlib: %s", stdlib)
        for sp in get_site_packages_sources():
            roots.append(sp)
            log.info("Including package: %s", sp)

    log.info("Collecting .py files from %d roots...", len(roots))
    files = collect_py_files(roots)
    log.info("Found %d files", len(files))

    total_bytes = sum(f.stat().st_size for f in files)
    log.info("Total source: %.1f MB", total_bytes / 1024 / 1024)

    log.info("Extracting snippets...")
    all_snippets = []
    for f in files:
        all_snippets.extend(extract_snippets(f))
    log.info("Extracted %d raw snippets", len(all_snippets))

    log.info("Building n-gram index (n=%d)...", NGRAM_SIZE)
    data = build_ngram_index(all_snippets)
    log.info(
        "Deduplicated to %d snippets, %d unique n-grams",
        len(data["snippets"]), len(data["ngram_index"]),
    )

    save_index(data, args.output)

    elapsed = time.perf_counter() - t0
    log.info("Build completed in %.2fs", elapsed)


if __name__ == "__main__":
    main()
