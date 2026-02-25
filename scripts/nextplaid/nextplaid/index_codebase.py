#!/usr/bin/env python3
"""Index project source code into NextPLAID for multi-vector retrieval.

Usage:
    python scripts/nextplaid/index_codebase.py [--reindex]

Phase 5: Dual-container architecture with AST-aware chunking.
  Code index → :8088 (LateOn-Code 130M, 128-dim)
  Docs index → :8089 (answerai-colbert-small-v1-onnx)

Each index is served by a dedicated NextPLAID container with a model
optimized for that content type. Embeddings are model-specific and
cannot be mixed across containers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/raid0/llm/claude")
CODE_URL = "http://localhost:8088"
DOCS_URL = "http://localhost:8089"

CODE_PATTERNS = [
    "src/**/*.py",
    "orchestration/**/*.py",
    "scripts/**/*.py",
    "tests/**/*.py",
]

DOC_PATTERNS = [
    "docs/**/*.md",
    "handoffs/**/*.md",
    "orchestration/*.yaml",
    "orchestration/*.json",
    "CLAUDE.md",
    "CHANGELOG.md",
]

SKIP_FRAGMENTS = {"__pycache__", ".pyc", "node_modules", ".git", "cache/"}

from ast_chunker import chunk_file  # AST-aware chunking (Phase 5)


def collect_files(patterns: list[str], root: Path) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if any(skip in str(path) for skip in SKIP_FRAGMENTS):
                continue
            if path.is_file():
                files.append(path)
    return sorted(set(files))


def index_files(
    client,
    index_name: str,
    patterns: list[str],
    reindex: bool = False,
) -> int:
    from next_plaid_client.models import IndexConfig

    # Handle existing index
    try:
        info = client.get_index(index_name)
        if reindex:
            print(f"  Deleting existing '{index_name}' index for reindex...")
            client.delete_index(index_name)
        else:
            nd = info.num_documents if hasattr(info, "num_documents") else 0
            print(f"  Index '{index_name}' exists with {nd} documents. Use --reindex to rebuild.")
            return nd
    except Exception:
        pass  # Index doesn't exist

    client.create_index(index_name, IndexConfig(nbits=4))

    files = collect_files(patterns, PROJECT_ROOT)
    print(f"  Collected {len(files)} files")

    all_texts: list[str] = []
    all_metadata: list[dict] = []

    for f in files:
        for chunk in chunk_file(f):
            all_texts.append(chunk["text"])
            all_metadata.append({
                "file": chunk["file"],
                "start_line": str(chunk["start_line"]),
                "end_line": str(chunk["end_line"]),
                "unit_type": chunk.get("unit_type", ""),
                "unit_name": chunk.get("unit_name", ""),
                "signature": chunk.get("signature", ""),
                "has_docstring": str(chunk.get("has_docstring", False)),
            })

    print(f"  Total chunks: {len(all_texts)}")

    if not all_texts:
        print("  No content to index.")
        return 0

    # Batch ingest — NextPLAID handles encoding server-side
    # Batch size 32 for LateOn-Code 130M (larger model needs more encoding time per batch)
    BATCH = 32
    for i in range(0, len(all_texts), BATCH):
        batch_docs = all_texts[i : i + BATCH]
        batch_meta = all_metadata[i : i + BATCH]
        client.update_documents_with_encoding(
            index_name, documents=batch_docs, metadata=batch_meta
        )
        done = min(i + BATCH, len(all_texts))
        print(f"    Indexed {done}/{len(all_texts)} chunks", end="\r")

    print()

    # Wait for async indexing to complete
    for attempt in range(30):
        time.sleep(1)
        try:
            info = client.get_index(index_name)
            nd = info.num_documents if hasattr(info, "num_documents") else 0
            if nd >= len(all_texts):
                print(f"  Index built: {nd} documents")
                return nd
        except Exception:
            pass
    print(f"  Warning: index may still be building (expected {len(all_texts)} docs)")
    return len(all_texts)


def _health_check(client, url: str) -> bool:
    """Print health info and return True if server is healthy."""
    try:
        health = client.health()
        print(f"  NextPLAID at {url}: v{health.version}, {health.loaded_indices} indices, {health.memory_usage_bytes // 1024 // 1024}MB RAM")
        return True
    except Exception as e:
        print(f"  Warning: Cannot reach NextPLAID at {url}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Index codebase into NextPLAID")
    parser.add_argument("--reindex", action="store_true", help="Delete and rebuild indices")
    parser.add_argument("--code-only", action="store_true", help="Only index code, skip docs")
    parser.add_argument("--docs-only", action="store_true", help="Only index docs, skip code")
    parser.add_argument("--code-url", default=CODE_URL, help="NextPLAID code server URL (default :8088)")
    parser.add_argument("--docs-url", default=DOCS_URL, help="NextPLAID docs server URL (default :8089)")
    # Legacy single-URL flag (overrides both if set)
    parser.add_argument("--url", default=None, help="Override both code and docs URLs (legacy)")
    args = parser.parse_args()

    if args.url:
        args.code_url = args.url
        args.docs_url = args.url

    try:
        from next_plaid_client import NextPlaidClient
    except ImportError:
        print("Error: pip install next-plaid-client", file=sys.stderr)
        sys.exit(1)

    total = 0
    t0 = time.time()

    # 120s timeout for LateOn-Code 130M (larger model, slower encoding per batch)
    if not args.docs_only:
        code_client = NextPlaidClient(args.code_url, timeout=120.0)
        if _health_check(code_client, args.code_url):
            print("\n[code] Indexing source files...")
            total += index_files(code_client, "code", CODE_PATTERNS, reindex=args.reindex)
        else:
            print(f"Error: Code server at {args.code_url} not reachable, skipping code index", file=sys.stderr)

    if not args.code_only:
        docs_client = NextPlaidClient(args.docs_url, timeout=120.0)
        if _health_check(docs_client, args.docs_url):
            print("\n[docs] Indexing documentation...")
            total += index_files(docs_client, "docs", DOC_PATTERNS, reindex=args.reindex)
        else:
            print(f"Error: Docs server at {args.docs_url} not reachable, skipping docs index", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"\nDone. {total} total documents indexed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
