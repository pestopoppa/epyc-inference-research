#!/usr/bin/env python3
"""Download AA-LCR source documents and extract text for benchmarking.

Fetches PDFs/HTML from AA-LCR dataset source URLs, extracts text via
pdf_router (born-digital) or curl+BeautifulSoup (HTML), and caches
results as individual text files keyed by URL hash.

Produces a ready-to-use JSONL where each row has the full document
context concatenated with the question.

Usage:
    python download_aa_lcr.py [--force] [--max-concurrent 4]

Cache directory: /mnt/raid0/llm/data/eval/aa_lcr/
Output JSONL:    /mnt/raid0/llm/data/eval/aa_lcr/aa_lcr.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

CACHE_DIR = Path("/mnt/raid0/llm/data/eval/aa_lcr/docs")
OUTPUT_DIR = Path("/mnt/raid0/llm/data/eval/aa_lcr")
JSONL_PATH = OUTPUT_DIR / "aa_lcr.jsonl"


def url_hash(url: str) -> str:
    """Deterministic short hash for cache filenames."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download a URL to a local file via curl with HTTP/2 then HTTP/1.1 fallback."""
    ua = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)"
    for extra_flags in [[], ["--http1.1"]]:
        try:
            result = subprocess.run(
                [
                    "curl", "-fsSL",
                    "--max-time", str(timeout),
                    "--retry", "2",
                    "-o", str(dest),
                    "-H", f"User-Agent: {ua}",
                    *extra_flags,
                    url,
                ],
                capture_output=True,
                timeout=timeout + 10,
            )
            if result.returncode == 0 and dest.stat().st_size > 0:
                return True
        except Exception:
            continue
    print(f"    [download] FAILED {url[:80]}")
    return False


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract text from PDF using pdf_router."""
    # Try importing pdf_router from orchestrator
    sys.path.insert(0, "/mnt/raid0/llm/epyc-orchestrator/src/services")
    try:
        from pdf_router import PDFRouter
        router = PDFRouter()
        result = router.extract_sync(str(pdf_path), extract_figures=False)
        return result.text.strip()
    except Exception:
        pass

    # Fallback: pdftotext directly
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), "-"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    # Fallback: PyMuPDF
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n\n".join(pages).strip()
    except Exception as e:
        print(f"    [extract] All methods failed for {pdf_path.name}: {e}")
        return ""


def extract_html_text(html_path: Path) -> str:
    """Extract text from HTML file."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        # Fallback: crude tag stripping
        import re
        text = html_path.read_text(errors="replace")
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        return " ".join(text.split()).strip()

    text = html_path.read_text(errors="replace")
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def is_pdf(path: Path) -> bool:
    """Check if file starts with PDF magic bytes."""
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def is_html(path: Path) -> bool:
    """Check if file looks like HTML."""
    try:
        with open(path, "rb") as f:
            head = f.read(512).lower()
            return b"<html" in head or b"<!doctype" in head or b"<head" in head
    except Exception:
        return False


def process_url(url: str, force: bool = False) -> str:
    """Download and extract text from a single URL. Returns cached text."""
    url = url.strip()
    if not url:
        return ""

    cache_file = CACHE_DIR / f"{url_hash(url)}.txt"

    # Check cache
    if cache_file.exists() and not force:
        text = cache_file.read_text()
        if text:
            return text

    print(f"  Processing: {url[:100]}...")

    # Handle ZIP files
    if url.lower().endswith(".zip"):
        return _process_zip(url, cache_file)

    # Download to temp file
    with tempfile.NamedTemporaryFile(suffix=".download", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        if not download_file(url, tmp_path):
            cache_file.write_text("")
            return ""

        # Detect content type and extract
        if is_pdf(tmp_path):
            text = extract_pdf_text(tmp_path)
        elif is_html(tmp_path):
            text = extract_html_text(tmp_path)
        else:
            # Try as PDF first (arxiv URLs don't have .pdf extension but serve PDF)
            text = extract_pdf_text(tmp_path)
            if not text:
                # Try as plain text
                try:
                    text = tmp_path.read_text(errors="replace").strip()
                except Exception:
                    text = ""

        cache_file.write_text(text)
        return text

    finally:
        tmp_path.unlink(missing_ok=True)


def _process_zip(url: str, cache_file: Path) -> str:
    """Download ZIP, extract PDFs inside, concatenate text."""
    import zipfile as zf

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "archive.zip"
        if not download_file(url, zip_path, timeout=120):
            cache_file.write_text("")
            return ""

        try:
            with zf.ZipFile(zip_path) as z:
                z.extractall(tmpdir)
        except Exception as e:
            print(f"    [zip] Extract failed: {e}")
            cache_file.write_text("")
            return ""

        # Process all PDFs found inside
        texts = []
        for root, _, files in os.walk(tmpdir):
            for fname in sorted(files):
                fpath = Path(root) / fname
                if fpath.suffix.lower() == ".pdf":
                    t = extract_pdf_text(fpath)
                    if t:
                        texts.append(f"--- {fname} ---\n{t}")

        text = "\n\n".join(texts)
        cache_file.write_text(text)
        return text


def build_jsonl(force: bool = False):
    """Build the final JSONL from cached document texts + AA-LCR questions."""
    import datasets as hf

    print("\nLoading AA-LCR dataset...")
    ds = hf.load_dataset("ArtificialAnalysis/AA-LCR", split="test")

    # Collect all unique URLs and process them
    all_urls = set()
    for row in ds:
        for url in row["data_source_urls"].split(";"):
            url = url.strip()
            if url:
                all_urls.add(url)

    print(f"Total unique URLs: {len(all_urls)}")

    # Process each URL (cached ones will be fast)
    failed = 0
    for i, url in enumerate(sorted(all_urls)):
        cache_file = CACHE_DIR / f"{url_hash(url)}.txt"
        if cache_file.exists() and not force:
            continue
        text = process_url(url, force=force)
        if not text:
            failed += 1
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(all_urls)} URLs processed")

    if failed:
        print(f"\n  WARNING: {failed}/{len(all_urls)} URLs failed to extract")

    # Build JSONL: concatenate document texts per question
    print("\nBuilding JSONL...")
    rows_written = 0
    rows_skipped = 0

    with open(JSONL_PATH, "w") as f:
        for row in ds:
            urls = [u.strip() for u in row["data_source_urls"].split(";") if u.strip()]

            # Concatenate document texts
            doc_texts = []
            for url in urls:
                cache_file = CACHE_DIR / f"{url_hash(url)}.txt"
                if cache_file.exists():
                    text = cache_file.read_text()
                    if text:
                        doc_texts.append(text)

            if not doc_texts:
                rows_skipped += 1
                continue

            context = "\n\n---\n\n".join(doc_texts)

            entry = {
                "id": f"aa_lcr_{row['document_category']}_{row['question_id']:03d}",
                "document_set_id": row["document_set_id"],
                "document_category": row["document_category"],
                "question": row["question"],
                "answer": row["answer"],
                "context": context,
                "context_tokens_expected": row["input_tokens"],
                "num_source_docs": len(urls),
                "num_docs_extracted": len(doc_texts),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            rows_written += 1

    print(f"\nDone: {rows_written} questions written to {JSONL_PATH}")
    if rows_skipped:
        print(f"  Skipped {rows_skipped} questions (no document text available)")

    # Write metadata
    meta = {
        "dataset": "ArtificialAnalysis/AA-LCR",
        "total_questions": len(ds),
        "questions_with_context": rows_written,
        "questions_skipped": rows_skipped,
        "unique_urls": len(all_urls),
        "urls_failed": failed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    meta_path = OUTPUT_DIR / "metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata written to {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Download AA-LCR documents")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and re-extract all documents")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    build_jsonl(force=args.force)


if __name__ == "__main__":
    main()
