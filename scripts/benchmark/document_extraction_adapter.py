#!/usr/bin/env python3
"""Document extraction benchmark adapter for OpenDataLoader-bench.

Provides NID, TEDS, and MHS scoring metrics for evaluating PDF extraction
quality against ground-truth markdown annotations.

Dataset: OpenDataLoader-bench 200-PDF corpus (Apache-2.0, ~200 real-world PDFs).
Dataset paths (PIP-05 collision resolution, 2026-08-25 — always cite full paths):
  * canonical OpenDataLoader-bench clone (repo + corpus):
    /mnt/raid0/llm/opendataloader-bench-upstream
  * /mnt/raid0/llm/omnidocbench is the OmniDocBench clone (renamed from
    opendataloader-bench); it also carries a 2026-07-21 copy of the 200-PDF
    corpus in {pdfs,ground-truth}/ which this adapter reads by default. The
    clone command below is informational only — do NOT clone into
    /mnt/raid0/llm/omnidocbench (that path is OmniDocBench's).

Metrics:
  NID (Normalized Information Distance): Reading order quality
  TEDS (Tree Edit Distance Similarity): Table structure accuracy
  MHS (Markdown Heading Similarity): Heading hierarchy accuracy

Usage:
    from document_extraction_adapter import (
        DocumentExtractionAdapter,
        score_nid, score_teds, score_mhs,
    )

    adapter = DocumentExtractionAdapter()
    problems = adapter.load_all()  # Returns (pdf_path, ground_truth_md) pairs

    score = score_nid(extracted_text, ground_truth)
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Default corpus root this adapter reads (a copy of the OpenDataLoader-bench
# 200-PDF corpus fetched 2026-07-21 into the OmniDocBench clone's directory).
# Reference: https://github.com/opendataloader-project/opendataloader-bench
ODL_BENCH_DIR = Path("/mnt/raid0/llm/omnidocbench")


@dataclass
class DocumentProblem:
    """A single document extraction problem."""

    id: str
    pdf_path: Path
    ground_truth: str  # Markdown ground truth
    category: str = ""  # e.g., "academic", "report", "form", "mixed"


class DocumentExtractionAdapter:
    """Adapter for the OpenDataLoader-bench corpus.

    Expected layout (the 2026-07-21 corpus copy inside the omnidocbench clone):
        <bench_dir>/
        ├── pdfs/           # Source PDF files (200)
        ├── ground-truth/   # Markdown ground truth (same basename as PDF)
        └── metadata.json   # Optional: per-document metadata

    Args:
        bench_dir: Path to a directory holding the {pdfs,ground-truth} corpus.
    """

    def __init__(self, bench_dir: Path = ODL_BENCH_DIR):
        self.bench_dir = bench_dir
        self.pdf_dir = bench_dir / "pdfs"
        self.gt_dir = bench_dir / "ground-truth"

    def is_available(self) -> bool:
        """Check if the benchmark dataset is cloned and accessible."""
        return self.pdf_dir.is_dir() and self.gt_dir.is_dir()

    def load_all(self) -> list[DocumentProblem]:
        """Load all document extraction problems.

        OpenDataLoader-bench uses local PDFs + ground-truth markdown.
        Returns:
            List of DocumentProblem with pdf_path (PDF) and ground_truth.
        """
        if not self.is_available():
            return []

        problems = []
        for gt_path in sorted(self.gt_dir.glob("*.md")):
            pdf_path = self.pdf_dir / f"{gt_path.stem}.pdf"
            if not pdf_path.exists():
                continue

            problems.append(DocumentProblem(
                id=gt_path.stem,
                pdf_path=pdf_path,
                ground_truth=gt_path.read_text(encoding="utf-8"),
            ))

        return problems

    def sample(self, n: int = 20, seed: int = 42) -> list[DocumentProblem]:
        """Sample n random problems."""
        import random
        all_problems = self.load_all()
        if not all_problems:
            return []
        rng = random.Random(seed)
        return rng.sample(all_problems, min(n, len(all_problems)))


class DocumentExtractionDatasetAdapter(DocumentExtractionAdapter):
    """Benchmark-suite adapter that emits prompt dictionaries.

    The base adapter intentionally exposes local PDF + ground-truth objects for
    document runners. The generic benchmark registry expects dict prompts, so
    this wrapper keeps that compatibility layer explicit.
    """

    suite_name = "document_extraction"
    has_real_tiers = False

    @property
    def total_available(self) -> int:
        return len(self.load_all())

    def _problem_to_prompt(self, problem: DocumentProblem) -> dict[str, Any]:
        return {
            "id": problem.id,
            "suite": self.suite_name,
            "name": problem.id,
            "prompt": (
                "Extract the content of the PDF at "
                f"{problem.pdf_path} as Markdown. Preserve reading order, "
                "tables, and heading hierarchy."
            ),
            "expected": problem.ground_truth,
            "scoring_method": "document_extraction",
            "scoring_config": {
                "pdf_path": str(problem.pdf_path),
                "metrics": ["nid", "teds", "mhs", "aggregate"],
            },
            "tier": 2,
        }

    def extract_all(self) -> list[dict[str, Any]]:
        return [self._problem_to_prompt(problem) for problem in self.load_all()]

    def sample(
        self,
        n: int = 20,
        seed: int = 42,
        stratify: bool = False,
    ) -> list[dict[str, Any]]:
        del stratify
        return [self._problem_to_prompt(problem) for problem in super().sample(n=n, seed=seed)]


# ── Scoring Metrics ────────────────────────────────────────────


def score_nid(extracted: str, ground_truth: str) -> float:
    """Normalized Information Distance (NID) — reading order quality.

    Computes the edit distance between the token sequences of extracted
    and ground-truth text, normalized by the maximum sequence length.
    Lower NID = better reading order preservation.

    Returns:
        NID score in [0.0, 1.0]. 0.0 = perfect match, 1.0 = completely different.
    """
    if not extracted and not ground_truth:
        return 0.0
    if not extracted or not ground_truth:
        return 1.0

    # Tokenize by whitespace-separated words
    tokens_ext = extracted.split()
    tokens_gt = ground_truth.split()

    # SequenceMatcher computes the ratio of matching tokens
    matcher = difflib.SequenceMatcher(None, tokens_ext, tokens_gt)
    similarity = matcher.ratio()

    return 1.0 - similarity


def score_teds(extracted: str, ground_truth: str) -> float:
    """Tree Edit Distance Similarity (TEDS) — table structure accuracy.

    Simplified TEDS: compares markdown table structures by parsing
    table rows and columns, computing edit distance on the table tree.

    For documents without tables, returns 1.0 (perfect — nothing to compare).

    Returns:
        TEDS score in [0.0, 1.0]. 1.0 = perfect table structure match.
    """
    gt_tables = _extract_markdown_tables(ground_truth)
    ext_tables = _extract_markdown_tables(extracted)

    if not gt_tables:
        return 1.0  # No tables to evaluate

    if not ext_tables:
        return 0.0  # Ground truth has tables but extraction found none

    # Compare tables pairwise (greedy matching by position)
    total_score = 0.0
    n_compared = min(len(gt_tables), len(ext_tables))

    for i in range(n_compared):
        gt_rows = gt_tables[i]
        ext_rows = ext_tables[i]

        # Compare row-by-row structure
        matcher = difflib.SequenceMatcher(None, gt_rows, ext_rows)
        total_score += matcher.ratio()

    # Penalize for missing or extra tables
    total_tables = max(len(gt_tables), len(ext_tables))
    return total_score / total_tables if total_tables > 0 else 1.0


def score_mhs(extracted: str, ground_truth: str) -> float:
    """Markdown Heading Similarity (MHS) — heading hierarchy accuracy.

    Compares the heading structure (ATX headings: # through ######)
    between extracted and ground-truth markdown.

    Returns:
        MHS score in [0.0, 1.0]. 1.0 = perfect heading match.
    """
    gt_headings = _extract_headings(ground_truth)
    ext_headings = _extract_headings(extracted)

    if not gt_headings:
        return 1.0  # No headings to evaluate

    if not ext_headings:
        return 0.0  # Ground truth has headings but extraction found none

    # Compare heading sequences
    matcher = difflib.SequenceMatcher(None, gt_headings, ext_headings)
    return matcher.ratio()


def score_document(extracted: str, ground_truth: str) -> dict[str, float]:
    """Compute all document extraction metrics.

    Returns:
        Dict with NID, TEDS, MHS scores and an aggregate.
    """
    nid = score_nid(extracted, ground_truth)
    teds = score_teds(extracted, ground_truth)
    mhs = score_mhs(extracted, ground_truth)

    # Aggregate: weighted average (reading order most important)
    aggregate = 0.5 * (1.0 - nid) + 0.3 * teds + 0.2 * mhs

    return {
        "nid": round(nid, 4),
        "teds": round(teds, 4),
        "mhs": round(mhs, 4),
        "aggregate": round(aggregate, 4),
    }


# ── Internal Helpers ───────────────────────────────────────────


def _extract_markdown_tables(text: str) -> list[list[str]]:
    """Extract markdown tables as lists of row strings."""
    tables = []
    current_table: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            # Skip separator rows (e.g., |---|---|)
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue
            current_table.append(stripped)
        else:
            if current_table:
                tables.append(current_table)
                current_table = []

    if current_table:
        tables.append(current_table)

    return tables


def _extract_headings(text: str) -> list[str]:
    """Extract ATX headings as (level, title) strings."""
    headings = []
    for line in text.splitlines():
        m = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            headings.append(f"h{level}:{title}")
    return headings


if __name__ == "__main__":
    adapter = DocumentExtractionAdapter()

    if not adapter.is_available():
        print(f"Dataset not found at {ODL_BENCH_DIR}")
        print(
            "Clone with: git clone "
            "https://github.com/opendataloader-project/opendataloader-bench "
            f"{ODL_BENCH_DIR}"
        )
    else:
        problems = adapter.load_all()
        print(f"Loaded {len(problems)} document extraction problems")

        if problems:
            p = problems[0]
            print(f"\nSample: {p.id}")
            print(f"  PDF: {p.pdf_path}")
            print(f"  Ground truth: {len(p.ground_truth)} chars")
            print(f"  Preview: {p.ground_truth[:200]}...")

    # Demo scoring
    print("\n=== Scoring Demo ===")
    gt = "# Introduction\n\nThis is a test document.\n\n## Methods\n\nWe used approach X."
    ext = "# Introduction\n\nThis is a test document.\n\n## Methods\n\nWe used approach Y."
    scores = score_document(ext, gt)
    print(f"Scores: {scores}")
