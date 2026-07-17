#!/usr/bin/env python3
"""Fixture tests for scripts/kb_rag/autowiki_writer.py.

Stdlib unittest (the research repo has no pytest under .venv). Builds a tiny
synthetic KB-RAG catalog.sqlite fixture, runs the deterministic writer, and
asserts:

* the section pages contain the expected H1 / Summary / Sections / Sources
  structure and per-chunk source citations,
* the INDEX clusters + cross-links the pages,
* output is byte-stable across two runs and idempotent on re-run,
* the JSON-export loader yields byte-identical pages to the SQLite loader,
* the --evidence-policy / cutoff knobs drop the right pages.

Run: .venv/bin/python scripts/kb_rag/tests/test_autowiki_writer.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

# Import the module under test from its parent dir (scripts/kb_rag/).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import autowiki_writer as aw  # noqa: E402

# Mirrors src/retrieval/kb_rag.py::_CATALOG_SCHEMA (the real KB-RAG catalog).
_CATALOG_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunk (
  chunk_id INTEGER PRIMARY KEY,
  file_path TEXT NOT NULL,
  heading_path TEXT NOT NULL,
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  content_hash TEXT NOT NULL,
  mtime REAL NOT NULL,
  emb_path TEXT NOT NULL,
  text_preview TEXT,
  token_count INTEGER NOT NULL DEFAULT 0
);
"""

# (file_path, heading_path, line_start, line_end, content_hash, text_preview)
_FIXTURE_ROWS = [
    # File A: 4 chunks -> 4 citations (survives --evidence-policy verified).
    ("/kb/handoffs/active/alpha-doc.md", ["Alpha Doc", "Overview"], 1, 10,
     "hashA1", "Overview line one\nOverview line two"),
    ("/kb/handoffs/active/alpha-doc.md", ["Alpha Doc", "Architecture"], 11, 20,
     "hashA2", "Alpha architecture preview."),
    ("/kb/handoffs/active/alpha-doc.md", ["Alpha Doc", "API"], 21, 30,
     "hashA3", "Alpha API preview."),
    ("/kb/handoffs/active/alpha-doc.md", ["Alpha Doc", "Setup"], 31, 40,
     "hashA4", "Alpha setup preview."),
    # File B: 2 chunks -> 2 citations (dropped under --evidence-policy verified).
    ("/kb/wiki/beta-notes.md", ["Beta Notes", "Intro"], 1, 5,
     "hashB1", "Beta intro preview."),
    ("/kb/wiki/beta-notes.md", ["Beta Notes", "Details"], 6, 12,
     "hashB2", "Beta details preview."),
]

_ALPHA_SLUG = "active-alpha-doc"
_BETA_SLUG = "wiki-beta-notes"


def _build_catalog(index_dir: Path) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    catalog = index_dir / "catalog.sqlite"
    conn = sqlite3.connect(str(catalog))
    conn.executescript(_CATALOG_SCHEMA)
    conn.executemany(
        "INSERT INTO chunk (file_path, heading_path, line_start, line_end, "
        "content_hash, mtime, emb_path, text_preview, token_count) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (fp, json.dumps(hp), ls, le, ch, 1_700_000_000.0,
             f"emb/{ch}.npz", preview, 42)
            for (fp, hp, ls, le, ch, preview) in _FIXTURE_ROWS
        ],
    )
    conn.commit()
    conn.close()
    return catalog


def _read_tree(root: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(root.glob("*.md"))}


class TestAutoWikiWriter(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.index_dir = self.tmp / "index"
        _build_catalog(self.index_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # -- structure / citations ------------------------------------------- #

    def test_section_pages_structure_and_citations(self) -> None:
        out = self.tmp / "wiki_out"
        summary = aw.generate_wiki(self.index_dir, out)

        self.assertTrue(summary["ok"])
        self.assertEqual(summary["chunks_read"], 6)
        self.assertEqual(summary["pages_written"], 2)
        self.assertEqual(summary["pages_dropped"], 0)
        self.assertEqual(summary["clusters"], ["active", "wiki"])

        alpha = (out / f"{_ALPHA_SLUG}.md").read_text(encoding="utf-8")
        # H1 + required sections (satisfies wiki_article_structure lint shape).
        self.assertTrue(alpha.startswith("# Alpha Doc\n"))
        self.assertIn("## Summary", alpha)
        self.assertIn("## Sections", alpha)
        self.assertIn("## Sources", alpha)
        # Heading breadcrumbs as subsections.
        self.assertIn("### Alpha Doc > Overview", alpha)
        self.assertIn("### Alpha Doc > Setup", alpha)
        # Multi-line preview is blockquoted line-by-line.
        self.assertIn("> Overview line one", alpha)
        self.assertIn("> Overview line two", alpha)
        # Per-chunk source citation (path:line_start-line_end (hash)).
        self.assertIn(
            "Source: `/kb/handoffs/active/alpha-doc.md:1-10` (`hashA1`)", alpha
        )
        # Sources section lists all four distinct citations.
        sources_block = alpha.split("## Sources", 1)[1]
        for h in ("hashA1", "hashA2", "hashA3", "hashA4"):
            self.assertIn(h, sources_block)

    def test_index_clusters_and_crosslinks(self) -> None:
        out = self.tmp / "wiki_out"
        aw.generate_wiki(self.index_dir, out)
        index = (out / "INDEX.md").read_text(encoding="utf-8")

        self.assertTrue(index.startswith("# KB AutoWiki Index\n"))
        self.assertIn("## Summary", index)
        self.assertIn("## Clusters", index)
        self.assertIn("## Sources", index)
        # Clusters, ordered active < wiki.
        self.assertIn("### active", index)
        self.assertIn("### wiki", index)
        self.assertLess(index.index("### active"), index.index("### wiki"))
        # Cross-links to both section pages.
        self.assertIn(f"[Alpha Doc]({_ALPHA_SLUG}.md)", index)
        self.assertIn(f"[Beta Notes]({_BETA_SLUG}.md)", index)

    # -- determinism / idempotency --------------------------------------- #

    def test_byte_stable_across_two_runs(self) -> None:
        out_a = self.tmp / "run_a"
        out_b = self.tmp / "run_b"
        aw.generate_wiki(self.index_dir, out_a)
        aw.generate_wiki(self.index_dir, out_b)
        self.assertEqual(_read_tree(out_a), _read_tree(out_b))
        self.assertEqual(len(_read_tree(out_a)), 3)  # INDEX + 2 pages

    def test_idempotent_rewrite_same_dir(self) -> None:
        out = self.tmp / "wiki_out"
        aw.generate_wiki(self.index_dir, out)
        first = _read_tree(out)
        aw.generate_wiki(self.index_dir, out)
        self.assertEqual(first, _read_tree(out))

    def test_json_export_matches_sqlite(self) -> None:
        # Same records via a JSON export -> byte-identical pages.
        export = [
            {"file_path": fp, "heading_path": hp, "line_start": ls,
             "line_end": le, "content_hash": ch, "text_preview": preview}
            for (fp, hp, ls, le, ch, preview) in _FIXTURE_ROWS
        ]
        json_path = self.tmp / "export.json"
        json_path.write_text(json.dumps(export), encoding="utf-8")

        out_sql = self.tmp / "from_sql"
        out_json = self.tmp / "from_json"
        aw.generate_wiki(self.index_dir, out_sql)
        aw.generate_wiki(json_path, out_json)

        # Section pages are byte-identical; only INDEX differs (source label).
        self.assertEqual(
            (out_sql / f"{_ALPHA_SLUG}.md").read_bytes(),
            (out_json / f"{_ALPHA_SLUG}.md").read_bytes(),
        )
        self.assertEqual(
            (out_sql / f"{_BETA_SLUG}.md").read_bytes(),
            (out_json / f"{_BETA_SLUG}.md").read_bytes(),
        )

    # -- evidence policy / cutoffs --------------------------------------- #

    def test_evidence_policy_verified_drops_thin_page(self) -> None:
        out = self.tmp / "verified"
        summary = aw.generate_wiki(self.index_dir, out, min_citations=3)
        self.assertEqual(summary["pages_written"], 1)
        self.assertEqual(summary["pages_dropped"], 1)
        self.assertFalse((out / f"{_BETA_SLUG}.md").exists())
        self.assertTrue((out / f"{_ALPHA_SLUG}.md").exists())
        dropped = summary["dropped"][0]
        self.assertEqual(dropped["reason"], "min_citations")
        self.assertEqual(dropped["citations"], 2)

    def test_top_k_sections_cutoff(self) -> None:
        out = self.tmp / "topk"
        summary = aw.generate_wiki(self.index_dir, out, top_k_sections=2)
        alpha_page = next(p for p in summary["pages"] if p["slug"] == _ALPHA_SLUG)
        self.assertEqual(alpha_page["sections"], 2)
        alpha = (out / f"{_ALPHA_SLUG}.md").read_text(encoding="utf-8")
        self.assertIn("### Alpha Doc > Overview", alpha)
        self.assertNotIn("### Alpha Doc > Setup", alpha)

    def test_dry_run_writes_nothing(self) -> None:
        out = self.tmp / "dry"
        summary = aw.generate_wiki(self.index_dir, out, dry_run=True)
        self.assertTrue(summary["dry_run"])
        self.assertEqual(summary["pages_written"], 2)
        self.assertFalse(out.exists())

    def test_cli_main_smoke(self) -> None:
        out = self.tmp / "cli_out"
        rc = aw.main(
            [
                "--index-dir", str(self.index_dir),
                "--output-dir", str(out),
                "--evidence-policy", "verified",
            ]
        )
        self.assertEqual(rc, 0)
        self.assertTrue((out / "INDEX.md").exists())
        self.assertTrue((out / f"{_ALPHA_SLUG}.md").exists())
        self.assertFalse((out / f"{_BETA_SLUG}.md").exists())


if __name__ == "__main__":
    unittest.main()
