#!/usr/bin/env python3
"""Tests for document extraction scoring metrics (NID, TEDS, MHS)."""

from document_extraction_adapter import (
    score_nid,
    score_teds,
    score_mhs,
    score_document,
    _extract_markdown_tables,
    _extract_headings,
)


class TestScoreNID:
    def test_identical_text(self):
        assert score_nid("hello world", "hello world") == 0.0

    def test_completely_different(self):
        assert score_nid("aaa bbb ccc", "xxx yyy zzz") > 0.9

    def test_partial_overlap(self):
        score = score_nid("the quick brown fox", "the slow brown fox")
        assert 0.0 < score < 1.0

    def test_empty_both(self):
        assert score_nid("", "") == 0.0

    def test_one_empty(self):
        assert score_nid("hello", "") == 1.0

    def test_reordered_worse(self):
        gt = "first second third fourth"
        assert score_nid("first second third fourth", gt) < score_nid("fourth third second first", gt)


class TestScoreTEDS:
    def test_no_tables(self):
        assert score_teds("no tables", "no tables") == 1.0

    def test_identical_tables(self):
        t = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert score_teds(t, t) == 1.0

    def test_gt_has_tables_ext_doesnt(self):
        assert score_teds("no tables", "| A | B |\n|---|---|\n| 1 | 2 |") == 0.0

    def test_partial_match(self):
        gt = "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
        ext = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert 0.0 < score_teds(ext, gt) < 1.0


class TestScoreMHS:
    def test_identical(self):
        t = "# Title\n## Section"
        assert score_mhs(t, t) == 1.0

    def test_no_headings(self):
        assert score_mhs("plain", "plain") == 1.0

    def test_missing_headings(self):
        assert score_mhs("plain", "# Title\n## Methods") == 0.0

    def test_wrong_level(self):
        assert score_mhs("## Title", "# Title") < 1.0


class TestScoreDocument:
    def test_perfect(self):
        t = "# Title\nContent"
        s = score_document(t, t)
        assert s["nid"] == 0.0
        assert s["aggregate"] > 0.9

    def test_all_keys(self):
        s = score_document("a", "b")
        assert set(s.keys()) == {"nid", "teds", "mhs", "aggregate"}


class TestHelpers:
    def test_extract_headings(self):
        assert _extract_headings("# A\n## B") == ["h1:A", "h2:B"]

    def test_extract_tables(self):
        t = "| A | B |\n|---|---|\n| 1 | 2 |"
        tables = _extract_markdown_tables(t)
        assert len(tables) == 1
        assert "| A | B |" in tables[0]


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
