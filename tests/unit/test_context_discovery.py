"""Unit tests for src/context_discovery.py (DCP-2 discovery/cost/assemble + DCP-3 codemap)."""

from __future__ import annotations

import hashlib
import textwrap

import pytest

from src.context_assembly import BundleEntry, ContextBundle, InclusionMode, LineRange
from src.context_discovery import (
    DiscoveredHit,
    parse_colgrep_json,
    discover_candidates,
    build_python_codemap,
    cost_candidates,
    assemble_delegation_bundle,
    render_bundle,
)


# ─── ColGREP JSON parsing ────────────────────────────────────────────────────────


def test_parse_colgrep_json_variants() -> None:
    payload = [
        {"path": "a.py", "start_line": 10, "end_line": 20, "score": 0.9},
        {"file": "b.py", "start": 5, "end": 5, "relevance": 0.5},  # alt field names
        {"nope": 1},  # no path → skipped
    ]
    hits = parse_colgrep_json(payload)
    assert [h.path for h in hits] == ["a.py", "b.py"]
    assert hits[0].line_ranges[0].start == 10 and hits[0].score == 0.9
    assert hits[1].line_ranges[0].end == 5


def test_parse_colgrep_json_string_and_garbage() -> None:
    assert parse_colgrep_json("not json") == []
    assert (
        parse_colgrep_json('[{"path": "x.py", "start_line": 1, "end_line": 2, "score": 0.3}]')[
            0
        ].path
        == "x.py"
    )
    assert parse_colgrep_json({"results": [{"path": "y.py"}]})[0].path == "y.py"  # nested


# ─── discovery (pass 1) ──────────────────────────────────────────────────────────


def test_discover_groups_merges_ranks_and_excludes() -> None:
    def fake_search(query, limit):
        return [
            DiscoveredHit("a.py", [LineRange(1, 5)], 0.4),
            DiscoveredHit("a.py", [LineRange(4, 9)], 0.8),  # same file → merge ranges, max score
            DiscoveredHit("node_modules/x.js", [LineRange(1, 2)], 0.99),  # policy-excluded
            DiscoveredHit("b.py", [LineRange(10, 12)], 0.6),
        ]

    hits = discover_candidates("q", code_search_fn=fake_search, max_files=8)
    assert [h.path for h in hits] == ["a.py", "b.py"]  # node_modules dropped; ranked by score
    a = next(h for h in hits if h.path == "a.py")
    assert [(r.start, r.end) for r in a.line_ranges] == [(1, 9)]  # merged
    assert a.score == 0.8


def test_discover_respects_max_files() -> None:
    def fake_search(q, limit):
        return [DiscoveredHit(f"f{i}.py", [LineRange(1, 1)], float(i)) for i in range(10)]

    hits = discover_candidates("q", code_search_fn=fake_search, max_files=3)
    assert len(hits) == 3
    assert [h.path for h in hits] == ["f9.py", "f8.py", "f7.py"]  # top-3 by score


# ─── DCP-3 codemap ───────────────────────────────────────────────────────────────


def test_python_codemap_signatures_only() -> None:
    src = textwrap.dedent('''
        import os

        def top(a: int, b: str = "x") -> bool:
            """Top-level fn docstring.
            second line ignored."""
            return True

        class Foo(Base):
            """Foo does things."""
            def method(self, n) -> None:
                secret = 42
                return None
    ''')
    cm = build_python_codemap(src)
    assert "def top(a: int, b: str='x') -> bool: ..." in cm
    assert "# Top-level fn docstring." in cm
    assert "class Foo(Base):" in cm
    assert "def method(self, n) -> None: ..." in cm
    # bodies are NOT included
    assert "secret = 42" not in cm
    assert "return True" not in cm


def test_python_codemap_syntax_error_returns_none() -> None:
    assert build_python_codemap("def broken(:\n") is None


def test_python_codemap_empty_module_returns_none() -> None:
    assert build_python_codemap("x = 1\n") is None  # no classes/functions


# ─── cost (pass 2) ───────────────────────────────────────────────────────────────


def test_cost_candidates_computes_modes() -> None:
    files = {
        "a.py": "def f():\n    return 1\n" + ("# pad\n" * 50),  # big-ish python
        "b.txt": "x" * 700,  # non-python → no codemap
    }
    hits = [
        DiscoveredHit("a.py", [LineRange(1, 2)], 0.9),
        DiscoveredHit("b.txt", [], 0.5),
    ]
    cands = cost_candidates(hits, file_reader_fn=lambda p: files[p])
    a = next(c for c in cands if c.path == "a.py")
    b = next(c for c in cands if c.path == "b.txt")
    assert a.desired_mode == InclusionMode.SLICES  # had ranges
    assert a.content_sha256 is not None and len(a.content_sha256) == 64
    assert a.cost_codemap < a.cost_full  # codemap cheaper than full body
    assert a.cost_slices < a.cost_full  # 2 lines < whole file
    assert b.desired_mode == InclusionMode.FULL  # no ranges
    assert b.content_sha256 is not None and len(b.content_sha256) == 64
    assert b.priority == 0.5


def test_cost_candidates_skips_unreadable() -> None:
    def reader(p):
        raise FileNotFoundError(p)

    assert cost_candidates([DiscoveredHit("gone.py", [], 0.5)], file_reader_fn=reader) == []


# ─── end-to-end assemble ─────────────────────────────────────────────────────────


def test_assemble_delegation_bundle_end_to_end() -> None:
    files = {
        "hot.py": "def hot():\n" + ("    x = 1\n" * 40),
        "warm.py": "def warm():\n" + ("    y = 2\n" * 40),
    }

    def fake_search(query, limit):
        return [
            DiscoveredHit("hot.py", [LineRange(1, 2)], 0.9),
            DiscoveredHit("warm.py", [LineRange(1, 2)], 0.3),
        ]

    bundle = assemble_delegation_bundle(
        "fix hot path",
        budget=10_000,
        code_search_fn=fake_search,
        file_reader_fn=lambda p: files[p],
        bundle_id="b1",
    )
    assert bundle.bundle_id == "b1"
    assert bundle.fits()
    paths = {e.path for e in bundle.included()}
    assert paths == {"hot.py", "warm.py"}
    m = bundle.manifest()
    assert m["total_tokens"] <= 10_000
    # hot.py (higher score) is packed; manifest carries per-entry provenance
    hot = next(e for e in m["entries"] if e["path"] == "hot.py")
    assert hot["source"] == "colgrep"
    assert hot["mode"] in ("slices", "full")
    assert hot["content_sha256"] is not None
    assert len(hot["content_sha256"]) == 64


def test_assemble_tight_budget_downgrades_or_excludes() -> None:
    files = {"a.py": "def a():\n" + ("    x=1\n" * 200)}  # large

    def fake_search(q, limit):
        return [DiscoveredHit("a.py", [LineRange(1, 1)], 0.9)]

    bundle = assemble_delegation_bundle(
        "q", budget=5, code_search_fn=fake_search, file_reader_fn=lambda p: files[p]
    )
    # budget of 5 tokens → can't fit full; ends up sliced/codemap or excluded — but never overflows
    assert bundle.total_tokens() <= 5


# ─── DCP-4 render identity ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "mode",
    [InclusionMode.FULL, InclusionMode.SLICES, InclusionMode.CODEMAP_ONLY],
)
def test_render_rejects_changed_bound_body_before_mode_processing(mode) -> None:
    planned_body = "def planned():\n    return 'old'\n"
    changed_body = "def changed():\n    return 'new'\n"
    bundle = ContextBundle(budget=1000)
    bundle.add_entry(
        BundleEntry(
            path="module.py",
            mode=mode,
            line_ranges=[LineRange(1, 2)] if mode == InclusionMode.SLICES else [],
            content_sha256=hashlib.sha256(planned_body.encode("utf-8")).hexdigest(),
        )
    )
    codemap_calls = []

    with pytest.raises(ValueError, match="content changed after context planning"):
        render_bundle(
            bundle,
            file_reader_fn=lambda _path: changed_body,
            codemap_fn=lambda body: codemap_calls.append(body) or "stale codemap",
        )
    assert codemap_calls == []


def test_render_keeps_same_bound_and_unbound_entries_compatible() -> None:
    body = "def current():\n    return 'same'\n"
    bundle = ContextBundle(budget=1000)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    bundle.add_entry(BundleEntry(path="bound.py", mode=InclusionMode.FULL, content_sha256=digest))
    bundle.add_entry(BundleEntry(path="unbound.py", mode=InclusionMode.FULL))

    rendered = render_bundle(bundle, file_reader_fn=lambda _path: body)

    assert "### bound.py (full)\n" + body in rendered
    assert "### unbound.py (full)\n" + body in rendered


@pytest.mark.parametrize("unreadable", ["none", "exception"])
def test_render_keeps_unreadable_bound_entries_as_skips(unreadable) -> None:
    bundle = ContextBundle(budget=1000)
    bundle.add_entry(
        BundleEntry(path="gone.py", mode=InclusionMode.FULL, content_sha256="0" * 64)
    )

    def reader(_path):
        if unreadable == "exception":
            raise OSError("unreadable")
        return None

    assert render_bundle(bundle, file_reader_fn=reader) == ""
