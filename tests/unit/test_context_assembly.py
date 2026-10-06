"""Unit tests for src/context_assembly.py (DCP-1 context bundle data model)."""

from __future__ import annotations

import pytest

from src.context_assembly import (
    InclusionMode,
    SourceKind,
    LineRange,
    merge_line_ranges,
    conservative_char_estimator,
    default_exclusion_reason,
    BundleEntry,
    BudgetBands,
    ContextBundle,
    Candidate,
    pack_to_budget,
)


# ─── line range merge ────────────────────────────────────────────────────────────

def test_merge_overlapping_and_adjacent() -> None:
    ranges = [LineRange(1, 10), LineRange(11, 20), LineRange(5, 8), LineRange(40, 50)]
    merged = merge_line_ranges(ranges)
    assert [(r.start, r.end) for r in merged] == [(1, 20), (40, 50)]


def test_merge_respects_gap() -> None:
    # 1-10 and 13-15: gap of 2 lines (11,12) — not merged at default gap=0.
    assert len(merge_line_ranges([LineRange(1, 10), LineRange(13, 15)])) == 2
    # With adjacency_gap=2, they merge.
    merged = merge_line_ranges([LineRange(1, 10), LineRange(13, 15)], adjacency_gap=2)
    assert [(r.start, r.end) for r in merged] == [(1, 15)]


def test_merge_empty() -> None:
    assert merge_line_ranges([]) == []


def test_invalid_line_range_rejected() -> None:
    with pytest.raises(ValueError):
        LineRange(0, 5)
    with pytest.raises(ValueError):
        LineRange(10, 5)


# ─── token estimator ─────────────────────────────────────────────────────────────

def test_conservative_estimator_overestimates() -> None:
    assert conservative_char_estimator("") == 0
    # 35 chars / 3.5 = 10; conservative ceil keeps it >= a real tokenizer for code.
    assert conservative_char_estimator("x" * 35) == 10
    assert conservative_char_estimator("a") == 1  # non-empty floor


# ─── exclusion policy ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("path,expected_substr", [
    ("node_modules/foo/bar.js", "vendored"),
    ("src/.venv/lib/x.py", "vendored"),
    ("models/model.gguf", "binary"),
    ("assets/logo.png", "binary"),
    ("package-lock.json", "lockfile"),
    ("Cargo.lock", "lockfile"),
    ("api/service_pb2.py", "generated"),
    (".env.production", "secrets"),
    ("keys/server.pem", "secrets"),
])
def test_exclusion_policy_flags(path: str, expected_substr: str) -> None:
    reason = default_exclusion_reason(path)
    assert reason is not None and expected_substr in reason


def test_exclusion_policy_allows_normal_source() -> None:
    assert default_exclusion_reason("src/orchestration/dispatcher.py") is None
    assert default_exclusion_reason("tests/unit/test_foo.py") is None


# ─── bundle entry ────────────────────────────────────────────────────────────────

def test_entry_normalizes_line_ranges() -> None:
    e = BundleEntry(
        path="a.py", mode=InclusionMode.SLICES,
        line_ranges=[LineRange(10, 20), LineRange(1, 5), LineRange(18, 25)],
        source=SourceKind.COLGREP,
    )
    assert [(r.start, r.end) for r in e.line_ranges] == [(1, 5), (10, 25)]


def test_entry_rejects_bad_mode_and_source() -> None:
    with pytest.raises(ValueError):
        BundleEntry(path="a.py", mode="bogus")
    with pytest.raises(ValueError):
        BundleEntry(path="a.py", source="bogus")


# ─── bundle accounting + policy application ──────────────────────────────────────

def test_add_entry_estimates_and_accounts() -> None:
    b = ContextBundle(budget=100)
    b.add_entry(BundleEntry(path="a.py", mode=InclusionMode.FULL, source=SourceKind.DIRECT_READ),
                body_text="x" * 35)  # → 10 tokens
    b.add_entry(BundleEntry(path="b.py", mode=InclusionMode.FULL, source=SourceKind.DIRECT_READ),
                body_text="y" * 70)  # → 20 tokens
    assert b.total_tokens() == 30
    assert b.remaining() == 70
    assert b.fits()


def test_add_entry_applies_exclusion_policy() -> None:
    b = ContextBundle(budget=100)
    e = b.add_entry(BundleEntry(path="node_modules/x.js", mode=InclusionMode.FULL,
                                source=SourceKind.COLGREP), body_text="z" * 350)
    assert e.mode == InclusionMode.EXCLUDED
    assert "vendored" in e.reason_downgraded_or_excluded
    # Excluded entries do not count toward budget.
    assert b.total_tokens() == 0


def test_manual_seed_bypasses_exclusion_policy() -> None:
    b = ContextBundle(budget=100)
    e = b.add_entry(BundleEntry(path=".env", mode=InclusionMode.FULL,
                                source=SourceKind.MANUAL_SEED), body_text="SECRET=1")
    assert e.mode == InclusionMode.FULL  # explicit manual seed wins
    assert b.total_tokens() > 0


def test_downgrade_and_exclude_with_reasons() -> None:
    b = ContextBundle(budget=10)
    b.add_entry(BundleEntry(path="big.py", mode=InclusionMode.FULL, source=SourceKind.DIRECT_READ),
                body_text="x" * 350)  # 100 tokens > budget
    assert not b.fits()
    assert b.downgrade("big.py", InclusionMode.CODEMAP_ONLY, "over budget; signatures only")
    # downgrade keeps it counting (codemap_only still counts), so set tokens to reflect codemap size
    for e in b.entries:
        if e.path == "big.py":
            e.estimated_tokens = 5
    assert b.fits()
    assert b.exclude("big.py", "still too big")
    assert b.total_tokens() == 0
    assert b.excluded()[0].path == "big.py"
    assert not b.downgrade("missing.py", InclusionMode.FULL, "x")  # miss returns False


def test_manifest_separates_metadata_from_text() -> None:
    b = ContextBundle(budget=100, bundle_id="bnd-1", repo_sha="abc", gitnexus_index_commit="def",
                      bands=BudgetBands(task=10, codemap=20, editable=40, tests=10, output_reserve=20))
    b.add_entry(BundleEntry(path="a.py", mode=InclusionMode.SLICES,
                            line_ranges=[LineRange(1, 5), LineRange(4, 9)],
                            symbol_ids=["a.py::foo"], content_sha256="sha",
                            source=SourceKind.GITNEXUS, reason_included="caller of target"),
                body_text="x" * 35)
    m = b.manifest()
    assert m["bundle_id"] == "bnd-1"
    assert m["repo_sha"] == "abc"
    assert m["bands"]["editable"] == 40
    assert m["total_tokens"] == 10
    entry = m["entries"][0]
    assert entry["line_ranges"] == [[1, 9]]  # merged
    assert entry["symbol_ids"] == ["a.py::foo"]
    assert entry["reason_included"] == "caller of target"
    # manifest carries no rendered body text — only metadata + token estimate
    assert "body_text" not in entry and "content" not in entry


def test_budget_bands_total() -> None:
    bands = BudgetBands(task=10, codemap=20, editable=40, tests=10, output_reserve=20)
    assert bands.total_reserved() == 100


# ─── DCP-2 packing ───────────────────────────────────────────────────────────────

def test_pack_includes_high_priority_full_then_downgrades() -> None:
    cands = [
        Candidate(path="hot.py", priority=10, cost_full=60, cost_slices=20, cost_codemap=5),
        Candidate(path="warm.py", priority=5, cost_full=60, cost_slices=20, cost_codemap=5),
        Candidate(path="cold.py", priority=1, cost_full=60, cost_slices=20, cost_codemap=5),
    ]
    bundle = pack_to_budget(cands, budget=70)
    by_path = {e.path: e for e in bundle.entries}
    assert by_path["hot.py"].mode == InclusionMode.FULL       # 60 fits first
    assert by_path["warm.py"].mode == InclusionMode.CODEMAP_ONLY  # only 5 left after 60 → codemap
    assert bundle.total_tokens() <= 70
    assert bundle.fits()


def test_pack_colgrep_spans_keep_allowed_slice_mode_without_codemap_fallback() -> None:
    candidate = Candidate(
        path="hit.py",
        priority=1,
        cost_full=60,
        cost_slices=20,
        cost_codemap=5,
        desired_mode=InclusionMode.FULL,
        line_ranges=[LineRange(10, 12)],
        source=SourceKind.COLGREP,
    )

    bundle = pack_to_budget([candidate], budget=20)

    entry = bundle.entries[0]
    assert entry.mode == InclusionMode.SLICES
    assert [(item.start, item.end) for item in entry.line_ranges] == [(10, 12)]
    assert entry.estimated_tokens == 20


def test_pack_colgrep_spans_exclude_when_only_codemap_would_fit() -> None:
    candidate = Candidate(
        path="hit.py",
        priority=1,
        cost_full=60,
        cost_slices=20,
        cost_codemap=5,
        desired_mode=InclusionMode.SLICES,
        line_ranges=[LineRange(10, 12)],
        source=SourceKind.COLGREP,
    )

    bundle = pack_to_budget([candidate], budget=5)

    entry = bundle.entries[0]
    assert entry.mode == InclusionMode.EXCLUDED
    assert [(item.start, item.end) for item in entry.line_ranges] == [(10, 12)]
    assert "missing evidence" in entry.reason_downgraded_or_excluded
    assert bundle.total_tokens() == 0


def test_pack_colgrep_without_spans_keeps_existing_codemap_fallback() -> None:
    candidate = Candidate(
        path="spanless.py",
        priority=1,
        cost_full=60,
        cost_slices=20,
        cost_codemap=5,
        source=SourceKind.COLGREP,
    )

    bundle = pack_to_budget([candidate], budget=5)

    assert bundle.entries[0].mode == InclusionMode.CODEMAP_ONLY


def test_pack_non_colgrep_spans_keep_existing_codemap_fallback() -> None:
    candidate = Candidate(
        path="direct.py",
        priority=1,
        cost_full=60,
        cost_slices=20,
        cost_codemap=5,
        desired_mode=InclusionMode.SLICES,
        line_ranges=[LineRange(10, 12)],
        source=SourceKind.DIRECT_READ,
    )

    bundle = pack_to_budget([candidate], budget=5)

    assert bundle.entries[0].mode == InclusionMode.CODEMAP_ONLY


def test_pack_excludes_when_nothing_fits() -> None:
    cands = [
        Candidate(path="a.py", priority=10, cost_full=100, cost_slices=100, cost_codemap=100),
    ]
    bundle = pack_to_budget(cands, budget=50)
    assert bundle.entries[0].mode == InclusionMode.EXCLUDED
    assert "fail-closed" in bundle.entries[0].reason_downgraded_or_excluded
    assert bundle.total_tokens() == 0


def test_pack_respects_desired_mode_ceiling() -> None:
    # desired_mode=SLICES means never include as FULL even if it fits
    cands = [Candidate(path="a.py", priority=10, cost_full=10, cost_slices=8, cost_codemap=2,
                       desired_mode=InclusionMode.SLICES)]
    bundle = pack_to_budget(cands, budget=100)
    assert bundle.entries[0].mode == InclusionMode.SLICES
    assert bundle.entries[0].estimated_tokens == 8


def test_pack_reserves_output_band() -> None:
    # budget 100 with output_reserve 80 → only 20 usable
    bands = BudgetBands(output_reserve=80)
    cands = [Candidate(path="a.py", priority=10, cost_full=50, cost_slices=15, cost_codemap=5)]
    bundle = pack_to_budget(cands, budget=100, bands=bands)
    assert bundle.entries[0].mode == InclusionMode.SLICES  # 50 doesn't fit in 20, 15 does
    assert bundle.total_tokens() == 15


def test_pack_applies_exclusion_policy_without_consuming_budget() -> None:
    cands = [
        Candidate(path="node_modules/x.js", priority=100, cost_full=5, cost_slices=5, cost_codemap=5),
        Candidate(path="real.py", priority=1, cost_full=10, cost_slices=5, cost_codemap=2),
    ]
    bundle = pack_to_budget(cands, budget=12)
    by_path = {e.path: e for e in bundle.entries}
    assert by_path["node_modules/x.js"].mode == InclusionMode.EXCLUDED  # policy, despite top priority
    assert by_path["real.py"].mode == InclusionMode.FULL  # budget not consumed by the vendored file
    assert bundle.total_tokens() == 10


def test_pack_priority_order_determines_richness() -> None:
    # Equal costs; the higher-priority file should get the richer mode when budget is tight.
    cands = [
        Candidate(path="low.py", priority=1, cost_full=40, cost_slices=10, cost_codemap=2),
        Candidate(path="high.py", priority=9, cost_full=40, cost_slices=10, cost_codemap=2),
    ]
    bundle = pack_to_budget(cands, budget=45)
    by_path = {e.path: e for e in bundle.entries}
    assert by_path["high.py"].mode == InclusionMode.FULL          # 40 first
    assert by_path["low.py"].mode in (InclusionMode.SLICES, InclusionMode.CODEMAP_ONLY)
    assert bundle.fits()
