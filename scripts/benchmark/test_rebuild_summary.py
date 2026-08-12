"""Suite-retirement stamping tests for the master summary rebuild path.

Property under test: rebuild_summary.py must never present a suite that is
retired-for-discrimination at a model's tier as a clean comparative number —
cells are stamped, cross-suite totals exclude them, and a missing/invalid
retirement sidecar aborts the rebuild instead of silently un-retiring
everything (fail-closed).
"""

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

import rebuild_summary  # noqa: E402
import score_with_claude  # noqa: E402


def test_retired_cell_stamp_applies_at_tier_only():
    """general is stamped at >=27B and for unresolvable sizes (fail-closed),
    clean below tier; a live suite (coder) is never stamped."""
    retirements = rebuild_summary.load_suite_retirements()
    at_tier = "Qwen3.6-27B-Q8_0"
    stamp = rebuild_summary.retired_suite_stamp("general", at_tier, retirements)
    assert stamp.startswith(rebuild_summary.RETIRED_CELL_STAMP)
    assert rebuild_summary.retired_suite_stamp("coder", at_tier, retirements) == ""
    assert rebuild_summary.retired_suite_stamp(
        "general", "Qwen2.5-7B-Instruct", retirements) == ""
    # Unresolvable model size cannot certify sub-tier: stamped.
    assert rebuild_summary.retired_suite_stamp(
        "general", "mystery-model", retirements).startswith(
            rebuild_summary.RETIRED_CELL_STAMP)


def test_aggregate_totals_exclude_stamped_suites():
    """The cross-suite total is a comparative number: stamped suites stay
    visible in their own cells but never feed it."""
    suite_scores = {
        "general": {"correct": 10, "total": 10,
                    "str": "10/10 !RETIRED-NONDISCRIMINATING@27B+",
                    "stamp": "!RETIRED-NONDISCRIMINATING@27B+"},
        "coder": {"correct": 5, "total": 9, "str": "5/9", "stamp": ""},
        "agentic": {"correct": 4, "total": 10, "str": "4/10", "stamp": ""},
    }
    assert rebuild_summary.aggregate_totals(suite_scores) == (9, 19)


def test_rebuild_refuses_without_retirement_sidecar(tmp_path, monkeypatch):
    """Deleting the retirement metadata aborts the rebuild loudly; no
    summary.csv is written from an un-retired view of the world."""
    monkeypatch.setattr(score_with_claude, "SUITE_RETIREMENTS_PATH",
                        tmp_path / "absent.json")
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setattr(rebuild_summary, "RUNS_DIR", str(runs_dir))
    out_file = tmp_path / "summary.csv"
    monkeypatch.setattr(rebuild_summary, "OUTPUT_FILE", str(out_file))

    with pytest.raises(SystemExit) as excinfo:
        rebuild_summary.main()

    assert excinfo.value.code  # non-zero / non-empty: a loud abort
    assert "FAIL-CLOSED" in str(excinfo.value.code)
    assert not out_file.exists()
