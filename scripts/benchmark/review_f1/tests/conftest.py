"""Shared test setup + fixtures for the review_f1 suite.

Dual-mode: works under pytest (auto-loaded conftest + fixtures) AND under the
stdlib runner embedded in each test file (research .venv has NO pytest). Tests
call the plain ``make_*`` builders directly so they never depend on fixture
injection; the ``@pytest.fixture`` wrappers exist only for pytest users.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parents[1]  # .../review_f1
TESTS_DIR = Path(__file__).resolve().parent
for p in (str(PKG_DIR), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Path to the checked-in synthetic golden set (assembled from raw fixtures).
RESEARCH_ROOT = PKG_DIR.parents[2]  # .../epyc-inference-research
SYNTHETIC_GOLDEN = RESEARCH_ROOT / "data" / "review_f1" / "fixtures" / "synthetic_golden_set.json"
RAW_SAMPLE_DIR = RESEARCH_ROOT / "data" / "review_f1" / "fixtures" / "raw_augment_sample"


def make_golden_cases() -> list[dict]:
    """Two-PR golden set with a low-severity finding on each PR (in-memory)."""
    return [
        {
            "case_id": "repoA__pr-1",
            "pr_ref": {"repo": "org/repoA", "number": 1, "diff": "<diff A>"},
            "golden_findings": [
                {"golden_id": "a-g0", "criterion": "logic_bug",
                 "location": {"file": "a.py", "line_start": 10, "line_end": 12}, "severity": "high"},
                {"golden_id": "a-g1", "criterion": "security",
                 "location": {"file": "a.py", "line_start": 30, "line_end": 30}, "severity": "medium"},
                {"golden_id": "a-g2", "criterion": "logic_bug",
                 "location": {"file": "a.py", "line_start": 5, "line_end": 5}, "severity": "low"},
            ],
        },
        {
            "case_id": "repoB__pr-2",
            "pr_ref": {"repo": "org/repoB", "number": 2, "diff": "<diff B>"},
            "golden_findings": [
                {"golden_id": "b-g0", "criterion": "runtime_error",
                 "location": {"file": "b.go", "line_start": 40, "line_end": 44}, "severity": "high"},
                {"golden_id": "b-g1", "criterion": "performance",
                 "location": {"file": "b.go", "line_start": 8, "line_end": 8}, "severity": "low"},
            ],
        },
    ]


def load_synthetic_golden() -> dict:
    return json.loads(SYNTHETIC_GOLDEN.read_text())


# ---- pytest fixture wrappers (only used when pytest is present) ---- #
try:
    import pytest

    @pytest.fixture
    def golden_cases():
        return make_golden_cases()

    @pytest.fixture
    def synthetic_golden():
        return load_synthetic_golden()
except ImportError:  # pragma: no cover - pytest not installed in research .venv
    pass
