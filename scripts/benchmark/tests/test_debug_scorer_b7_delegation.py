"""The research-repo debug_scorer is a delegation shim to the orchestrator B7 scorer.

Regression pin for the 2026-08-23 port (epyc-root
`handoffs/active/scorer-fork-drift-audit-2026-07-22.md`, residual row "Research-repo
`debug_scorer.py` is fully pre-B7 (10/10 defect classes, off routing path) — port B7 or
stamp research benchmarks scored with it as pre-B7-scorer era", ticked 2026-08-23).

This repo's `debug_scorer.py` was a fully pre-B7 fork (10/10 defect classes). It is now
a thin shim that loads the orchestrator's B7-hardened `debug_scorer.py` by absolute
path (the A2 pattern proven by `seeding_scoring.py`) and re-exports its public API.
These tests pin, through the research import path (`from debug_scorer import score_answer`),
that the B7 verdicts are inherited — the audit's proof cases P-01…P-13 (orch column) —
and that a missing orchestrator copy fails CLOSED with ImportError, never silently
falling back to pre-B7 semantics.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

BENCHMARK_DIR = Path(__file__).resolve().parents[1]
for _p in (str(BENCHMARK_DIR), str(BENCHMARK_DIR.parent), str(BENCHMARK_DIR.parents[1])):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import debug_scorer  # noqa: E402  (the shim, imported as research consumers do)


ORCH_SCORER_PATH = Path(
    "/mnt/raid0/llm/epyc-orchestrator/scripts/benchmark/debug_scorer.py")


def _shim_module() -> Path:
    return Path(debug_scorer.__file__).resolve()


def test_shim_resolves_to_the_orchestrator_file() -> None:
    """The research import path must reach the orchestrator B7 copy, not a local fork."""
    assert _shim_module() == BENCHMARK_DIR / "debug_scorer.py"
    orch = debug_scorer._load_orchestrator_scorer()
    assert Path(orch.__file__).resolve() == ORCH_SCORER_PATH.resolve()
    assert orch.__name__ == debug_scorer._ORCH_SCORER_KEY
    assert sys.modules[debug_scorer._ORCH_SCORER_KEY] is orch


def test_score_answer_is_bound_from_the_orchestrator_copy() -> None:
    orch = debug_scorer._load_orchestrator_scorer()
    assert debug_scorer.score_answer is orch.score_answer
    assert debug_scorer._extract_code_block is orch._extract_code_block


# ── P-proofs (audit table, orch column) through the research import path ──


def test_p01_score06_boundary_substring_not_plain_in() -> None:
    assert debug_scorer.score_answer("The count is 630 items", "63", "substring") is False


def test_p02_score06b_digit_separator_strip() -> None:
    assert debug_scorer.score_answer("Result: 479,001,600.", "479001600", "substring") is True


def test_p03_score03_final_answer_region_gates_colon_fallback() -> None:
    assert debug_scorer.score_answer(
        "The answer: 42\nActually the final result is 43", "42", "exact_match") is False


def test_p04_score24b_capture_group_guard_raises_value_error() -> None:
    with pytest.raises(ValueError):
        debug_scorer.score_answer(
            "cat cat cat", "cat cat dog", "f1", {"extract_pattern": "(a)(b)"})


def test_p05_score24_multiset_f1() -> None:
    assert debug_scorer.score_answer("cat cat cat", "cat cat dog", "f1", {}) is True


def test_p06_score16_nested_boxed_answer() -> None:
    assert debug_scorer.score_answer(
        r"The result is \boxed{\frac{1}{2}}", r"\frac{1}{2}", "exact_match") is True


def test_p06b_multiple_choice_textual_label() -> None:
    assert debug_scorer.score_answer(
        "The answer is black cat.", "black cat", "multiple_choice",
        {"choices": ["black cat", "cat", "dog"]}) is True


def test_p07_score21_vacuous_oracle_rejected() -> None:
    assert debug_scorer.score_answer(
        "def solve():\n    return 1", "", "code_execution",
        {"language": "python", "test_code": "assert True"}) is False


def test_p08_score0405_entry_point_without_oracle_raises() -> None:
    with pytest.raises(debug_scorer.ScoringUnavailableError):
        debug_scorer.score_answer(
            "def solve():\n    return 42", "42", "code_execution",
            {"language": "python", "timeout": 10, "entry_point": "solve"})


def test_p09_score25_unknown_verifier_raises() -> None:
    with pytest.raises(ValueError):
        debug_scorer.score_answer(
            "x", "x", "programmatic", {"verifier": "typoed_verifier_name"})


def test_p12_score23_str_wrap_non_string_expected() -> None:
    assert debug_scorer.score_answer("42", 42, "exact_match") is True


def test_p13_llm_judge_fastpath_boundary_anchored() -> None:
    assert debug_scorer._contains_text_unit("we concatenate strings", "cat") is False


def test_math_verify_method_is_present_not_unknown() -> None:
    """Pre-B7 fork: `ValueError: Unknown scoring method: math_verify` (absent method).

    B7 has a real `_score_math_verify`; if math_verify is unavailable in this
    environment it raises ScoringUnavailableError instead — but never the
    pre-B7 "Unknown scoring method" ValueError.
    """
    try:
        result = debug_scorer.score_answer("x", "x", "math_verify")
    except debug_scorer.ScoringUnavailableError:
        pass
    else:
        assert isinstance(result, bool)


# ── Fail-closed behavior ──


def test_missing_orchestrator_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing orchestrator copy must raise ImportError, never degrade."""
    monkeypatch.setattr(
        debug_scorer, "_ORCH_SCORER_PATH",
        Path("/nonexistent/epyc-orchestrator/scripts/benchmark/debug_scorer.py"))
    monkeypatch.delitem(sys.modules, debug_scorer._ORCH_SCORER_KEY, raising=False)

    with pytest.raises(ImportError, match="refusing to fall back"):
        debug_scorer._load_orchestrator_scorer()
    assert debug_scorer._ORCH_SCORER_KEY not in sys.modules

    # Import-time failure: reload re-executes the module body, so the module-level
    # `_ORCH_SCORER_PATH` assignment would clobber a setattr patch — patch the
    # filesystem probe instead to simulate a missing orchestrator copy.
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(ImportError):
        importlib.reload(debug_scorer)


# ── Re-export surface ──


def test_all_consumed_names_resolve() -> None:
    assert callable(debug_scorer.score_answer)
    assert callable(debug_scorer._extract_code_block)
    assert callable(debug_scorer.score_batch)
    assert callable(debug_scorer._contains_text_unit)
    assert issubclass(debug_scorer.ScoringUnavailableError, RuntimeError)


def test_unknown_name_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = debug_scorer.definitely_not_a_real_name_xyz


def test_package_style_import_matches_bare_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """memory_viability_runner imports `from benchmark.debug_scorer import score_answer`.

    Both import styles must resolve to the SAME orchestrator-bound function.
    Warm the shared cache first so the package-style import's fresh shim exec
    hits it (test order must not matter).
    """
    debug_scorer._load_orchestrator_scorer()
    monkeypatch.delitem(sys.modules, "benchmark.debug_scorer", raising=False)
    monkeypatch.delitem(sys.modules, "benchmark", raising=False)
    from benchmark.debug_scorer import score_answer as pkg_score_answer

    assert pkg_score_answer is debug_scorer.score_answer
    assert pkg_score_answer("The count is 630 items", "63", "substring") is False


def test_extract_code_block_helper_still_works() -> None:
    code = debug_scorer._extract_code_block("```python\nx = 1\n```")
    assert code is not None
    assert "x = 1" in code
