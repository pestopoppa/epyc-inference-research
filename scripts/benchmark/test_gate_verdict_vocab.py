#!/usr/bin/env python3
"""Lock the vendored CJ-8 vocabulary against its canonical definition.

``scripts/benchmark/gate_verdict_vocab.py`` duplicates the verdict and cause
strings from ``epyc-root:scripts/benchmark/gate_verdict.py`` on purpose (see that
module's docstring: no import edge from a scorer to a sibling repository that may
not be checked out). Duplication without a lock is drift, so the lock is here.

The conformance test SKIPS when epyc-root is not on disk. It must never pass
vacuously: the local-invariant tests below run unconditionally, and the
conformance test asserts a non-empty comparison before comparing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_MODULE_PATH = _HERE.parent / "gate_verdict_vocab.py"
_SPEC = importlib.util.spec_from_file_location("gate_verdict_vocab", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
gvv = importlib.util.module_from_spec(_SPEC)
sys.modules.setdefault("gate_verdict_vocab", gvv)
_SPEC.loader.exec_module(gvv)


def _canonical():
    """Load epyc-root's gate_verdict.py by path, or return None if absent."""
    for candidate in (
        Path("/workspace/scripts/benchmark/gate_verdict.py"),
        _HERE.parents[3] / "scripts" / "benchmark" / "gate_verdict.py",
    ):
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "_canonical_gate_verdict", candidate)
            if spec is None or spec.loader is None:  # pragma: no cover
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register BEFORE exec: the module defines dataclasses, and
            # @dataclass resolves annotations via sys.modules[cls.__module__].
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            return mod
    return None


# --------------------------------------------------------------------------- #
# Conformance against the canonical module (skips when epyc-root is absent)
# --------------------------------------------------------------------------- #
def test_vocabulary_matches_canonical_gate_verdict() -> None:
    canonical = _canonical()
    if canonical is None:
        pytest.skip("epyc-root scripts/benchmark/gate_verdict.py not on disk")
    # Guard against a vacuous comparison: if the canonical module ever stopped
    # exporting these, an equality over two empty tuples would "pass".
    assert canonical.VERDICTS, "canonical VERDICTS is empty — vacuous comparison"
    assert canonical.CAUSES, "canonical CAUSES is empty — vacuous comparison"

    assert gvv.VERDICTS == canonical.VERDICTS
    assert gvv.DECIDED_VERDICTS == canonical.DECIDED_VERDICTS
    assert gvv.CAUSES == canonical.CAUSES
    assert dict(gvv.CAUSE_MEANINGS) == {
        c: canonical.CAUSE_MEANINGS[c] for c in canonical.ALL_CAUSES
    }
    assert gvv.SUITE_LEVEL_CAUSES == canonical.SUITE_LEVEL_CAUSES
    assert gvv.ALL_CAUSES == canonical.ALL_CAUSES
    # The suite-level cause is never admissible on a single item.
    assert set(canonical.SUITE_LEVEL_CAUSES).isdisjoint(gvv.CAUSES)


# --------------------------------------------------------------------------- #
# Local invariants — run unconditionally
# --------------------------------------------------------------------------- #
def test_out_of_coverage_exit_code_is_non_zero() -> None:
    """THE SAFETY RULE. An undecidable input must keep BLOCKING; it is renamed,
    never downgraded to a pass."""
    assert gvv.EXIT_OUT_OF_COVERAGE != gvv.EXIT_PASS
    assert gvv.EXIT_BY_VERDICT[gvv.VERDICT_OUT_OF_COVERAGE] != 0


def test_out_of_coverage_requires_a_registered_cause() -> None:
    with pytest.raises(gvv.MissingCauseCodeError):
        gvv.out_of_coverage("item", "made_up_cause")
    with pytest.raises(gvv.MissingCauseCodeError):
        gvv.verdict_of("item", gvv.VERDICT_OUT_OF_COVERAGE)


def test_suite_level_cause_is_refused_on_a_single_item() -> None:
    with pytest.raises(gvv.MissingCauseCodeError):
        gvv.out_of_coverage("q1", gvv.CAUSE_INSUFFICIENT_COVERAGE)
    # Mutation guard: it IS admissible at suite granularity, so the refusal
    # above is about the granularity and not about an unknown string.
    rec = gvv.out_of_coverage("suite:x", gvv.CAUSE_INSUFFICIENT_COVERAGE,
                              suite_level=True)
    assert rec["cause"] == gvv.CAUSE_INSUFFICIENT_COVERAGE


def test_decided_verdict_refuses_a_cause_code() -> None:
    with pytest.raises(gvv.SpuriousCauseCodeError):
        gvv.verdict_of("item", gvv.VERDICT_FAIL, cause=gvv.CAUSE_ABSENT)


def test_two_valued_verdict_is_refused() -> None:
    for bad in (True, False, 0, 1):
        with pytest.raises(gvv.GateVerdictError):
            gvv.verdict_of("item", bad)


def test_out_of_coverage_is_not_decided() -> None:
    assert not gvv.is_decided(gvv.VERDICT_OUT_OF_COVERAGE)
    assert gvv.is_decided(gvv.VERDICT_PASS)
    assert gvv.is_decided(gvv.VERDICT_FAIL)


def test_record_carries_the_remedy_not_just_the_cause() -> None:
    rec = gvv.out_of_coverage("q1", gvv.CAUSE_NO_REFERENCE, detail="no gold")
    assert rec["verdict"] == gvv.VERDICT_OUT_OF_COVERAGE
    assert rec["cause"] == gvv.CAUSE_NO_REFERENCE
    assert "corpus join" in rec["cause_means"]
    assert rec["detail"] == "no gold"
