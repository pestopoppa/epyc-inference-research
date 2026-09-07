#!/usr/bin/env python3
"""The CJ-8 / CJ-9 gate verdict VOCABULARY, vendored for this repository.

WHY THIS FILE EXISTS AND WHY IT IS NOT AN IMPORT
------------------------------------------------
The canonical module is ``epyc-root:scripts/benchmark/gate_verdict.py`` (landed
``9c86adb8``). It is the authority for the vocabulary, the closed cause registry,
the per-suite coverage contract and the suite fold. Nothing here re-decides any
of that.

This repository does not import it, for the same reason ``gate_verdict.py``
itself refuses to import ``dashboard/``: an import edge from a measurement path
to a tree that may not be checked out beside it turns a missing sibling
repository into a scoring-time crash. epyc-root is an umbrella repo; a scorer in
this repo must run without it on disk.

So the *strings* are duplicated and the duplication is LOCKED BY A TEST rather
than by an import: ``test_gate_verdict_vocab.py`` loads the canonical module by
path when epyc-root is present beside this checkout and asserts the vocabularies
are identical, and skips (never passes vacuously) when it is not.

WHAT IS VENDORED AND WHAT IS DELIBERATELY NOT
---------------------------------------------
Vendored: the three verdict values, the closed cause registry with its meanings,
the mandatory-cause enforcement, and the 0/1/2 exit contract.

NOT vendored: ``SuiteCoverageContract`` / ``resolve_suite`` (CJ-9). Declaring a
suite's ``min_resolved_coverage`` is a measurement-parameter decision belonging
to that suite's owner, and a *second* place to declare it would be exactly the
silent re-declaration ``DuplicateSuiteContractError`` exists to refuse. A suite
in this repo that needs the fold should use the canonical module directly, at a
boundary where epyc-root is known to be present.

``CAUSE_INSUFFICIENT_COVERAGE`` IS vendored, but stays SUITE-LEVEL ONLY, exactly
as upstream: an item is not undecided because the *suite* was thin, so it is
admissible only under ``suite_level=True``.

Pure: stdlib only, no I/O, no clock, no network.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

__all__ = [
    "VERDICT_PASS", "VERDICT_FAIL", "VERDICT_OUT_OF_COVERAGE", "VERDICTS",
    "DECIDED_VERDICTS", "CAUSES", "CAUSE_MEANINGS",
    "CAUSE_ABSENT", "CAUSE_EMPTY", "CAUSE_UNPARSED", "CAUSE_NO_REFERENCE",
    "CAUSE_UNSUPPORTED", "CAUSE_ABSTAINED", "CAUSE_CHECKER_ERROR",
    "CAUSE_TIMEOUT", "CAUSE_SKIPPED", "CAUSE_INSUFFICIENT_COVERAGE",
    "SUITE_LEVEL_CAUSES", "ALL_CAUSES",
    "EXIT_PASS", "EXIT_FAIL", "EXIT_OUT_OF_COVERAGE", "EXIT_BY_VERDICT",
    "GateVerdictError", "MissingCauseCodeError", "SpuriousCauseCodeError",
    "out_of_coverage", "verdict_of", "is_decided",
]

#: The gate reached a decision and the assertion held.
VERDICT_PASS = "pass"
#: The gate reached a decision and the assertion did not hold.
VERDICT_FAIL = "fail"
#: The gate reached NO decision. NOT a negative label. Carries a mandatory cause
#: code, because the reasons below have different remedies and only the cause
#: code tells them apart.
VERDICT_OUT_OF_COVERAGE = "out-of-coverage"

VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_OUT_OF_COVERAGE)
#: The verdicts that constitute a DECISION — the resolved-coverage numerator.
DECIDED_VERDICTS = frozenset({VERDICT_PASS, VERDICT_FAIL})

CAUSE_ABSENT = "absent"
CAUSE_EMPTY = "empty"
CAUSE_UNPARSED = "unparsed"
CAUSE_NO_REFERENCE = "no_reference"
CAUSE_UNSUPPORTED = "unsupported"
CAUSE_ABSTAINED = "abstained"
CAUSE_CHECKER_ERROR = "checker_error"
CAUSE_TIMEOUT = "timeout"
CAUSE_SKIPPED = "skipped"
#: SUITE-LEVEL ONLY. The suite's own resolved coverage fell below its declared
#: threshold, so the SUITE verdict is undecided. Never valid on a single item.
CAUSE_INSUFFICIENT_COVERAGE = "insufficient_coverage"

#: Causes valid on a single item's verdict. Closed: a free-text cause is not
#: foldable and collapses the taxonomy into one "other" bucket.
CAUSES = (
    CAUSE_ABSENT, CAUSE_EMPTY, CAUSE_UNPARSED, CAUSE_NO_REFERENCE,
    CAUSE_UNSUPPORTED, CAUSE_ABSTAINED, CAUSE_CHECKER_ERROR, CAUSE_TIMEOUT,
    CAUSE_SKIPPED,
)
#: Causes valid only on a suite-level (folded) verdict.
SUITE_LEVEL_CAUSES = (CAUSE_INSUFFICIENT_COVERAGE,)
#: Every cause the vocabulary admits, at any granularity.
ALL_CAUSES = CAUSES + SUITE_LEVEL_CAUSES

#: Cause -> the remedy it points at. The whole point of the taxonomy: a reader
#: who sees the count must be able to tell which subsystem to go fix.
CAUSE_MEANINGS: Mapping[str, str] = {
    CAUSE_ABSENT: "no response/artifact was produced for this item — fix the "
                  "generation or transport path, not the model's quality",
    CAUSE_EMPTY: "a response arrived and carried nothing — fix generation "
                 "(budget, stop tokens, template), not the scorer",
    CAUSE_UNPARSED: "a non-empty response arrived and no answer could be "
                    "extracted — fix the extractor; a parse-failure rate read "
                    "as a quality gap is a scoring artifact",
    CAUSE_NO_REFERENCE: "no gold/oracle exists for this item — fix the corpus "
                        "join, not the subject",
    CAUSE_UNSUPPORTED: "the item is outside this checker's competence — either "
                       "build the substrate or exclude the slice explicitly",
    CAUSE_ABSTAINED: "the checker ran and declined to decide — an abstention, "
                     "not a negative label",
    CAUSE_CHECKER_ERROR: "the checker raised — the instrument is defective",
    CAUSE_TIMEOUT: "the checker exceeded its budget before deciding — raise the "
                   "budget or shrink the item",
    CAUSE_SKIPPED: "deliberately not run (outside the declared slice) — "
                   "compliant, declared, and still not a decision",
    CAUSE_INSUFFICIENT_COVERAGE: "the suite's resolved coverage fell below its "
                                 "declared threshold, so the suite verdict "
                                 "itself is undecided",
}

#: 0 pass / 1 fail / 2 could-not-check, per
#: ``epyc-root:scripts/validate/check_ratification_receipts.py``. A wrapper that
#: only tests ``!= 0`` is unaffected; one that wants the distinction can have it.
#: OUT-OF-COVERAGE IS NON-ZERO AND STAYS NON-ZERO: an undecidable input must keep
#: BLOCKING. It is renamed, never downgraded to a pass.
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_OUT_OF_COVERAGE = 2

EXIT_BY_VERDICT: Mapping[str, int] = {
    VERDICT_PASS: EXIT_PASS,
    VERDICT_FAIL: EXIT_FAIL,
    VERDICT_OUT_OF_COVERAGE: EXIT_OUT_OF_COVERAGE,
}


class GateVerdictError(ValueError):
    """Base for every refusal here. Subclasses ``ValueError`` so a caller that
    already guards ``ValueError`` does not swallow a compliance refusal as a
    different class of bug."""


class MissingCauseCodeError(GateVerdictError):
    """``out-of-coverage`` with no cause, or a cause outside :data:`CAUSES`."""


class SpuriousCauseCodeError(GateVerdictError):
    """A decided verdict carrying a cause code — the caller does not know which
    of the two verdicts it emitted."""


def out_of_coverage(item_id: str, cause: str,
                    detail: Optional[str] = None,
                    *, suite_level: bool = False) -> dict:
    """Build one ``out-of-coverage`` record. ``cause`` is positional on purpose:
    there is no call shape that omits it.

    ``suite_level=True`` widens the admissible cause set by exactly
    :data:`SUITE_LEVEL_CAUSES` and nothing else."""
    admissible = ALL_CAUSES if suite_level else CAUSES
    if cause not in admissible:
        extra = (" (that cause is SUITE-LEVEL only and is never valid on a "
                 "single item)" if cause in SUITE_LEVEL_CAUSES else "")
        raise MissingCauseCodeError(
            f"item {item_id!r}: cause {cause!r} is outside the closed registry "
            f"{admissible!r}{extra}. A free-text cause is not foldable."
        )
    record: dict[str, Any] = {
        "item_id": item_id,
        "verdict": VERDICT_OUT_OF_COVERAGE,
        "cause": cause,
        "cause_means": CAUSE_MEANINGS[cause],
    }
    if detail is not None:
        record["detail"] = detail
    return record


def verdict_of(item_id: str, verdict: str,
               cause: Optional[str] = None,
               detail: Optional[str] = None,
               *, suite_level: bool = False) -> dict:
    """Build any one verdict record, enforcing the cause asymmetry: MANDATORY on
    ``out-of-coverage``, FORBIDDEN on the two decided verdicts."""
    if isinstance(verdict, bool) or verdict in (0, 1):
        raise GateVerdictError(
            f"verdict {verdict!r} is TWO-VALUED (a bool / 0-1 score). It has no "
            f"way to spell {VERDICT_OUT_OF_COVERAGE!r}, so an item the checker "
            f"never decided is reported as {VERDICT_FAIL!r}."
        )
    if verdict not in VERDICTS:
        raise GateVerdictError(f"verdict {verdict!r} is not one of {VERDICTS!r}.")
    if verdict == VERDICT_OUT_OF_COVERAGE:
        if cause is None:
            raise MissingCauseCodeError(
                f"item {item_id!r}: {VERDICT_OUT_OF_COVERAGE!r} without a cause "
                f"code. An undecided count with no cause names no remedy."
            )
        return out_of_coverage(item_id, cause, detail, suite_level=suite_level)
    if cause is not None:
        raise SpuriousCauseCodeError(
            f"item {item_id!r}: verdict {verdict!r} is a DECISION and must not "
            f"carry a cause code (got {cause!r})."
        )
    record: dict[str, Any] = {"item_id": item_id, "verdict": verdict}
    if detail is not None:
        record["detail"] = detail
    return record


def is_decided(verdict: str) -> bool:
    """True iff this verdict counts toward the resolved-coverage numerator."""
    return verdict in DECIDED_VERDICTS
