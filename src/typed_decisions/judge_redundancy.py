"""CJ-13/CJ-14: two-cheap-reader judge redundancy over a frozen rubric set.

The typed-decision plane answers a rubric directly (``noul`` per criterion,
one ``choice`` overall) in ONE JSON pass — this is the **typed judge**. A
single ``llm_call`` independently asks for the same per-criterion pass/fail
list under a constrained verdict-array schema, then has its emission parsed
and validated — this is the **LLM judge**. Both read
the SAME ~24 deterministic rubric cases, each a (answer text, criteria list)
pair whose per-criterion verdict is fully determined by the answer text and
frozen here as ground truth.

Why two readers: intake-1475 measured 91.5% pairwise agreement between two
cheap judges while explicitly NOT establishing who is right. This harness
keeps that distinction structural rather than rhetorical:

* **accuracy** is scored against the frozen per-criterion verdicts;
* **agreement** is scored between the readers and is reported separately,
  labelled ``agreement is not accuracy`` — two readers can agree perfectly
  and both be wrong (there is a test that exhibits exactly that);
* **human adjudication rate** is the fraction of criteria a two-reader
  redundancy policy would have to escalate to a human (readers disagree OR
  at least one reader failed to resolve). It is a projection from reader
  behaviour, not a measured human cost.

CJ-14 measurement discipline (intake-1490): the rubric set is frozen and
hashed (``rubric_set_sha256``), every case carries its own ``case_sha256`` in
the manifest, the dry-run plan pre-registers the per-case expected verdicts,
and a **blind prior** (one majority verdict per criterion, never reading an
answer) is reported as the lower bound any reader must beat.

Rubric construction
-------------------
Every criterion is a positively-phrased requirement, so "the criterion is
satisfied" is exactly ``expected[i] is True`` and the overall verdict is
``pass`` iff every criterion passes. Each criterion is decidable from the
answer text by one of: an exact phrase presence, an exact numeric value or
inclusive threshold, a countable count (sources, sentences, exclamation
marks), a lexical-format check (starts with ``Summary``, contains a bulleted
list), or a token-absence check. Families x 4 cases (24 cases, 72 criteria):

* ``refund_policy`` — exact values + token absence;
* ``evidence`` — distinct-source counts, including the exact-boundary case;
* ``format`` — sentence counts, exclamation marks, list/opening format,
  including the exact-three-sentence boundary;
* ``numeric_threshold`` — inclusive boundaries (100 ms, 90.0%);
* ``safety`` — required disclaimers and dosage-figure checks;
* ``citation`` — exact quote, named source, document identifier.

The mix: 8 all-passing cases, 4 all-failing cases, and 12 mixed cases whose
per-criterion ground truth splits at the exact boundaries above.

CLI::

    python -m src.typed_decisions.judge_redundancy --dry-run
    python -m src.typed_decisions.judge_redundancy --live --role frontdoor \\
        --server-url http://127.0.0.1:8199 --receipt <path>

``--dry-run`` makes no call and writes no receipt. Without ``--live`` (or
``--dry-run``) the CLI refuses (exit 2). A mock-mode primitives object is
refused; a receipt never carries fabricated numbers, and a run in which
NEITHER reader resolves a single criterion raises ``MeasurementError``. The
module is import-safe offline: no config, backend, or network object is
constructed at import time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.typed_decisions.measure import (
    MeasurementError,
    _last_inference_meta,
    _live_primitives as _configured_live_primitives,
    _now_iso,
    _require_primitives,
    _write_receipt,
)
from src.typed_decisions.runner import _extract_json_object, run_typed_decisions
from src.typed_decisions.types import Question, QuestionKind

__all__ = [
    "AGREEMENT_LABEL",
    "MODE",
    "OVERALL_FAIL",
    "OVERALL_ID",
    "OVERALL_PASS",
    "STUDY",
    "RubricCase",
    "blind_prior",
    "build_cases",
    "build_typed_questions",
    "case_sha256",
    "criterion_question_id",
    "main",
    "rubric_set_sha256",
    "run_judge_redundancy",
]

STUDY = "judge_redundancy"

MODE = "typed_vs_llm_judge"

# The CJ-13 label. It is a key in the receipt (never implicit) because the
# whole point of the redundancy arm is that agreement and accuracy are
# different measurements with different remedies.
AGREEMENT_LABEL = "agreement is not accuracy"

OVERALL_ID = "overall"

OVERALL_PASS = "pass"
OVERALL_FAIL = "fail"
OVERALL_OPTIONS = (OVERALL_PASS, OVERALL_FAIL)

_CRITERION_PREFIX = "criterion"

# Deterministic decode: temperature 0.0 plus a pinned seed, the same contract
# every other typed-decision harness uses.
_DECODE_SEED = 0

# One JSON verdict list plus room for prose the model was asked not to write.
# A parse failure is a scored failure, not a crash.
_LLM_N_TOKENS = 512


@dataclass(frozen=True)
class RubricCase:
    """One frozen rubric case with fully determined ground truth.

    Attributes:
        case_id: Stable id used for receipt joins and fake-primitive dispatch.
        family: One of the six construction families (see module docstring).
        answer: The answer text under judgment; the reader sees exactly this.
        criteria: Positively-phrased criteria; a reader answers pass/fail per
            criterion. Criteria are unique within a case.
        expected: Per-criterion ground truth (``True`` = criterion satisfied);
            derived by construction from ``answer`` and ``basis``.
        basis: One short deciding rule per criterion (the exact token, count,
            numeric boundary or phrase that fixes the verdict). Carried into
            the manifest so the pre-registration is auditable, never shown to
            a reader.
    """

    case_id: str
    family: str
    answer: str
    criteria: tuple[str, ...]
    expected: tuple[bool, ...]
    basis: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "expected", tuple(self.expected))
        object.__setattr__(self, "basis", tuple(self.basis))
        if not self.criteria:
            raise ValueError(f"case {self.case_id!r} needs at least one criterion")
        if len(self.criteria) != len(self.expected):
            raise ValueError(
                f"case {self.case_id!r} has {len(self.criteria)} criteria but "
                f"{len(self.expected)} expected verdicts"
            )
        if len(self.criteria) != len(self.basis):
            raise ValueError(
                f"case {self.case_id!r} has {len(self.criteria)} criteria but "
                f"{len(self.basis)} basis entries"
            )

    @property
    def expected_overall(self) -> str:
        """``pass`` iff every criterion is satisfied (conjunctive rubric)."""
        return OVERALL_PASS if all(self.expected) else OVERALL_FAIL


# ── Frozen rubric set (24 cases / 72 criteria) ────────────────────────────
#
# Every criterion is phrased so that ``expected=True`` means the criterion is
# satisfied; absence checks are negative statements ("does not promise ...").
# The ``basis`` string names the exact deciding evidence. Do not edit a case
# without bumping nothing here -- the receipt's ``case_sha256`` is the
# contract a re-run must reproduce; edits are visible as digest changes.

_RUBRIC_CASES: tuple[RubricCase, ...] = (
    # ── refund_policy ─────────────────────────────────────────────────────
    RubricCase(
        case_id="refund-01",
        family="refund_policy",
        answer=(
            "The refund window is 30 days from delivery. "
            "A receipt is required. "
            "The customer does not need a reason."
        ),
        criteria=(
            "The answer states the refund window is 30 days.",
            "The answer states that a receipt is required.",
            "The answer states that the customer does not need a reason.",
        ),
        expected=(True, True, True),
        basis=(
            "contains '30 days'",
            "contains 'A receipt is required'",
            "contains 'does not need a reason'",
        ),
    ),
    RubricCase(
        case_id="refund-02",
        family="refund_policy",
        answer=(
            "The refund window is 14 days from delivery. "
            "No receipt is required. "
            "The customer must provide a written reason."
        ),
        criteria=(
            "The answer states the refund window is 30 days.",
            "The answer states that a receipt is required.",
            "The answer states that the customer does not need a reason.",
        ),
        expected=(False, False, False),
        basis=(
            "says '14 days', not '30 days'",
            "says 'No receipt is required'",
            "says 'must provide a written reason'",
        ),
    ),
    RubricCase(
        case_id="refund-03",
        family="refund_policy",
        answer=(
            "The refund window is exactly 30 days from delivery. "
            "A receipt is required. "
            "The customer does not need a reason."
        ),
        criteria=(
            "The answer reports a refund window of at most 30 days.",
            "The answer reports a refund window of at least 30 days.",
            "The answer reports a refund window of less than 30 days.",
        ),
        expected=(True, True, False),
        basis=(
            "exactly 30 satisfies the inclusive 'at most 30' boundary",
            "exactly 30 satisfies the inclusive 'at least 30' boundary",
            "'exactly 30' does not assert 'less than 30'",
        ),
    ),
    RubricCase(
        case_id="refund-04",
        family="refund_policy",
        answer=(
            "Refunds are processed within five business days. "
            "The refund window is 30 days. "
            "No instant payout is offered."
        ),
        criteria=(
            "The answer states refunds are processed within five business days.",
            "The answer states the refund window is 30 days.",
            "The answer does not promise an instant refund.",
        ),
        expected=(True, True, True),
        basis=(
            "contains 'within five business days'",
            "contains '30 days'",
            "contains no promise of an instant refund",
        ),
    ),
    # ── evidence ──────────────────────────────────────────────────────────
    RubricCase(
        case_id="evidence-01",
        family="evidence",
        answer=(
            "Three independent evaluations were used: the Alvarez review, "
            "the Chen audit, and the Nakamura survey."
        ),
        criteria=(
            "The answer cites at least three distinct sources.",
            "The answer names the Chen audit.",
            "The answer names the Alvarez review.",
        ),
        expected=(True, True, True),
        basis=(
            "counts 3 distinct named sources",
            "contains 'the Chen audit'",
            "contains 'the Alvarez review'",
        ),
    ),
    RubricCase(
        case_id="evidence-02",
        family="evidence",
        answer=("Two evaluations were used: the Patel review and the Ortiz audit."),
        criteria=(
            "The answer cites at least three distinct sources.",
            "The answer names the Chen audit.",
            "The answer names the Nakamura survey.",
        ),
        expected=(False, False, False),
        basis=(
            "counts 2 distinct named sources",
            "contains no 'Chen audit'",
            "contains no 'Nakamura survey'",
        ),
    ),
    RubricCase(
        case_id="evidence-03",
        family="evidence",
        answer=(
            "Exactly three sources were used: the Alvarez review, "
            "the Chen audit, and the Nakamura survey."
        ),
        criteria=(
            "The answer cites at least three distinct sources.",
            "The answer cites exactly four distinct sources.",
            "The answer cites more than three distinct sources.",
        ),
        expected=(True, False, False),
        basis=(
            "counts 3, so the inclusive 'at least 3' boundary holds",
            "counts 3, not 4",
            "counts 3, not more than 3",
        ),
    ),
    RubricCase(
        case_id="evidence-04",
        family="evidence",
        answer=(
            "Four sources were used: the Alvarez review, the Chen audit, "
            "the Nakamura survey, and the Patel memo."
        ),
        criteria=(
            "The answer cites at least four distinct sources.",
            "The answer cites at most four distinct sources.",
            "The answer cites five distinct sources.",
        ),
        expected=(True, True, False),
        basis=(
            "counts 4, so the inclusive 'at least 4' boundary holds",
            "counts 4, so the inclusive 'at most 4' boundary holds",
            "counts 4, not 5",
        ),
    ),
    # ── format ────────────────────────────────────────────────────────────
    RubricCase(
        case_id="format-01",
        family="format",
        answer=(
            "Summary: the migration is safe. The backups are current. The rollback plan is tested."
        ),
        criteria=(
            "The answer is at most three sentences.",
            "The answer contains no exclamation mark.",
            "The answer begins with the word Summary.",
        ),
        expected=(True, True, True),
        basis=(
            "3 sentence-ending periods",
            "no '!' character",
            "starts with 'Summary'",
        ),
    ),
    RubricCase(
        case_id="format-02",
        family="format",
        answer=(
            "Overview of the migration. "
            "The backups are current. "
            "The rollback plan is tested. "
            "No risks were found! "
            "Please approve."
        ),
        criteria=(
            "The answer is at most three sentences.",
            "The answer contains no exclamation mark.",
            "The answer begins with the word Summary.",
        ),
        expected=(False, False, False),
        basis=(
            "5 sentences (4 periods + 1 exclamation)",
            "contains '!'",
            "starts with 'Overview'",
        ),
    ),
    RubricCase(
        case_id="format-03",
        family="format",
        answer=("Summary: the migration is safe. The rollback plan is tested. This is important!"),
        criteria=(
            "The answer is at most three sentences.",
            "The answer contains no exclamation mark.",
            "The answer begins with the word Summary.",
        ),
        expected=(True, False, True),
        basis=(
            "exactly 3 sentences satisfies the inclusive 'at most 3' boundary",
            "contains '!'",
            "starts with 'Summary'",
        ),
    ),
    RubricCase(
        case_id="format-04",
        family="format",
        answer=("Summary: the migration is ready.\n- Backups are current.\n- Rollback is tested."),
        criteria=(
            "The answer contains a bulleted list.",
            "The answer is at most three sentences.",
            "The answer begins with the word Summary.",
        ),
        expected=(True, True, True),
        basis=(
            "contains two '- ' bullet items",
            "3 sentence-ending periods",
            "starts with 'Summary'",
        ),
    ),
    # ── numeric_threshold ─────────────────────────────────────────────────
    RubricCase(
        case_id="threshold-01",
        family="numeric_threshold",
        answer="The p95 latency is 100 ms.",
        criteria=(
            "The answer reports a p95 latency of at most 100 ms.",
            "The answer reports a p95 latency below 100 ms.",
            "The answer reports a p95 latency of at least 100 ms.",
        ),
        expected=(True, False, True),
        basis=(
            "100 satisfies the inclusive 'at most 100' boundary",
            "100 is not below 100",
            "100 satisfies the inclusive 'at least 100' boundary",
        ),
    ),
    RubricCase(
        case_id="threshold-02",
        family="numeric_threshold",
        answer="The p95 latency is 101 ms.",
        criteria=(
            "The answer reports a p95 latency of at most 100 ms.",
            "The answer reports a p95 latency below 100 ms.",
            "The answer reports a p95 latency of at least 100 ms.",
        ),
        expected=(False, False, True),
        basis=(
            "101 exceeds 100",
            "101 is not below 100",
            "101 satisfies 'at least 100'",
        ),
    ),
    RubricCase(
        case_id="threshold-03",
        family="numeric_threshold",
        answer="The success rate is 90.0 percent.",
        criteria=(
            "The answer reports a success rate above 90 percent.",
            "The answer reports a success rate of at least 90 percent.",
            "The answer reports a success rate below 90 percent.",
        ),
        expected=(False, True, False),
        basis=(
            "90.0 is not above 90",
            "90.0 satisfies the inclusive 'at least 90' boundary",
            "90.0 is not below 90",
        ),
    ),
    RubricCase(
        case_id="threshold-04",
        family="numeric_threshold",
        answer="The success rate is 90.1 percent.",
        criteria=(
            "The answer reports a success rate above 90 percent.",
            "The answer reports a success rate of at most 90 percent.",
            "The answer reports a success rate of at least 90 percent.",
        ),
        expected=(True, False, True),
        basis=(
            "90.1 is above 90",
            "90.1 exceeds 90",
            "90.1 satisfies 'at least 90'",
        ),
    ),
    # ── safety ────────────────────────────────────────────────────────────
    RubricCase(
        case_id="safety-01",
        family="safety",
        answer=(
            "General information only; this is not medical advice. "
            "No dosage is provided. "
            "Ask a clinician before changing any treatment."
        ),
        criteria=(
            "The answer states that it is not medical advice.",
            "The answer does not give a dosage figure.",
            "The answer tells the reader to ask a clinician.",
        ),
        expected=(True, True, True),
        basis=(
            "contains 'not medical advice'",
            "contains no dosage figure",
            "contains 'Ask a clinician'",
        ),
    ),
    RubricCase(
        case_id="safety-02",
        family="safety",
        answer=(
            "Take 500 mg twice daily. This will cure the infection. Stop other treatments now."
        ),
        criteria=(
            "The answer states that it is not medical advice.",
            "The answer does not give a dosage figure.",
            "The answer tells the reader to ask a clinician.",
        ),
        expected=(False, False, False),
        basis=(
            "contains no 'not medical advice' disclaimer",
            "contains the dosage figure '500 mg'",
            "contains no instruction to ask a clinician",
        ),
    ),
    RubricCase(
        case_id="safety-03",
        family="safety",
        answer=("Do not take 500 mg without clinician approval. This is not medical advice."),
        criteria=(
            "The answer states that it is not medical advice.",
            "The answer does not give a dosage figure.",
            "The answer warns against taking a dose without approval.",
        ),
        expected=(True, False, True),
        basis=(
            "contains 'not medical advice'",
            "contains the dosage figure '500 mg'",
            "contains 'Do not take 500 mg without clinician approval'",
        ),
    ),
    RubricCase(
        case_id="safety-04",
        family="safety",
        answer=(
            "General wellness advice only. "
            "No treatments are recommended here. "
            "Ask your doctor about any changes."
        ),
        criteria=(
            "The answer states that no treatments are recommended.",
            "The answer does not name a medication.",
            "The answer tells the reader to ask a doctor.",
        ),
        expected=(True, True, True),
        basis=(
            "contains 'No treatments are recommended'",
            "names no medication",
            "contains 'Ask your doctor'",
        ),
    ),
    # ── citation ──────────────────────────────────────────────────────────
    RubricCase(
        case_id="citation-01",
        family="citation",
        answer=('The NIST report states "the control failed" and recommends review.'),
        criteria=(
            'The answer quotes the exact phrase "the control failed".',
            "The answer names NIST as the source.",
            "The answer mentions a page number.",
        ),
        expected=(True, True, False),
        basis=(
            'contains the quoted phrase "the control failed"',
            "contains 'NIST'",
            "contains no page number",
        ),
    ),
    RubricCase(
        case_id="citation-02",
        family="citation",
        answer=(
            "A review states that the control failed, but the source organization is not named."
        ),
        criteria=(
            "The answer reports that the control failed.",
            "The answer names NIST as the source.",
            "The answer mentions a page number.",
        ),
        expected=(True, False, False),
        basis=(
            "contains 'the control failed'",
            "contains no 'NIST'",
            "contains no page number",
        ),
    ),
    RubricCase(
        case_id="citation-03",
        family="citation",
        answer=("According to NIST SP 800-53, the control failed and a finding was raised."),
        criteria=(
            "The answer names NIST as the source.",
            "The answer gives a document identifier.",
            "The answer reports that the control failed.",
        ),
        expected=(True, True, True),
        basis=(
            "contains 'NIST'",
            "contains the identifier 'SP 800-53'",
            "contains 'the control failed'",
        ),
    ),
    RubricCase(
        case_id="citation-04",
        family="citation",
        answer=("According to ISO 27001, the control passed and no finding was raised."),
        criteria=(
            "The answer names NIST as the source.",
            "The answer reports that the control failed.",
            "The answer states that no finding was raised.",
        ),
        expected=(False, False, True),
        basis=(
            "names 'ISO 27001', not 'NIST'",
            "says the control passed",
            "contains 'no finding was raised'",
        ),
    ),
)


def build_cases() -> list[RubricCase]:
    """Return the frozen catalogue (24 cases, 72 criteria) in a fixed order.

    Pure function of module constants: no clock, no randomness, no
    environment, so two invocations address the same cases in the same order
    and their receipts are comparable. The returned dataclasses are frozen.
    """
    return list(_RUBRIC_CASES)


def criterion_question_id(index: int) -> str:
    """Stable typed-question id for criterion ``index`` (zero-based)."""
    return f"{_CRITERION_PREFIX}-{index:02d}"


def build_typed_questions(case: RubricCase) -> list[Question]:
    """Project one case into the typed judge's question catalogue.

    One ``noul`` question per criterion, in criterion order, plus one ``choice``
    overall question (``pass`` iff every criterion is satisfied). The answer
    text is passed as the runner's ``state`` (see ``_judge_state``).
    """
    questions = [
        Question(
            id=criterion_question_id(index),
            kind=QuestionKind.NOUL,
            text=f"Does the ANSWER satisfy criterion {index + 1}: {criterion}",
        )
        for index, criterion in enumerate(case.criteria)
    ]
    questions.append(
        Question(
            id=OVERALL_ID,
            kind=QuestionKind.CHOICE,
            text=(
                "Overall verdict: does the ANSWER satisfy EVERY criterion above? "
                "Answer pass only when all criteria are satisfied."
            ),
            options=OVERALL_OPTIONS,
        )
    )
    return questions


def _judge_state(case: RubricCase) -> str:
    return f"ANSWER UNDER REVIEW:\n{case.answer}"


# ── Digest / manifest (CJ-14) ─────────────────────────────────────────────


def _case_payload(case: RubricCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "family": case.family,
        "answer": case.answer,
        "criteria": list(case.criteria),
        "expected": list(case.expected),
        "basis": list(case.basis),
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def case_sha256(case: RubricCase) -> str:
    """SHA-256 of the canonical case payload (answer + criteria + truth)."""
    payload = json.dumps(
        _case_payload(case), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return _sha256_text(payload)


def rubric_set_sha256(cases: Sequence[RubricCase]) -> str:
    """SHA-256 over the ordered canonical case payloads: the frozen identity."""
    payloads = [
        json.dumps(_case_payload(case), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        for case in cases
    ]
    return _sha256_text("[" + ",".join(payloads) + "]")


def blind_prior(cases: Sequence[RubricCase]) -> dict[str, Any]:
    """The answer-blind lower bound: one majority verdict per criterion.

    Computed from the frozen expected distribution, never from an answer.
    Any reader whose accuracy does not beat this floor has demonstrated
    nothing; a reader that only matches it has learned nothing from the
    answer.
    """
    criteria_total = sum(len(case.criteria) for case in cases)
    pass_count = sum(sum(1 for verdict in case.expected if verdict) for case in cases)
    fail_count = criteria_total - pass_count
    criterion_majority_pass = pass_count >= fail_count
    criterion_correct = max(pass_count, fail_count)

    overall_pass = sum(1 for case in cases if case.expected_overall == OVERALL_PASS)
    overall_fail = len(cases) - overall_pass
    overall_majority_pass = overall_pass >= overall_fail
    overall_correct = max(overall_pass, overall_fail)

    return {
        "description": (
            "answer-blind prior: emits the majority verdict for every criterion "
            "without reading any answer; the lower bound a reader must beat"
        ),
        "criterion_majority_verdict": OVERALL_PASS if criterion_majority_pass else OVERALL_FAIL,
        "criterion_accuracy": criterion_correct / criteria_total if criteria_total else None,
        "overall_majority_verdict": OVERALL_PASS if overall_majority_pass else OVERALL_FAIL,
        "overall_accuracy": overall_correct / len(cases) if cases else None,
    }


# ── Reader A: typed judge (run_typed_decisions) ───────────────────────────


def _run_typed_arm(
    primitives: Any,
    cases: Sequence[RubricCase],
    role: str,
) -> tuple[dict[str, dict[str, Any]], float]:
    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for case in cases:
        records[case.case_id] = _run_typed_case(primitives, case, role)
    return records, (time.perf_counter() - started) * 1000.0


def _run_typed_case(primitives: Any, case: RubricCase, role: str) -> dict[str, Any]:
    criterion_ids = [criterion_question_id(index) for index in range(len(case.criteria))]
    record: dict[str, Any] = {
        "values": [None] * len(case.criteria),
        "overall": None,
        "confidence": None,
        "failures": [],
        "error": None,
        "prompt_sha256": None,
        "raw_text_chars": 0,
        "wall_ms": None,
        "tokens": None,
        "meta": None,
    }
    started = time.perf_counter()
    try:
        result = run_typed_decisions(
            primitives,
            state=_judge_state(case),
            questions=build_typed_questions(case),
            role=role,
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a scored failure
        record["error"] = f"{type(exc).__name__}: {exc}"
    else:
        decisions = {decision.question_id: decision for decision in result.decisions}
        record["values"] = [
            decisions[question_id].value if question_id in decisions else None
            for question_id in criterion_ids
        ]
        overall = decisions.get(OVERALL_ID)
        record["overall"] = overall.value if overall is not None else None
        confidences = [
            float(decisions[question_id].confidence)
            for question_id in criterion_ids
            if question_id in decisions
        ]
        record["confidence"] = sum(confidences) / len(confidences) if confidences else None
        record["failures"] = [
            {"reason": failure.reason, "detail": failure.detail} for failure in result.failures
        ]
        record["prompt_sha256"] = result.prompt_sha256
        record["raw_text_chars"] = len(result.raw_text)
    record["wall_ms"] = (time.perf_counter() - started) * 1000.0
    _attach_tokens(record, primitives)
    return record


# ── Reader B: plain LLM judge (one llm_call per case) ─────────────────────


def _llm_prompt(case: RubricCase) -> str:
    criteria_lines = "\n".join(
        f"{index + 1}. {criterion}" for index, criterion in enumerate(case.criteria)
    )
    count = len(case.criteria)
    return (
        "Grade ONE answer against a numbered rubric.\n\n"
        f"ANSWER:\n{case.answer}\n\n"
        f"RUBRIC CRITERIA:\n{criteria_lines}\n\n"
        "Judge each criterion independently as true (satisfied) or false "
        "(not satisfied).\n"
        f'Return EXACTLY ONE JSON object of the form {{"verdicts": [true, false, ...]}} '
        f"with exactly {count} boolean values, in criterion order, and nothing else "
        "(no prose, no markdown fences).\n"
    )


def _llm_response_schema(expected_count: int) -> dict[str, Any]:
    """Constrain reader B's own shape without changing its independent prompt."""
    return {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "array",
                "items": {"type": "boolean"},
                "minItems": expected_count,
                "maxItems": expected_count,
            }
        },
        "required": ["verdicts"],
        "additionalProperties": False,
    }


def _llm_instrument(cases: Sequence[RubricCase]) -> dict[str, Any]:
    """Identify constrained reader B separately from historical free-form runs."""
    return {
        "id": "independent_llm_judge_constrained_verdicts_v1",
        "calls_per_case": 1,
        "repair_calls": 0,
        "response_schemas": [
            {
                "case_id": case.case_id,
                "schema_sha256": _sha256_text(
                    json.dumps(
                        _llm_response_schema(len(case.criteria)),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
            }
            for case in cases
        ],
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys instead of silently keeping the last value."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_verdicts(
    raw_text: str, expected_count: int
) -> tuple[tuple[bool, ...] | None, str | None]:
    """Extract + validate one LLM-judge emission into ``(verdicts, error)``.

    The accepted emission is exactly one balanced JSON object with a
    ``"verdicts"`` list holding one boolean per criterion in order. Anything
    else -- no balanced JSON, a non-list ``verdicts``, wrong length,
    non-boolean entries -- returns an error and no verdicts; the caller
    counts every criterion in that case as unresolved (wrong for accuracy,
    not compared for agreement).
    """
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        return None, "no balanced JSON object found"
    try:
        parsed = json.loads(candidate, object_pairs_hook=_unique_object)
    except ValueError as exc:
        return None, f"JSON decode error: {exc}"
    if not isinstance(parsed, Mapping) or set(parsed) != {"verdicts"}:
        return None, "expected exactly one object key: verdicts"
    parsed = parsed["verdicts"]
    if not isinstance(parsed, list):
        return None, "'verdicts' is not a list"
    if len(parsed) != expected_count:
        return None, f"expected {expected_count} verdicts, got {len(parsed)}"
    invalid = [index for index, value in enumerate(parsed) if not isinstance(value, bool)]
    if invalid:
        return None, f"verdicts at positions {invalid} are not booleans"
    return tuple(parsed), None


def _llm_overall(verdicts: Sequence[bool]) -> str:
    """The LLM reader's overall verdict: the conjunction of its own list."""
    return OVERALL_PASS if all(verdicts) else OVERALL_FAIL


def _run_llm_arm(
    primitives: Any,
    cases: Sequence[RubricCase],
    role: str,
) -> tuple[dict[str, dict[str, Any]], float]:
    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for case in cases:
        records[case.case_id] = _run_llm_case(primitives, case, role)
    return records, (time.perf_counter() - started) * 1000.0


def _run_llm_case(primitives: Any, case: RubricCase, role: str) -> dict[str, Any]:
    prompt = _llm_prompt(case)
    record: dict[str, Any] = {
        "verdicts": None,
        "overall": None,
        "error": None,
        "raw_text": "",
        "prompt_sha256": _sha256_text(prompt),
        "wall_ms": None,
        "tokens": None,
        "meta": None,
    }
    started = time.perf_counter()
    try:
        raw_text = str(
            primitives.llm_call(
                prompt,
                role=role,
                n_tokens=_LLM_N_TOKENS,
                json_schema=_llm_response_schema(len(case.criteria)),
                temperature=0.0,
                seed=_DECODE_SEED,
            )
            or ""
        )
    except Exception as exc:  # noqa: BLE001 - any failure is a scored failure
        record["error"] = f"{type(exc).__name__}: {exc}"
    else:
        record["raw_text"] = raw_text
        verdicts, error = _parse_verdicts(raw_text, len(case.criteria))
        record["error"] = error
        if verdicts is not None:
            record["verdicts"] = [bool(verdict) for verdict in verdicts]
            record["overall"] = _llm_overall(verdicts)
    record["wall_ms"] = (time.perf_counter() - started) * 1000.0
    _attach_tokens(record, primitives)
    return record


def _attach_tokens(record: dict[str, Any], primitives: Any) -> None:
    meta = _last_inference_meta(primitives)
    record["meta"] = meta
    if isinstance(meta, Mapping):
        tokens = meta.get("tokens")
        if isinstance(tokens, (int, float)) and not isinstance(tokens, bool):
            record["tokens"] = float(tokens)


# ── Scoring: accuracy per reader ──────────────────────────────────────────


def _values_for(reader: str, case: RubricCase, record: Mapping[str, Any]) -> list[Any]:
    if reader == "typed":
        return list(record["values"])
    verdicts = record["verdicts"]
    if verdicts is None:
        return [None] * len(case.criteria)
    return list(verdicts)


def _reader_summary(
    cases: Sequence[RubricCase],
    records: Mapping[str, Mapping[str, Any]],
    *,
    reader: str,
    wall_ms: float,
    with_confidence: bool,
) -> dict[str, Any]:
    criteria_total = sum(len(case.criteria) for case in cases)
    criteria_resolved = 0
    criteria_correct = 0
    case_correct = 0
    overall_resolved = 0
    overall_correct = 0
    errored_cases = 0
    tokens: list[float] = []
    confidences: list[float] = []

    for case in cases:
        record = records[case.case_id]
        values = _values_for(reader, case, record)
        resolved = 0
        correct = 0
        for value, expected in zip(values, case.expected, strict=True):
            if value is None:
                continue
            resolved += 1
            if isinstance(value, bool) and value == expected:
                correct += 1
        criteria_resolved += resolved
        criteria_correct += correct
        if correct == len(case.criteria):
            case_correct += 1
        overall = record["overall"]
        if overall is not None:
            overall_resolved += 1
            if overall == case.expected_overall:
                overall_correct += 1
        if record["error"] is not None or resolved == 0:
            errored_cases += 1
        case_tokens = record["tokens"]
        if isinstance(case_tokens, (int, float)) and not isinstance(case_tokens, bool):
            tokens.append(float(case_tokens))
        confidence = record.get("confidence")
        if with_confidence and isinstance(confidence, (int, float)):
            confidences.append(float(confidence))

    criteria_unresolved = criteria_total - criteria_resolved
    return {
        "reader": reader,
        "criteria_total": criteria_total,
        "criteria_resolved": criteria_resolved,
        "criteria_correct": criteria_correct,
        "criteria_wrong": criteria_resolved - criteria_correct,
        "criteria_unresolved": criteria_unresolved,
        "failures": criteria_unresolved,
        "errored_cases": errored_cases,
        "criterion_accuracy": criteria_correct / criteria_total if criteria_total else None,
        "case_accuracy": case_correct / len(cases) if cases else None,
        "overall_resolved": overall_resolved,
        "overall_correct": overall_correct,
        "overall_accuracy": overall_correct / len(cases) if cases else None,
        "wall_ms": wall_ms,
        "tokens_generated": sum(tokens) if tokens else None,
        "calls_with_token_meta": len(tokens),
        "mean_confidence": (
            sum(confidences) / len(confidences) if with_confidence and confidences else None
        ),
    }


# ── Scoring: reader agreement (separate from accuracy) ────────────────────


def _agreement(
    cases: Sequence[RubricCase],
    typed_records: Mapping[str, Mapping[str, Any]],
    llm_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Pairwise criterion agreement between the readers, unresolved-aware.

    Only criteria resolved by BOTH readers are compared; a pair missing on
    either side is ``unresolved``, never an agreement and never a
    disagreement. ``human_adjudication_*`` counts criteria that a two-reader
    redundancy policy cannot settle (readers disagree OR at least one reader
    failed); it is a projection from reader behaviour, not a measured human
    cost. Agreement is explicitly NOT accuracy: neither reader is ground
    truth, so this block carries its own label.
    """
    criteria_total = sum(len(case.criteria) for case in cases)
    compared = 0
    agreeing = 0
    unresolved = 0
    overall_compared = 0
    overall_agreeing = 0
    disagreements: list[dict[str, Any]] = []
    per_case: dict[str, dict[str, int]] = {}

    for case in cases:
        typed_values = _values_for("typed", case, typed_records[case.case_id])
        llm_values = _values_for("llm", case, llm_records[case.case_id])
        case_compared = 0
        case_agreeing = 0
        for index, (typed_value, llm_value) in enumerate(
            zip(typed_values, llm_values, strict=True)
        ):
            if typed_value is None or llm_value is None:
                unresolved += 1
                continue
            compared += 1
            case_compared += 1
            if type(typed_value) is type(llm_value) and typed_value == llm_value:
                agreeing += 1
                case_agreeing += 1
            else:
                disagreements.append(
                    {
                        "case_id": case.case_id,
                        "criterion_index": index,
                        "typed": typed_value,
                        "llm": llm_value,
                    }
                )
        per_case[case.case_id] = {"compared": case_compared, "agreeing": case_agreeing}
        typed_overall = typed_records[case.case_id]["overall"]
        llm_overall = llm_records[case.case_id]["overall"]
        if typed_overall is not None and llm_overall is not None:
            overall_compared += 1
            if typed_overall == llm_overall:
                overall_agreeing += 1

    adjudication = criteria_total - agreeing
    return {
        "label": AGREEMENT_LABEL,
        "note": (
            "Agreement is between two cheap readers that may share failure modes; "
            "it is NOT accuracy against the frozen ground truth. Accuracy appears "
            "only in the per-reader summaries and in blind_prior."
        ),
        "compared": compared,
        "agreeing": agreeing,
        "disagreeing": compared - agreeing,
        "unresolved": unresolved,
        "rate": agreeing / compared if compared else None,
        "human_adjudication_criteria": adjudication,
        "human_adjudication_rate": (adjudication / criteria_total if criteria_total else None),
        "overall": {
            "compared": overall_compared,
            "agreeing": overall_agreeing,
            "rate": overall_agreeing / overall_compared if overall_compared else None,
        },
        "disagreements": disagreements,
        "per_case": per_case,
    }


# ── Run ───────────────────────────────────────────────────────────────────


def run_judge_redundancy(
    primitives: Any,
    *,
    role: str,
    receipt_path: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run both readers over the frozen rubric set and return the receipt.

    Args:
        primitives: The ``LLMPrimitives`` seam (anything exposing
            ``llm_call``). Required unless ``dry_run``; a mock-mode object is
            refused because the receipt would be fabricated.
        role: Registry role every call is charged to.
        receipt_path: Explicit receipt path; default is
            ``<artifacts_dir or tmp>/typed_decisions/judge_redundancy-<stamp>.json``.
        artifacts_dir: Base directory for the default receipt path.
        dry_run: Return the pre-registration plan without calling a model and
            without writing a receipt (CLI ``--dry-run``).

    Returns:
        The receipt dict (``study``, ``timestamp``, ``mode``, ``role``,
        ``agreement_label``, ``manifest``, ``counts``, ``results``,
        ``metric_directions``, ``prompt_sha256``) including the resolved
        ``receipt_path``.

    Raises:
        MeasurementError: no live primitives, mock-mode primitives, or a run
            in which neither reader resolved a single criterion.
    """
    cases = build_cases()

    if dry_run:
        return _dry_run_plan(cases, role)

    _require_primitives(primitives, "judge-redundancy")
    if getattr(primitives, "mock_mode", False):
        raise MeasurementError(
            "primitives is in mock_mode; judge-redundancy numbers would be fabricated"
        )

    typed_records, typed_wall_ms = _run_typed_arm(primitives, cases, role)
    llm_records, llm_wall_ms = _run_llm_arm(primitives, cases, role)

    typed_summary = _reader_summary(
        cases, typed_records, reader="typed", wall_ms=typed_wall_ms, with_confidence=True
    )
    llm_summary = _reader_summary(
        cases, llm_records, reader="llm", wall_ms=llm_wall_ms, with_confidence=False
    )
    if typed_summary["criteria_resolved"] == 0 and llm_summary["criteria_resolved"] == 0:
        raise MeasurementError(
            "judge-redundancy study resolved no criterion in either reader; "
            "there is nothing to score or compare"
        )

    agreement = _agreement(cases, typed_records, llm_records)
    prior = blind_prior(cases)
    rows = _case_rows(cases, typed_records, llm_records, agreement)
    expected_pass = sum(sum(1 for verdict in case.expected if verdict) for case in cases)
    criteria_total = sum(len(case.criteria) for case in cases)

    receipt = {
        "study": STUDY,
        "timestamp": _now_iso(),
        "mode": MODE,
        "role": role,
        "agreement_label": AGREEMENT_LABEL,
        "manifest": {
            "llm_instrument": _llm_instrument(cases),
            "rubric_set_sha256": rubric_set_sha256(cases),
            "cases": [
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "case_sha256": case_sha256(case),
                    "criteria": len(case.criteria),
                    "expected_pass": sum(1 for verdict in case.expected if verdict),
                    "expected_fail": sum(1 for verdict in case.expected if not verdict),
                }
                for case in cases
            ],
        },
        "counts": {
            "cases": len(cases),
            "criteria": criteria_total,
            "expected_pass_criteria": expected_pass,
            "expected_fail_criteria": criteria_total - expected_pass,
            "typed_criteria_resolved": typed_summary["criteria_resolved"],
            "typed_criteria_correct": typed_summary["criteria_correct"],
            "typed_failures": typed_summary["failures"],
            "llm_criteria_resolved": llm_summary["criteria_resolved"],
            "llm_criteria_correct": llm_summary["criteria_correct"],
            "llm_failures": llm_summary["failures"],
            "agreement_compared_criteria": agreement["compared"],
            "agreement_agreeing_criteria": agreement["agreeing"],
            "agreement_unresolved_criteria": agreement["unresolved"],
            "human_adjudication_criteria": agreement["human_adjudication_criteria"],
        },
        "results": {
            "readers": {"typed": typed_summary, "llm": llm_summary},
            "agreement": agreement,
            "blind_prior": prior,
            "cases": rows,
        },
        "metric_directions": dict(_METRIC_DIRECTIONS),
        "prompt_sha256": {
            "typed": [typed_records[case.case_id]["prompt_sha256"] for case in cases],
            "llm": [llm_records[case.case_id]["prompt_sha256"] for case in cases],
        },
    }
    _write_receipt(receipt, receipt_path=receipt_path, artifacts_dir=artifacts_dir)
    return receipt


_METRIC_DIRECTIONS: dict[str, str] = {
    "typed_criterion_accuracy": "higher_better",
    "typed_case_accuracy": "higher_better",
    "typed_overall_accuracy": "higher_better",
    "llm_criterion_accuracy": "higher_better",
    "llm_case_accuracy": "higher_better",
    "llm_overall_accuracy": "higher_better",
    "typed_failures": "lower_better",
    "llm_failures": "lower_better",
    "typed_wall_ms": "lower_better",
    "llm_wall_ms": "lower_better",
    "typed_tokens_generated": "lower_better",
    "llm_tokens_generated": "lower_better",
    "agreement_rate": "higher_better",
    "human_adjudication_rate": "lower_better",
    "blind_prior_criterion_accuracy": "higher_better",
}


def _dry_run_plan(cases: Sequence[RubricCase], role: str) -> dict[str, Any]:
    criteria_total = sum(len(case.criteria) for case in cases)
    expected_pass = sum(sum(1 for verdict in case.expected if verdict) for case in cases)
    return {
        "study": STUDY,
        "dry_run": True,
        "plan": {
            "mode": MODE,
            "role": role,
            "readers": ["typed", "llm"],
            "llm_instrument": _llm_instrument(cases),
            "agreement_label": AGREEMENT_LABEL,
            "rubric_set_sha256": rubric_set_sha256(cases),
            "cases": [
                {
                    "case_id": case.case_id,
                    "family": case.family,
                    "answer_sha256": _sha256_text(case.answer),
                    "case_sha256": case_sha256(case),
                    "criteria": list(case.criteria),
                    "expected": list(case.expected),
                    "expected_overall": case.expected_overall,
                    "basis": list(case.basis),
                }
                for case in cases
            ],
            "counts": {
                "cases": len(cases),
                "criteria": criteria_total,
                "expected_pass_criteria": expected_pass,
                "expected_fail_criteria": criteria_total - expected_pass,
            },
            "blind_prior": blind_prior(cases),
        },
    }


def _case_rows(
    cases: Sequence[RubricCase],
    typed_records: Mapping[str, Mapping[str, Any]],
    llm_records: Mapping[str, Mapping[str, Any]],
    agreement: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        typed = typed_records[case.case_id]
        llm = llm_records[case.case_id]
        typed_values = _values_for("typed", case, typed)
        llm_values = _values_for("llm", case, llm)
        typed_flags = _correct_flags(typed_values, case.expected)
        llm_flags = _correct_flags(llm_values, case.expected)
        per_case = agreement["per_case"][case.case_id]
        rows.append(
            {
                "case_id": case.case_id,
                "family": case.family,
                "answer_sha256": _sha256_text(case.answer),
                "case_sha256": case_sha256(case),
                "criteria": list(case.criteria),
                "expected": list(case.expected),
                "expected_overall": case.expected_overall,
                "typed": {
                    "values": typed_values,
                    "correct": typed_flags,
                    "failures": list(typed["failures"]),
                    "overall": typed["overall"],
                    "overall_correct": (
                        typed["overall"] == case.expected_overall
                        if typed["overall"] is not None
                        else None
                    ),
                    "confidence": typed["confidence"],
                    "wall_ms": typed["wall_ms"],
                    "tokens": typed["tokens"],
                    "error": typed["error"],
                    "prompt_sha256": typed["prompt_sha256"],
                },
                "llm": {
                    "verdicts": llm_values,
                    "criterion_verdicts": {
                        criterion_question_id(index): value
                        for index, value in enumerate(llm_values)
                    },
                    "correct": llm_flags,
                    "overall": llm["overall"],
                    "overall_correct": (
                        llm["overall"] == case.expected_overall
                        if llm["overall"] is not None
                        else None
                    ),
                    "wall_ms": llm["wall_ms"],
                    "tokens": llm["tokens"],
                    "error": llm["error"],
                    "raw_text": llm["raw_text"],
                    "prompt_sha256": llm["prompt_sha256"],
                },
                "agreement": {
                    "compared": per_case["compared"],
                    "agreeing": per_case["agreeing"],
                },
            }
        )
    return rows


def _correct_flags(values: Sequence[Any], expected: Sequence[bool]) -> list[bool | None]:
    flags: list[bool | None] = []
    for value, verdict in zip(values, expected, strict=True):
        if value is None:
            flags.append(None)
        else:
            flags.append(isinstance(value, bool) and value == verdict)
    return flags


# ── CLI ───────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.typed_decisions.judge_redundancy",
        description=(
            "CJ-13/CJ-14 judge redundancy: typed judge vs plain LLM judge over a "
            "frozen 24-case / 72-criterion rubric set. Real model calls require --live."
        ),
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="allow real model calls; without this (or --dry-run) the harness refuses to run",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the pre-registered plan without calling the model or writing a receipt",
    )
    parser.add_argument("--role", default="worker", help="registry role the calls are charged to")
    parser.add_argument(
        "--server-url",
        default=None,
        help=(
            "optional single-endpoint override (e.g. http://127.0.0.1:8199): binds "
            "--role to exactly this llama-server after a /health probe; without it "
            "the configured server map is used"
        ),
    )
    parser.add_argument(
        "--receipt",
        default=None,
        help=(
            "receipt path (default: <artifacts-dir or tmp>/typed_decisions/"
            "judge_redundancy-<stamp>.json)"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="directory for the default receipt path",
    )
    return parser


def _live_primitives(*, server_url: str | None = None, role: str = "worker") -> Any:
    """Build live primitives; an optional single-endpoint override.

    Without ``server_url`` this defers to the measure harness's configured
    primitives (the live stack's server map). With a URL, ``role`` is bound to
    exactly that endpoint after a ``/health`` readiness probe — the TD-7
    convention — so a live run against an ad-hoc GPU server does not depend on
    the live config.
    """
    if server_url is None:
        return _configured_live_primitives()

    import httpx

    try:
        health = httpx.get(f"{server_url.rstrip('/')}/health", timeout=5.0)
        health.raise_for_status()
        payload = health.json()
    except Exception as exc:  # noqa: BLE001 - any failure means "server not usable"
        raise MeasurementError(f"llama-server health check failed at {server_url}: {exc}") from exc
    if str(payload.get("status")) != "ok":
        raise MeasurementError(f"llama-server at {server_url} is not ready: {payload!r}")

    from src.llm_primitives import LLMPrimitives

    primitives = LLMPrimitives(
        mock_mode=False,
        server_urls={role: server_url},
        num_slots=4,
    )
    if not getattr(primitives, "_backends", None):
        raise MeasurementError(
            f"no LLM backends available for {server_url!r} (role {role!r}); is the server up?"
        )
    return primitives


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 ok, 1 study error, 2 gate)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.live and not args.dry_run:
        print(
            "refusing to run: pass --live for real model calls or --dry-run to print the plan",
            file=sys.stderr,
        )
        return 2

    try:
        primitives = (
            _live_primitives(server_url=args.server_url, role=args.role)
            if (args.live and not args.dry_run)
            else None
        )
        receipt = run_judge_redundancy(
            primitives,
            role=args.role,
            receipt_path=args.receipt,
            artifacts_dir=args.artifacts_dir,
            dry_run=args.dry_run,
        )
    except (MeasurementError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(receipt, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
