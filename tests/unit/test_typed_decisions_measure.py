"""Unit tests for the TD-2/TD-3 measurement harnesses.

All studies run against ``_FakePrimitives`` (the canned-response pattern from
``tests/unit/test_typed_decisions.py``): no model or server call is made. The
calibration fixture is constructed so ECE and Brier are exactly representable
binary fractions, and the contamination fixture's answers depend on question
POSITION, so the expected per-question flip counts are recomputed in-test from
the orderings the receipt reports.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from src.typed_decisions.measure import (
    MeasurementError,
    _load_questions,
    _last_inference_meta,
    main,
    run_calibration_study,
    run_contamination_study,
    run_fanout_study,
    run_parallel_fanout_study,
)
from src.typed_decisions.types import Question, QuestionKind
from src.typed_decisions.bench import _load_questions as _load_bench_questions

ROLE = "worker"
STATE = "unit-test state"


class _FakePrimitives:
    """Ordered canned-response stand-in with a fake inference-meta channel."""

    def __init__(self, responder: Callable[..., str]):
        self.responder = responder
        self.calls: list[dict] = []
        self._last_inference_meta: dict = {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        self._last_inference_meta = {
            "tokens": 11,
            "prompt_ms": 1.5,
            "gen_ms": 2.5,
            "elapsed_ms": 4.0,
        }
        return self.responder(prompt, **kwargs)


@pytest.mark.parametrize("loader", [_load_questions, _load_bench_questions])
def test_question_json_loader_preserves_optional_descriptions_and_legacy_default(
    loader, tmp_path: Path
):
    described_path = tmp_path / "described.json"
    described_path.write_text(
        json.dumps({"questions": [{
            "id": "q", "kind": "choice", "text": "Pick.",
            "options": ["alpha", "beta"],
            "option_descriptions": ["first meaning", None],
        }]}),
        encoding="utf-8",
    )
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(
        json.dumps([{
            "id": "q", "kind": "choice", "text": "Pick.",
            "options": ["alpha", "beta"],
        }]),
        encoding="utf-8",
    )

    assert loader(described_path)[0].option_descriptions == ("first meaning", None)
    assert loader(legacy_path)[0].option_descriptions == ()


@pytest.mark.parametrize("loader", [_load_questions, _load_bench_questions])
def test_question_json_loader_rejects_string_as_description_sequence(loader, tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps({"questions": [{
            "id": "q", "kind": "choice", "text": "Pick.",
            "options": ["x", "y"], "option_descriptions": "xy",
        }]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sequence, not text or bytes"):
        loader(path)


def _parse_catalog(prompt: str) -> list[dict]:
    """Extract ``(id, kind, candidates)`` per question, in catalogue order."""
    entries: list[dict] = []
    current: dict | None = None
    for line in prompt.splitlines():
        match = re.match(r"^Q\d+ id=(\S+) kind=(\S+)$", line)
        if match:
            current = {"id": match.group(1), "kind": match.group(2), "candidates": []}
            entries.append(current)
            continue
        if current is not None and line.startswith("  candidates: "):
            current["candidates"] = line[len("  candidates: ") :].split(" | ")
    return entries


def _rotation_responder(prompt: str, **kwargs) -> str:
    """Answer by catalogue POSITION, so reordering changes answers."""
    answers: dict[str, dict] = {}
    for position, entry in enumerate(_parse_catalog(prompt)):
        candidates = entry["candidates"]
        if entry["kind"] == "choice":
            answers[entry["id"]] = {
                "choice": candidates[position % len(candidates)],
                "probabilities": {label: 1.0 / len(candidates) for label in candidates},
                "confidence": 0.5,
            }
        elif entry["kind"] == "score":
            levels = [int(candidate) for candidate in candidates]
            answers[entry["id"]] = {
                "score": levels[position % len(levels)],
                "probabilities": {str(level): 1.0 / len(levels) for level in levels},
                "confidence": 0.5,
            }
        else:
            answers[entry["id"]] = {
                "noul": position % 2 == 0,
                "probabilities": {"true": 0.5, "false": 0.5},
                "confidence": 0.5,
            }
    return json.dumps({"answers": answers})


def _stable_responder(prompt: str, **kwargs) -> str:
    """Answer identically regardless of catalogue position or batching."""
    answers: dict[str, dict] = {}
    for entry in _parse_catalog(prompt):
        candidates = entry["candidates"]
        if entry["kind"] == "choice":
            answers[entry["id"]] = {
                "choice": candidates[0],
                "probabilities": {label: 1.0 / len(candidates) for label in candidates},
                "confidence": 0.5,
            }
        elif entry["kind"] == "score":
            levels = [int(candidate) for candidate in candidates]
            answers[entry["id"]] = {
                "score": levels[0],
                "probabilities": {str(level): 1.0 / len(levels) for level in levels},
                "confidence": 0.5,
            }
        else:
            answers[entry["id"]] = {
                "noul": True,
                "probabilities": {"true": 0.5, "false": 0.5},
                "confidence": 0.5,
            }
    return json.dumps({"answers": answers})


def _same_value(left: object, right: object) -> bool:
    return type(left) is type(right) and left == right


# ── 1. Contamination (TD-2) ───────────────────────────────────────────────

CONTAMINATION_QUESTIONS = (
    Question(
        id="choice-0",
        kind=QuestionKind.CHOICE,
        text="Pick the deployment strategy.",
        options=("alpha", "beta", "gamma"),
    ),
    Question(
        id="choice-1",
        kind=QuestionKind.CHOICE,
        text="Pick the rollback strategy.",
        options=("revert", "hold"),
    ),
    Question(
        id="score-0",
        kind=QuestionKind.SCORE,
        text="Rate the change risk.",
        levels=(0, 1, 2, 3),
    ),
    Question(id="noul-0", kind=QuestionKind.NOUL, text="Is the invariant satisfied?"),
)


def _expected_answer(question: Question, position: int) -> object:
    if question.kind is QuestionKind.CHOICE:
        return question.options[position % len(question.options)]
    if question.kind is QuestionKind.SCORE:
        return question.levels[position % len(question.levels)]
    return position % 2 == 0


class TestContaminationStudy:
    def test_flip_rate_matches_recomputed_expectation(self, tmp_path: Path):
        primitives = _FakePrimitives(_rotation_responder)
        receipt_path = tmp_path / "contamination.json"

        receipt = run_contamination_study(
            primitives,
            state=STATE,
            questions=CONTAMINATION_QUESTIONS,
            role=ROLE,
            order_permutations=4,
            seed=7,
            receipt_path=receipt_path,
        )

        results = receipt["results"]
        canonical_order = results["canonical"]["order"]
        assert canonical_order == [question.id for question in CONTAMINATION_QUESTIONS]
        orders = [canonical_order] + [record["order"] for record in results["permutations"]]
        assert len(orders) == 4
        assert len({tuple(order) for order in orders}) == 4
        assert all(order != canonical_order for order in orders[1:])

        by_id = {question.id: question for question in CONTAMINATION_QUESTIONS}
        answers_per_order = [
            {
                question_id: _expected_answer(by_id[question_id], position)
                for position, question_id in enumerate(order)
            }
            for order in orders
        ]
        canonical_answers = answers_per_order[0]
        expected_flips: dict[str, int] = {}
        expected_compared = 0
        expected_total = 0
        for question_id in canonical_order:
            flips = 0
            for answers in answers_per_order[1:]:
                if not _same_value(answers[question_id], canonical_answers[question_id]):
                    flips += 1
            expected_flips[question_id] = flips
            expected_compared += len(answers_per_order) - 1
            expected_total += flips

        assert expected_total > 0, "fixture must produce at least one flip"
        reported = {row["question_id"]: row for row in results["per_question"]}
        for question_id, flips in expected_flips.items():
            assert reported[question_id]["flips"] == flips
            assert reported[question_id]["compared"] == 3
        assert receipt["counts"] == {
            "questions": 4,
            "orderings": 4,
            "comparable_pairs": expected_compared,
            "flips": expected_total,
            "unresolved_pairs": 0,
            "canonical_unresolved": 0,
        }
        assert results["flip_rate"] == expected_total / expected_compared

        expected_hashes = [
            hashlib.sha256(call["prompt"].encode("utf-8")).hexdigest() for call in primitives.calls
        ]
        assert len(primitives.calls) == 4
        assert receipt["prompt_sha256"] == expected_hashes

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["receipt_path"] == str(receipt_path)
        assert receipt["study"] == "contamination"
        assert receipt["role"] == ROLE
        assert receipt["mode"] == "json"
        assert receipt["timestamp"]

    def test_orderings_are_deterministic_for_a_seed(self):
        first = run_contamination_study(
            None,
            state=STATE,
            questions=CONTAMINATION_QUESTIONS,
            role=ROLE,
            seed=3,
            dry_run=True,
        )
        second = run_contamination_study(
            None,
            state=STATE,
            questions=CONTAMINATION_QUESTIONS,
            role=ROLE,
            seed=3,
            dry_run=True,
        )
        other_seed = run_contamination_study(
            None,
            state=STATE,
            questions=CONTAMINATION_QUESTIONS,
            role=ROLE,
            seed=4,
            dry_run=True,
        )

        assert first["plan"]["orderings"] == second["plan"]["orderings"]
        assert first["plan"]["orderings"][0] == [
            question.id for question in CONTAMINATION_QUESTIONS
        ]
        assert other_seed["plan"]["orderings"] != first["plan"]["orderings"]

    def test_dry_run_makes_no_calls_and_writes_no_receipt(self, tmp_path: Path):
        primitives = _FakePrimitives(_rotation_responder)
        receipt_path = tmp_path / "contamination.json"

        receipt = run_contamination_study(
            primitives,
            state=STATE,
            questions=CONTAMINATION_QUESTIONS,
            role=ROLE,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_live_primitives_are_required(self):
        with pytest.raises(MeasurementError):
            run_contamination_study(
                None,
                state=STATE,
                questions=CONTAMINATION_QUESTIONS,
                role=ROLE,
            )
        with pytest.raises(MeasurementError):
            run_contamination_study(
                object(),
                state=STATE,
                questions=CONTAMINATION_QUESTIONS,
                role=ROLE,
            )

    def test_requires_at_least_two_orderings_and_two_questions(self):
        with pytest.raises(ValueError):
            run_contamination_study(
                _FakePrimitives(_rotation_responder),
                state=STATE,
                questions=CONTAMINATION_QUESTIONS,
                role=ROLE,
                order_permutations=1,
            )
        with pytest.raises(MeasurementError):
            run_contamination_study(
                _FakePrimitives(_rotation_responder),
                state=STATE,
                questions=CONTAMINATION_QUESTIONS[:1],
                role=ROLE,
            )


# ── 2. Calibration (TD-2) ─────────────────────────────────────────────────

CALIBRATION_QUESTIONS = tuple(
    Question(
        id=f"cal-{index}",
        kind=QuestionKind.CHOICE,
        text=f"Calibration question {index}.",
        options=("a", "b"),
    )
    for index in range(4)
)

# Right/wrong at two confidence levels:
#   cal-0 confidence 0.75 correct, cal-1 confidence 0.75 incorrect,
#   cal-2 confidence 0.25 correct, cal-3 confidence 0.25 incorrect.
CALIBRATION_LABELS = {"cal-0": "a", "cal-1": "a", "cal-2": "a", "cal-3": "a"}


def _choice_entry(value: str, probability_a: float) -> dict:
    return {
        "choice": value,
        "probabilities": {"a": probability_a, "b": 1.0 - probability_a},
        "confidence": max(probability_a, 1.0 - probability_a),
    }


def _calibration_response() -> str:
    return json.dumps(
        {
            "answers": {
                "cal-0": _choice_entry("a", 0.875),
                "cal-1": _choice_entry("b", 0.875),
                "cal-2": _choice_entry("a", 0.625),
                "cal-3": _choice_entry("b", 0.625),
            }
        }
    )


class TestCalibrationStudy:
    def test_exact_ece_brier_and_reliability_bins(self, tmp_path: Path):
        primitives = _FakePrimitives(lambda prompt, **kwargs: _calibration_response())
        receipt_path = tmp_path / "calibration.json"

        receipt = run_calibration_study(
            primitives,
            state=STATE,
            questions=CALIBRATION_QUESTIONS,
            labels=CALIBRATION_LABELS,
            role=ROLE,
            receipt_path=receipt_path,
        )

        metrics = receipt["results"]["metrics"]
        assert metrics["n"] == 4
        assert metrics["ece"] == 0.25
        assert metrics["brier"] == 0.3125
        assert metrics["accuracy"] == 0.5
        assert metrics["mean_confidence"] == 0.5

        bins = receipt["results"]["reliability_bins"]
        assert len(bins) == 10
        assert [entry["count"] for entry in bins].count(2) == 2
        assert sum(entry["count"] for entry in bins) == 4
        low = bins[2]
        high = bins[7]
        assert (low["lower"], low["upper"]) == (0.2, 0.3)
        assert low["mean_confidence"] == 0.25
        assert low["accuracy"] == 0.5
        assert (high["lower"], high["upper"]) == (0.7, 0.8)
        assert high["mean_confidence"] == 0.75
        assert high["accuracy"] == 0.5
        assert bins[0]["count"] == 0
        assert bins[0]["mean_confidence"] is None
        assert bins[0]["accuracy"] is None

        rows = {row["question_id"]: row for row in receipt["results"]["rows"]}
        assert rows["cal-0"]["correct"] is True
        assert rows["cal-1"]["correct"] is False
        assert rows["cal-0"]["confidence"] == 0.75
        assert rows["cal-2"]["confidence"] == 0.25
        assert receipt["counts"] == {
            "questions": 4,
            "resolved": 4,
            "labeled": 4,
            "scored": 4,
            "unlabeled": 0,
            "unresolved": 0,
            "unknown_labels": 0,
            "failures": 0,
        }

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["mode"] == "json"
        assert receipt["role"] == ROLE
        expected_hash = hashlib.sha256(primitives.calls[0]["prompt"].encode()).hexdigest()
        assert receipt["prompt_sha256"] == [expected_hash]

    def test_unknown_labels_are_reported_not_scored(self, tmp_path: Path):
        primitives = _FakePrimitives(lambda prompt, **kwargs: _calibration_response())

        receipt = run_calibration_study(
            primitives,
            state=STATE,
            questions=CALIBRATION_QUESTIONS,
            labels={**CALIBRATION_LABELS, "ghost": "a"},
            role=ROLE,
            receipt_path=tmp_path / "calibration.json",
        )

        assert receipt["counts"]["unknown_labels"] == 1
        assert receipt["results"]["unknown_label_ids"] == ["ghost"]
        assert receipt["counts"]["scored"] == 4

    def test_no_matching_labels_fails_loudly(self):
        primitives = _FakePrimitives(lambda prompt, **kwargs: _calibration_response())

        with pytest.raises(MeasurementError):
            run_calibration_study(
                primitives,
                state=STATE,
                questions=CALIBRATION_QUESTIONS,
                labels={},
                role=ROLE,
            )

    def test_dry_run_makes_no_calls_and_writes_no_receipt(self, tmp_path: Path):
        primitives = _FakePrimitives(lambda prompt, **kwargs: _calibration_response())
        receipt_path = tmp_path / "calibration.json"

        receipt = run_calibration_study(
            primitives,
            state=STATE,
            questions=CALIBRATION_QUESTIONS,
            labels=CALIBRATION_LABELS,
            role=ROLE,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        assert receipt["plan"]["labeled_questions"] == ["cal-0", "cal-1", "cal-2", "cal-3"]
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_live_primitives_are_required(self):
        with pytest.raises(MeasurementError):
            run_calibration_study(
                None,
                state=STATE,
                questions=CALIBRATION_QUESTIONS,
                labels=CALIBRATION_LABELS,
                role=ROLE,
            )


# ── 3. Fan-out (TD-3) ─────────────────────────────────────────────────────

FANOUT_STATES = ("state-a", "state-b", "state-c")


class TestFanoutStudy:
    def test_stable_responder_agrees_across_arms(self, tmp_path: Path):
        primitives = _FakePrimitives(_stable_responder)
        receipt_path = tmp_path / "fanout.json"

        receipt = run_fanout_study(
            primitives,
            states=FANOUT_STATES,
            role=ROLE,
            questions_per_state=4,
            receipt_path=receipt_path,
        )

        assert receipt["counts"] == {
            "states": 3,
            "questions_per_state": 4,
            "batched_calls": 3,
            "singleton_calls": 12,
            "comparable_pairs": 12,
            "agreeing_pairs": 12,
            "disagreements": 0,
            "unresolved_pairs": 0,
        }
        results = receipt["results"]
        assert results["agreement_rate"] == 1.0
        assert results["questions"] == [f"fanout-{index:03d}" for index in range(4)]
        assert results["batched"]["tokens_generated"] == 33.0
        assert results["batched"]["calls_with_token_meta"] == 3
        assert results["singleton"]["tokens_generated"] == 132.0
        assert results["singleton"]["calls_with_token_meta"] == 12
        assert results["batched"]["serial_sum_ms"] >= 0.0
        assert len(results["batched"]["per_call_ms"]) == 3
        assert len(results["singleton"]["per_call_ms"]) == 12
        assert len(primitives.calls) == 15
        assert len(receipt["prompt_sha256"]) == 15

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["timestamp"]

    def test_position_dependent_answers_show_as_disagreements(self):
        receipt = run_fanout_study(
            _FakePrimitives(_rotation_responder),
            states=("state-a",),
            role=ROLE,
            questions_per_state=4,
        )

        assert receipt["counts"]["comparable_pairs"] == 4
        assert receipt["counts"]["agreeing_pairs"] == 1
        assert receipt["counts"]["disagreements"] == 3
        assert receipt["results"]["agreement_rate"] == 0.25
        by_question = {entry["question_id"]: entry for entry in receipt["results"]["disagreements"]}
        assert by_question["fanout-001"] == {
            "state_index": 0,
            "question_id": "fanout-001",
            "batched_value": "ship",
            "singleton_value": "hold",
        }
        assert by_question["fanout-002"]["batched_value"] == 2
        assert by_question["fanout-002"]["singleton_value"] == 0
        assert by_question["fanout-003"]["batched_value"] is False
        assert by_question["fanout-003"]["singleton_value"] is True
        assert receipt["results"]["per_question_agreement"]["fanout-001"] == {
            "compared": 1,
            "agreeing": 0,
        }

    def test_probe_question_ids_are_stable_across_calls(self):
        first = run_fanout_study(
            None,
            states=("state-a",),
            role=ROLE,
            questions_per_state=3,
            dry_run=True,
        )
        second = run_fanout_study(
            None,
            states=("state-b", "state-c"),
            role=ROLE,
            questions_per_state=3,
            dry_run=True,
        )

        assert first["plan"]["questions"] == ["fanout-000", "fanout-001", "fanout-002"]
        assert first["plan"]["questions"] == second["plan"]["questions"]

    def test_dry_run_via_direct_function_args(self, tmp_path: Path):
        primitives = _FakePrimitives(_stable_responder)
        receipt_path = tmp_path / "fanout.json"

        receipt = run_fanout_study(
            primitives,
            states=FANOUT_STATES,
            role=ROLE,
            questions_per_state=2,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        assert receipt["plan"]["batched_calls"] == 3
        assert receipt["plan"]["singleton_calls"] == 6
        assert receipt["plan"]["questions"] == ["fanout-000", "fanout-001"]
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_live_primitives_and_states_are_required(self):
        with pytest.raises(MeasurementError):
            run_fanout_study(
                None,
                states=("state-a",),
                role=ROLE,
                questions_per_state=1,
            )
        with pytest.raises(MeasurementError):
            run_fanout_study(
                _FakePrimitives(_stable_responder),
                states=(),
                role=ROLE,
                questions_per_state=1,
            )
        with pytest.raises(ValueError):
            run_fanout_study(
                _FakePrimitives(_stable_responder),
                states=("state-a",),
                role=ROLE,
                questions_per_state=0,
            )


# ── 4. Fan-out with a provided catalogue (TD-3b) ──────────────────────────

PROVIDED_QUESTIONS = (
    Question(
        id="provided-choice",
        kind=QuestionKind.CHOICE,
        text="Which deployment ring carries the canary?",
        options=("ring-a", "ring-b"),
        criteria=("choose exactly one ring",),
    ),
    Question(
        id="provided-score",
        kind=QuestionKind.SCORE,
        text="How many replicas should the canary use?",
        levels=(2, 3, 4),
    ),
    Question(
        id="provided-noul",
        kind=QuestionKind.NOUL,
        text="Is the pre-flight check green?",
        criteria=("answer from the state alone",),
    ),
)


def _candidates(question: Question) -> list[str]:
    if question.kind is QuestionKind.CHOICE:
        return list(question.options)
    if question.kind is QuestionKind.SCORE:
        return [str(level) for level in question.levels]
    return ["true", "false"]


class TestFanoutProvidedCatalogue:
    def test_provided_catalogue_reaches_primitives_verbatim(self, tmp_path: Path):
        primitives = _FakePrimitives(_stable_responder)
        receipt_path = tmp_path / "fanout-provided.json"

        receipt = run_fanout_study(
            primitives,
            states=("state-a", "state-b"),
            role=ROLE,
            questions=PROVIDED_QUESTIONS,
            receipt_path=receipt_path,
        )

        assert receipt["probe_source"] == "provided"
        assert receipt["results"]["questions"] == [q.id for q in PROVIDED_QUESTIONS]
        assert receipt["counts"] == {
            "states": 2,
            "questions_per_state": 3,
            "batched_calls": 2,
            "singleton_calls": 6,
            "comparable_pairs": 6,
            "agreeing_pairs": 6,
            "disagreements": 0,
            "unresolved_pairs": 0,
        }
        assert receipt["results"]["agreement_rate"] == 1.0
        expected_ids = [question.id for question in PROVIDED_QUESTIONS]
        expected_candidates = [_candidates(question) for question in PROVIDED_QUESTIONS]

        assert len(primitives.calls) == 8
        batched_prompts = [call["prompt"] for call in primitives.calls[:2]]
        singleton_prompts = [call["prompt"] for call in primitives.calls[2:]]
        for prompt in batched_prompts:
            entries = _parse_catalog(prompt)
            assert [entry["id"] for entry in entries] == expected_ids
            assert [entry["candidates"] for entry in entries] == expected_candidates
            for question in PROVIDED_QUESTIONS:
                assert f"question: {question.text}" in prompt
            assert "question: Fan-out probe" not in prompt
        for index, prompt in enumerate(singleton_prompts):
            question = PROVIDED_QUESTIONS[index % len(PROVIDED_QUESTIONS)]
            entries = _parse_catalog(prompt)
            assert [entry["id"] for entry in entries] == [question.id]
            assert entries[0]["candidates"] == _candidates(question)
            assert f"question: {question.text}" in prompt
        for question in PROVIDED_QUESTIONS:
            for criterion in question.criteria:
                assert any(f"criterion: {criterion}" in call["prompt"] for call in primitives.calls)

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["mode"] == "json"

    def test_questions_per_state_is_derived_and_mismatch_rejected(self, tmp_path: Path):
        matching = run_fanout_study(
            _FakePrimitives(_stable_responder),
            states=("state-a",),
            role=ROLE,
            questions=PROVIDED_QUESTIONS,
            questions_per_state=3,
            receipt_path=tmp_path / "matching.json",
        )
        assert matching["counts"]["questions_per_state"] == 3
        assert matching["probe_source"] == "provided"

        primitives = _FakePrimitives(_stable_responder)
        derived = run_fanout_study(
            primitives,
            states=("state-a",),
            role=ROLE,
            questions=PROVIDED_QUESTIONS,
            dry_run=True,
        )
        assert derived["plan"]["questions_per_state"] == 3
        assert primitives.calls == []

        with pytest.raises(ValueError):
            run_fanout_study(
                None,
                states=("state-a",),
                role=ROLE,
                questions=PROVIDED_QUESTIONS,
                questions_per_state=4,
                dry_run=True,
            )

    def test_empty_or_duplicate_provided_catalogues_fail_loudly(self):
        with pytest.raises(ValueError):
            run_fanout_study(
                None,
                states=("state-a",),
                role=ROLE,
                questions=[],
                dry_run=True,
            )
        with pytest.raises(ValueError):
            run_fanout_study(
                None,
                states=("state-a",),
                role=ROLE,
                questions=[PROVIDED_QUESTIONS[0], PROVIDED_QUESTIONS[0]],
                dry_run=True,
            )
        with pytest.raises(ValueError):
            run_fanout_study(
                None,
                states=("state-a",),
                role=ROLE,
                dry_run=True,
            )

    def test_provided_dry_run_makes_no_calls(self, tmp_path: Path):
        primitives = _FakePrimitives(_stable_responder)
        receipt_path = tmp_path / "fanout-provided.json"

        receipt = run_fanout_study(
            primitives,
            states=("state-a",),
            role=ROLE,
            questions=PROVIDED_QUESTIONS,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        assert receipt["probe_source"] == "provided"
        assert receipt["plan"]["probe_source"] == "provided"
        assert receipt["plan"]["questions"] == [q.id for q in PROVIDED_QUESTIONS]
        assert receipt["plan"]["batched_calls"] == 1
        assert receipt["plan"]["singleton_calls"] == 3
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_generated_path_marks_probe_source_and_keeps_ids(self):
        receipt = run_fanout_study(
            _FakePrimitives(_stable_responder),
            states=("state-a",),
            role=ROLE,
            questions_per_state=3,
        )

        assert receipt["probe_source"] == "generated"
        assert receipt["results"]["questions"] == [
            "fanout-000",
            "fanout-001",
            "fanout-002",
        ]


# ── 5. Parallel fan-out (TD-3b) ───────────────────────────────────────────


def _delayed_responder(delay_s: float, responder: Callable[..., str]) -> Callable[..., str]:
    """Wrap a responder with a fixed sleep so concurrency changes wall time."""

    def wrapped(prompt: str, **kwargs) -> str:
        time.sleep(delay_s)
        return responder(prompt, **kwargs)

    return wrapped


class TestParallelFanoutStudy:
    def test_concurrent_arm_is_faster_than_sequential_and_agrees(self, tmp_path: Path):
        primitives = _FakePrimitives(_delayed_responder(0.05, _stable_responder))
        created: list[_FakePrimitives] = []

        def factory() -> _FakePrimitives:
            fake = _FakePrimitives(_delayed_responder(0.05, _stable_responder))
            created.append(fake)
            return fake

        receipt_path = tmp_path / "parallel.json"
        receipt = run_parallel_fanout_study(
            primitives,
            states=("state-a",),
            questions=PROVIDED_QUESTIONS,
            role=ROLE,
            workers=3,
            primitives_factory=factory,
            receipt_path=receipt_path,
        )

        assert receipt["study"] == "parallel_fanout"
        assert receipt["counts"] == {
            "states": 1,
            "questions_per_state": 3,
            "workers": 3,
            "batched_calls": 1,
            "sequential_singleton_calls": 3,
            "concurrent_singleton_calls": 3,
        }
        arms = receipt["results"]["arms"]
        assert arms["batched"]["calls"] == 1
        assert arms["sequential_singleton"]["calls"] == 3
        assert arms["concurrent_singleton"]["calls"] == 3
        assert arms["batched"]["tokens_generated"] == 11.0
        assert arms["sequential_singleton"]["tokens_generated"] == 33.0
        assert arms["concurrent_singleton"]["tokens_generated"] == 33.0
        assert arms["concurrent_singleton"]["calls_with_token_meta"] == 3

        for comparison in receipt["results"]["agreement"].values():
            assert comparison["agreement_rate"] == 1.0
            assert comparison["comparable_pairs"] == 3
            assert comparison["unresolved_pairs"] == 0

        assert receipt["results"]["prompt_sha256_identical_concurrent_vs_sequential"] is True
        assert (
            receipt["prompt_sha256"]["concurrent_singleton"]
            == receipt["prompt_sha256"]["sequential_singleton"]
        )
        assert len(receipt["prompt_sha256"]["batched"]) == 1

        # The artificial per-call sleep makes Arm C provably faster in
        # wall-clock while the per-call serial sums stay equal.
        assert arms["concurrent_singleton"]["wall_ms"] < (
            arms["sequential_singleton"]["wall_ms"] * 0.75
        )
        assert arms["concurrent_singleton"]["serial_sum_ms"] == pytest.approx(
            arms["sequential_singleton"]["serial_sum_ms"], rel=0.25
        )
        assert receipt["results"]["speedup_concurrent_vs_sequential"] > 1.0
        assert receipt["results"]["speedup_concurrent_vs_batched"] > 0.0

        assert receipt["metric_directions"] == {
            "agreement_rate_vs_batched": "higher_better",
            "speedup_concurrent_vs_batched": "higher_better",
            "speedup_concurrent_vs_sequential": "higher_better",
            "wall_ms": "lower_better",
            "serial_sum_ms": "lower_better",
        }

        # One primitives object per worker, each used for exactly one call;
        # the shared object carries the batched + sequential arms.
        assert len(created) == 3
        assert all(len(fake.calls) == 1 for fake in created)
        assert len(primitives.calls) == 4

        # All arms share the runner's canonical instruction prefix.
        all_prompts = [call["prompt"] for call in primitives.calls] + [
            call["prompt"] for fake in created for call in fake.calls
        ]
        shared_prefix = all_prompts[0].split("RESPONSE JSON SCHEMA (authoritative):")[0]
        assert shared_prefix
        assert all(prompt.startswith(shared_prefix) for prompt in all_prompts)

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["timestamp"]

    def test_missing_factory_refuses_concurrent_arm_before_any_call(self):
        primitives = _FakePrimitives(_stable_responder)

        with pytest.raises(MeasurementError, match="primitives_factory"):
            run_parallel_fanout_study(
                primitives,
                states=("state-a",),
                questions=PROVIDED_QUESTIONS,
                role=ROLE,
                workers=2,
            )

        assert primitives.calls == []

    def test_factory_returning_the_same_object_is_rejected(self):
        primitives = _FakePrimitives(_stable_responder)
        shared = _FakePrimitives(_stable_responder)

        with pytest.raises(MeasurementError, match="same object twice"):
            run_parallel_fanout_study(
                primitives,
                states=("state-a",),
                questions=PROVIDED_QUESTIONS,
                role=ROLE,
                workers=2,
                primitives_factory=lambda: shared,
            )

        assert primitives.calls == []
        assert shared.calls == []

    def test_dry_run_plans_all_three_arms_and_makes_no_calls(self, tmp_path: Path):
        primitives = _FakePrimitives(_stable_responder)
        receipt_path = tmp_path / "parallel.json"

        receipt = run_parallel_fanout_study(
            primitives,
            states=("state-a", "state-b"),
            questions=PROVIDED_QUESTIONS,
            role=ROLE,
            workers=4,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        assert receipt["study"] == "parallel_fanout"
        plan = receipt["plan"]
        assert plan["workers"] == 4
        assert plan["states"] == 2
        assert plan["batched_calls"] == 2
        assert plan["sequential_singleton_calls"] == 6
        assert plan["concurrent_singleton_calls"] == 6
        assert plan["questions"] == [question.id for question in PROVIDED_QUESTIONS]
        assert plan["primitives_factory_supplied"] is False
        assert len(plan["state_sha256"]) == 2
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_states_questions_and_workers_are_validated(self):
        with pytest.raises(MeasurementError):
            run_parallel_fanout_study(
                None,
                states=(),
                questions=PROVIDED_QUESTIONS,
                role=ROLE,
                dry_run=True,
            )
        with pytest.raises(ValueError):
            run_parallel_fanout_study(
                None,
                states=("state-a",),
                questions=[],
                role=ROLE,
                dry_run=True,
            )
        with pytest.raises(ValueError):
            run_parallel_fanout_study(
                None,
                states=("state-a",),
                questions=PROVIDED_QUESTIONS,
                role=ROLE,
                workers=0,
                dry_run=True,
            )


# ── 6. CLI surface ────────────────────────────────────────────────────────


class TestCli:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture):
        with pytest.raises(SystemExit) as excinfo:
            main(["--help"])

        assert excinfo.value.code == 0
        assert "contamination" in capsys.readouterr().out

    def test_refuses_without_live_or_dry_run(self, tmp_path: Path):
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(STATE), encoding="utf-8")
        questions_file = tmp_path / "questions.json"
        questions_file.write_text(
            json.dumps([{"id": "q0", "kind": "noul", "text": "Is it so?"}]),
            encoding="utf-8",
        )

        code = main(
            [
                "contamination",
                "--state-file",
                str(state_file),
                "--questions-file",
                str(questions_file),
            ]
        )

        assert code == 2

    def test_fanout_cli_requires_a_catalogue(self, tmp_path: Path):
        states_file = tmp_path / "states.json"
        states_file.write_text(json.dumps(["state-a"]), encoding="utf-8")

        with pytest.raises(SystemExit) as excinfo:
            main(["fanout", "--states-file", str(states_file)])

        assert excinfo.value.code == 2

    def test_fanout_cli_questions_file_dry_run(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        states_file = tmp_path / "state.json"
        states_file.write_text(json.dumps({"state": "single state"}), encoding="utf-8")
        questions_file = tmp_path / "questions.json"
        questions_file.write_text(
            json.dumps(
                [
                    {
                        "id": "q-choice",
                        "kind": "choice",
                        "text": "Pick a lane.",
                        "options": ["a", "b"],
                    },
                    {"id": "q-score", "kind": "score", "text": "Rate it.", "levels": [0, 1, 2]},
                ]
            ),
            encoding="utf-8",
        )

        code = main(
            [
                "fanout",
                "--states-file",
                str(states_file),
                "--questions-file",
                str(questions_file),
                "--dry-run",
            ]
        )

        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["probe_source"] == "provided"
        assert printed["plan"]["questions"] == ["q-choice", "q-score"]
        assert printed["plan"]["questions_per_state"] == 2
        assert printed["plan"]["states"] == 1

    def test_parallel_cli_dry_run_prints_plan(self, tmp_path: Path, capsys: pytest.CaptureFixture):
        states_file = tmp_path / "states.json"
        states_file.write_text(json.dumps(["state-a"]), encoding="utf-8")
        questions_file = tmp_path / "questions.json"
        questions_file.write_text(
            json.dumps(
                [
                    {
                        "id": "q-choice",
                        "kind": "choice",
                        "text": "Pick a lane.",
                        "options": ["a", "b"],
                    },
                    {"id": "q-noul", "kind": "noul", "text": "Is it so?"},
                ]
            ),
            encoding="utf-8",
        )

        code = main(
            [
                "parallel",
                "--states-file",
                str(states_file),
                "--questions-file",
                str(questions_file),
                "--workers",
                "2",
                "--dry-run",
            ]
        )

        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["study"] == "parallel_fanout"
        assert printed["dry_run"] is True
        assert printed["plan"]["questions"] == ["q-choice", "q-noul"]
        assert printed["plan"]["workers"] == 2
        assert printed["plan"]["batched_calls"] == 1
        assert printed["plan"]["sequential_singleton_calls"] == 2
        assert printed["plan"]["concurrent_singleton_calls"] == 2

    def test_parallel_cli_refuses_without_live_or_dry_run(self, tmp_path: Path):
        states_file = tmp_path / "states.json"
        states_file.write_text(json.dumps(["state-a"]), encoding="utf-8")
        questions_file = tmp_path / "questions.json"
        questions_file.write_text(
            json.dumps([{"id": "q0", "kind": "noul", "text": "Is it so?"}]),
            encoding="utf-8",
        )

        code = main(
            [
                "parallel",
                "--states-file",
                str(states_file),
                "--questions-file",
                str(questions_file),
            ]
        )

        assert code == 2


# ── TD-21.33: _last_inference_meta's real-LLMPrimitives branch ────────────
class TestLastInferenceMetaRealPrimitives:
    """`_last_inference_meta` (used by this module's own fanout records and by
    `judge_redundancy.py`/`tool_args_pilot.py`) prefers the per-call-safe
    `get_last_inference_meta()` getter for a REAL `LLMPrimitives`, falling back
    to the plain attribute for `_FakePrimitives`-shaped test doubles (covered
    by every other test in this file). No live requests; a fake backend only.
    """

    def test_real_primitives_uses_the_per_call_safe_getter(self):
        from unittest.mock import Mock

        from src.llm_primitives import LLMPrimitives
        from src.model_server import InferenceResult

        prims = LLMPrimitives(mock_mode=False, server_urls={ROLE: "http://localhost:9201"})
        backend = Mock(spec=[])
        backend.infer = Mock(return_value=InferenceResult(
            role=ROLE, output="ignored", tokens_generated=7, generation_speed=1.0,
            elapsed_time=0.001, success=True, prompt_eval_ms=1.0, generation_ms=2.0,
            http_overhead_ms=0.0, completion_reason="stop",
        ))
        prims._backends[ROLE] = backend

        prims._real_call("a prompt", ROLE, n_tokens=8)

        meta = _last_inference_meta(prims)
        assert meta is not None
        assert meta["completion_reason"] == "stop"
        assert meta["tokens"] == 7

    def test_non_llmprimitives_double_falls_back_to_the_plain_attribute(self):
        class _Bare:
            def __init__(self) -> None:
                self._last_inference_meta = {"completion_reason": "length"}

        meta = _last_inference_meta(_Bare())
        assert meta == {"completion_reason": "length"}

    def test_missing_meta_returns_none(self):
        class _Bare:
            pass

        assert _last_inference_meta(_Bare()) is None
