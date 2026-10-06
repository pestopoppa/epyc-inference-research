"""Unit tests for the CJ-13/CJ-14 judge redundancy harness.

All runs use ``_FakePrimitives``: calls whose schema contains ``answers`` are
the typed arm and are answered with a schema-valid runner response built from
the case's frozen verdicts; schemas containing ``verdicts`` identify the LLM
arm and are answered with
a canned ``{"verdicts": [...]}`` emission (or a deliberate failure shape in
the failure tests). No model or server is touched.

The load-bearing tests are the last ones in ``TestRun``: readers that agree
perfectly while both deviate from the frozen ground truth must report
``agreement.rate == 1.0`` and ``criterion_accuracy < 1.0`` — the CJ-13
"agreement is not accuracy" property, encoded rather than asserted in prose.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from src.typed_decisions.judge_redundancy import (
    AGREEMENT_LABEL,
    OVERALL_PASS,
    OVERALL_ID,
    RubricCase,
    _llm_response_schema,
    _parse_verdicts,
    _run_llm_case,
    blind_prior,
    build_cases,
    build_typed_questions,
    case_sha256,
    criterion_question_id,
    main,
    rubric_set_sha256,
    run_judge_redundancy,
)
from src.typed_decisions.measure import MeasurementError
from src.typed_decisions.types import QuestionKind

ROLE = "frontdoor"

_CASES = build_cases()
_CASE_BY_ANSWER = {case.answer: case for case in _CASES}
_CRITERIA_TOTAL = sum(len(case.criteria) for case in _CASES)

_EXPECTED_DIRECTIONS = {
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


def _typed_response(
    case: RubricCase,
    *,
    verdicts: list[bool] | None = None,
    overall: str | None = None,
) -> str:
    """Schema-valid typed-runner response built from the case's truth (or an override)."""
    values = list(case.expected) if verdicts is None else list(verdicts)
    overall_value = case.expected_overall if overall is None else overall
    answers: dict[str, dict] = {}
    for index, value in enumerate(values):
        answers[criterion_question_id(index)] = {
            "noul": bool(value),
            "probabilities": {
                "true": 0.82 if value else 0.18,
                "false": 0.18 if value else 0.82,
            },
            # Deliberately NOT the local statistic (the runner derives that
            # from the probabilities above): 0.82/0.18 -> choice_confidence 0.64.
            "confidence": 0.9,
        }
    answers[OVERALL_ID] = {
        "choice": overall_value,
        "probabilities": {
            "pass": 0.82 if overall_value == OVERALL_PASS else 0.18,
            "fail": 0.18 if overall_value == OVERALL_PASS else 0.82,
        },
        "confidence": 0.8,
    }
    return json.dumps({"answers": answers})


def _llm_response(case: RubricCase, *, verdicts: list[bool] | None = None) -> str:
    values = list(case.expected) if verdicts is None else list(verdicts)
    return json.dumps({"verdicts": values})


class _FakePrimitives:
    """Canned per-case responder split by which arm is calling."""

    def __init__(self, *, typed=None, llm=None, tokens: float = 7.0, mock_mode: bool = False):
        self._typed = typed if typed is not None else _typed_response
        self._llm = llm if llm is not None else _llm_response
        self._tokens = tokens
        self.mock_mode = mock_mode
        self.calls: list[dict] = []
        self._last_inference_meta: dict = {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        self._last_inference_meta = {"tokens": self._tokens, "elapsed_ms": 1.0}
        case = self._case_for(prompt)
        if "answers" in kwargs["json_schema"]["properties"]:
            return self._typed(case)
        return self._llm(case)

    @staticmethod
    def _case_for(prompt: str) -> RubricCase:
        for answer, case in _CASE_BY_ANSWER.items():
            if answer in prompt:
                return case
        raise AssertionError("prompt does not contain any known rubric answer")


def _case_rows(receipt: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in receipt["results"]["cases"]}


# ── 1. Frozen catalogue ───────────────────────────────────────────────────


class TestRubricCatalogue:
    def test_deterministic_and_sized(self):
        assert build_cases() == _CASES
        assert len(_CASES) >= 20
        assert _CRITERIA_TOTAL == 72
        assert {case.family for case in _CASES} == {
            "refund_policy",
            "evidence",
            "format",
            "numeric_threshold",
            "safety",
            "citation",
        }
        assert len({case.case_id for case in _CASES}) == len(_CASES)
        assert len({case.answer for case in _CASES}) == len(_CASES)

    def test_every_case_is_internally_consistent(self):
        for case in _CASES:
            assert len(case.criteria) == len(case.expected) == len(case.basis) >= 2
            assert len(set(case.criteria)) == len(case.criteria)
            assert case.expected_overall == ("pass" if all(case.expected) else "fail"), case.case_id
            assert all(item for item in case.basis), case.case_id

    def test_documented_pass_fail_borderline_mix(self):
        all_pass = [case for case in _CASES if all(case.expected)]
        all_fail = [case for case in _CASES if not any(case.expected)]
        mixed = [case for case in _CASES if any(case.expected) and not all(case.expected)]
        assert len(all_pass) >= 4
        assert len(all_fail) >= 3
        assert len(mixed) >= 6
        assert len(all_pass) + len(all_fail) + len(mixed) == len(_CASES)
        # The exact-boundary cases are the borderline-but-determinate family.
        assert {"refund-03", "evidence-03", "format-03", "threshold-01"} <= {
            case.case_id for case in mixed
        }

    def test_digests_are_deterministic_and_sensitive(self):
        assert rubric_set_sha256(_CASES) == rubric_set_sha256(build_cases())
        # Frozen CJ-13 receipt: artifacts/typed_decisions/run_20260918/
        # judge-redundancy-20260918.json (root repository).
        assert rubric_set_sha256(_CASES) == (
            "bcfc61909aa84c79cf46d9cc7954ec1b31dedbd0ed9e02c34f036088f17b4e85"
        )
        assert len({case_sha256(case) for case in _CASES}) == len(_CASES)
        original = _CASES[0]
        altered = RubricCase(
            case_id=original.case_id,
            family=original.family,
            answer=original.answer + " Extra.",
            criteria=original.criteria,
            expected=original.expected,
            basis=original.basis,
        )
        assert case_sha256(altered) != case_sha256(original)

    def test_typed_question_catalogue(self):
        case = _CASES[0]
        questions = build_typed_questions(case)

        assert [question.id for question in questions] == [
            criterion_question_id(index) for index in range(len(case.criteria))
        ] + [OVERALL_ID]
        assert [question.kind for question in questions[:-1]] == [QuestionKind.NOUL] * len(
            case.criteria
        )
        assert questions[-1].kind is QuestionKind.CHOICE
        assert questions[-1].options == ("pass", "fail")
        for index, question in enumerate(questions[:-1]):
            assert case.criteria[index] in question.text

    def test_blind_prior_is_the_majority_floor(self):
        prior = blind_prior(_CASES)
        pass_count = sum(sum(1 for verdict in case.expected if verdict) for case in _CASES)
        fail_count = _CRITERIA_TOTAL - pass_count

        assert prior["criterion_majority_verdict"] == (
            "pass" if pass_count >= fail_count else "fail"
        )
        assert prior["criterion_accuracy"] == pytest.approx(
            max(pass_count, fail_count) / _CRITERIA_TOTAL
        )
        assert prior["overall_accuracy"] is not None


# ── 2. Both readers, accuracy and agreement ───────────────────────────────


class TestRun:
    def test_perfect_readers_and_receipt_shape(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt_path = tmp_path / "redundancy.json"

        receipt = run_judge_redundancy(primitives, role=ROLE, receipt_path=receipt_path)

        assert receipt["study"] == "judge_redundancy"
        assert receipt["mode"] == "typed_vs_llm_judge"
        assert receipt["role"] == ROLE
        assert receipt["timestamp"]
        assert receipt["agreement_label"] == AGREEMENT_LABEL

        counts = receipt["counts"]
        assert counts["cases"] == 24
        assert counts["criteria"] == _CRITERIA_TOTAL
        assert counts["typed_criteria_resolved"] == _CRITERIA_TOTAL
        assert counts["typed_criteria_correct"] == _CRITERIA_TOTAL
        assert counts["typed_failures"] == 0
        assert counts["llm_criteria_resolved"] == _CRITERIA_TOTAL
        assert counts["llm_criteria_correct"] == _CRITERIA_TOTAL
        assert counts["llm_failures"] == 0
        assert counts["agreement_compared_criteria"] == _CRITERIA_TOTAL
        assert counts["agreement_agreeing_criteria"] == _CRITERIA_TOTAL
        assert counts["human_adjudication_criteria"] == 0

        typed = receipt["results"]["readers"]["typed"]
        llm = receipt["results"]["readers"]["llm"]
        assert typed["criterion_accuracy"] == 1.0
        assert typed["case_accuracy"] == 1.0
        assert typed["overall_accuracy"] == 1.0
        assert typed["criteria_wrong"] == 0
        # The runner REPLACES the model-stated 0.9 with the local statistic
        # derived from the 0.82/0.18 probe: 0.64, not 0.9.
        assert typed["mean_confidence"] == pytest.approx(0.64)
        assert typed["tokens_generated"] == 7.0 * 24
        assert typed["calls_with_token_meta"] == 24
        assert llm["criterion_accuracy"] == 1.0
        assert llm["case_accuracy"] == 1.0
        assert llm["overall_accuracy"] == 1.0
        assert llm["mean_confidence"] is None
        assert llm["tokens_generated"] == 7.0 * 24

        agreement = receipt["results"]["agreement"]
        assert agreement["label"] == AGREEMENT_LABEL
        assert agreement["note"]
        assert agreement["compared"] == _CRITERIA_TOTAL
        assert agreement["agreeing"] == _CRITERIA_TOTAL
        assert agreement["disagreeing"] == 0
        assert agreement["unresolved"] == 0
        assert agreement["rate"] == 1.0
        assert agreement["human_adjudication_rate"] == 0.0
        assert agreement["overall"]["rate"] == 1.0
        assert agreement["disagreements"] == []

        assert len(primitives.calls) == 48
        typed_calls = [
            call
            for call in primitives.calls
            if "answers" in call["json_schema"]["properties"]
        ]
        llm_calls = [
            call
            for call in primitives.calls
            if "verdicts" in call["json_schema"]["properties"]
        ]
        assert len(typed_calls) == 24
        assert len(llm_calls) == 24
        assert all(call.get("temperature") == 0.0 for call in primitives.calls)
        assert all(call.get("seed") == 0 for call in primitives.calls)

        rows = _case_rows(receipt)
        assert set(rows) == {case.case_id for case in _CASES}
        for case in _CASES:
            row = rows[case.case_id]
            assert row["typed"]["values"] == list(case.expected)
            assert row["llm"]["verdicts"] == list(case.expected)
            assert row["llm"]["criterion_verdicts"] == {
                criterion_question_id(index): value
                for index, value in enumerate(case.expected)
            }
            assert row["typed"]["correct"] == [True] * len(case.criteria)
            assert row["llm"]["correct"] == [True] * len(case.criteria)
            assert row["agreement"] == {
                "compared": len(case.criteria),
                "agreeing": len(case.criteria),
            }

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["receipt_path"] == str(receipt_path)

    def test_typed_one_criterion_wrong_splits_accuracy_from_agreement(self, tmp_path: Path):
        target = _CASES[0]
        flipped = list(target.expected)
        flipped[0] = not flipped[0]

        def typed(case: RubricCase) -> str:
            if case.case_id == target.case_id:
                return _typed_response(case, verdicts=flipped)
            return _typed_response(case)

        receipt = run_judge_redundancy(
            _FakePrimitives(typed=typed), role=ROLE, receipt_path=tmp_path / "r.json"
        )

        typed = receipt["results"]["readers"]["typed"]
        llm = receipt["results"]["readers"]["llm"]
        assert typed["criteria_correct"] == _CRITERIA_TOTAL - 1
        assert typed["criteria_wrong"] == 1
        assert typed["criterion_accuracy"] == pytest.approx((_CRITERIA_TOTAL - 1) / _CRITERIA_TOTAL)
        assert llm["criterion_accuracy"] == 1.0

        agreement = receipt["results"]["agreement"]
        assert agreement["compared"] == _CRITERIA_TOTAL
        assert agreement["agreeing"] == _CRITERIA_TOTAL - 1
        assert agreement["disagreeing"] == 1
        assert agreement["rate"] == pytest.approx((_CRITERIA_TOTAL - 1) / _CRITERIA_TOTAL)
        assert agreement["human_adjudication_criteria"] == 1
        assert agreement["disagreements"] == [
            {
                "case_id": target.case_id,
                "criterion_index": 0,
                "typed": flipped[0],
                "llm": target.expected[0],
            }
        ]

    def test_llm_parse_failure_is_wrong_and_not_compared(self, tmp_path: Path):
        target = _CASES[1]

        def llm(case: RubricCase) -> str:
            if case.case_id == target.case_id:
                return "I believe every criterion is satisfied, but here is no JSON."
            return _llm_response(case)

        receipt = run_judge_redundancy(
            _FakePrimitives(llm=llm), role=ROLE, receipt_path=tmp_path / "r.json"
        )

        llm_summary = receipt["results"]["readers"]["llm"]
        n_target = len(target.criteria)
        assert llm_summary["criteria_resolved"] == _CRITERIA_TOTAL - n_target
        assert llm_summary["failures"] == n_target
        assert llm_summary["criteria_unresolved"] == n_target
        assert llm_summary["errored_cases"] == 1
        assert llm_summary["criterion_accuracy"] == pytest.approx(
            (_CRITERIA_TOTAL - n_target) / _CRITERIA_TOTAL
        )

        agreement = receipt["results"]["agreement"]
        assert agreement["unresolved"] == n_target
        assert agreement["compared"] == _CRITERIA_TOTAL - n_target
        assert agreement["agreeing"] == _CRITERIA_TOTAL - n_target
        assert agreement["rate"] == 1.0
        # Agreement excludes the failed case; accuracy does not. This is
        # exactly why the receipt labels them separately.
        assert llm_summary["criterion_accuracy"] < agreement["rate"]
        assert agreement["human_adjudication_criteria"] == n_target
        assert agreement["human_adjudication_rate"] == pytest.approx(n_target / _CRITERIA_TOTAL)

        row = _case_rows(receipt)[target.case_id]
        assert row["llm"]["verdicts"] == [None] * n_target
        assert row["llm"]["error"] == "no balanced JSON object found"
        assert row["llm"]["correct"] == [None] * n_target
        assert row["llm"]["overall"] is None

    def test_llm_wrong_verdict_count_is_rejected(self, tmp_path: Path):
        target = _CASES[2]

        def llm(case: RubricCase) -> str:
            if case.case_id == target.case_id:
                return json.dumps({"verdicts": [True]})
            return _llm_response(case)

        receipt = run_judge_redundancy(
            _FakePrimitives(llm=llm), role=ROLE, receipt_path=tmp_path / "r.json"
        )

        row = _case_rows(receipt)[target.case_id]
        expected_error = f"expected {len(target.criteria)} verdicts, got 1"
        assert row["llm"]["error"] == expected_error
        assert row["llm"]["verdicts"] == [None] * len(target.criteria)

    def test_typed_runner_failure_counts_every_criterion_wrong(self, tmp_path: Path):
        target = _CASES[2]

        def typed(case: RubricCase) -> str:
            if case.case_id == target.case_id:
                return "the verdict is pass, but this is not JSON"
            return _typed_response(case)

        primitives = _FakePrimitives(typed=typed)
        receipt = run_judge_redundancy(primitives, role=ROLE, receipt_path=tmp_path / "r.json")

        typed = receipt["results"]["readers"]["typed"]
        n_target = len(target.criteria)
        assert typed["criteria_unresolved"] == n_target
        assert typed["failures"] == n_target
        assert typed["errored_cases"] == 1
        assert typed["criterion_accuracy"] == pytest.approx(
            (_CRITERIA_TOTAL - n_target) / _CRITERIA_TOTAL
        )
        row = _case_rows(receipt)[target.case_id]
        assert row["typed"]["values"] == [None] * n_target
        assert row["typed"]["failures"]
        assert row["typed"]["correct"] == [None] * n_target
        assert row["typed"]["overall"] is None
        # The runner retried once (corrective pass); both attempts failed.
        target_calls = [
            call
            for call in primitives.calls
            if "answers" in call["json_schema"]["properties"]
            and target.answer in call["prompt"]
        ]
        assert len(target_calls) == 2

    def test_agreement_is_not_accuracy_when_both_readers_agree_and_are_wrong(self, tmp_path: Path):
        def typed(case: RubricCase) -> str:
            return _typed_response(case, verdicts=[True] * len(case.criteria))

        def llm(case: RubricCase) -> str:
            return json.dumps({"verdicts": [True] * len(case.criteria)})

        receipt = run_judge_redundancy(
            _FakePrimitives(typed=typed, llm=llm),
            role=ROLE,
            receipt_path=tmp_path / "r.json",
        )

        typed = receipt["results"]["readers"]["typed"]
        llm = receipt["results"]["readers"]["llm"]
        prior = receipt["results"]["blind_prior"]
        assert typed["criterion_accuracy"] == pytest.approx(prior["criterion_accuracy"])
        assert llm["criterion_accuracy"] == pytest.approx(prior["criterion_accuracy"])
        assert typed["criterion_accuracy"] < 1.0

        agreement = receipt["results"]["agreement"]
        assert agreement["rate"] == 1.0
        assert agreement["compared"] == _CRITERIA_TOTAL
        assert agreement["disagreeing"] == 0
        assert receipt["agreement_label"] == AGREEMENT_LABEL

    def test_no_resolution_in_either_reader_raises(self):
        primitives = _FakePrimitives(typed=lambda case: "garbage", llm=lambda case: "garbage")
        with pytest.raises(MeasurementError):
            run_judge_redundancy(primitives, role=ROLE)

    def test_mock_mode_primitives_are_refused(self):
        with pytest.raises(MeasurementError, match="mock_mode"):
            run_judge_redundancy(_FakePrimitives(mock_mode=True), role=ROLE)

    def test_live_primitives_are_required(self):
        with pytest.raises(MeasurementError):
            run_judge_redundancy(None, role=ROLE)
        with pytest.raises(MeasurementError):
            run_judge_redundancy(object(), role=ROLE)


# ── 3. Receipt directions / manifest / parse contract ─────────────────────


class TestReceiptAndParse:
    def test_metric_directions_are_explicit(self, tmp_path: Path):
        receipt = run_judge_redundancy(
            _FakePrimitives(), role=ROLE, receipt_path=tmp_path / "r.json"
        )
        assert receipt["metric_directions"] == _EXPECTED_DIRECTIONS

    def test_manifest_hashes_every_case(self, tmp_path: Path):
        receipt = run_judge_redundancy(
            _FakePrimitives(), role=ROLE, receipt_path=tmp_path / "r.json"
        )
        manifest = receipt["manifest"]
        assert manifest["rubric_set_sha256"] == rubric_set_sha256(_CASES)
        by_id = {entry["case_id"]: entry for entry in manifest["cases"]}
        assert set(by_id) == {case.case_id for case in _CASES}
        for case in _CASES:
            entry = by_id[case.case_id]
            assert entry["case_sha256"] == case_sha256(case)
            assert entry["criteria"] == len(case.criteria)
            assert entry["expected_pass"] + entry["expected_fail"] == len(case.criteria)

        hashes = receipt["prompt_sha256"]
        assert len(hashes["typed"]) == len(_CASES)
        assert len(hashes["llm"]) == len(_CASES)
        assert all(len(value) == 64 for value in hashes["typed"] + hashes["llm"])
        assert hashes["llm"] == [row["llm"]["prompt_sha256"] for row in receipt["results"]["cases"]]

    def test_parse_verdicts_accepts_the_contract_shapes(self):
        assert _parse_verdicts('{"verdicts": [true, false, true]}', 3) == (
            (True, False, True),
            None,
        )
        assert _parse_verdicts('Here you go: {"verdicts": [false]}', 1) == ((False,), None)

    def test_parse_verdicts_rejects_everything_else(self):
        for raw in (
            "no json at all",
            '{"verdicts": [1, 0, 1]}',
            '{"verdicts": [true]}',
            '{"verdicts": "pass"}',
            '{"other": []}',
            '{"verdicts": [true, false, "maybe"]}',
        ):
            verdicts, error = _parse_verdicts(raw, 3)
            assert verdicts is None
            assert error


# ── 4. Dry-run and CLI surface ────────────────────────────────────────────


class TestDryRunAndCli:
    def test_dry_run_pre_registers_the_plan_with_no_calls(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt_path = tmp_path / "redundancy.json"

        receipt = run_judge_redundancy(
            primitives, role=ROLE, dry_run=True, receipt_path=receipt_path
        )

        assert receipt["dry_run"] is True
        plan = receipt["plan"]
        assert plan["mode"] == "typed_vs_llm_judge"
        assert plan["readers"] == ["typed", "llm"]
        assert plan["agreement_label"] == AGREEMENT_LABEL
        assert plan["rubric_set_sha256"] == rubric_set_sha256(_CASES)
        assert [entry["case_id"] for entry in plan["cases"]] == [case.case_id for case in _CASES]
        first = plan["cases"][0]
        assert first["criteria"] == list(_CASES[0].criteria)
        assert first["expected"] == list(_CASES[0].expected)
        assert first["expected_overall"] == _CASES[0].expected_overall
        assert first["case_sha256"] == case_sha256(_CASES[0])
        assert first["basis"] == list(_CASES[0].basis)
        assert plan["counts"]["cases"] == 24
        assert plan["counts"]["criteria"] == _CRITERIA_TOTAL
        assert plan["blind_prior"]["criterion_accuracy"] is not None

        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_cli_refuses_without_live_or_dry_run(self):
        assert main([]) == 2

    def test_cli_dry_run_prints_the_plan(self, capsys: pytest.CaptureFixture):
        code = main(["--dry-run"])

        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["dry_run"] is True
        assert printed["plan"]["counts"]["cases"] == 24

    def test_cli_live_writes_the_receipt(self, tmp_path: Path, monkeypatch):
        primitives = _FakePrimitives()
        monkeypatch.setattr(
            "src.typed_decisions.judge_redundancy._live_primitives",
            lambda **_: primitives,
        )
        receipt_path = tmp_path / "live.json"

        code = main(["--live", "--role", ROLE, "--receipt", str(receipt_path)])

        assert code == 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["role"] == ROLE
        assert receipt["counts"]["agreement_agreeing_criteria"] == _CRITERIA_TOTAL
        assert receipt["results"]["agreement"]["label"] == AGREEMENT_LABEL


class TestConstrainedReaderB:
    def test_schema_closes_shape_and_pins_criterion_count(self):
        assert _llm_response_schema(3) == {
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "array",
                    "items": {"type": "boolean"},
                    "minItems": 3,
                    "maxItems": 3,
                }
            },
            "required": ["verdicts"],
            "additionalProperties": False,
        }

    @pytest.mark.parametrize(
        "raw",
        [
            '{}',
            '{"verdicts": [true, true, true], "verdicts": [false, false, false]}',
            '{"verdicts": [true, true]}',
            '{"verdicts": [true, true, true, true]}',
            '{"verdicts": [true, true, true], "extra": false}',
            '{"verdicts": [true, true, "false"]}',
            '{"verdicts": [true, true, 0]}',
            '{"verdicts": [true, true, null]}',
        ],
    )
    def test_invalid_emission_stays_unresolved_without_retry(self, raw):
        primitives = _FakePrimitives(llm=lambda case: raw)
        record = _run_llm_case(primitives, _CASES[0], ROLE)
        assert len(primitives.calls) == 1
        assert record["verdicts"] is None
        assert record["overall"] is None
        assert record["error"]
        assert record["tokens"] == 7.0
        assert primitives.calls[0]["json_schema"] == _llm_response_schema(3)

    def test_transport_failure_is_single_call(self):
        def fail(case):
            raise RuntimeError("injected transport failure")

        primitives = _FakePrimitives(llm=fail)
        record = _run_llm_case(primitives, _CASES[0], ROLE)
        assert len(primitives.calls) == 1
        assert record["verdicts"] is None
        assert record["error"] == "RuntimeError: injected transport failure"

    def test_missing_metadata_is_not_zero(self):
        class NoMetadata(_FakePrimitives):
            def llm_call(self, prompt, **kwargs):
                result = super().llm_call(prompt, **kwargs)
                self._last_inference_meta = {}
                return result

        primitives = NoMetadata()
        record = _run_llm_case(primitives, _CASES[0], ROLE)
        assert len(primitives.calls) == 1
        assert record["verdicts"] == list(_CASES[0].expected)
        assert record["tokens"] is None


    @pytest.mark.parametrize(
        "payload,valid",
        [
            ({"verdicts": [True, False, True]}, True),
            ({"verdicts": [False, False, False]}, True),
            ({}, False),
            ({"verdicts": [True]}, False),
            ({"verdicts": [True] * 4}, False),
            ({"verdicts": [True, False, 0]}, False),
            ({"verdicts": [True, False, "true"]}, False),
            ({"verdicts": [True, False, None]}, False),
            ({"verdicts": [True] * 3, "extra": False}, False),
        ],
    )
    def test_schema_semantics(self, payload, valid):
        schema = _llm_response_schema(3)
        Draft202012Validator.check_schema(schema)
        assert Draft202012Validator(schema).is_valid(payload) is valid

    def test_instrument_identity_matches_dry_run_and_wire(self, tmp_path):
        primitives = _FakePrimitives()
        receipt = run_judge_redundancy(
            primitives, role=ROLE, receipt_path=tmp_path / "r.json"
        )
        instrument = receipt["manifest"]["llm_instrument"]
        plan = run_judge_redundancy(None, role=ROLE, dry_run=True)
        assert instrument == plan["plan"]["llm_instrument"]
        assert instrument["id"] == "independent_llm_judge_constrained_verdicts_v1"
        assert instrument["calls_per_case"] == 1
        assert instrument["repair_calls"] == 0
        llm_calls = [
            call
            for call in primitives.calls
            if "verdicts" in call["json_schema"]["properties"]
        ]
        assert len(instrument["response_schemas"]) == len(llm_calls) == len(_CASES)
        for case, call, entry in zip(
            _CASES, llm_calls, instrument["response_schemas"], strict=True
        ):
            encoded = json.dumps(
                call["json_schema"], sort_keys=True, separators=(",", ":")
            )
            assert entry == {
                "case_id": case.case_id,
                "schema_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            }
