"""Unit tests for the TD-4 live pilot harness (tool_args_pilot.py).

All runs use a split fake primitives object: the closed-set arm is identified
by the ``json_schema`` kwarg the typed-decision runner passes and is answered
from the case's expected argument dict (or a deliberately partial/unparseable
response in the failure tests); the free-form arm is answered with a canned
JSON arguments object. No model or server is touched.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from src.typed_decisions.measure import MeasurementError
from src.typed_decisions.tool_args import tool_schema_to_questions
from src.typed_decisions.tool_args_pilot import (
    PilotCase,
    build_cases,
    main,
    run_tool_args_pilot,
)
from src.typed_decisions.types import QuestionKind

ROLE = "worker"

_CASES = build_cases()
_CASE_BY_STATE = {case.state: case for case in _CASES}
_CASE_BY_ID = {case.case_id: case for case in _CASES}
_EXPECTED_ARGS = sum(len(case.expected) for case in _CASES)


def _mapping(case: PilotCase):
    return tool_schema_to_questions(case.tool, case.parameters)


def _closed_response(case: PilotCase) -> str:
    """Schema-valid runner response whose values equal the case ground truth."""
    answers: dict[str, dict] = {}
    for question in _mapping(case).questions:
        if question.kind is QuestionKind.CHOICE:
            value = case.expected[question.id]
            answers[question.id] = {
                "choice": value,
                "probabilities": {
                    label: 1.0 if label == value else 0.0 for label in question.options
                },
                "confidence": 1.0,
            }
        elif question.kind is QuestionKind.SCORE:
            value = case.expected[question.id]
            labels = [str(level) for level in question.levels]
            answers[question.id] = {
                "score": value,
                "probabilities": {label: 1.0 if label == str(value) else 0.0 for label in labels},
                "confidence": 1.0,
            }
        else:
            arg, separator, label = question.id.partition("__")
            if separator:
                selected = [str(item) for item in case.expected[arg]]
                value = label in selected
            else:
                value = bool(case.expected[question.id])
            answers[question.id] = {
                "noul": value,
                "probabilities": {
                    "true": 1.0 if value else 0.0,
                    "false": 0.0 if value else 1.0,
                },
                "confidence": 1.0,
            }
    return json.dumps({"answers": answers})


class _FakePrimitives:
    """Canned per-case responder split by which arm is calling."""

    def __init__(
        self,
        *,
        closed: Callable[[PilotCase], str] = _closed_response,
        free: Callable[[PilotCase], str] | None = None,
    ):
        self.closed = closed
        self.free = free if free is not None else (lambda case: json.dumps(case.expected))
        self.calls: list[dict] = []
        self._last_inference_meta: dict = {}

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        self._last_inference_meta = {"tokens": 9, "elapsed_ms": 1.0}
        case = self._case_for(prompt)
        if kwargs.get("json_schema") is not None:
            return self.closed(case)
        return self.free(case)

    @staticmethod
    def _case_for(prompt: str) -> PilotCase:
        for state, case in _CASE_BY_STATE.items():
            if state in prompt:
                return case
        raise AssertionError("prompt does not contain any known pilot case state")


def _case_rows(receipt: dict) -> dict[str, dict]:
    return {row["case_id"]: row for row in receipt["results"]["cases"]}


# ── 1. Case catalogue ─────────────────────────────────────────────────────


class TestBuildCases:
    def test_catalogue_is_deterministic_and_sized(self):
        first = build_cases()
        second = build_cases()

        assert first == second
        assert len(first) >= 18
        assert {case.tool for case in first} == {
            "schedule_meeting",
            "deploy_service",
            "file_ticket",
        }
        assert len({case.case_id for case in first}) == len(first)

    def test_expected_dicts_cover_the_mapping_rules(self):
        kinds: set[QuestionKind] = set()
        for case in _CASES:
            mapping = _mapping(case)
            kinds.update(question.kind for question in mapping.questions)
            Draft202012Validator(case.parameters).validate(case.expected)

            assert len(mapping.skipped) == 1, (case.case_id, mapping.skipped)
            mapped_args = {question.id.split("__", 1)[0] for question in mapping.questions}
            assert set(case.expected) == mapped_args

            for value in case.expected.values():
                if isinstance(value, bool):
                    continue
                if isinstance(value, list):
                    for item in value:
                        assert str(item) in case.state
                else:
                    assert str(value) in case.state

        assert kinds == {QuestionKind.CHOICE, QuestionKind.SCORE, QuestionKind.NOUL}
        arrays = [
            value for case in _CASES for value in case.expected.values() if isinstance(value, list)
        ]
        assert any(value == [] for value in arrays)
        assert any(value for value in arrays)
        booleans = [
            value for case in _CASES for value in case.expected.values() if isinstance(value, bool)
        ]
        assert True in booleans and False in booleans


# ── 2. Both arms scoring ──────────────────────────────────────────────────


class TestPilotArms:
    def test_both_arms_score_perfectly_and_receipt_shape(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt_path = tmp_path / "pilot.json"

        receipt = run_tool_args_pilot(
            primitives,
            role=ROLE,
            receipt_path=receipt_path,
        )

        assert receipt["study"] == "tool_args_pilot"
        assert receipt["mode"] == "closed_set_vs_free_form"
        assert receipt["role"] == ROLE
        assert receipt["timestamp"]
        assert receipt["metric_directions"] == {
            "exact_match": "higher_better",
            "per_arg_exact_match": "higher_better",
            "agreement": "higher_better",
            "wall_ms": "lower_better",
        }
        counts = receipt["counts"]
        assert counts["cases"] == 18
        assert counts["tools"] == 3
        assert counts["closed_set_resolved"] == 18
        assert counts["closed_set_exact_match"] == 18
        assert counts["closed_set_failures"] == 0
        assert counts["free_form_resolved"] == 18
        assert counts["free_form_exact_match"] == 18
        assert counts["free_form_failures"] == 0
        assert counts["agreement_compared"] == 18
        assert counts["agreement_agreeing"] == 18

        closed = receipt["results"]["arms"]["closed_set"]
        free = receipt["results"]["arms"]["free_form"]
        assert closed["decode"] == "typed_json"
        assert free["decode"] == "free_form_text"
        assert closed["exact_match_rate"] == 1.0
        assert closed["per_arg_total"] == _EXPECTED_ARGS
        assert closed["per_arg_exact_match"] == _EXPECTED_ARGS
        assert closed["per_arg_exact_match_rate"] == 1.0
        assert closed["wall_ms"] >= 0.0
        assert len(closed["per_case_ms"]) == 18
        assert closed["tokens_generated"] == 9.0 * 18
        assert closed["calls_with_token_meta"] == 18
        assert free["exact_match_rate"] == 1.0
        assert free["tokens_generated"] == 9.0 * 18
        assert receipt["results"]["agreement"] == {"compared": 18, "agreeing": 18, "rate": 1.0}

        rows = _case_rows(receipt)
        assert set(rows) == {case.case_id for case in _CASES}
        for case in _CASES:
            row = rows[case.case_id]
            assert row["expected"] == case.expected
            assert row["closed_set"]["exact_match"] is True
            assert row["free_form"]["exact_match"] is True
        assert rows["deploy-02"]["closed_set"]["skipped"] == [
            "notes: unsupported JSON-schema type 'string'"
        ]

        assert len(primitives.calls) == 36
        assert all(call.get("json_schema") is not None for call in primitives.calls[:18])
        assert all(call.get("json_schema") is None for call in primitives.calls[18:])
        assert len(receipt["prompt_sha256"]) == 36
        assert len(set(receipt["prompt_sha256"])) == 36

        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert loaded == receipt
        assert receipt["receipt_path"] == str(receipt_path)

    def test_closed_set_partial_answers_count_as_wrong(self, tmp_path: Path):
        def partial(case: PilotCase) -> str:
            payload = json.loads(_closed_response(case))
            if case.case_id == "meeting-01":
                del payload["answers"]["priority"]
            return json.dumps(payload)

        primitives = _FakePrimitives(closed=partial)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["closed_set_resolved"] == 17
        assert counts["closed_set_failures"] == 1
        assert counts["closed_set_exact_match"] == 17
        assert counts["free_form_exact_match"] == 18
        assert counts["agreement_compared"] == 17
        assert counts["agreement_agreeing"] == 17

        closed = receipt["results"]["arms"]["closed_set"]
        assert closed["per_arg_total"] == _EXPECTED_ARGS
        assert closed["per_arg_exact_match"] == _EXPECTED_ARGS - len(
            _CASE_BY_ID["meeting-01"].expected
        )
        row = _case_rows(receipt)["meeting-01"]
        assert row["closed_set"]["resolved"] is False
        assert row["closed_set"]["failure_count"] >= 1
        assert row["closed_set"]["error"]
        assert row["closed_set"]["exact_match"] is False

        closed_calls = [call for call in primitives.calls if call.get("json_schema") is not None]
        assert len(closed_calls) == 19

    def test_free_form_wrong_value_reduces_exact_match_and_agreement(self, tmp_path: Path):
        def wrong(case: PilotCase) -> str:
            payload = dict(case.expected)
            if case.case_id == "meeting-02":
                payload["priority"] = "normal"
            return json.dumps(payload)

        primitives = _FakePrimitives(free=wrong)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["closed_set_exact_match"] == 18
        assert counts["free_form_resolved"] == 18
        assert counts["free_form_failures"] == 0
        assert counts["free_form_exact_match"] == 17
        assert counts["agreement_compared"] == 18
        assert counts["agreement_agreeing"] == 17

        free = receipt["results"]["arms"]["free_form"]
        assert free["per_arg_total"] == _EXPECTED_ARGS
        assert free["per_arg_exact_match"] == _EXPECTED_ARGS - 1
        row = _case_rows(receipt)["meeting-02"]
        assert row["free_form"]["resolved"] is True
        assert row["free_form"]["exact_match"] is False
        assert row["free_form"]["arguments"]["priority"] == "normal"

    def test_free_form_parse_failure_counts_as_wrong(self, tmp_path: Path):
        def unparseable(case: PilotCase) -> str:
            if case.case_id == "ticket-03":
                return "the arguments are severity p4 and nothing else"
            return json.dumps(case.expected)

        primitives = _FakePrimitives(free=unparseable)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["free_form_resolved"] == 17
        assert counts["free_form_failures"] == 1
        assert counts["free_form_exact_match"] == 17
        assert counts["agreement_compared"] == 17
        row = _case_rows(receipt)["ticket-03"]
        assert row["free_form"]["resolved"] is False
        assert "no balanced JSON object" in row["free_form"]["error"]
        free = receipt["results"]["arms"]["free_form"]
        assert free["per_arg_exact_match"] == _EXPECTED_ARGS - len(row["expected"])

    def test_no_resolution_in_either_arm_reports_no_agreement(self, tmp_path: Path):
        primitives = _FakePrimitives(
            closed=lambda case: "not json",
            free=lambda case: "still not json",
        )
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "pilot.json")

        counts = receipt["counts"]
        assert counts["closed_set_resolved"] == 0
        assert counts["closed_set_failures"] == 18
        assert counts["free_form_resolved"] == 0
        assert counts["free_form_failures"] == 18
        assert receipt["results"]["agreement"] == {"compared": 0, "agreeing": 0, "rate": None}
        assert receipt["results"]["arms"]["closed_set"]["exact_match_rate"] == 0.0
        assert receipt["results"]["arms"]["closed_set"]["per_arg_exact_match"] == 0
        assert receipt["results"]["arms"]["free_form"]["per_arg_total"] == _EXPECTED_ARGS
        assert receipt["results"]["arms"]["free_form"]["tokens_generated"] == 9.0 * 18


# ── 3. Dry-run and primitives gate ────────────────────────────────────────


class TestPilotGates:
    def test_dry_run_issues_no_calls_and_writes_no_receipt(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt_path = tmp_path / "pilot.json"

        receipt = run_tool_args_pilot(
            primitives,
            role=ROLE,
            dry_run=True,
            receipt_path=receipt_path,
        )

        assert receipt["dry_run"] is True
        plan = receipt["plan"]
        assert plan["mode"] == "closed_set_vs_free_form"
        assert plan["arms"] == ["closed_set", "free_form"]
        assert [entry["case_id"] for entry in plan["cases"]] == [case.case_id for case in _CASES]
        assert plan["cases"][0]["expected"] == _CASES[0].expected
        assert plan["cases"][0]["question_ids"]
        assert plan["cases"][0]["skipped"]
        assert plan["counts"] == {"cases": 18, "tools": 3}
        assert primitives.calls == []
        assert not receipt_path.exists()

    def test_live_primitives_are_required(self):
        with pytest.raises(MeasurementError):
            run_tool_args_pilot(None, role=ROLE)
        with pytest.raises(MeasurementError):
            run_tool_args_pilot(object(), role=ROLE)


# ── 4. CLI surface ────────────────────────────────────────────────────────


class TestPilotCli:
    def test_refuses_without_live_or_dry_run(self):
        assert main([]) == 2

    def test_dry_run_prints_the_plan(self, capsys: pytest.CaptureFixture):
        code = main(["--dry-run"])

        assert code == 0
        printed = json.loads(capsys.readouterr().out)
        assert printed["dry_run"] is True
        assert printed["plan"]["counts"]["cases"] == 18

    def test_live_runs_with_injected_primitives(self, tmp_path: Path, monkeypatch):
        primitives = _FakePrimitives()
        monkeypatch.setattr(
            "src.typed_decisions.tool_args_pilot._live_primitives",
            lambda: primitives,
        )
        receipt_path = tmp_path / "pilot.json"

        code = main(["--live", "--role", ROLE, "--receipt", str(receipt_path)])

        assert code == 0
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["counts"]["cases"] == 18
        assert receipt["counts"]["closed_set_exact_match"] == 18


# ── 5. TD-29 / TD-29.M0: per-call accounting, arm subsets, native forwarding ──


class _AccountingPrimitives(_FakePrimitives):
    """Publishes server-style usage meta; call N reports N tokens, 1000 prompt, 600 cached."""

    def llm_call(self, prompt: str, **kwargs):
        result = super().llm_call(prompt, **kwargs)
        number = len(self.calls)
        self._last_inference_meta = {
            "tokens": number,
            "prompt_tokens": 1000,
            "cached_prompt_tokens": 600,
            "prompt_ms": 5.0,
            "gen_ms": 7.0,
        }
        return result


class TestPilotCallAccounting:
    def test_retry_tokens_are_summed_and_last_call_kept(self, tmp_path: Path):
        attempts: dict[str, int] = {}

        def flaky(case: PilotCase) -> str:
            attempts[case.case_id] = attempts.get(case.case_id, 0) + 1
            if case.case_id == "meeting-01" and attempts[case.case_id] == 1:
                return "not json at all"
            return _closed_response(case)

        primitives = _AccountingPrimitives(closed=flaky)
        receipt = run_tool_args_pilot(primitives, role=ROLE, receipt_path=tmp_path / "p.json")

        row = _case_rows(receipt)["meeting-01"]["closed_set"]
        assert row["resolved"] is True
        assert row["call_count"] == 2
        assert row["tokens"] == 1.0 + 2.0
        assert row["tokens_last_call"] == 2.0
        assert row["prompt_tokens"] == 2000.0
        assert row["cache_n"] == 1200.0
        assert row["prompt_n"] == 800.0
        assert row["prompt_ms"] == 10.0
        assert len(row["calls"]) == 2

        closed = receipt["results"]["arms"]["closed_set"]
        assert closed["calls_total"] == 19
        assert closed["per_case_calls"][0] == 2
        assert closed["prompt_n_total"] == 19 * 400.0
        assert closed["prompt_tokens_total"] == 19 * 1000.0
        assert closed["cache_n_total"] == 19 * 600.0
        assert closed["tokens_generated"] == float(sum(range(1, 20)))
        assert receipt["token_accounting"].startswith("per-case tokens = SUM")

    def test_single_call_cases_keep_the_pre_m0_token_figure(self, tmp_path: Path):
        receipt = run_tool_args_pilot(
            _FakePrimitives(), role=ROLE, receipt_path=tmp_path / "p.json"
        )
        rows = _case_rows(receipt)
        assert all(row["closed_set"]["call_count"] == 1 for row in rows.values())
        assert all(row["closed_set"]["tokens"] == 9.0 for row in rows.values())
        assert all(row["closed_set"]["prompt_n"] is None for row in rows.values())


class TestPilotArmSubsets:
    def test_closed_only_run_reports_absent_arm_as_none(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt = run_tool_args_pilot(
            primitives, role=ROLE, receipt_path=tmp_path / "p.json", arms=["closed_set"]
        )

        assert receipt["arms_run"] == ["closed_set"]
        assert receipt["closed_mode"] == "json"
        counts = receipt["counts"]
        assert counts["closed_set_exact_match"] == 18
        assert counts["free_form_resolved"] is None
        assert counts["agreement_compared"] is None
        assert receipt["results"]["agreement"] is None
        assert receipt["results"]["arms"]["free_form"] is None
        assert all(row["free_form"] is None for row in receipt["results"]["cases"])
        assert len(primitives.calls) == 18
        assert all(call.get("json_schema") is not None for call in primitives.calls)

    def test_free_only_run(self, tmp_path: Path):
        primitives = _FakePrimitives()
        receipt = run_tool_args_pilot(
            primitives, role=ROLE, receipt_path=tmp_path / "p.json", arms=["free_form"]
        )
        assert receipt["arms_run"] == ["free_form"]
        assert receipt["closed_mode"] is None
        assert receipt["counts"]["free_form_exact_match"] == 18
        assert receipt["results"]["arms"]["closed_set"] is None
        assert all(call.get("json_schema") is None for call in primitives.calls)

    def test_unknown_or_empty_arms_are_rejected(self):
        with pytest.raises(ValueError):
            run_tool_args_pilot(_FakePrimitives(), role=ROLE, arms=["bogus"])
        with pytest.raises(ValueError):
            run_tool_args_pilot(_FakePrimitives(), role=ROLE, arms=[])

    def test_case_log_is_written_per_case(self, tmp_path: Path):
        log = tmp_path / "cases.jsonl"
        run_tool_args_pilot(
            _FakePrimitives(), role=ROLE, receipt_path=tmp_path / "p.json", case_log_path=log
        )
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 36
        assert [line["arm"] for line in lines] == ["closed_set"] * 18 + ["free_form"] * 18
        assert lines[0]["case_id"] == _CASES[0].case_id
        assert "call_count" in lines[0]

    def test_provenance_is_embedded(self, tmp_path: Path):
        receipt = run_tool_args_pilot(
            _FakePrimitives(),
            role=ROLE,
            receipt_path=tmp_path / "p.json",
            provenance={"server_commit": "90c12df42", "binary_sha256": "abc"},
        )
        assert receipt["provenance"] == {"server_commit": "90c12df42", "binary_sha256": "abc"}


class TestPilotClosedMode:
    def test_native_mode_and_cue_style_reach_the_runner(self, tmp_path: Path, monkeypatch):
        import src.typed_decisions.tool_args_pilot as pilot

        seen: list[tuple[Any, Any]] = []
        real = pilot.run_typed_decisions

        def spy(primitives, **kwargs):
            seen.append((kwargs.pop("mode", None), kwargs.pop("cue_style", None)))
            return real(primitives, **kwargs)

        monkeypatch.setattr(pilot, "run_typed_decisions", spy)
        receipt = run_tool_args_pilot(
            _FakePrimitives(),
            role=ROLE,
            receipt_path=tmp_path / "p.json",
            closed_mode="native",
            cue_style="id_only",
            arms=["closed_set"],
        )

        assert seen == [("native", "id_only")] * 18
        assert receipt["closed_mode"] == "native"
        assert receipt["cue_style"] == "id_only"
        assert receipt["results"]["arms"]["closed_set"]["decode"] == "typed_native:id_only"

    def test_json_mode_passes_no_mode_kwarg(self, tmp_path: Path, monkeypatch):
        import src.typed_decisions.tool_args_pilot as pilot

        seen: list[dict] = []
        real = pilot.run_typed_decisions

        def spy(primitives, **kwargs):
            seen.append(dict(kwargs))
            return real(primitives, **kwargs)

        monkeypatch.setattr(pilot, "run_typed_decisions", spy)
        run_tool_args_pilot(
            _FakePrimitives(), role=ROLE, receipt_path=tmp_path / "p.json", arms=["closed_set"]
        )
        assert all("mode" not in call and "cue_style" not in call for call in seen)

    def test_receipt_records_native_key_and_resolved_value(self, tmp_path: Path, monkeypatch):
        # TD-29 single-token keys: a re-keyed decision carries its key, and the
        # case record carries key AND resolved value; un-keyed cases omit it.
        import dataclasses

        import src.typed_decisions.tool_args_pilot as pilot

        real = pilot.run_typed_decisions

        def keyed(primitives, **kwargs):
            kwargs.pop("mode", None)
            kwargs.pop("cue_style", None)
            result = real(primitives, **kwargs)
            decisions = tuple(
                dataclasses.replace(d, native_key="B") if d.question_id == "severity" else d
                for d in result.decisions
            )
            return dataclasses.replace(result, decisions=decisions)

        monkeypatch.setattr(pilot, "run_typed_decisions", keyed)
        receipt = run_tool_args_pilot(
            _FakePrimitives(),
            role=ROLE,
            receipt_path=tmp_path / "p.json",
            closed_mode="native",
            arms=["closed_set"],
        )

        rows = {row["case_id"]: row["closed_set"] for row in receipt["results"]["cases"]}
        ticket = rows["ticket-01"]
        assert ticket["native_keys"] == {"severity": {"key": "B", "value": "p2"}}
        assert ticket["exact_match"] is True
        assert "native_keys" not in rows["meeting-01"]

    def test_json_mode_records_carry_no_native_keys(self, tmp_path: Path):
        receipt = run_tool_args_pilot(
            _FakePrimitives(), role=ROLE, receipt_path=tmp_path / "p.json", arms=["closed_set"]
        )
        assert all("native_keys" not in row["closed_set"] for row in receipt["results"]["cases"])

    def test_invalid_mode_combinations_are_rejected(self):
        with pytest.raises(ValueError):
            run_tool_args_pilot(_FakePrimitives(), role=ROLE, closed_mode="bogus")
        with pytest.raises(ValueError):
            run_tool_args_pilot(_FakePrimitives(), role=ROLE, cue_style="id_only")


class TestPilotCliSidecar:
    def test_cli_forwards_sidecar_options(self, tmp_path: Path, monkeypatch):
        primitives = _FakePrimitives()
        captured: dict[str, Any] = {}

        def fake_live(**kwargs):
            captured.update(kwargs)
            return primitives

        monkeypatch.setattr("src.typed_decisions.tool_args_pilot._live_primitives", fake_live)
        provenance = tmp_path / "prov.json"
        provenance.write_text(json.dumps({"server_commit": "90c12df42"}), encoding="utf-8")
        receipt_path = tmp_path / "pilot.json"
        case_log = tmp_path / "cases.jsonl"

        code = main(
            [
                "--live",
                "--role",
                "frontdoor",
                "--receipt",
                str(receipt_path),
                "--arm",
                "free_form",
                "--server-url",
                "frontdoor=http://127.0.0.1:8199",
                "--provenance-file",
                str(provenance),
                "--case-log",
                str(case_log),
            ]
        )

        assert code == 0
        assert captured == {"server_url_overrides": {"frontdoor": "http://127.0.0.1:8199"}}
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["arms_run"] == ["free_form"]
        assert receipt["provenance"] == {"server_commit": "90c12df42"}
        assert len(case_log.read_text(encoding="utf-8").splitlines()) == 18

    def test_cli_rejects_bad_server_url(self):
        assert main(["--live", "--server-url", "frontdoor"]) == 1

    def test_cli_rejects_cue_style_without_native(self, monkeypatch):
        monkeypatch.setattr(
            "src.typed_decisions.tool_args_pilot._live_primitives", lambda: _FakePrimitives()
        )
        assert main(["--live", "--cue-style", "id_only"]) == 1
