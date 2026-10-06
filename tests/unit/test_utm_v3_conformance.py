"""Offline UTM-V3 synthetic structural fixtures; never semantic validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orchestration.repl_memory.memory_actions import MemoryAction, MemoryActionStore
from src.typed_decisions.runner import run_typed_decisions
from src.typed_decisions.types import Question, QuestionKind

FIXTURES = Path(__file__).parent / "fixtures" / "utm_v3"
CASES_PATH = FIXTURES / "synthetic_cases.json"
RUBRIC_PATH = FIXTURES / "rubric.json"
GENERATED_OUTPUT = Path("ni07_utm_v3_synthetic_readout.json")
AXES = ("source_resolution", "privacy", "scope", "freshness", "poisoning", "manipulation")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def _action(spec: dict) -> MemoryAction:
    values = dict(spec)
    values.setdefault("channel", "log")
    return MemoryAction(
        action=values["action"], channel=values["channel"], coordinate="synthetic-utm-v3",
        key=values["key"], content=values["content"],
        title=values.get("title", ""), description=values.get("description", ""),
        source_event_ids=tuple(values.get("source_event_ids", ())),
        embedding_version=values.get("embedding_version", ""),
        when_not_to_use=values.get("when_not_to_use", ""),
        retention_state=values.get("retention_state", "active"),
    )


def test_eight_synthetic_classes_have_only_source_supported_store_expectations(tmp_path: Path):
    cases = _load_cases()
    rubric = json.loads(RUBRIC_PATH.read_text(encoding="utf-8"))
    assert rubric["semantic_gold"] is None
    assert set(rubric["semantic_axes"]) == set(AXES)
    assert {case["case_id"] for case in cases} == {
        "conflicting", "stale", "poisoned", "private", "strategically manipulated",
        "duplicate", "missing-source", "delete/restore",
    }
    observed = {}
    for case in cases:
        assert case["semantic_gold"] is None
        assert set(case["semantic_axes"]) == set(AXES)
        assert set(case["semantic_axes"].values()) == {"not_evaluated"}
        store = MemoryActionStore(tmp_path / case["case_id"])
        results = []
        projection_stages = []
        for action_spec in case["actions"]:
            results.append(store.apply(_action(action_spec)))
            channel = action_spec.get("channel", "log")
            projection_stages.append(
                (store.path / f"{channel}.md").read_text(encoding="utf-8")
            )
        statuses = [result.status for result in results]
        assert statuses == case["expected_statuses"], case["case_id"]
        rows = [json.loads(line) for line in store.ledger_path.read_text().splitlines()]
        current_content = case.get("expected_current")
        projection = store.path / f"{rows[-1]['channel']}.md"
        if case["case_id"] == "conflicting":
            assert "synthetic first" in projection_stages[0]
            assert "synthetic second" in projection_stages[1]
            assert "synthetic first" not in projection_stages[1]
        if case["case_id"] == "delete/restore":
            assert sum(row["action"] == "DELETE" for row in rows) == case["expected_tombstones"]
            assert "synthetic before delete" in projection_stages[0]
            assert "synthetic before delete" not in projection_stages[1]
            assert "synthetic reinsert after delete" not in projection_stages[1]
            assert "synthetic reinsert after delete" in projection_stages[2]
            assert "synthetic before delete" not in projection_stages[2]
        if case["case_id"] == "missing-source":
            assert rows[-1]["source_event_ids"] == ["synthetic-dangling-source-id"]
        if case["case_id"] == "duplicate":
            assert len(rows) == 1  # CREATE exists is a no-op, not a second event.
        if current_content is not None:
            assert current_content in projection.read_text(encoding="utf-8")
        observed[case["case_id"]] = {
            "statuses": statuses,
            "ledger_events": len(rows),
            "structural_disposition": "accepted_by_existing_store_contract",
            "semantic_axes": case["semantic_axes"],
        }

    # The emitted file is declared with native_conformance --generated-output;
    # it is an execution readout, not a tracked source or semantic result.
    parser = _parser_dispositions()
    readout = {
        "schema": "utm-v3-synthetic-readout/1",
        "case_manifest_sha256": _sha(CASES_PATH),
        "rubric_sha256": _sha(RUBRIC_PATH),
        "case_counts": {"total": len(cases), "by_class": {case["case_id"]: 1 for case in cases}},
        "case_sha256": {
            case["case_id"]: hashlib.sha256(
                json.dumps(
                    case, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                ).encode("utf-8")
            ).hexdigest()
            for case in cases
        },
        "structural_dispositions": observed,
        "resolved_denominator": {"basis": "structural outcomes only", "count": len(observed)},
        "unresolved_denominator": {"basis": "structural outcomes only", "count": 0},
        "parser_dispositions": parser,
        "reader_agreement_not_accuracy": {
            "label": "agreement is not accuracy",
            "measured": False,
            "reason": "no paired semantic readers in this structural fixture",
        },
        "semantic_gold_accuracy": None,
        "semantic_calibration": "not_measured",
        "semantic_axes": {axis: "not_evaluated" for axis in AXES},
    }
    assert set(rubric["required_readout"]) <= set(readout)
    encoded = json.dumps(readout, sort_keys=True, separators=(",", ":")) + "\n"
    with GENERATED_OUTPUT.open("x", encoding="utf-8") as output:
        output.write(encoded)
    assert GENERATED_OUTPUT.read_text(encoding="utf-8") == encoded


class _FakePrimitives:
    def __init__(self, emissions: list[str]):
        self.emissions = list(emissions)

    def llm_call(self, prompt: str, **kwargs) -> str:
        assert kwargs["temperature"] == 0.0
        return self.emissions.pop(0)


def _valid_emission(question_id: str = "structural_parse", value: object = True) -> str:
    return json.dumps(
        {
            "answers": {
                question_id: {
                    "noul": value,
                    "probabilities": {"true": 0.9, "false": 0.1},
                    "confidence": 0.8,
                }
            }
        }
    )


def _parser_dispositions() -> dict:
    question = Question(
        id="structural_parse",
        kind=QuestionKind.NOUL,
        text="Is the synthetic envelope parseable?",
    )
    recovered = run_typed_decisions(
        _FakePrimitives(["not-json", _valid_emission()]),
        state="synthetic fixture",
        questions=[question],
        role="frontdoor",
    )
    assert recovered.decisions and recovered.failures
    assert recovered.failures[0].reason == "no_json"

    unresolved = run_typed_decisions(
        _FakePrimitives(["not-json"]),
        state="synthetic fixture",
        questions=[question],
        role="frontdoor",
        max_retries=0,
    )
    assert not unresolved.decisions and len(unresolved.failures) == 1

    wrong_id = run_typed_decisions(
        _FakePrimitives([_valid_emission(question_id="unexpected_question")]),
        state="synthetic fixture",
        questions=[question],
        role="frontdoor",
        max_retries=0,
    )
    assert not wrong_id.decisions and wrong_id.failures[0].reason == "schema_violation"

    wrong_value = run_typed_decisions(
        _FakePrimitives([_valid_emission(value="true")]),
        state="synthetic fixture",
        questions=[question],
        role="frontdoor",
        max_retries=0,
    )
    assert not wrong_value.decisions and wrong_value.failures[0].reason == "schema_violation"

    transport = run_typed_decisions(
        _FakePrimitives(["[ERROR: synthetic transport refusal]"]),
        state="synthetic fixture",
        questions=[question],
        role="frontdoor",
    )
    assert not transport.decisions and transport.failures[0].reason == "transport_error"
    return {
        "attempted_batches": 5,
        "attempts_total": sum(
            len(result.failures) + bool(result.decisions)
            for result in (recovered, unresolved, wrong_id, wrong_value, transport)
        ),
        "resolved_batches": sum(
            bool(result.decisions)
            for result in (recovered, unresolved, wrong_id, wrong_value, transport)
        ),
        "unresolved_batches": sum(
            not result.decisions
            for result in (recovered, unresolved, wrong_id, wrong_value, transport)
        ),
        "recovered_with_failure_history": {
            "decisions": len(recovered.decisions), "failure_history": len(recovered.failures)
        },
        "unresolved_output": {
            "decisions": len(unresolved.decisions), "failures": len(unresolved.failures)
        },
        "wrong_question_id": {
            "decisions": len(wrong_id.decisions), "failures": len(wrong_id.failures)
        },
        "wrong_typed_value": {
            "decisions": len(wrong_value.decisions), "failures": len(wrong_value.failures)
        },
        "transport_refusal": {
            "decisions": len(transport.decisions), "failures": len(transport.failures)
        },
    }


def test_typed_parser_separates_recovered_unresolved_and_transport_dispositions():
    observed = _parser_dispositions()
    assert observed["resolved_batches"] == 1
    assert observed["unresolved_batches"] == 4
    assert observed["recovered_with_failure_history"] == {"decisions": 1, "failure_history": 1}
