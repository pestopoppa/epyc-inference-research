"""Mock-only contracts for the provisional CS-24 conversation shadow."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from src.features import Features, reset_features, set_features
from src.typed_decisions import QuestionKind
from src.typed_decisions import conversation_shadow as conversation
from src.typed_decisions import shadow


class _FakePrimitives:
    def __init__(self, responses: str | Sequence[str]):
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[dict] = []

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        response = self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]
        if isinstance(response, BaseException):
            raise response
        return response


def _response(choice: str) -> str:
    return json.dumps(
        {
            "answers": {
                "conversation_route": {
                    "choice": choice,
                    "probabilities": {"LOCAL": 0.35, "ORCH": 0.65},
                    "confidence": 0.65,
                }
            }
        }
    )


@pytest.fixture(autouse=True)
def _reset_flags(monkeypatch):
    reset_features()
    monkeypatch.delenv(shadow.ENV_LOG_PATH, raising=False)
    with shadow._state_lock:
        shadow._pending = 0
        shadow._dropped = 0
    yield
    assert shadow._executor is None  # these fixtures never start an executor
    with shadow._state_lock:
        shadow._pending = 0
        shadow._dropped = 0
    reset_features()


def test_builder_emits_only_provisional_binary_choice():
    question = conversation.build_conversation_question()
    assert question.kind is QuestionKind.CHOICE
    assert question.options == ("LOCAL", "ORCH")
    assert "provisional" in question.criteria[0].lower()
    assert "social/backchannel/control" in question.criteria[0]
    assert "factual/reasoning/code/memory/current-events" in question.criteria[0]


@pytest.mark.parametrize(
    "labels",
    [
        ("LOCAL",),
        ("LOCAL", "ORCH", "UNKNOWN"),
        ("LOCAL", "ORCH", "ABSTAIN"),
        ("local", "ORCH"),
        ("LOCAL", "LOCAL"),
    ],
)
def test_builder_refuses_unverified_or_unsupported_catalog(labels):
    with pytest.raises(ValueError, match="provisional"):
        conversation.build_conversation_question(labels=labels)


def test_submit_delegates_json_and_exact_incumbent_without_mutation(monkeypatch):
    incumbent = {"route": "host-route", "confidence": 0.12}
    before = incumbent.copy()
    captured = {}

    def fake_submit(primitives, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(conversation, "submit_shadow", fake_submit)
    assert conversation.submit_conversation_shadow(
        object(), state="synthetic state", incumbent=incumbent, role="worker_general"
    )
    assert captured["incumbent"] is incumbent
    assert captured["mode"] == "json"
    assert captured["questions"][0].options == ("LOCAL", "ORCH")
    assert incumbent == before


def test_invalid_catalog_fails_open_without_submit_or_incumbent_change(monkeypatch):
    incumbent = {"route": "host-route"}
    called = []
    monkeypatch.setattr(conversation, "submit_shadow", lambda *args, **kwargs: called.append(1))
    assert not conversation.submit_conversation_shadow(
        object(), state="synthetic", incumbent=incumbent, role="worker_general", labels=("LOCAL", "ORCH", "UNKNOWN")
    )
    assert called == []
    assert incumbent == {"route": "host-route"}


@pytest.mark.parametrize("reason", ["flag_off", "missing_sink", "queue_full", "submit_error"])
def test_submission_refusals_are_nonblocking_and_fail_open(monkeypatch, tmp_path, reason):
    incumbent = {"route": "host-route"}
    if reason == "flag_off":
        assert not conversation.submit_conversation_shadow(
            object(), state="synthetic", incumbent=incumbent, role="worker_general", log_path="/tmp/unused-shadow.jsonl"
        )
    elif reason == "missing_sink":
        set_features(Features(typed_decisions_shadow=True))
        assert not conversation.submit_conversation_shadow(
            object(), state="synthetic", incumbent=incumbent, role="worker_general"
        )
    elif reason == "queue_full":
        set_features(Features(typed_decisions_shadow=True))
        monkeypatch.setattr(shadow, "_get_executor", lambda: pytest.fail("queue-full path started executor"))
        with shadow._state_lock:
            shadow._pending = shadow.MAX_PENDING
            shadow._dropped = 0
        primitives = _FakePrimitives(_response("LOCAL"))
        assert not shadow.submit_shadow(
            primitives,
            surface="unit.queue_full",
            state="synthetic",
            questions=(conversation.build_conversation_question(),),
            incumbent=incumbent,
            role="worker_general",
            log_path=tmp_path / "queue-full.jsonl",
            mode="json",
        )
        assert shadow.shadow_stats() == {"pending": shadow.MAX_PENDING, "dropped": 1, "bound": shadow.MAX_PENDING}
        assert primitives.calls == []
    else:
        monkeypatch.setattr(
            conversation,
            "submit_shadow",
            (lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("submit failed")))
            if reason == "submit_error"
            else (lambda *args, **kwargs: False),
        )
        assert not conversation.submit_conversation_shadow(
            object(), state="synthetic", incumbent=incumbent, role="worker_general"
        )
    assert incumbent == {"route": "host-route"}


def test_actual_runner_accepts_valid_binary_and_keeps_confidence_observational(tmp_path):
    incumbent = {"route": "host-route", "confidence": 0.01}
    primitives = _FakePrimitives(_response("ORCH"))
    path = tmp_path / "shadow.jsonl"
    shadow.shadow_decision(
        primitives,
        surface="unit.cs24",
        state="synthetic state",
        questions=(conversation.build_conversation_question(),),
        incumbent=incumbent,
        role="worker_general",
        log_path=path,
        mode="json",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["decisions"][0]["value"] == "ORCH"
    assert record["decisions"][0]["confidence"] == pytest.approx(0.65)
    assert record["incumbent"] == {"route": "host-route", "confidence": 0.01}
    assert incumbent == {"route": "host-route", "confidence": 0.01}


def test_actual_runner_recovers_schema_failure_and_preserves_attempt_history(tmp_path):
    primitives = _FakePrimitives([_response("UNKNOWN"), _response("LOCAL")])
    path = tmp_path / "recovered.jsonl"
    shadow.shadow_decision(
        primitives,
        surface="unit.cs24",
        state="synthetic state",
        questions=(conversation.build_conversation_question(),),
        incumbent={"route": "host-route"},
        role="worker_general",
        log_path=path,
        mode="json",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert len(primitives.calls) == 2
    assert record["decisions"][0]["value"] == "LOCAL"
    assert record["failures"]  # recovered failures remain attempt history


def test_actual_runner_unresolved_invalid_and_transport_failures_remain_observational(tmp_path):
    incumbent = {"route": "host-route"}
    path = tmp_path / "invalid.jsonl"
    shadow.shadow_decision(
        _FakePrimitives(_response("UNKNOWN")),
        surface="unit.cs24",
        state="synthetic state",
        questions=(conversation.build_conversation_question(),),
        incumbent=incumbent,
        role="worker_general",
        log_path=path,
        mode="json",
    )
    invalid_record = json.loads(path.read_text(encoding="utf-8"))
    assert invalid_record["decisions"] == []
    assert invalid_record["failures"]  # unresolved output remains explicit
    assert invalid_record["incumbent"] == incumbent
    assert incumbent == {"route": "host-route"}

    path = tmp_path / "transport.jsonl"
    shadow.shadow_decision(
        _FakePrimitives("[ERROR: offline]"),
        surface="unit.cs24",
        state="synthetic state",
        questions=(conversation.build_conversation_question(),),
        incumbent=incumbent,
        role="worker_general",
        log_path=path,
        mode="json",
    )
    transport_record = json.loads(path.read_text(encoding="utf-8"))
    assert transport_record["decisions"] == []
    assert transport_record["failures"][0]["reason"] == "transport_error"
    assert transport_record["incumbent"] == incumbent


@pytest.mark.parametrize(
    "raw,expected_reason",
    [
        ("not JSON", "no_json"),
        (
            json.dumps({"answers": {"not_conversation_route": {
                "choice": "LOCAL", "probabilities": {"LOCAL": 0.35, "ORCH": 0.65}, "confidence": 0.65,
            }}}),
            "schema_violation",
        ),
        (
            json.dumps({"answers": {"conversation_route": {
                "choice": ["LOCAL"], "probabilities": {"LOCAL": 0.35, "ORCH": 0.65}, "confidence": 0.65,
            }}}),
            "schema_violation",
        ),
    ],
)
def test_actual_runner_records_malformed_wrong_id_and_wrong_type_controls(tmp_path, raw, expected_reason):
    path = tmp_path / "negative-control.jsonl"
    shadow.shadow_decision(
        _FakePrimitives(raw),
        surface="unit.cs24",
        state="synthetic negative control",
        questions=(conversation.build_conversation_question(),),
        incumbent={"route": "host-route"},
        role="worker_general",
        log_path=path,
        mode="json",
    )
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["decisions"] == []
    assert record["failures"]
    assert {failure["reason"] for failure in record["failures"]} == {expected_reason}
    assert record["incumbent"] == {"route": "host-route"}


def test_abstention_is_not_a_supported_third_label():
    for label in ("UNKNOWN", "ABSTAIN"):
        with pytest.raises(ValueError, match="provisional"):
            conversation.build_conversation_question(labels=("LOCAL", "ORCH", label))
