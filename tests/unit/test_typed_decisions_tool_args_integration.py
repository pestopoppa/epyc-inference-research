"""Unit tests for the TD-4 tool-argument integration (fail-open, flag-gated).

Covers the integration contract in
``src/typed_decisions/tool_args_integration.py``: the flag gate, the
all-arguments-mappable gate, the exact assembled dict from canned primitives,
typed/transport/assembly failures, and the call-site wiring in
``src/repl_environment/context.py`` (typed path skipped when off, typed
arguments used when on). No model/server call: the primitives seam is a
canned-response fake, mirroring ``tests/unit/test_typed_decisions_tool_args.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from src.features import Features, reset_features, set_features
from src.typed_decisions import tool_args_integration as integration
from src.typed_decisions.native import (
    REASON_NATIVE_TOKENIZER_UNAVAILABLE,
    REASON_NATIVE_UNKNOWN_CANDIDATE,
    REASON_NATIVE_UNSUPPORTED_CANDIDATES,
)
from src.typed_decisions.tool_args import tool_schema_to_questions
from src.typed_decisions.tool_args_integration import (
    maybe_typed_arguments,
    registry_parameters_to_schema,
)
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
)

ROLE = "worker"
STATE = "unit-test state: deploy the service safely"

_FULLY_MAPPABLE: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["fast", "safe"]},
        "dry_run": {"type": "boolean"},
    },
    "required": ["mode", "dry_run"],
}

# One unmappable optional argument -> the whole schema must decline, because
# the assembled dict replaces the model's argument object wholesale.
_PARTIAL: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "mode": {"type": "string", "enum": ["fast", "safe"]},
        "notes": {"type": "string"},
    },
    "required": ["mode"],
}

# The required argument itself is unmappable -> decline without a call.
_REQUIRED_UNMAPPABLE: dict[str, Any] = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

# Valid decisions, invalid assembly: both array-enum members true, which the
# documented single-select approximation rejects.
_MULTI_SELECT: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "tags": {"type": "array", "items": {"type": "string", "enum": ["alpha", "beta"]}},
    },
    "required": ["tags"],
}

_REGISTRY_PARAMETERS: dict[str, Any] = {
    "mode": {"type": "string", "enum": ["fast", "safe"], "required": True},
    "dry_run": {"type": "boolean", "required": True},
}


@pytest.fixture
def enabled():
    set_features(Features(typed_decisions_tool_args=True))
    try:
        yield
    finally:
        reset_features()


@pytest.fixture
def disabled():
    set_features(Features(typed_decisions_tool_args=False))
    try:
        yield
    finally:
        reset_features()


class _FakePrimitives:
    """Canned-response stand-in for ``LLMPrimitives`` (last response repeats)."""

    def __init__(self, responses: str | Sequence[str] = "{}"):
        self.responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls: list[dict] = []

    def llm_call(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _questions(parameters: Mapping[str, Any], tool_name: str = "deploy") -> list[Question]:
    return tool_schema_to_questions(tool_name, parameters).questions


def _canned_response(questions: Sequence[Question], values: Mapping[str, object]) -> str:
    answers: dict[str, dict] = {}
    for question in questions:
        value = values[question.id]
        if question.kind is QuestionKind.CHOICE:
            labels = list(question.options)
            answers[question.id] = {
                "choice": value,
                "probabilities": {label: 1.0 / len(labels) for label in labels},
                "confidence": 1.0,
            }
        elif question.kind is QuestionKind.SCORE:
            labels = [str(level) for level in question.levels]
            answers[question.id] = {
                "score": value,
                "probabilities": {label: 1.0 / len(labels) for label in labels},
                "confidence": 1.0,
            }
        else:
            answers[question.id] = {
                "noul": value,
                "probabilities": {"true": 1.0, "false": 0.0}
                if value
                else {"true": 0.0, "false": 1.0},
                "confidence": 1.0,
            }
    return json.dumps({"answers": answers})


def _decision(question: Question, value: object, mode: str = "native") -> Decision:
    return Decision(
        question_id=question.id,
        kind=question.kind,
        value=value,
        probabilities={},
        confidence=1.0,
        mode=mode,
    )


def _result(decisions: Sequence[Decision], failures: Sequence[ParseFailure], mode: str):
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(failures),
        raw_text="",
        mode=mode,
        elapsed_ms=0.0,
        prompt_sha256="deadbeef",
    )


# ── 1. Flag gate ──────────────────────────────────────────────────────────


class TestFlagGate:
    def test_flag_off_returns_none_without_calling_the_model(self, disabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_flag_off_returns_none_even_when_everything_else_is_ready(self, disabled):
        mapping = _questions(_FULLY_MAPPABLE)
        primitives = _FakePrimitives(_canned_response(mapping, {"mode": "safe", "dry_run": True}))

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []


# ── 2. Engagement gate ────────────────────────────────────────────────────


class TestEngagementGate:
    def test_partially_mappable_schema_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_PARTIAL,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_required_unmappable_argument_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_REQUIRED_UNMAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_non_object_schema_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters={"type": "array", "items": {"type": "string"}},
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_missing_state_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state="   ",
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls == []

    def test_missing_primitives_declines(self, enabled):
        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=None,
            role=ROLE,
        )

        assert result is None

    def test_unusable_primitives_decline(self, enabled):
        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=object(),
            role=ROLE,
        )

        assert result is None


# ── 3. Happy path through the real runner ─────────────────────────────────


class TestHappyPath:
    def test_fully_mappable_schema_returns_the_validated_arguments(self, enabled):
        mapping = tool_schema_to_questions("deploy", _FULLY_MAPPABLE)
        primitives = _FakePrimitives(
            _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
        )

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result == {"mode": "safe", "dry_run": True}
        # One JSON call: native mode could not resolve a tokenizer from the
        # fake primitives (no base URL), so it failed closed with zero model
        # calls and the JSON rerun produced the answer.
        assert len(primitives.calls) == 1
        assert "grammar" not in primitives.calls[0]

    def test_json_preferred_mode_skips_the_native_attempt(self, enabled):
        mapping = tool_schema_to_questions("deploy", _FULLY_MAPPABLE)
        primitives = _FakePrimitives(
            _canned_response(mapping.questions, {"mode": "fast", "dry_run": False})
        )

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
            mode="json",
        )

        assert result == {"mode": "fast", "dry_run": False}
        assert len(primitives.calls) == 1


# ── 4. Failure contract ───────────────────────────────────────────────────


class TestFailureContract:
    def test_typed_failure_returns_none(self, enabled):
        primitives = _FakePrimitives("this is not a JSON object")

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert primitives.calls

    def test_transport_failure_returns_none(self, enabled):
        primitives = _FakePrimitives("[ERROR: backend is down]")

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None
        assert len(primitives.calls) == 1

    def test_assembly_error_returns_none(self, enabled):
        mapping = tool_schema_to_questions("deploy", _MULTI_SELECT)
        primitives = _FakePrimitives(
            _canned_response(
                mapping.questions,
                {"tags__alpha": True, "tags__beta": True},
            )
        )

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_MULTI_SELECT,
            state=STATE,
            primitives=primitives,
            role=ROLE,
        )

        assert result is None

    def test_unknown_preferred_mode_declines_without_a_call(self, enabled):
        primitives = _FakePrimitives()

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=primitives,
            role=ROLE,
            mode="batch",
        )

        assert result is None
        assert primitives.calls == []


# ── 5. Mode preference (native first, JSON on eligibility failure) ────────


class TestModePreference:
    def test_complete_native_pass_wins_without_a_json_rerun(self, monkeypatch, enabled):
        mapping = tool_schema_to_questions("deploy", _FULLY_MAPPABLE)
        values = {"mode": "safe", "dry_run": True}
        native_result = _result(
            [_decision(question, values[question.id]) for question in mapping.questions],
            [],
            "native",
        )
        calls: list[str] = []

        def fake_run(primitives, *, state, questions, role, mode="json", **kwargs):
            calls.append(mode)
            return native_result

        monkeypatch.setattr(integration, "run_typed_decisions", fake_run)

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=_FakePrimitives(),
            role=ROLE,
        )

        assert result == {"mode": "safe", "dry_run": True}
        assert calls == ["native"]

    @pytest.mark.parametrize(
        "reason",
        [REASON_NATIVE_UNSUPPORTED_CANDIDATES, REASON_NATIVE_TOKENIZER_UNAVAILABLE],
    )
    def test_native_eligibility_failure_reruns_in_json(self, monkeypatch, enabled, reason):
        values = {"mode": "fast", "dry_run": False}
        calls: list[str] = []

        def fake_run(primitives, *, state, questions, role, mode="json", **kwargs):
            calls.append(mode)
            if mode == "native":
                return _result([], [ParseFailure(reason, "not eligible")], "native")
            return _result(
                [_decision(question, values[question.id]) for question in questions],
                [],
                "json",
            )

        monkeypatch.setattr(integration, "run_typed_decisions", fake_run)

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=_FakePrimitives(),
            role=ROLE,
        )

        assert result == {"mode": "fast", "dry_run": False}
        assert calls == ["native", "json"]

    def test_other_native_failure_declines_without_a_json_rerun(self, monkeypatch, enabled):
        calls: list[str] = []

        def fake_run(primitives, *, state, questions, role, mode="json", **kwargs):
            calls.append(mode)
            return _result([], [ParseFailure(REASON_NATIVE_UNKNOWN_CANDIDATE, "no row")], mode)

        monkeypatch.setattr(integration, "run_typed_decisions", fake_run)

        result = maybe_typed_arguments(
            tool_name="deploy",
            parameters=_FULLY_MAPPABLE,
            state=STATE,
            primitives=_FakePrimitives(),
            role=ROLE,
        )

        assert result is None
        assert calls == ["native"]


# ── 6. Registry-shape projection ──────────────────────────────────────────


class TestRegistryProjection:
    def test_projection_carries_closed_sets_and_required(self):
        schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)

        assert schema["type"] == "object"
        assert schema["required"] == ["mode", "dry_run"]
        assert schema["properties"]["mode"]["enum"] == ["fast", "safe"]
        assert schema["properties"]["dry_run"]["type"] == "boolean"

    def test_projection_carries_bounds_and_items(self):
        schema = registry_parameters_to_schema(
            {
                "retries": {"type": "integer", "minimum": 0, "maximum": 3},
                "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
            }
        )

        assert schema["properties"]["retries"]["minimum"] == 0
        assert schema["properties"]["tags"]["items"]["enum"] == ["a", "b"]
        assert "required" not in schema

    def test_projection_declines_malformed_maps(self):
        assert registry_parameters_to_schema(None) == {}
        assert registry_parameters_to_schema({}) == {}
        assert registry_parameters_to_schema({"bad": "not-a-spec"}) == {}

    def test_projection_defaults_missing_type_to_string(self):
        schema = registry_parameters_to_schema({"query": {"description": "free text"}})

        assert schema["properties"]["query"]["type"] == "string"


# ── 7. Call-site wiring (src/repl_environment/context.py) ─────────────────


class _Tool:
    def __init__(self, name: str, parameters: Mapping[str, Any]):
        self.name = name
        self.parameters = parameters


class _FakeRegistry:
    def __init__(self, tool: _Tool):
        self._tools = {tool.name: tool}
        self.invocations: list[dict] = []
        self.allowed = True

    def can_use_tool(self, role, tool_name, context=None):
        return self.allowed and tool_name in self._tools

    def invoke(
        self,
        tool_name,
        role,
        *,
        caller_type=None,
        chain_id=None,
        chain_index=0,
        context=None,
        **kwargs,
    ):
        if not self.can_use_tool(role, tool_name, context=context):
            raise PermissionError("tool denied")
        self.invocations.append({"tool_name": tool_name, "role": role, "kwargs": dict(kwargs)})
        return "tool-result"


class _Repl:
    def __init__(self, *, registry, primitives, context, role="worker"):
        self.tool_registry = registry
        self.llm_primitives = primitives
        self.context = context
        self.role = role
        self._active_tool_chain_id = None
        self._active_tool_chain_index = 0


def _make_repl(primitives, registry) -> Any:
    from src.repl_environment.context import _ContextMixin

    class _TestRepl(_ContextMixin, _Repl):
        pass

    return _TestRepl(registry=registry, primitives=primitives, context=STATE)


def test_call_site_skips_typed_path_when_off(monkeypatch, disabled):
    """Flag off: the typed-args path is never entered (no dead work per dispatch)."""
    recorded: list[dict] = []
    helper_calls: list[str] = []

    def _recording(**kwargs):
        recorded.append(kwargs)
        return None

    monkeypatch.setattr(integration, "maybe_typed_arguments", _recording)

    primitives = _FakePrimitives("{}")
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)
    original_helper = repl._typed_tool_arguments

    def _helper_spy(tool_name):
        helper_calls.append(tool_name)
        return original_helper(tool_name)

    monkeypatch.setattr(repl, "_typed_tool_arguments", _helper_spy)

    result = repl._dispatch_tool("deploy", mode="fast", dry_run=False)

    assert result == "tool-result"
    assert helper_calls == []
    assert recorded == []
    assert primitives.calls == []
    assert getattr(repl, "_last_decision_receipt", None) is None
    assert registry.invocations == [
        {"tool_name": "deploy", "role": "worker", "kwargs": {"mode": "fast", "dry_run": False}}
    ]


def test_call_site_forwards_schema_and_state_when_on(enabled, monkeypatch, tmp_path):
    """Flag on: the call site hands the tool's projected schema and task state over."""
    from types import SimpleNamespace

    import src.config

    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    recorded: list[dict] = []
    original = integration.maybe_typed_arguments

    def _recording(**kwargs):
        recorded.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(integration, "maybe_typed_arguments", _recording)

    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    primitives = _FakePrimitives(
        _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)

    repl._dispatch_tool("deploy", mode="fast", dry_run=False)

    assert len(recorded) == 1
    assert recorded[0]["tool_name"] == "deploy"
    assert recorded[0]["parameters"]["properties"]["mode"]["enum"] == ["fast", "safe"]
    assert recorded[0]["state"] == STATE


def test_call_site_uses_typed_arguments_when_on(enabled, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import src.config

    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    schema = registry_parameters_to_schema(_REGISTRY_PARAMETERS)
    mapping = tool_schema_to_questions("deploy", schema)
    primitives = _FakePrimitives(
        _canned_response(mapping.questions, {"mode": "safe", "dry_run": True})
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(primitives, registry)

    result = repl._dispatch_tool("deploy", mode="fast", dry_run=False)

    assert result == "tool-result"
    assert registry.invocations == [
        {"tool_name": "deploy", "role": "worker", "kwargs": {"mode": "safe", "dry_run": True}}
    ]
    rows = (tmp_path / "typed_decisions" / "decision_receipts.jsonl").read_text().splitlines()
    receipt = json.loads(rows[0])
    assert len(rows) == 1
    assert receipt["schema_version"] == "decision_receipt.v1"
    assert receipt["selected_id"] == "deploy"
    assert receipt["validation_result"] == "accepted"
    assert receipt["authorization_result"] == "allowed"
    assert receipt["fallback"] is None
    assert receipt["observed_downstream_outcome"]["status"] == "returned"


def test_call_site_rechecks_task_revision_and_uses_incumbent(enabled, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import src.config

    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(_FakePrimitives(), registry)

    def _mutating_choice(tool_name):
        repl.context = "changed after selection"
        return {"mode": "safe", "dry_run": True}

    monkeypatch.setattr(repl, "_typed_tool_arguments", _mutating_choice)
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}
    receipt = json.loads(
        (tmp_path / "typed_decisions" / "decision_receipts.jsonl").read_text().strip()
    )
    assert receipt["validation_result"] == "stale"
    assert receipt["fallback"] == "incumbent_fallback"


@pytest.mark.parametrize("mutation", ["catalog", "model", "circuit"])
def test_call_site_rechecks_catalog_and_model(enabled, monkeypatch, tmp_path, mutation):
    from types import SimpleNamespace

    import src.config

    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS.copy()))
    primitives = _FakePrimitives()
    circuit = {"available": True}
    if mutation == "circuit":
        primitives.server_urls = {ROLE: "http://selector.invalid"}
        primitives.health_tracker = SimpleNamespace(is_available=lambda url: circuit["available"])
    repl = _make_repl(primitives, registry)

    def _mutating_choice(tool_name):
        if mutation == "catalog":
            registry._tools[tool_name].parameters = {"changed": {"type": "string"}}
        elif mutation == "model":
            primitives.llm_call = None
        else:
            circuit["available"] = False
        return {"mode": "safe", "dry_run": True}

    monkeypatch.setattr(repl, "_typed_tool_arguments", _mutating_choice)
    assert repl._dispatch_tool("deploy", mode="fast", dry_run=False) == "tool-result"
    assert registry.invocations[0]["kwargs"] == {"mode": "fast", "dry_run": False}
    receipt = json.loads(
        (tmp_path / "typed_decisions" / "decision_receipts.jsonl").read_text().strip()
    )
    assert receipt["validation_result"] == (
        "stale" if mutation == "catalog" else "model_unavailable"
    )
    assert receipt["fallback"] == "incumbent_fallback"


def test_call_site_selection_cannot_authorize_tool(enabled, monkeypatch, tmp_path):
    from types import SimpleNamespace

    import src.config

    monkeypatch.setattr(
        src.config,
        "get_config",
        lambda: SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path)),
    )
    registry = _FakeRegistry(_Tool("deploy", _REGISTRY_PARAMETERS))
    repl = _make_repl(_FakePrimitives(), registry)

    def _revoked_choice(tool_name):
        registry.allowed = False
        return {"mode": "safe", "dry_run": True}

    monkeypatch.setattr(repl, "_typed_tool_arguments", _revoked_choice)
    with pytest.raises(PermissionError, match="tool denied"):
        repl._dispatch_tool("deploy", mode="fast", dry_run=False)
    assert registry.invocations == []
    receipt = json.loads(
        (tmp_path / "typed_decisions" / "decision_receipts.jsonl").read_text().strip()
    )
    assert receipt["validation_result"] == "unauthorized"
    assert receipt["authorization_result"] == "denied"
    assert receipt["fallback"] == "incumbent_fallback"
    assert receipt["observed_downstream_outcome"]["detail"] == "PermissionError"
