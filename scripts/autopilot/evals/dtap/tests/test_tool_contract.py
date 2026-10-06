"""Opt-in native tool schemas and fake HTTP transport; zero endpoint access."""
from __future__ import annotations

import json

import pytest

from harness.endpoint import ChatEndpoint, ChatResult
from harness.outcomes import HarnessFailure, ParseFailure, ToolFailure
from harness.runner import CaseRegistry, DEFAULT_ARM_CONFIG, run_case
from harness.tool_contract import (
    CONTRACT_VERSION,
    NativeToolContract,
    load_native_tool_contract,
    validate_tool_calls,
)
from harness.trace import verify_trace


def _schema(name="lookup", parameters=None):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": "Public test function.",
            "parameters": parameters or {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }


class _Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.body).encode()


def _transport(response=None):
    captured = []

    def opener(request, *, timeout):
        captured.append({
            "url": request.full_url,
            "body": json.loads(request.data),
            "timeout": timeout,
        })
        return _Response(response or {"choices": [{"message": {"content": "done"}}]})

    return opener, captured


def test_legacy_request_stays_message_only_and_schema_is_opt_in():
    opener, captured = _transport()
    endpoint = ChatEndpoint("http://fake/v1", model="tiny", opener=opener)
    endpoint.complete([{"role": "user", "content": "hi"}])
    assert set(captured[0]["body"]) == {"model", "messages", "temperature", "max_tokens"}


@pytest.mark.parametrize(
    ("endpoint_mode", "expects_client_marker"),
    [("openai-compatible", False), ("orchestrator-client", True)],
)
def test_fake_transport_sends_schema_and_only_explicit_orchestrator_mode_marker(
    endpoint_mode, expects_client_marker,
):
    opener, captured = _transport()
    endpoint = ChatEndpoint("http://fake/v1", opener=opener)
    tools = [_schema()]
    endpoint.complete([], tools=tools, tool_choice="auto", endpoint_mode=endpoint_mode)
    body = captured[0]["body"]
    assert body["tools"] == tools and body["tool_choice"] == "auto"
    assert (body.get("x_tool_mode") == "client") is expects_client_marker


def test_invalid_mode_is_rejected_before_fake_transport():
    opener, captured = _transport()
    endpoint = ChatEndpoint("http://fake/v1", opener=opener)
    with pytest.raises(ValueError, match="explicit endpoint mode"):
        endpoint.complete([], tools=[_schema()], endpoint_mode="guess-from-url")
    assert captured == []


@pytest.mark.parametrize("arguments", ["{bad", "[]"])
def test_response_parser_rejects_malformed_or_nonobject_json_arguments(arguments):
    endpoint = ChatEndpoint("http://fake/v1")
    envelope = {"choices": [{"message": {"content": None, "tool_calls": [{
        "id": "call-1", "function": {"name": "lookup", "arguments": arguments},
    }]}}]}
    with pytest.raises(ParseFailure):
        endpoint._parse(envelope)


def test_pinned_catalog_covers_registry_and_uses_plain_unique_names():
    registry = CaseRegistry()
    contract = load_native_tool_contract(registry.cases, registry_path=registry.path)
    assert contract.version == CONTRACT_VERSION
    assert set(contract.cases) == set(registry.cases)
    assert all("." not in item["function"]["name"] for tools in contract.cases.values() for item in tools)


def test_argument_validation_checks_required_types_and_outer_unknowns_but_allows_nested_maps():
    schema = _schema(parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "count": {"type": "integer"},
            "attributes": {"type": "object", "additionalProperties": {}},
        },
        "required": ["query", "count"],
        "additionalProperties": False,
    })
    tools = [schema]
    validate_tool_calls([{"name": "lookup", "arguments": {
        "query": "x", "count": 2, "attributes": {"arbitrary": [1, {"ok": True}]},
    }}], tools)
    for arguments in (
        {"count": 2},
        {"query": "x", "count": True},
        {"query": "x", "count": 2, "unexpected": 1},
    ):
        with pytest.raises(ToolFailure):
            validate_tool_calls([{"name": "lookup", "arguments": arguments}], tools)
    with pytest.raises(ToolFailure, match="must be an object"):
        validate_tool_calls([{"name": "lookup", "arguments": []}], tools)


def test_anyof_keeps_sibling_type_and_enum_constraints():
    tools = [_schema(parameters={
        "type": "object",
        "properties": {
            "value": {
                "type": "string",
                "enum": ["safe"],
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
        },
        "required": ["value"],
        "additionalProperties": False,
    })]
    validate_tool_calls([{"name": "lookup", "arguments": {"value": "safe"}}], tools)
    for value in (None, "other"):
        with pytest.raises(ToolFailure):
            validate_tool_calls([{"name": "lookup", "arguments": {"value": value}}], tools)


def test_loaded_contract_returns_independent_schema_values():
    registry = CaseRegistry()
    contract = load_native_tool_contract(registry.cases, registry_path=registry.path)
    first = contract.for_case("crm-benign-001")
    original = contract.tools_sha256("crm-benign-001")
    first[0]["function"]["name"] = "mutated"
    second = contract.for_case("crm-benign-001")
    assert second[0]["function"]["name"] != "mutated"
    assert contract.tools_sha256("crm-benign-001") == original


def test_registry_bytes_must_match_the_pinned_sidecar(tmp_path):
    registry = CaseRegistry()
    changed = tmp_path / "cases.json"
    changed.write_bytes(registry.path.read_bytes() + b" ")
    with pytest.raises(HarnessFailure, match="case-registry bytes"):
        load_native_tool_contract(registry.cases, registry_path=changed)


def test_declared_function_uses_fake_parser_and_preserves_native_judge_result(tmp_path):
    registry = CaseRegistry()
    case_id = "crm-benign-001"
    contract = load_native_tool_contract(registry.cases, registry_path=registry.path)
    calls = []
    response_bodies = [
        {"choices": [{"finish_reason": "tool_calls", "message": {"content": None,
         "tool_calls": [{"id": "call-native", "type": "function",
                         "function": {"name": "get_available_modules", "arguments": "{}"}}]}}]},
        {"choices": [{"finish_reason": "stop", "message": {"content": "Completed."}}]},
    ]

    class QueueResponse(_Response):
        pass

    def opener(request, *, timeout):
        calls.append(json.loads(request.data))
        return QueueResponse(response_bodies[len(calls) - 1])

    endpoint = ChatEndpoint("http://fake/v1", opener=opener)
    result = run_case(
        case_id, "done", 0, endpoint, DEFAULT_ARM_CONFIG, tmp_path, registry,
        native_tool_contract=contract,
    )
    assert result.status == "ok" and len(calls) == 2
    assert calls[0]["tools"] == contract.for_case(case_id)
    assert calls[0]["tools"][0]["type"] == "function"
    records = verify_trace(result.trace_path)
    request = next(record["payload"] for record in records if record["event"] == "endpoint_request")
    assert request["request_tools_sha256"] == contract.tools_sha256(case_id)
    accepted = [record["payload"] for record in records if record["event"] == "tool_call"]
    assert [call["tool"] for call in accepted] == ["get_available_modules"]
    verdict = next(record["payload"]["verdict"] for record in records if record["event"] == "judge_result")
    assert result.task_success == verdict["task_success"]
    assert result.attack_success == verdict["attack_success"]


def test_unknown_function_is_rejected_before_runner_records_accepted_call(tmp_path):
    registry = CaseRegistry()
    case_id = "crm-benign-001"
    contract = NativeToolContract.from_mappings(
        CONTRACT_VERSION,
        {"repository": "fixture", "commit": "0" * 40, "tree": "0" * 40},
        {case_id: [_schema("known")]},
        "a" * 64,
    )

    class FakeEndpoint:
        def complete(self, messages, seed=0, **kwargs):
            assert kwargs["tools"] == contract.for_case(case_id)
            assert kwargs["endpoint_mode"] == "openai-compatible"
            return ChatResult("", [{"name": "not-advertised", "arguments": {}}], "tool_calls")

    result = run_case(
        case_id, "done", 0, FakeEndpoint(), DEFAULT_ARM_CONFIG, tmp_path, registry,
        native_tool_contract=contract,
    )
    assert result.status == "failed" and result.failure["type"] == "tool"
    records = verify_trace(result.trace_path)
    assert not any(record["event"] == "tool_call" for record in records)


def test_contract_loader_rejects_invalid_required_property_with_valid_control(tmp_path):
    registry = CaseRegistry()
    source_path = Path(__file__).resolve().parents[1] / "harness" / "tool_schemas.json"
    raw = json.loads(source_path.read_text())
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(raw))
    control = load_native_tool_contract(registry.cases, path, registry_path=registry.path)
    assert control.version == CONTRACT_VERSION

    malformed = json.loads(path.read_text())
    first_tool = malformed["cases"]["crm-benign-001"][0]["function"]["parameters"]
    first_tool["required"] = ["__undeclared_required_property__"]
    path.write_text(json.dumps(malformed))
    with pytest.raises(HarnessFailure, match="required names must be declared properties"):
        load_native_tool_contract(registry.cases, path, registry_path=registry.path)
