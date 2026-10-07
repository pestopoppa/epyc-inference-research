"""Synthetic registration controls for the read-only UTM-B1 MCP wrappers."""

import inspect
import json
import ast
from pathlib import Path

from src.trace.mcp_tools import register_trace_navigation_tools


class _FakeMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *, name):
        def register(function):
            self.tools[name] = function
            return function

        return register


class _FakeNavigation:
    def __init__(self):
        self.calls = []

    def search_records(self, text):
        self.calls.append(("search_records", text))
        return [{"id": 4, "summary": "synthetic match"}]

    def get_records(self, event_ids):
        self.calls.append(("get_records", list(event_ids)))
        return [{"id": event_id, "summary": "synthetic exact record"} for event_id in event_ids]


def test_orchestrator_mcp_server_registers_the_trace_navigation_surface():
    repository = Path(__file__).resolve().parents[2]
    source = ast.parse((repository / "src/mcp_server.py").read_text(encoding="utf-8"))
    registrations = [
        node.value
        for node in source.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "register_trace_navigation_tools"
    ]

    assert len(registrations) == 1
    assert len(registrations[0].args) == 1
    assert isinstance(registrations[0].args[0], ast.Name)
    assert registrations[0].args[0].id == "mcp"


def test_registration_exposes_only_existing_search_and_expand_functions():
    server = _FakeMCP()
    navigation = _FakeNavigation()

    register_trace_navigation_tools(server, navigation=navigation)

    assert set(server.tools) == {"ms.search", "ms.expand"}
    assert list(inspect.signature(server.tools["ms.search"]).parameters) == [
        "text", "session_id"
    ]
    assert list(inspect.signature(server.tools["ms.expand"]).parameters) == [
        "event_ids", "session_id"
    ]


def test_search_delegates_without_exposing_store_path_or_vector_inputs():
    server = _FakeMCP()
    navigation = _FakeNavigation()
    register_trace_navigation_tools(server, navigation=navigation)

    response = json.loads(server.tools["ms.search"]("synthetic query", session_id="client"))

    assert response == {"records": [{"id": 4, "summary": "synthetic match"}]}
    assert navigation.calls == [("search_records", "synthetic query")]


def test_expand_preserves_requested_event_order_and_is_read_only():
    server = _FakeMCP()
    navigation = _FakeNavigation()
    register_trace_navigation_tools(server, navigation=navigation)

    response = json.loads(server.tools["ms.expand"]([9, 4], session_id="client"))

    assert [row["id"] for row in response["records"]] == [9, 4]
    assert navigation.calls == [("get_records", [9, 4])]
