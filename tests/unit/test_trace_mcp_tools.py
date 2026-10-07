"""Off-host FastMCP controls over a temporary synthetic trace database."""

import asyncio
import ast
import json
from pathlib import Path
from types import SimpleNamespace

from fastmcp import Client, FastMCP

from src.trace import navigation
from src.trace.store import Event, ensure_schema, upsert_events
from src.trace.mcp_tools import register_trace_navigation_tools


def _run(coro):
    return asyncio.run(coro)


def _synthetic_navigation(tmp_path):
    db_path = tmp_path / "synthetic-trace.sqlite"
    conn = ensure_schema(db_path)
    try:
        upsert_events(conn, [
            Event(
                ts_utc="2026-10-07T00:00:00+00:00",
                source="synthetic",
                source_path="synthetic-source.jsonl",
                source_line=1,
                session_id="synthetic-session",
                category="observe",
                summary="needle first synthetic event",
                detail_json=json.dumps({"private": "synthetic payload"}),
            ),
            Event(
                ts_utc="2026-10-07T00:00:01+00:00",
                source="synthetic",
                source_path="synthetic-source.jsonl",
                source_line=2,
                session_id="synthetic-session",
                category="observe",
                summary="needle second synthetic event",
                detail_json=json.dumps({"private": "another payload"}),
            ),
        ])
    finally:
        conn.close()
    return SimpleNamespace(
        search_records=lambda text: navigation.search_records(text, db_path=db_path),
        get_records=lambda ids: navigation.get_records(ids, db_path=db_path),
    )


def test_default_orchestrator_catalog_does_not_activate_candidate_tools():
    repository = Path(__file__).resolve().parents[2]
    source = ast.parse((repository / "src/mcp_server.py").read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_trace_navigation_tools"
        for node in ast.walk(source)
    )


def test_opt_in_registration_exposes_dotted_names_and_exact_optional_schemas(tmp_path):
    server = FastMCP("synthetic-trace-navigation")
    register_trace_navigation_tools(server, navigation=_synthetic_navigation(tmp_path))

    async def _schemas():
        async with Client(server) as client:
            return {tool.name: tool.inputSchema for tool in await client.list_tools()}

    schemas = _run(_schemas())

    assert set(schemas) == {"ms.search", "ms.expand"}
    search = schemas["ms.search"]
    assert search["required"] == ["text"]
    assert search["properties"]["text"]["type"] == "string"
    assert search["properties"]["session_id"]["type"] == "string"
    assert search["properties"]["session_id"]["default"] == ""
    assert search["additionalProperties"] is False
    expand = schemas["ms.expand"]
    assert expand["required"] == ["event_ids"]
    assert expand["properties"]["event_ids"] == {"items": {"type": "integer"}, "type": "array"}
    assert expand["properties"]["session_id"]["default"] == ""
    assert expand["additionalProperties"] is False


def test_fastmcp_calls_reach_existing_navigation_against_synthetic_sqlite(tmp_path):
    navigation_api = _synthetic_navigation(tmp_path)
    server = FastMCP("synthetic-trace-navigation")
    register_trace_navigation_tools(server, navigation=navigation_api)

    async def _calls():
        async with Client(server) as client:
            search = await client.call_tool("ms.search", {"text": "needle"})
            # Deliberately omit session_id to verify its default remains callable.
            search_rows = json.loads(search.content[0].text)["records"]
            ids = [row["id"] for row in search_rows]
            expand = await client.call_tool("ms.expand", {"event_ids": ids[::-1]})
            expanded_rows = json.loads(expand.content[0].text)["records"]
            return search_rows, expanded_rows

    search_rows, expanded_rows = _run(_calls())

    assert len(search_rows) == 2
    assert [row["id"] for row in expanded_rows] == [row["id"] for row in search_rows][::-1]
    assert all("synthetic" in row["source_path"] for row in expanded_rows)
