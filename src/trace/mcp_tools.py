"""Thin model-facing registration for the existing read-only trace navigation API."""

from __future__ import annotations

import json
from typing import Any


def register_trace_navigation_tools(server: Any, *, navigation: Any | None = None) -> None:
    """Register the two UTM-B1 lookup tools over the existing navigation API.

    ``session_id`` is accepted for the orchestrator MCP client contract but is
    not a trace filter. No database path, write operation, embedding source, or
    retrieval policy is exposed as a model-controlled argument.
    """
    if navigation is None:
        from src.trace import navigation as navigation_module

        navigation = navigation_module

    @server.tool(name="ms.search")
    def ms_search(text: str, session_id: str = "") -> str:
        """Search the existing trace FTS store and return ranked event rows."""
        rows = navigation.search_records(text)
        return json.dumps({"records": rows}, ensure_ascii=False, sort_keys=True)

    @server.tool(name="ms.expand")
    def ms_expand(event_ids: list[int], session_id: str = "") -> str:
        """Expand exact trace event IDs in caller-supplied order."""
        rows = navigation.get_records(event_ids)
        return json.dumps({"records": rows}, ensure_ascii=False, sort_keys=True)
