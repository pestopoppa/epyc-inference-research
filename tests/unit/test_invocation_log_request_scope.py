"""RTG-02 / INV-LOG: the process-global tool-invocation log must never be the
source of a per-request durable record, and must not grow without bound.

Two defects, one root cause (the registry is ONE object per API process —
``src/api/__init__.py`` builds a single ``state.tool_registry``):

1. LEAK. ``chat_pipeline/stages.py`` (ReAct), ``chat_delegation.py``,
   ``chat.py`` and ``stream_adapter.py`` read
   ``tool_registry.get_invocation_log()`` to build per-request telemetry. Under
   concurrency that log interleaves every in-flight request's calls, so a
   durable record picked up OTHER requests' tool calls. A global
   ``clear_invocation_log()`` cannot fix this: clearing shared state to scope one
   request is itself racy. The fix scopes by construction —
   ``REPLEnvironment._invoked_tools``, one list per REPL, hence one per request.
2. UNBOUNDED GROWTH. The log was a plain ``list`` nothing ever cleared
   (``clear_invocation_log`` had zero callers in ``src/``), so it grew for the
   lifetime of the uvicorn process. It is now a bounded ring.
"""
from __future__ import annotations

import pytest

from src.registry.tool_registry import (
    INVOCATION_LOG_MAX_DEFAULT,
    Tool,
    ToolCategory,
    ToolPermissions,
    ToolRegistry,
)
from src.repl_environment import REPLConfig, REPLEnvironment


def _shared_registry() -> ToolRegistry:
    """One registry, as the API has one — shared by every request."""
    registry = ToolRegistry()
    for name in ("alpha", "beta", "boom"):
        registry.register_tool(
            Tool(
                name=name,
                description=name,
                category=ToolCategory.DATA,
                parameters={},
                handler=(
                    (lambda: (_ for _ in ()).throw(RuntimeError("tool blew up")))
                    if name == "boom"
                    else (lambda n=name: f"{n}-result")
                ),
            )
        )
    registry.set_role_permissions(
        "worker", ToolPermissions(allowed_categories=[ToolCategory.DATA]),
    )
    return registry


def _repl(registry: ToolRegistry) -> REPLEnvironment:
    return REPLEnvironment(
        context="x",
        config=REPLConfig(structured_mode=True),
        tool_registry=registry,
        role="worker",
    )


# ── Fix A: per-request scope ────────────────────────────────────────────────


def test_interleaved_requests_each_see_only_their_own_invocations():
    """Two concurrent requests, calls interleaved on ONE shared registry."""
    registry = _shared_registry()
    req_a = _repl(registry)
    req_b = _repl(registry)

    # Interleave: A, B, B, A — exactly what uvicorn concurrency produces.
    req_a.execute('TOOL("alpha")')
    req_b.execute('TOOL("beta")')
    req_b.execute('TOOL("beta")')
    req_a.execute('TOOL("alpha")')

    # The shared registry ring holds ALL FOUR, mixed together. This is the
    # precondition that made every reader of it wrong.
    shared = [inv.tool_name for inv in registry.get_invocation_log()]
    assert shared == ["alpha", "beta", "beta", "alpha"]

    assert [r.tool_name for r in req_a._invoked_tools] == ["alpha", "alpha"]
    assert [r.tool_name for r in req_b._invoked_tools] == ["beta", "beta"]


def test_durable_record_contains_only_its_own_requests_tool_calls():
    """The `work` payload built for request A must not name request B's tools."""
    from src.api.routes.chat_pipeline.repl_executor import _repl_work_meta

    registry = _shared_registry()
    req_a = _repl(registry)
    req_b = _repl(registry)

    req_a.execute('TOOL("alpha")')
    req_b.execute('TOOL("beta")')

    meta_a = _repl_work_meta(req_a, "answer A")
    meta_b = _repl_work_meta(req_b, "answer B")

    names_a = [c["tool_name"] for c in meta_a["work"]["tool_calls"]]
    names_b = [c["tool_name"] for c in meta_b["work"]["tool_calls"]]
    assert names_a == ["alpha"], f"request A's record leaked: {names_a}"
    assert names_b == ["beta"], f"request B's record leaked: {names_b}"


def test_no_tool_request_after_a_tool_request_records_nothing():
    """The original leak shape: a later NO-TOOL request must report no tools."""
    from src.api.routes.chat_pipeline.repl_executor import _repl_work_meta

    registry = _shared_registry()
    with_tool = _repl(registry)
    without_tool = _repl(registry)

    with_tool.execute('TOOL("alpha")')
    assert registry.get_invocation_log(), "precondition: shared ring is non-empty"

    without_tool.execute("x = 1 + 1")
    meta = _repl_work_meta(without_tool, "answer")
    assert meta["work"].get("tool_calls") in (None, [])


def test_failing_tool_still_scopes_and_cleans_up():
    """An exception mid-request must not leave the failing call attributed to a
    neighbouring request, and the failing request's own state dies with it."""
    registry = _shared_registry()
    failing = _repl(registry)
    neighbour = _repl(registry)

    neighbour.execute('TOOL("alpha")')
    failing.execute('TOOL("boom")')  # handler raises; recorded success=False

    failing_records = failing._invoked_tools
    assert [r.tool_name for r in failing_records] == ["boom"]
    assert failing_records[0].success is False

    # The neighbour is untouched by the failure.
    assert [r.tool_name for r in neighbour._invoked_tools] == ["alpha"]

    # Cleanup is by construction: the per-request buffer is owned by the REPL, so
    # dropping the request drops the buffer. Nothing request-scoped survives in
    # the process-global registry beyond its bounded diagnostic ring.
    del failing
    assert [r.tool_name for r in neighbour._invoked_tools] == ["alpha"]


# ── Fix A: boundedness ──────────────────────────────────────────────────────


def test_registry_log_is_bounded_after_many_requests(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX", "5")
    registry = _shared_registry()

    for _ in range(50):
        _repl(registry).execute('TOOL("alpha")')

    log = registry.get_invocation_log()
    assert len(log) == 5, f"ring must evict, got {len(log)} entries"
    assert all(inv.tool_name == "alpha" for inv in log)


def test_registry_log_default_cap_is_finite():
    registry = ToolRegistry()
    assert registry._invocation_log.maxlen == INVOCATION_LOG_MAX_DEFAULT
    assert registry._invocation_log_enabled is True


def test_registry_log_can_be_disabled(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX", "0")
    registry = _shared_registry()
    registry.invoke("alpha", "worker")
    assert registry.get_invocation_log() == []


def test_malformed_cap_override_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_TOOL_INVOCATION_LOG_MAX", "not-a-number")
    registry = ToolRegistry()
    assert registry._invocation_log.maxlen == INVOCATION_LOG_MAX_DEFAULT


def test_clear_invocation_log_still_works():
    """Kept as a diagnostic/test API (it is NOT a per-request mechanism), so it
    must stay functional rather than becoming dead code."""
    registry = _shared_registry()
    registry.invoke("alpha", "worker")
    assert registry.get_invocation_log()
    registry.clear_invocation_log()
    assert registry.get_invocation_log() == []


def test_get_invocation_log_returns_a_snapshot():
    registry = _shared_registry()
    registry.invoke("alpha", "worker")
    snapshot = registry.get_invocation_log()
    snapshot.clear()
    assert len(registry.get_invocation_log()) == 1


# ── No route may read the shared log for per-request telemetry ──────────────


def _global_invocation_log_calls(source: str) -> list[int]:
    """Return call lines that read the process-global invocation log.

    Parse Python syntax, rather than grep: comments and strings that explain the
    hazard are harmless. Request-local ``repl._invoked_tools`` remains allowed.
    """
    import ast

    tree = ast.parse(source)
    return [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_invocation_log"
    ]


def test_request_paths_do_not_read_the_process_global_log():
    """Scan every tracked route source so a new endpoint joins the guard."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    route_root = root / "src/api/routes"
    offenders = {}
    for path in sorted(route_root.rglob("*.py")):
        module_path = path.relative_to(root).as_posix()
        lines = _global_invocation_log_calls(path.read_text(encoding="utf-8"))
        if lines:
            offenders[module_path] = lines

    assert not offenders, (
        "route code must not read the process-global log for request telemetry; "
        f"per-request records must come from repl._invoked_tools: {offenders}"
    )


def test_route_guard_detects_calls_but_ignores_comments_strings_and_request_local_log():
    """Pin both the forbidden syntax and the legitimate request-local spelling."""
    assert _global_invocation_log_calls(
        "registry.get_invocation_log()\n"
    ) == [1]
    assert _global_invocation_log_calls(
        "# registry.get_invocation_log()\n"
        "note = 'registry.get_invocation_log()'\n"
        "calls = list(getattr(repl, '_invoked_tools', None) or [])\n"
    ) == []
