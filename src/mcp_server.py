#!/usr/bin/env python3
"""Read-only MCP server exposing orchestrator info to Claude Code.

Provides introspection tools for querying model configurations,
orchestrator roles, server status, benchmark results, and canvas operations.

Uses plugin-based tool loading with hot-reload support.

Usage (stdio transport, launched by Claude Code):
    python src/mcp_server.py

Configuration (.mcp.json):
    {
      "mcpServers": {
        "orchestrator": {
          "command": "/mnt/raid0/llm/epyc-orchestrator/.venv/bin/python",
          "args": ["src/mcp_server.py"]
        }
      }
    }
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
import time
from pathlib import Path

from fastmcp import FastMCP
from fastmcp import Context

logger = logging.getLogger(__name__)

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

mcp = FastMCP("orchestrator-info")

# UTM-B1: expose only the existing read-only FTS lookup and exact event expand
# functions. Navigation owns its fixed default store and allowlist; the tools
# do not accept caller-selected databases, writes, or embedding/vector inputs.
from src.trace.mcp_tools import register_trace_navigation_tools

register_trace_navigation_tools(mcp)

# HS-4 P0-MCP-b: every tool declares ``session_id``.
# OpenCode names MCP tools "<server>_<tool>", and the epyc OpenCode plugin stamps
# ``session_id`` onto every ``orchestrator_*`` call. fastmcp>=3 refuses undeclared
# arguments (tests/unit/test_mcp_undeclared_args_refused.py), so a tool without the
# parameter would refuse every stamped call. The chat tools forward it to /chat as
# ``ChatRequest.session_id``. The other tools read static state and do not depend on
# the session, so they accept the value and ignore it.

# Plugin loader for dynamic tool discovery
_plugin_loader = None


def _get_plugin_loader():
    """Get or create the plugin loader."""
    global _plugin_loader
    if _plugin_loader is None:
        from src.tool_loader import ToolPluginLoader

        _plugin_loader = ToolPluginLoader()
        tools_dir = PROJECT_ROOT / "src" / "tools"
        count = _plugin_loader.discover_plugins(tools_dir)
        logger.info("Discovered %d plugins from %s", count, tools_dir)
    return _plugin_loader


@mcp.tool()
def lookup_model(role: str, session_id: str = "") -> str:
    """Look up model config for an orchestrator role.

    Args:
        role: Role name (e.g., "coder_escalation", "architect_general").

    Returns:
        Formatted string with role configuration details.
    """
    try:
        from src.registry_loader import RegistryLoader, format_tps

        registry = RegistryLoader(validate_paths=False)
        role_config = registry.get_role(role)

        speed = format_tps(role_config.performance)
        speedup = role_config.performance.speedup or "N/A"

        lines = [
            f"Role: {role_config.name}",
            f"Tier: {role_config.tier}",
            f"Description: {role_config.description}",
            f"Model: {role_config.model.name}",
            f"Quant: {role_config.model.quant}",
            f"Size: {role_config.model.size_gb} GB",
            f"Acceleration: {role_config.acceleration.type}",
            f"Speed: {speed}",
            f"Speedup: {speedup}",
        ]

        if role_config.constraints and role_config.constraints.forbid:
            lines.append(f"Forbidden: {', '.join(role_config.constraints.forbid)}")
        if role_config.notes:
            lines.append(f"Notes: {role_config.notes}")

        return "\n".join(lines)

    except KeyError:
        logger.debug("Role not found: %s", role)
        return f"Role not found: {role}"
    except Exception as e:
        logger.warning("Error loading role '%s': %s: %s", role, type(e).__name__, e)
        return f"Error loading role '{role}': {type(e).__name__}: {e}"


@mcp.tool()
def list_roles(session_id: str = "") -> str:
    """List all configured orchestrator roles by tier.

    Returns:
        Formatted string with all roles grouped by tier.
    """
    try:
        from src.registry_loader import RegistryLoader, format_tps

        registry = RegistryLoader(validate_paths=False)
        lines = []

        for tier in ["A", "B", "C", "D"]:
            roles = registry.get_roles_by_tier(tier)
            if roles:
                lines.append(f"\n--- Tier {tier} ---")
                for r in roles:
                    speed = format_tps(r.performance)
                    accel = r.acceleration.type
                    lines.append(f"  {r.name}: {r.model.name} ({accel}, {speed})")

        return "\n".join(lines) if lines else "No roles configured."

    except Exception as e:
        logger.warning("Error listing roles: %s: %s", type(e).__name__, e)
        return f"Error listing roles: {type(e).__name__}: {e}"


@mcp.tool()
def server_status(session_id: str = "") -> str:
    """Get current status of all orchestrator services.

    Returns:
        Formatted string with running service information.
    """
    state_file = PROJECT_ROOT / "logs" / "orchestrator_state.json"

    if not state_file.exists():
        return "No orchestrator state file found. Stack may not be running."

    try:
        state = json.loads(state_file.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Error reading state file %s: %s", state_file, e)
        return f"Error reading state file: {e}"

    if not state:
        return "No services running."

    lines = []
    for name, info in state.items():
        if isinstance(info, dict):
            pid = info.get("pid", "?")
            port = info.get("port", "?")
            started = info.get("started_at", "?")
            lines.append(f"{name}: PID {pid} on port {port} (started {started})")
        else:
            lines.append(f"{name}: {info}")

    return "\n".join(lines)


@mcp.tool()
def query_benchmarks(model_name: str = "", suite: str = "", session_id: str = "") -> str:
    """Query benchmark results from the summary CSV.

    Args:
        model_name: Filter by model name (case-insensitive substring match).
                    Empty string returns all models.
        suite: Filter by suite name (e.g., "thinking", "coder", "math").
               Empty string returns all suites.

    Returns:
        Formatted benchmark results.
    """
    csv_path = PROJECT_ROOT / "benchmarks" / "results" / "reviews" / "summary.csv"

    if not csv_path.exists():
        return "No benchmark summary found."

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except (OSError, csv.Error) as e:
        logger.warning("Error reading benchmark CSV %s: %s", csv_path, e)
        return f"Error reading benchmark CSV: {e}"

    # Filter
    if model_name:
        model_lower = model_name.lower()
        rows = [r for r in rows if model_lower in r.get("model", "").lower()]

    if not rows:
        return f"No results matching model='{model_name}'"

    # Format output
    lines = []
    for r in rows:
        model = r.get("model", "?")
        pct = r.get("pct_str", "?")
        tps = r.get("avg_tps", "?")

        if suite and suite in r:
            score = r.get(suite, "N/A")
            lines.append(f"{model}: {suite}={score}, overall={pct} ({tps} t/s)")
        else:
            lines.append(f"{model}: {pct} ({tps} t/s)")

    return "\n".join(lines)


# Canvas tools - integrated via plugin system
@mcp.tool()
def export_reasoning_canvas(
    graph_type: str = "hypothesis",
    include_evidence: bool = True,
    output_path: str = "",
    session_id: str = "",
) -> str:
    """Export a reasoning graph to JSON Canvas format.

    Args:
        graph_type: Type of graph: 'hypothesis', 'failure', or 'session'
        include_evidence: Whether to include evidence/symptom nodes
        output_path: Optional custom output path

    Returns:
        Path to the created canvas file, or error message.
    """
    try:
        from src.tools.canvas_tools import export_reasoning_canvas as _export

        return _export(
            graph_type=graph_type,  # type: ignore
            include_evidence=include_evidence,
            output_path=output_path or None,
        )
    except Exception as e:
        logger.exception("Canvas export failed: %s", e)
        return f"Error: {type(e).__name__}: {e}"


@mcp.tool()
def import_canvas_edits(canvas_path: str, baseline_path: str = "", session_id: str = "") -> str:
    """Import an edited canvas and extract planning constraints.

    Args:
        canvas_path: Path to the edited canvas file
        baseline_path: Optional path to baseline canvas for diff

    Returns:
        JSON-encoded constraints extracted from the canvas.
    """
    try:
        from src.tools.canvas_tools import import_canvas_edits as _import

        return _import(canvas_path, baseline_path or None)
    except Exception as e:
        logger.exception("Canvas import failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


@mcp.tool()
def list_canvases(directory: str = "", session_id: str = "") -> str:
    """List available canvas files.

    Args:
        directory: Directory to search. Defaults to logs/canvases.

    Returns:
        Formatted list of canvas files with metadata.
    """
    try:
        from src.tools.canvas_tools import list_canvases as _list

        return _list(directory or None)
    except Exception as e:
        logger.exception("Canvas list failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


# Plugin management tools
@mcp.tool()
def list_plugins(enabled_only: bool = True, session_id: str = "") -> str:
    """List all available tool plugins.

    Args:
        enabled_only: Only show enabled plugins

    Returns:
        JSON-formatted list of plugins and their tools.
    """
    try:
        loader = _get_plugin_loader()
        tools = loader.list_tools(enabled_only=enabled_only)

        # Group by plugin
        by_plugin: dict[str, list] = {}
        for tool in tools:
            plugin = tool.get("plugin", "unknown")
            by_plugin.setdefault(plugin, []).append(tool)

        result = {
            "status": "success",
            "plugin_count": len(by_plugin),
            "tool_count": len(tools),
            "plugins": {
                name: {
                    "tools": [t["name"] for t in tools_list],
                    "enabled": all(t.get("enabled", True) for t in tools_list),
                }
                for name, tools_list in by_plugin.items()
            },
        }
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception("List plugins failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


@mcp.tool()
def reload_plugins(session_id: str = "") -> str:
    """Reload any changed plugins (hot-reload).

    Returns:
        Status message with number of plugins reloaded.
    """
    try:
        loader = _get_plugin_loader()
        count = loader.reload_changed()
        return json.dumps(
            {
                "status": "success",
                "reloaded": count,
                "message": f"Reloaded {count} plugin(s)",
            }
        )
    except Exception as e:
        logger.exception("Reload plugins failed: %s", e)
        return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})


# ---------------------------------------------------------------------------
# CC Local Integration — chat delegation tools
# ---------------------------------------------------------------------------

_ORCHESTRATOR_API_URL = None
_MCP_PROGRESS_INTERVAL_S = 5
_MAX_MCP_CHAT_TIMEOUT_S = 120


def _get_api_url() -> str:
    """Get the orchestrator API base URL from environment."""
    global _ORCHESTRATOR_API_URL
    if _ORCHESTRATOR_API_URL is None:
        import os

        _ORCHESTRATOR_API_URL = os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8000")
    return _ORCHESTRATOR_API_URL


def _is_mcp_chat_enabled() -> bool:
    """Check if the claude_code_mcp_chat feature flag is enabled."""
    import os

    val = os.environ.get("ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT", "0").lower()
    return val in ("1", "true", "yes", "on")


def _post_chat(payload: dict) -> dict:
    """POST a JSON payload to the orchestrator /chat endpoint.

    Returns parsed JSON response dict, or a dict with 'error' key on failure.
    """
    import urllib.request
    import urllib.error

    url = f"{_get_api_url()}/chat"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    timeout_s = payload.get("timeout_s", 120)
    try:
        with urllib.request.urlopen(req, timeout=timeout_s + 5) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"error": f"Orchestrator not reachable at {url}: {exc.reason}"}
    except TimeoutError:
        return {"error": f"Request timed out after {timeout_s}s"}
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON response: {exc}"}


async def _post_chat_async(payload: dict) -> dict:
    """POST asynchronously so cancellation closes the request and its socket."""
    import httpx

    url = f"{_get_api_url()}/chat"
    timeout_s = payload.get("timeout_s", 120)
    try:
        async with httpx.AsyncClient(timeout=timeout_s + 5) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {timeout_s}s"}
    except httpx.HTTPStatusError as exc:
        return {"error": f"Orchestrator returned HTTP {exc.response.status_code}"}
    except httpx.RequestError as exc:
        return {"error": f"Orchestrator not reachable at {url}: {exc}"}
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid JSON response: {exc}"}


async def _await_chat_request(payload: dict, ctx: Context | None) -> dict:
    """Keep a single cancellable HTTP task and report progress while it runs."""
    request_task = asyncio.create_task(_post_chat_async(payload))
    started = time.monotonic()
    timeout_s = payload.get("timeout_s", 120)
    progress_total = timeout_s + 5
    try:
        while True:
            done, _ = await asyncio.wait(
                {request_task}, timeout=min(_MCP_PROGRESS_INTERVAL_S, timeout_s)
            )
            if done:
                return request_task.result()
            if ctx is not None:
                elapsed = min(time.monotonic() - started, progress_total)
                await ctx.report_progress(
                    elapsed, progress_total, "Orchestrator request is running"
                )
    finally:
        if not request_task.done():
            request_task.cancel()
            try:
                await request_task
            except asyncio.CancelledError:
                pass


def _format_chat_response(resp: dict) -> str:
    """Format a ChatResponse dict into a readable MCP tool result."""
    if "error" in resp:
        return f"Error: {resp['error']}"

    lines = []

    # Answer
    answer = resp.get("answer", "")
    if answer:
        lines.append(answer)

    # Error from orchestrator
    error_code = resp.get("error_code")
    if error_code:
        detail = resp.get("error_detail", "")
        lines.append(f"\n[Error {error_code}] {detail}")

    # Routing metadata
    routed_to = resp.get("routed_to", "")
    strategy = resp.get("routing_strategy", "")
    mode = resp.get("mode", "")
    elapsed = resp.get("elapsed_seconds")

    meta_parts = []
    if routed_to:
        meta_parts.append(f"role={routed_to}")
    if strategy:
        meta_parts.append(f"strategy={strategy}")
    if mode:
        meta_parts.append(f"mode={mode}")
    if elapsed is not None:
        meta_parts.append(f"elapsed={elapsed:.1f}s")

    if meta_parts:
        lines.append(f"\n[Routing: {', '.join(meta_parts)}]")

    return "\n".join(lines) if lines else "(empty response)"


@mcp.tool()
async def orchestrator_chat(
    prompt: str,
    context: str = "",
    force_role: str = "",
    force_mode: str = "",
    timeout_s: int = 120,
    session_id: str = "",
    ctx: Context | None = None,
) -> str:
    """Send a prompt to the local orchestrator for inference via the full routing pipeline.

    Routes through MemRL routing, factual risk scoring, mode selection, and
    delegation. Requires the orchestrator stack to be running.

    Args:
        prompt: The user's prompt or question.
        context: Additional context to include.
        force_role: Force a specific role (e.g., "architect_general", "frontdoor").
                    Empty string uses automatic routing.
        force_mode: Force a mode ("direct", "repl", "delegated").
                    Empty string uses automatic selection.
        timeout_s: Request timeout in seconds (default 120).
        session_id: Shell session identifier. Forwarded as ``ChatRequest.session_id``
                    when non-empty (the OpenCode plugin stamps it on every call).
        ctx: MCP request context used for progress notifications.

    Returns:
        The model's response with routing metadata.
    """
    if not _is_mcp_chat_enabled():
        return "CC Local integration disabled. Set ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT=1 to enable."

    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, int)
        or not 1 <= timeout_s <= _MAX_MCP_CHAT_TIMEOUT_S
    ):
        return f"Error: timeout_s must be an integer from 1 to {_MAX_MCP_CHAT_TIMEOUT_S} seconds"

    payload: dict = {
        "prompt": prompt,
        "real_mode": True,
        "mock_mode": False,
        "timeout_s": timeout_s,
        "client_deadline_unix_s": time.time() + timeout_s,
    }
    if context:
        payload["context"] = context
    if force_role:
        payload["force_role"] = force_role
    if force_mode:
        payload["force_mode"] = force_mode
    if session_id:
        payload["session_id"] = session_id

    if ctx is not None:
        progress_total = timeout_s + 5
        await ctx.report_progress(0, progress_total, "Submitting request to the orchestrator")
    try:
        async with asyncio.timeout(timeout_s + 5):
            resp = await _await_chat_request(payload, ctx)
    except TimeoutError:
        resp = {"error": f"Request timed out after {timeout_s}s"}
    if ctx is not None:
        await ctx.report_progress(progress_total, progress_total, "Orchestrator request finished")
    return _format_chat_response(resp)


@mcp.tool()
def orchestrator_route_explain(
    prompt: str,
    context: str = "",
    session_id: str = "",
) -> str:
    """Explain how the orchestrator would route a prompt WITHOUT running inference.

    Uses mock mode to execute the routing pipeline (MemRL, risk scoring,
    mode selection) and return the routing decision without generating a response.

    Args:
        prompt: The prompt to analyze.
        context: Additional context to include.
        session_id: Shell session identifier, forwarded as ``ChatRequest.session_id``
                    when non-empty.

    Returns:
        Routing analysis showing selected role, strategy, mode, and timeout.
    """
    if not _is_mcp_chat_enabled():
        return "CC Local integration disabled. Set ORCHESTRATOR_CLAUDE_CODE_MCP_CHAT=1 to enable."

    payload: dict = {
        "prompt": prompt,
        "real_mode": False,
        "mock_mode": True,
    }
    if context:
        payload["context"] = context
    if session_id:
        payload["session_id"] = session_id

    resp = _post_chat(payload)

    if "error" in resp:
        return f"Error: {resp['error']}"

    lines = ["Routing Analysis:"]
    routed_to = resp.get("routed_to", "unknown")
    lines.append(f"  Role: {routed_to}")
    lines.append(f"  Strategy: {resp.get('routing_strategy', 'unknown')}")
    lines.append(f"  Mode: {resp.get('mode', 'unknown')}")
    lines.append(f"  Tool required: {resp.get('tool_required', False)}")
    lines.append(f"  Timeout: {resp.get('timeout_s', '?')}s")

    delegation = resp.get("delegation_success")
    if delegation is not None:
        lines.append(f"  Delegation: {'success' if delegation else 'no'}")

    return "\n".join(lines)


if __name__ == "__main__":
    # Initialize plugin loader on startup
    _get_plugin_loader()
    mcp.run(transport="stdio")
