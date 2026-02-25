"""Canonical log format for 3-way eval terminal output.

ALL eval log formatting lives here. seed_specialist_routing.py calls
these functions and logs the returned lines. To change the format,
edit this module and update test_eval_log_format.py.

Design:
  - Pure functions: take data, return list[str] (no logger dependency)
  - Caller does `for line in format_X(...): logger.info(line)`
  - Tested directly without mocking the evaluation pipeline
"""

from __future__ import annotations

from collections import Counter
from typing import Any


def compute_tps(resp: dict[str, Any], elapsed: float = 0.0) -> float:
    """Compute tokens/second from response dict.

    When *elapsed* (wall-clock seconds) is provided and positive, uses
    ``tokens_generated / elapsed`` for accurate pipeline throughput.
    Falls back to ``generation_ms``-based calculation, then ``predicted_tps``.
    """
    tokens = int(resp.get("tokens_generated", 0) or 0)
    if elapsed > 0 and tokens > 0:
        return tokens / elapsed
    gen_ms = resp.get("generation_ms", 0.0)
    if gen_ms > 0 and tokens > 0:
        return tokens / (gen_ms / 1000.0)
    tps = resp.get("predicted_tps", 0.0)
    if tps and tps > 0:
        return tps
    return 0.0


def _tps_str(tps: float) -> str:
    return f", {tps:.1f} t/s" if tps > 0 else ""


def _token_str(resp: dict[str, Any], status: str) -> str:
    tokens = int(resp.get("tokens_generated", 0) or 0)
    est = int(resp.get("tokens_generated_estimate", 0) or 0)
    if status == "INFRA" and tokens == 0 and est > 0:
        return f"{tokens} tok, est {est} tok"
    return f"{tokens} tok"


def _status_str(passed: bool | None, error: str | None) -> str:
    if passed is None:
        return "INFRA"
    if passed:
        return "PASS"
    if error:
        return "ERROR"
    return "FAIL"


def _dedup_consecutive(items: list[str]) -> list[str]:
    """Deduplicate consecutive repeated items for readability."""
    deduped: list[str] = []
    for item in items:
        if not deduped or deduped[-1] != item:
            deduped.append(item)
    return deduped


# ── Per-configuration formatters ──────────────────────────────────────


def format_self_direct(
    action_key: str,
    passed: bool,
    error: str | None,
    elapsed: float,
    resp: dict[str, Any],
    infra: bool = False,
) -> list[str]:
    """Format SELF:direct result line.

    Output:
        SELF:direct → PASS (4.6s, 23.5 t/s, 85 tok)
    """
    status = _status_str(None if infra else passed, error)
    tps = compute_tps(resp, elapsed)
    token_info = _token_str(resp, status)
    return [
        f"    {action_key} → {status} ({elapsed:.1f}s{_tps_str(tps)}, {token_info})"
    ]


def format_self_repl(
    action_key: str,
    passed: bool,
    error: str | None,
    elapsed: float,
    resp: dict[str, Any],
    infra: bool = False,
) -> list[str]:
    """Format SELF:repl result with tools and per-tool timing.

    Output:
        SELF:repl → PASS (16.2s, 18.3 t/s, 240 tok, 3 tools)
          tools: peek, grep, FINAL
          peek: 120ms (ok)
          grep: 85ms (ok)
          FINAL: 10ms (ok)
    """
    status = _status_str(None if infra else passed, error)
    tps = compute_tps(resp, elapsed)
    token_info = _token_str(resp, status)
    tools_used = resp.get("tools_used", 0)
    tools_called = resp.get("tools_called", [])
    tool_timings = resp.get("tool_timings", [])

    lines = [
        f"    {action_key} → {status} ({elapsed:.1f}s{_tps_str(tps)}, {token_info}, {tools_used} tools)"
    ]
    if tools_used > 0 and tools_called:
        lines.append(f"      tools: {', '.join(_dedup_consecutive(tools_called))}")
    if tool_timings:
        lines.extend(_format_tool_timings(tool_timings))
    return lines


def format_architect_result(
    action_key: str,
    passed: bool | None,
    error: str | None,
    elapsed: float,
    resp: dict[str, Any],
) -> list[str]:
    """Format one ARCHITECT result with tools, delegation, and chain.

    Output:
        ARCHITECT → PASS (106.7s, 6.8 t/s, 724 tok)
          tools: peek, grep
          peek: 200ms (ok)
          grep: 150ms (ok)
          delegates: 2 (coder_escalation, worker_explore)
          delegate: coder_escalation → ok (42300ms, 18.3 t/s, 774 tok)
          delegate: worker_explore → ok (8200ms, 44.1 t/s, 362 tok)
          chain: architect_general → coder_escalation → worker_explore
    """
    status = _status_str(passed, error)
    tps = compute_tps(resp, elapsed)
    token_info = _token_str(resp, status)
    tools_used = resp.get("tools_used", 0)
    tools_called = resp.get("tools_called", [])
    tool_timings = resp.get("tool_timings", [])
    deleg_events = resp.get("delegation_events", [])
    role_history = resp.get("role_history", [])

    lines = [
        f"    {action_key} → {status} ({elapsed:.1f}s{_tps_str(tps)}, {token_info})"
    ]

    # Tool list
    if tools_used > 0 and tools_called:
        lines.append(f"      tools: {', '.join(_dedup_consecutive(tools_called))}")

    # Per-tool timing
    if tool_timings:
        lines.extend(_format_tool_timings(tool_timings))

    # Delegation events
    if deleg_events:
        lines.extend(_format_delegation_events(deleg_events))

    # Role chain
    if len(role_history) > 1:
        lines.append(f"      chain: {' → '.join(role_history)}")

    return lines


def format_reward_skip(action_key: str, reason: str = "INFRA_SKIP") -> list[str]:
    """Format a reward skip line.

    Output:
        SELF:direct -> INFRA_SKIP (not injecting reward)
    """
    return [f"    {action_key} -> {reason} (not injecting reward)"]


def format_all_infra_skip(action_key: str) -> list[str]:
    """Format when all architect results are infra errors.

    Output:
        ARCHITECT -> ALL INFRA_SKIP
    """
    return [f"    {action_key} -> ALL INFRA_SKIP"]


# ── Shared sub-formatters ─────────────────────────────────────────────


def _format_tool_timings(tool_timings: list[dict]) -> list[str]:
    """Format per-tool timing lines.

    Output:
        peek: 120ms (ok)
        grep: 85ms (fail)
    """
    lines = []
    for tt in tool_timings:
        name = tt.get("tool_name", "?")
        ms = tt.get("elapsed_ms", 0.0)
        ok = "ok" if tt.get("success", True) else "fail"
        lines.append(f"      {name}: {ms:.0f}ms ({ok})")
    return lines


def _format_delegation_events(events: list[dict]) -> list[str]:
    """Format delegation event lines with optional summary.

    Output (multiple delegates):
        delegates: 3 (2x worker_explore, coder_escalation)
        delegate: worker_explore → ok (8200ms, 44.1 t/s, 362 tok)
        delegate: worker_explore → ok (7800ms, 42.3 t/s, 340 tok)
        delegate: coder_escalation → ok (31500ms, 22.7 t/s, 715 tok)

    Output (single delegate):
        delegate: coder_escalation → ok (31500ms, 22.7 t/s, 715 tok)
    """
    lines: list[str] = []

    # Summary line for multiple delegates
    if len(events) > 1:
        role_counts = Counter(de.get("to_role", "?") for de in events)
        parts = []
        for role, count in role_counts.most_common():
            parts.append(f"{count}x {role}" if count > 1 else role)
        lines.append(f"      delegates: {len(events)} ({', '.join(parts)})")

    # Individual delegate lines
    for de in events:
        to_role = de.get("to_role", "?")
        ms = de.get("elapsed_ms", 0.0)
        tok = de.get("tokens_generated", 0)
        success = de.get("success")
        tps = (tok / (ms / 1000.0)) if ms > 0 and tok > 0 else 0.0
        ok = "ok" if success else ("fail" if success is False else "?")
        lines.append(
            f"      delegate: {to_role} → {ok} ({ms:.0f}ms{_tps_str(tps)}, {tok} tok)"
        )

    return lines
