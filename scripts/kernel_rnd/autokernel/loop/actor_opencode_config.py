"""Per-run opencode config for the autokernel planner/author seat.

The first 27B-on-GPU proposal made 64 serial tool calls, read whole files, dumped
unbounded perf output, overflowed its 98,304-token slot in 26 minutes, compacted, and
took ~39 minutes per proposal without ever using a subagent. This module writes the
config that fixes the shape of that session, not the model: bounded MCP tools
(`actor_tools_mcp`), a global cap on built-in tool output, a step cap, a tool-discipline
system prompt, and (as an A/B knob) fan-out through the `task` tool.

The caller passes the file as ``OPENCODE_CONFIG`` and selects the agent with
``opencode run --agent <AGENT_NAMES[role]>``. opencode (1.18.31) loads the global
``~/.config/opencode/opencode.jsonc`` FIRST and merges ``OPENCODE_CONFIG`` on top, so
the global providers (``qwen-gpu``), the bash deny-list and the filesystem-containment
plugin all survive. This file deliberately sets no top-level ``permission``, no
``plugin`` list and no per-agent ``bash`` rule: an agent's permission is appended AFTER
the global rules and the last match wins, so an agent-level ``bash: allow`` would
silently re-allow every globally denied verb.

Key names are verified against the opencode 1.18.31 binary's config schema
(``ConfigV1.Info`` / ``AgentConfig`` / ``McpLocalConfig`` / ``PermissionConfig``):

* ``tool_output.{max_lines,max_bytes}`` -- global truncation thresholds (defaults
  2000 lines / 51200 bytes); over either, the full text goes to disk and a preview
  is returned.
* ``agent.<name>.{description,mode,steps,prompt,permission}`` -- ``tools`` is
  deprecated in favour of ``permission`` (the decoder converts it), and ``maxSteps``
  in favour of ``steps``. A set ``prompt`` REPLACES opencode's provider system prompt
  rather than appending to it, so the prompt here is self-contained.
* ``mcp.<name>.{type,command,cwd,environment,enabled,timeout}`` -- ``cwd`` is
  supported (relative paths resolve from the workspace); ``environment`` is merged
  over the parent environment; ``timeout`` is per-request ms (default 5000).
* ``permission.task`` patterns match the ``subagent_type``; MCP tools are named
  ``<server>_<tool>`` with non ``[A-Za-z0-9_-]`` characters replaced by ``_``.
* ``compaction.prune`` -- prune old tool outputs (default false).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Sequence

#: role -> the agent name the caller passes as ``opencode run --agent``.
AGENT_NAMES = {"planner": "autokernel-planner", "author": "autokernel-author"}

#: The fan-out subagent (``task`` tool ``subagent_type``); defined only when fan_out.
SCOUT_AGENT = "autokernel-scout"

#: MCP server key; its tools surface as ``autokernel-tools_<tool>``.
MCP_SERVER = "autokernel-tools"
MCP_MODULE = "scripts.kernel_rnd.autokernel.loop.actor_tools_mcp"
MCP_TOOLS = ("read_range", "grep", "outline", "code_search", "profile_top",
             "symbol_annotate")

#: The GPU llama-server runs -np 2: more concurrent subagents than this just queue.
MAX_CONCURRENT_SUBAGENTS = 2

#: Per-request MCP timeout. The 5 s default is too short for profile_top on a large
#: perf.data; the tool server caps its own output, not its wall time.
MCP_TIMEOUT_MS = 120_000

#: <repo>/scripts/kernel_rnd/autokernel/loop/this_file.py -> <repo>
RESEARCH_ROOT = Path(__file__).resolve().parents[4]


def _tool(name: str) -> str:
    return f"{MCP_SERVER}_{name}"


_FINAL_JSON = ("6. Stop investigating as soon as you can answer. Print the requested single "
               "JSON object as the LAST thing in your reply, with nothing after it.")
_FINAL_SUMMARY = ("6. Stop as soon as you can answer. Reply with a summary of at most ~15 "
                  "lines with file:line references -- no file dumps.")


def _discipline(final: str = _FINAL_JSON) -> str:
    t = _tool
    return "\n".join([
        "Tool discipline -- your context is one ~98k-token slot and every byte a tool "
        "returns stays in it:",
        f"1. Locate before you read: {t('outline')} for a file's structure, "
        f"{t('grep')} or {t('code_search')} with a NARROW pattern, then "
        f"{t('read_range')} on only the lines you need.",
        "2. Never read more than ~200 lines at once and never read a whole file. If you "
        "use the built-in read tool, always pass offset and limit.",
        f"3. Profiles: use {t('profile_top')} and {t('symbol_annotate')}. Never run perf "
        "report/annotate/script or print a profile through bash.",
        "4. bash is for short, bounded commands only (e.g. `git log --oneline -n 20`, "
        "`git diff --stat`); bound every command's output.",
        "5. Do not re-read what you have already read; keep notes in your reasoning.",
        final,
    ])


def _fan_out() -> str:
    n = MAX_CONCURRENT_SUBAGENTS
    return "\n".join([
        "Parallel investigation -- keep your own context small:",
        f"- When questions are independent (e.g. one per candidate hotspot or file), "
        f"delegate them with the `task` tool, subagent_type \"{SCOUT_AGENT}\", issuing "
        f"up to {n} task calls in the SAME message so they run concurrently. The server "
        f"has {n} slots: never more than {n} subagents at once.",
        "- Give each scout one precise question and ask for a summary of at most ~15 "
        "lines with file:line references.",
        "- Work from the summaries; do not repeat a scout's reads yourself.",
    ])


def _role_intro(role: str) -> str:
    if role == "planner":
        return ("You are the autokernel PLANNER for a llama.cpp kernel lane worktree. "
                "You investigate the code and profiles and PROPOSE one change; you cannot "
                "edit files.")
    return ("You are the autokernel AUTHOR for a llama.cpp kernel lane worktree. You "
            "implement the requested change with minimal, targeted edits.")


def _scout_prompt() -> str:
    return "\n\n".join([
        "You are an autokernel SCOUT subagent. Answer the one question you were given "
        "by reading code and profiles; you cannot edit files and cannot start subagents.",
        _discipline(_FINAL_SUMMARY),
    ])


def build_actor_config(*, role: str, lane: Path, profiles: Sequence[Path] = (),
                       python: str = sys.executable,
                       research_root: Path = RESEARCH_ROOT, steps: int = 60,
                       tool_output_max_lines: int = 250,
                       tool_output_max_bytes: int = 12000,
                       fan_out: bool = True,
                       instructions_path: Path | None = None,
                       replace_system_prompt: bool = False) -> dict:
    """The opencode config (a dict ready for ``json.dump``) for one actor run.

    By default the seat's guidance is ADDED to opencode's own system prompt through
    the top-level ``instructions`` file list (``instructions_path``; its text is
    ``actor_instructions(role, fan_out)``). An agent ``prompt`` REPLACES that system
    prompt, and opencode's default is the part that keeps the model terse: with it
    replaced (v1, ``replace_system_prompt=True``), the 27B planner decoded a median
    1,626 tokens per step against 266 on the plain seat and filled its 98k slot in 12
    steps instead of ~45 (DS41 seat A/B, 2026-09-24)."""
    if role not in AGENT_NAMES:
        raise ValueError(f"unknown actor role {role!r}; expected one of "
                         f"{sorted(AGENT_NAMES)}")
    for label, value in (("steps", steps), ("tool_output_max_lines", tool_output_max_lines),
                         ("tool_output_max_bytes", tool_output_max_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive int, got {value!r}")

    root = Path(research_root)
    command = [str(python), "-m", MCP_MODULE, "--root", str(lane)]
    for profile in profiles:
        command += ["--profiles", str(profile)]

    mcp_tools = {f"{MCP_SERVER}_*": "allow"}
    permission: dict = {"edit": "deny" if role == "planner" else "allow", **mcp_tools}
    if fan_out:
        permission["task"] = {"*": "deny", SCOUT_AGENT: "allow"}
    else:
        permission["task"] = "deny"

    if not replace_system_prompt and instructions_path is None:
        raise ValueError("instructions_path is required unless replace_system_prompt=True")

    primary: dict = {
        "description": f"autokernel {role} seat (bounded tools, step-capped)",
        "mode": "primary",
        "steps": steps,
        "permission": permission,
    }
    if replace_system_prompt:
        primary["prompt"] = actor_instructions(role, fan_out, scout_note=False)
    agents = {AGENT_NAMES[role]: primary}
    if fan_out:
        scout: dict = {
            "description": ("Read-only investigator for ONE independent question about "
                            "the lane's code or profiles; returns a <=15-line summary."),
            "mode": "subagent",
            "hidden": True,
            "steps": max(1, min(20, steps)),
            "permission": {"edit": "deny", "task": "deny", **mcp_tools},
        }
        if replace_system_prompt:
            scout["prompt"] = _scout_prompt()
        agents[SCOUT_AGENT] = scout

    return {
        "$schema": "https://opencode.ai/config.json",
        "tool_output": {"max_lines": tool_output_max_lines,
                        "max_bytes": tool_output_max_bytes},
        "compaction": {"prune": True},
        "mcp": {
            MCP_SERVER: {
                "type": "local",
                "command": command,
                "cwd": str(root),
                # Merged over the parent env by opencode. Both roots: the module is
                # launched by its repo-root path, and loop code imports `autokernel.*`.
                "environment": {"PYTHONPATH": os.pathsep.join(
                    [str(root), str(root / "scripts" / "kernel_rnd")])},
                "enabled": True,
                "timeout": MCP_TIMEOUT_MS,
            },
        },
        "agent": agents,
        **({} if replace_system_prompt else {"instructions": [str(instructions_path)]}),
    }


def actor_instructions(role: str, fan_out: bool, *, scout_note: bool = True) -> str:
    """The seat's guidance text: role, tool discipline, fan-out. As an ``instructions``
    file it is loaded by every agent in the run, scouts included, hence the note."""
    sections = [_role_intro(role), _discipline()]
    if fan_out:
        sections.append(_fan_out())
        if scout_note:
            sections.append(f"If you are a {SCOUT_AGENT} subagent: ignore the role and fan-out "
                            "text above; answer only your one question, read-only, in a summary "
                            "of at most ~15 lines with file:line references -- no file dumps.")
    return "\n\n".join(sections)


def write_actor_config(path: Path, **kw) -> Path:
    """Build the config and write it to ``path`` atomically (temp file + rename)."""
    path = Path(path)
    if kw.get("role") not in AGENT_NAMES:
        raise ValueError(f"unknown actor role {kw.get('role')!r}; expected one of "
                         f"{sorted(AGENT_NAMES)}")
    if not kw.get("replace_system_prompt"):
        instructions = path.with_name(path.stem + ".instructions.md")
        kw.setdefault("instructions_path", instructions)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(Path(kw["instructions_path"]),
                      actor_instructions(kw["role"], kw.get("fan_out", True)) + "\n")
    config = build_actor_config(**kw)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return path


def _atomic_write(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


__all__ = ["actor_instructions", "AGENT_NAMES", "MAX_CONCURRENT_SUBAGENTS", "MCP_SERVER", "SCOUT_AGENT",
           "build_actor_config", "write_actor_config"]
