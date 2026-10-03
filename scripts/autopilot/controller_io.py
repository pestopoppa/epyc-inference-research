"""Controller invocation + action extraction + scope validation.

Extracted from autopilot.py during the 2026-05-22 Tranche-5 refactor. The
controller is the Claude CLI subprocess that proposes the next action;
this module handles invocation, JSON action extraction, and AP-9
single-variable scope validation.

2026-05-22 streaming overhaul:
- Switched `--output-format json` (single final JSON, fully buffered) to
  `stream-json` (line-delimited events emitted live). Caller can now tail
  the planner output as it streams instead of waiting up to 300s.
- Each line is teed to a per-call planner tap file at
  PLANNER_TAP_PATH so the dashboard can SSE-stream it. The tap file is
  appended across calls (with section separators) so the recent planning
  history survives across trials.

`autopilot.py` keeps the public function names as thin re-imports.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from src.structured_output.repair import (
    CompleteFn,
    RepairResult,
    http_chat_completer,
    parse_with_repair,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

log = logging.getLogger("autopilot")

# Tap file the planner subprocess streams into. Dashboard tails this via
# /dashboard/events/planner_tap. The path is fixed (not per-invocation) so
# a single SSE consumer can watch every planner session.
PLANNER_TAP_PATH = Path("/mnt/raid0/llm/tmp/planner_tap.log")

# Persistent JSONL archive of planner sessions — survives /tmp wipes and
# is queryable for reasoning-trace research. One line per session with
# the full event list. Lives alongside other autopilot logs so it's
# included in the same backup/retention scheme.
PLANNER_ARCHIVE_PATH = _REPO_ROOT / "logs/planner_archive.jsonl"
PLANNER_SUBPROCESS_STATUS_PATH = Path("/mnt/raid0/llm/tmp/autopilot_planner_subprocess.json")

# Do not inherit the operator's last interactive Claude model. Fable access is
# metered/temporary, and a stale global default can brick AutoPilot planning.
DEFAULT_CLAUDE_MODEL = "opus"
DEFAULT_CLAUDE_FALLBACK_MODEL = "sonnet"
PLANNER_ALLOWED_TOOLS = {"Read", "Grep", "Glob"}
PLANNER_DISALLOWED_TOOLS = {
    "Bash",
    "CronCreate",
    "CronDelete",
    "CronList",
    "DesignSync",
    "Edit",
    "EnterWorktree",
    "ExitPlanMode",
    "MultiEdit",
    "NotebookEdit",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
}
PLANNER_CLI_DISALLOWED_TOOLS = PLANNER_DISALLOWED_TOOLS - {
    # Current Claude Code rejects this legacy alias in --disallowedTools, but
    # keep it in PLANNER_DISALLOWED_TOOLS so stream-level detection still fails
    # closed if an older planner event emits it.
    "MultiEdit",
}

_FALLBACK_NUMERIC_SURFACES = {"memrl_retrieval", "think_harder", "monitor", "escalation"}


def _configured_numeric_surfaces() -> set[str]:
    try:
        from species.numeric_swarm import SURFACES as _NS_SURFACES
    except Exception:
        return set(_FALLBACK_NUMERIC_SURFACES)

    surfaces = {surface for surface in _NS_SURFACES if isinstance(surface, str) and surface.strip()}
    return surfaces or set(_FALLBACK_NUMERIC_SURFACES)


_RAW_NUMERIC_SURFACES = _configured_numeric_surfaces()
_SUPPRESSED_NUMERIC_SURFACES: set[str] = set()
_NUMERIC_SURFACES = set(_RAW_NUMERIC_SURFACES)
_PROMPT_MUTATIONS = {"targeted_fix", "compress", "few_shot_evolution"}
_CODE_MUTATIONS = {"targeted_fix", "new_file"}
_SLOT_SCORERS = {"expected_attention", "knorm"}
DEEP_EVAL_TIERS = (0, 1, 2, 3)


def set_suppressed_numeric_surfaces(surfaces: set[str] | list[str] | tuple[str, ...]) -> None:
    """Update startup-scoped numeric surfaces hidden from planner validation."""
    global _NUMERIC_SURFACES
    _SUPPRESSED_NUMERIC_SURFACES.clear()
    _SUPPRESSED_NUMERIC_SURFACES.update(
        str(surface).strip()
        for surface in surfaces
        if str(surface).strip() in _RAW_NUMERIC_SURFACES
    )
    _NUMERIC_SURFACES = set(_RAW_NUMERIC_SURFACES) - _SUPPRESSED_NUMERIC_SURFACES
    _ACTION_SCHEMAS["numeric_trial"]["enums"]["surface"] = _NUMERIC_SURFACES


def suppressed_numeric_surfaces() -> set[str]:
    """Return a snapshot of numeric surfaces withheld from live dispatch."""
    return set(_SUPPRESSED_NUMERIC_SURFACES)


def _write_planner_subprocess_status(
    *,
    status: str,
    prompt: str,
    cmd: list[str],
    child_pid: int | None,
    started_at: float,
    returncode: int | None = None,
    error: str = "",
) -> None:
    """Best-effort heartbeat for planner subprocess lifetime diagnostics."""
    payload = {
        "status": status,
        "provider": "claude",
        "parent_pid": os.getpid(),
        "child_pid": child_pid,
        "started_at": started_at,
        "updated_at": time.time(),
        "duration_s": max(0.0, time.time() - started_at),
        "prompt_chars": len(prompt),
        "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "cmd": [cmd[0], *["<prompt>" if item == prompt else item for item in cmd[1:]]],
        "returncode": returncode,
        "error": error[:1000],
    }
    try:
        PLANNER_SUBPROCESS_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLANNER_SUBPROCESS_STATUS_PATH.write_text(
            json.dumps(payload, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception:
        log.debug("planner subprocess heartbeat write failed", exc_info=True)


_ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "seed_batch": {
        "allowed": {"type", "n_questions", "suites"},
    },
    "numeric_trial": {
        "allowed": {"type", "surface", "params"},
        "enums": {"surface": _NUMERIC_SURFACES},
    },
    "prompt_mutation": {
        "allowed": {"type", "file", "mutation", "description"},
        "required": {"file"},
        "enums": {"mutation": _PROMPT_MUTATIONS},
    },
    "gepa_optimize": {
        "allowed": {"type", "file", "max_evals", "description"},
        "required": {"file"},
    },
    "code_mutation": {
        "allowed": {"type", "file", "mutation", "description"},
        "required": {"file"},
        "enums": {"mutation": _CODE_MUTATIONS},
    },
    "structural_experiment": {
        "allowed": {"type", "flags"},
        "required": {"flags"},
    },
    "consult_gate_probe": {
        "allowed": {"type", "task_suite", "turns", "tier"},
        "enums": {"task_suite": {"targeted", "bep"}},
    },
    "structural_prune": {
        "allowed": {"type", "file", "block", "description"},
        "required": {"file", "block"},
    },
    "slot_compact": {
        "allowed": {
            "type",
            "port",
            "slot_id",
            "keep_ratio",
            "scorer",
            "keep_first",
            "n_future",
            "use_covariance",
            "layer_weights",
            "threshold",
        },
        "enums": {"scorer": _SLOT_SCORERS},
    },
    "train_routing_models": {
        "allowed": {"type", "min_memories"},
    },
    "distill_skillbank": {
        "allowed": {"type", "teacher", "categories"},
    },
    "reset_memories": {
        "allowed": {"type", "keep_seen", "keep_skills"},
    },
    "deep_eval": {
        "allowed": {"type", "tier"},
        "required": {"tier"},
        "enums": {"tier": set(DEEP_EVAL_TIERS)},
    },
    "rollback": {
        "allowed": {"type", "to_checkpoint"},
        "enums": {"to_checkpoint": {"production_best"}},
    },
    "distill_knowledge": {
        "allowed": {"type", "last_n"},
    },
}


def _open_planner_tap() -> Any:
    """Return an append-mode handle on the planner tap, creating dirs if needed."""
    try:
        PLANNER_TAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        return open(PLANNER_TAP_PATH, "a", buffering=1)  # line-buffered
    except Exception as exc:
        log.warning("Could not open planner tap %s: %s", PLANNER_TAP_PATH, exc)
        return None


def _append_planner_archive(record: dict) -> None:
    """Append one planner-session record to the persistent JSONL archive.

    Best-effort: silent on failure (the tap file is the live source of
    truth; archive is for after-the-fact analysis). One JSONL line per
    session with timestamp, duration, session_id, prompt hash + length,
    captured events, and final result.
    """
    try:
        PLANNER_ARCHIVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PLANNER_ARCHIVE_PATH, "a") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    except Exception as exc:
        log.debug("planner archive append failed: %s", exc)


def _summarize_event(line: str) -> str:
    """Produce a human-readable one-line summary of a stream-json event.

    Stream-json events look like:
        {"type":"system","subtype":"init", ...}
        {"type":"assistant","message":{"content":[{"type":"text","text":"..."}]}}
        {"type":"assistant","message":{"content":[{"type":"tool_use","name":"Read",...}]}}
        {"type":"user","message":{"content":[{"type":"tool_result","content":"..."}]}}
        {"type":"result","subtype":"success","total_cost_usd":...,"result":"..."}

    We summarize so the tap file is readable even at a glance.
    """
    try:
        evt = json.loads(line)
    except json.JSONDecodeError:
        return line.rstrip()

    t = evt.get("type", "?")
    if t == "system":
        sub = evt.get("subtype", "")
        # For the init event surface model + tools + cwd so the operator
        # sees the planner's environment at a glance.
        sid = evt.get("session_id", "")[:8]
        extras = []
        if evt.get("model"):
            extras.append(f"model={evt['model']}")
        tools = evt.get("tools") or []
        if tools:
            extras.append(f"tools={','.join(tools[:8])}{'…' if len(tools) > 8 else ''}")
        if evt.get("cwd"):
            extras.append(f"cwd={evt['cwd']}")
        suffix = (" " + " ".join(extras)) if extras else ""
        return f"[system:{sub}] session={sid}…{suffix}"
    if t == "assistant":
        msg = evt.get("message", {})
        parts = []
        for c in msg.get("content", []):
            ctype = c.get("type")
            if ctype == "text":
                txt = c.get("text", "").strip()
                if txt:
                    # 1200c (was 300c) — assistant text is the planner's
                    # reasoning the operator wants to read mid-stream.
                    parts.append(txt[:1200])
            elif ctype == "tool_use":
                name = c.get("name", "?")
                inp = c.get("input", {})
                # 800c (was 160c) — args include file paths / search patterns
                # / shell commands that show WHAT the planner is doing.
                arg_preview = json.dumps(inp)[:800]
                parts.append(f"TOOL_USE {name}({arg_preview})")
            elif ctype == "thinking":
                # Extended-thinking content (when the planner model emits it)
                think = c.get("thinking", "").strip()
                if think:
                    parts.append(f"THINKING: {think[:800]}")
        return "[assistant] " + " | ".join(parts) if parts else "[assistant] (empty)"
    if t == "user":
        msg = evt.get("message", {})
        for c in msg.get("content", []):
            if c.get("type") == "tool_result":
                content = c.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        (b.get("text", "") if isinstance(b, dict) else str(b)) for b in content
                    )
                is_error = c.get("is_error", False)
                # 1500c (was 240c) — tool results are the planner's evidence;
                # truncating them hides the inputs to its next decision.
                # Newlines preserved — the tap renderer wraps the content.
                preview = str(content)[:1500]
                tag = "[tool_error] " if is_error else "[tool_result] "
                return tag + preview
        return "[user] (no tool_result)"
    if t == "result":
        sub = evt.get("subtype", "")
        cost = evt.get("total_cost_usd")
        dur = evt.get("duration_ms")
        turns = evt.get("num_turns")
        usage = evt.get("usage") or {}
        in_tok = usage.get("input_tokens")
        out_tok = usage.get("output_tokens")
        parts = [f"[result:{sub}]"]
        if isinstance(cost, (int, float)):
            parts.append(f"cost=${cost:.4f}")
        if isinstance(dur, (int, float)):
            parts.append(f"duration={dur}ms")
        if isinstance(turns, (int, float)):
            parts.append(f"turns={turns}")
        if isinstance(in_tok, (int, float)) and isinstance(out_tok, (int, float)):
            parts.append(f"tokens={in_tok}in/{out_tok}out")
        return " ".join(parts)
    return f"[{t}] {line.rstrip()[:200]}"


def invoke_controller(
    prompt: str,
    session_id: str | None = None,
    timeout: int = 300,
    *,
    cwd: Path | str | None = None,
) -> tuple[str, str | None]:
    """Invoke Claude CLI for meta-reasoning with live streaming to planner tap.

    Returns (response_text, session_id). `cwd` is the working directory
    Claude runs in; defaults to current process cwd if not provided.

    Streams each event to PLANNER_TAP_PATH as it's emitted so the dashboard
    can watch the planner reason in real time.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--output-format",
        "stream-json",
        "--verbose",  # required by claude CLI for stream-json output
        # This is a JSON controller call, not Claude Code's interactive Plan
        # Mode. Plan mode can steer the CLI toward writing ~/.claude/plans/*
        # when it wants to escalate, which wastes a planner turn and returns an
        # empty action. Keep the available tool surface explicitly read-only.
        "--permission-mode",
        "default",
        "--safe-mode",
        "--tools",
        "Read,Grep,Glob",
        "--allowedTools",
        "Read,Grep,Glob",
        "--disallowedTools",
        ",".join(sorted(PLANNER_CLI_DISALLOWED_TOOLS)),
    ]
    planner_model = os.environ.get("AUTOPILOT_CLAUDE_MODEL", DEFAULT_CLAUDE_MODEL).strip()
    if planner_model:
        cmd.extend(["--model", planner_model])
    fallback_model = os.environ.get(
        "AUTOPILOT_CLAUDE_FALLBACK_MODEL",
        DEFAULT_CLAUDE_FALLBACK_MODEL,
    ).strip()
    if fallback_model:
        cmd.extend(["--fallback-model", fallback_model])
    if session_id:
        cmd.extend(["--resume", session_id])

    tap = _open_planner_tap()
    if tap is not None:
        try:
            tap.write(
                f"\n{'=' * 72}\n[{datetime.now().isoformat(timespec='seconds')}] PLANNER session start\n"
            )
            if session_id:
                tap.write(f"resume_session: {session_id}\n")
            tap.write(f"prompt_chars: {len(prompt)}\n")
            tap.write(f"{'-' * 72}\n")
            tap.flush()
        except Exception:
            pass

    result_text = ""
    final_session_id = session_id
    proc: subprocess.Popen | None = None
    reader_thread: threading.Thread | None = None
    # Captured events for the archive write at end. Each entry is the
    # one-line summary; full raw JSON would bloat the JSONL too much for
    # routine grep, and the user can always reconstruct via the live tap.
    archive_events: list[str] = []
    archive_meta: dict[str, Any] = {}
    disallowed_tool_uses: list[str] = []
    session_start_ts = time.time()

    def _archive_controller_call(
        *,
        status: str,
        ok: bool,
        error: str = "",
    ) -> None:
        import hashlib

        _append_planner_archive(
            {
                "ts": session_start_ts,
                "ts_iso": datetime.fromtimestamp(session_start_ts).isoformat(timespec="seconds"),
                "type": "planner_provider_call",
                "provider": "claude",
                "role": "draft",
                "status": status,
                "ok": ok,
                "error": error,
                "duration_s": time.time() - session_start_ts,
                "session_id": final_session_id,
                "resume_session_id": session_id,
                "prompt_chars": len(prompt),
                "prompt_sha256_16": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "result_chars": len(result_text),
                "result_preview": (result_text or "")[:500],
                "n_events": len(archive_events),
                "events": archive_events[-200:],
                **archive_meta,
            }
        )

    def _drain_stdout(p: subprocess.Popen):
        """Read p.stdout line-by-line; tee each line to tap; capture result."""
        nonlocal result_text, final_session_id
        assert p.stdout is not None
        try:
            for raw_line in p.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                # Tee summarized + raw to tap, and also remember the
                # summary for the archive write.
                summary = _summarize_event(line)
                archive_events.append(summary)
                if tap is not None:
                    try:
                        tap.write(summary + "\n")
                        tap.flush()
                    except Exception:
                        pass
                # Capture the final result + session_id from the result event
                try:
                    evt = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if evt.get("type") == "result":
                    result_text = evt.get("result", "") or result_text
                    final_session_id = evt.get("session_id", final_session_id)
                    archive_meta["total_cost_usd"] = evt.get("total_cost_usd")
                    archive_meta["duration_ms"] = evt.get("duration_ms")
                    archive_meta["subtype"] = evt.get("subtype")
                elif evt.get("type") == "system" and evt.get("subtype") == "init":
                    final_session_id = evt.get("session_id", final_session_id)
                elif evt.get("type") == "assistant":
                    msg = evt.get("message") or {}
                    for content in msg.get("content", []):
                        if not isinstance(content, dict) or content.get("type") != "tool_use":
                            continue
                        name = str(content.get("name") or "")
                        if name and name not in PLANNER_ALLOWED_TOOLS:
                            disallowed_tool_uses.append(name)
        except Exception as exc:
            log.warning("Planner stdout drain failed: %s", exc)

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
            cwd=str(cwd) if cwd else None,
            env={k: v for k, v in os.environ.items() if k != "CLAUDECODE"},
        )
        _write_planner_subprocess_status(
            status="running",
            prompt=prompt,
            cmd=cmd,
            child_pid=proc.pid,
            started_at=session_start_ts,
        )

        reader_thread = threading.Thread(target=_drain_stdout, args=(proc,), daemon=True)
        reader_thread.start()

        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            log.error("Controller timed out after %ds", timeout)
            _write_planner_subprocess_status(
                status="timeout",
                prompt=prompt,
                cmd=cmd,
                child_pid=proc.pid,
                started_at=session_start_ts,
                returncode=proc.returncode,
                error=f"timeout after {timeout}s",
            )
            if tap is not None:
                try:
                    tap.write(f"[TIMEOUT after {timeout}s]\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            _archive_controller_call(
                status="timeout",
                ok=False,
                error=f"timeout after {timeout}s",
            )
            return "", session_id

        reader_thread.join(timeout=5)

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            log.error("Controller failed (rc=%d): %s", proc.returncode, stderr[:500])
            _write_planner_subprocess_status(
                status="process_failed",
                prompt=prompt,
                cmd=cmd,
                child_pid=proc.pid,
                started_at=session_start_ts,
                returncode=proc.returncode,
                error=stderr,
            )
            if tap is not None:
                try:
                    tap.write(f"[FAIL rc={proc.returncode}] {stderr[:400]}\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            # 2026-05-23: detect stale --resume target. claude CLI emits
            # various stderr patterns when the resumed session has been
            # pruned / wasn't persisted; returning the same stale
            # session_id back to the caller causes it to save it again
            # into autopilot state, repeating the failure every trial.
            # Clear it so the next call starts fresh. Broadened pattern
            # set to catch CLI wording drift across versions.
            _stderr_low = (stderr or "").lower()
            _stale_session_phrases = (
                "no conversation found",
                "session expired",
                "session not found",
                "conversation not found",
                "unknown session",
                "invalid session",
                "could not resume",
                "session has been deleted",
            )
            _stale_id_combo = "session id" in _stderr_low and (
                "not found" in _stderr_low
                or "expired" in _stderr_low
                or "invalid" in _stderr_low
                or "no such" in _stderr_low
            )
            if _stale_id_combo or any(p in _stderr_low for p in _stale_session_phrases):
                log.warning(
                    "Clearing stale planner session_id=%s (CLI reports it no longer "
                    "exists / expired); next trial will start a fresh conversation",
                    (session_id or "")[:12],
                )
                _archive_controller_call(
                    status="stale_session",
                    ok=False,
                    error=stderr[:1000],
                )
                return "", None
            _archive_controller_call(
                status="process_failed",
                ok=False,
                error=stderr[:1000],
            )
            return "", session_id

        if disallowed_tool_uses:
            tools = sorted(set(disallowed_tool_uses))
            error = "planner used disallowed tool(s): " + ", ".join(tools)
            log.error(error)
            _write_planner_subprocess_status(
                status="disallowed_tool_use",
                prompt=prompt,
                cmd=cmd,
                child_pid=proc.pid,
                started_at=session_start_ts,
                returncode=proc.returncode,
                error=error,
            )
            if tap is not None:
                try:
                    tap.write(f"[FAIL disallowed_tool_use] {error}\n{'=' * 72}\n")
                    tap.flush()
                except Exception:
                    pass
            _archive_controller_call(
                status="disallowed_tool_use",
                ok=False,
                error=error,
            )
            return "", None

        if tap is not None:
            try:
                tap.write(
                    f"[END] result_chars={len(result_text)} session={(final_session_id or '')[:8]}…\n{'=' * 72}\n"
                )
                tap.flush()
            except Exception:
                pass

        # Archive write (persistent JSONL, survives /tmp wipe)
        _archive_controller_call(status="success", ok=True)
        _write_planner_subprocess_status(
            status="success",
            prompt=prompt,
            cmd=cmd,
            child_pid=proc.pid,
            started_at=session_start_ts,
            returncode=proc.returncode,
        )

        return result_text, final_session_id

    except FileNotFoundError:
        log.error("Claude CLI not found")
        _write_planner_subprocess_status(
            status="missing_cli",
            prompt=prompt,
            cmd=cmd,
            child_pid=None,
            started_at=session_start_ts,
            error="Claude CLI not found",
        )
        _archive_controller_call(
            status="missing_cli",
            ok=False,
            error="Claude CLI not found",
        )
        return "", session_id
    finally:
        if tap is not None:
            try:
                tap.close()
            except Exception:
                pass


def _unwrap_action(data: Any) -> dict[str, Any] | None:
    """Unwrap action from list or validate it's a dict with a 'type' field."""
    if isinstance(data, list) and len(data) > 0:
        data = data[0]
    if isinstance(data, dict) and "type" in data:
        return data
    return None


def _loads_json_payload(payload: str) -> Any:
    """Load a JSON payload, tolerating only trivial trailing bracket noise.

    Local planner models sometimes emit a valid fenced object followed by a
    duplicate closing brace. Recover that narrow case so a usable action still
    reaches schema validation, but do not accept arbitrary trailing prose.
    """
    stripped = payload.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        try:
            data, end = decoder.raw_decode(stripped)
        except json.JSONDecodeError:
            raise original_error
        trailing = stripped[end:].strip()
        if trailing and set(trailing) <= {"}", "]", ")"}:
            log.warning(
                "Recovered planner JSON payload with trailing bracket noise: %r",
                trailing[:80],
            )
            return data
        raise original_error


def _loads_leading_action(text: str) -> dict[str, Any] | None:
    """Recover a strict leading JSON action with optional rationale sidecar.

    Local planner models sometimes obey the JSON object shape but miss the
    fenced ``json:autopilot_actions`` wrapper. Accept only an object/list at the
    very start of the response, and only if the trailing text is empty or the
    rationale fence. Arbitrary prose after the object remains unusable.
    """
    stripped = text.strip()
    if not stripped or stripped[0] not in "{[":
        return None
    decoder = json.JSONDecoder()
    try:
        data, end = decoder.raw_decode(stripped)
    except json.JSONDecodeError:
        return None
    action = _unwrap_action(data)
    if action is None:
        return None
    trailing = stripped[end:].strip()
    if not trailing or trailing.startswith("```json:autopilot_rationale"):
        log.warning("Recovered planner action from leading JSON object without action fence")
        return action
    return None


def extract_action(text: str) -> dict[str, Any] | None:
    """Extract structured action from controller response.

    Looks for ```json:autopilot_actions``` block first; falls back to any
    ```json``` block whose payload is a dict with a 'type' field. As a narrow
    local-planner recovery path, also accepts a response that starts with a JSON
    action object and is followed only by an optional rationale fence.
    """
    marker = "```json:autopilot_actions"
    if marker in text:
        start = text.index(marker) + len(marker)
        try:
            end = text.index("```", start)
        except ValueError:
            # A truncated generation opened the fence but never closed it --
            # not a JSONDecodeError (there is no JSON to even attempt), but
            # the same "nothing usable here" outcome every other malformed
            # shape in this function returns None for, not a crash. Found
            # live 2026-09-24 alongside the require_evidence fix: a
            # truncated ACTION block is exactly the shape that motivated
            # both.
            log.warning("autopilot_actions block has no closing fence")
            return None
        try:
            data = _loads_json_payload(text[start:end])
            return _unwrap_action(data)
        except json.JSONDecodeError as e:
            log.error("Failed to parse action JSON: %s", e)
            return None

    # Fallback: look for any JSON block
    if "```json" in text:
        start = text.index("```json") + len("```json")
        try:
            end = text.index("```", start)
        except ValueError:
            end = None
        if end is not None:
            try:
                data = _loads_json_payload(text[start:end])
                if isinstance(data, dict) and "type" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

    leading = _loads_leading_action(text)
    if leading is not None:
        return leading

    return None


def extract_rationale(text: str) -> dict[str, Any]:
    """Extract the optional rationale sidecar from the controller response.

    Looks for a ```json:autopilot_rationale``` fenced block. The block carries
    the chosen action's falsifier + self-scored rubric, e.g.

        ```json:autopilot_rationale
        {"falsifier": "...", "rubric_scores": {"info_gain": 4,
         "coherence": 5, "usefulness": 3, "synthesis_note": "..."}}
        ```

    Returns `falsifier` (str), `rubric_scores` (dict), and an optional
    `vidya_claim_ids` list when explicitly supplied. The first two default to
    empty when the block is missing or malformed. The
    contract is intentionally soft: rationale capture is observability, not a
    gate, so a missing block must not abort the trial.
    """
    empty: dict[str, Any] = {"falsifier": "", "rubric_scores": {}}
    marker = "```json:autopilot_rationale"
    if marker not in text:
        return empty
    start = text.index(marker) + len(marker)
    try:
        end = text.index("```", start)
    except ValueError:
        log.warning("autopilot_rationale block has no closing fence")
        return empty
    try:
        data = json.loads(text[start:end].strip())
    except json.JSONDecodeError as e:
        log.warning("Failed to parse autopilot_rationale JSON: %s", e)
        return empty
    if not isinstance(data, dict):
        return empty
    falsifier = data.get("falsifier", "")
    rubric = data.get("rubric_scores", {})
    if not isinstance(falsifier, str):
        falsifier = str(falsifier)
    if not isinstance(rubric, dict):
        rubric = {}
    result = {"falsifier": falsifier, "rubric_scores": rubric}
    claim_ids = data.get("vidya_claim_ids")
    if isinstance(claim_ids, list) and all(isinstance(cid, str) for cid in claim_ids):
        result["vidya_claim_ids"] = claim_ids
    return result


# --------------------------------------------------------------------------- TD-21.2/.3 repair
#
# Fish first (the functions above, unchanged); on a miss, ONE constrained
# completion turn back to a llama-server directly, via the shared
# `src.structured_output.repair` helper.
#
# NOT the orchestrator's own :8000 `/v1/chat/completions` -- that seam
# REFUSES `response_format` by design (HS-OD-1,
# `src/api/models/openai.py:_UNHONOURED_SEMANTIC_FIELDS["response_format"]`
# -> 422 "JSON mode is not implemented on this seam"; a first draft of this
# module pointed here and every repair attempt would have 422'd, landing as
# a typed "failed" with zero benefit -- caught in review, 2026-09-24). A
# llama-server's OWN `/v1` DOES honour `response_format json_schema`
# (verified live, e.g. the AutoKernel reference on :8083 and :8070 today),
# so the repair turn targets one directly, resolved via the orchestrator's
# own role->server config (`_resolve_role_server_base_url`, same source
# `src/api/routes/health.py:_first_backend_url` and
# `src/llm_primitives/backend.py:_normalise_role_urls` already read) --
# never a hardcoded port. The repair turn is available regardless of which
# provider (Claude CLI, Codex CLI, or a local HTTP provider) produced the
# unparseable draft: its only job is to re-express a reply that already
# exists, not to reproduce the draft provider's reasoning.
# `planner_coordinator.py` is expected to inject a fake/raising completer in
# tests so no test ever reaches a real server.

_ACTION_REPAIR_URL_ENV = "AUTOPILOT_LOCAL_PLANNER_URL"
_ACTION_REPAIR_ROLE_ENV = "AUTOPILOT_LOCAL_PLANNER_ROLE"
_ACTION_REPAIR_DEFAULT_ROLE = "frontdoor"

_ACTION_REPAIR_SITE = "autopilot.controller_io.extract_action"
_RATIONALE_REPAIR_SITE = "autopilot.controller_io.extract_rationale"

_ACTION_REPAIR_INSTRUCTION = (
    "The reply given as the user message is an AutoPilot controller response "
    "that failed to parse as its intended fenced next-action JSON object. "
    "Extract ONLY the action the reply intends to take next -- ignore any "
    "rationale, falsifier, or rubric commentary elsewhere in the reply -- as "
    "exactly one JSON object matching the given schema. Copy the reply's own "
    "field values faithfully; do not invent an action it did not already "
    "propose."
)

_RATIONALE_REPAIR_INSTRUCTION = (
    "The reply given as the user message is the payload of an AutoPilot "
    "controller's rationale sidecar that failed to parse. Convert it into "
    "exactly one JSON object with `falsifier` (string) and `rubric_scores` "
    "(object) matching the given schema, copying the reply's own wording and "
    "values faithfully -- do not invent a falsifier or scores it did not "
    "already give."
)


def _get_orchestrator_config() -> Any:
    """Indirection point so tests can substitute a fake config without
    touching the real (`lru_cache`d) `src.config.get_config()`. Deferred
    import (as `planner_coordinator.py` did for its chat review thresholds before
    RI-18c) so this module's import does not pull
    in the full orchestrator config stack eagerly."""
    from src.config import get_config

    return get_config()


def _resolve_role_server_base_url(role: str) -> str:
    """Resolve `role` to a llama-server base URL via `get_config().server_urls`
    -- the SAME role->server resolution `src/api/routes/health.py`
    (`_first_backend_url`) and `src/llm_primitives/backend.py`
    (`_normalise_role_urls`) already read -- never a hardcoded port. A role's
    configured value may be a `full:`-prefixed, comma-separated
    multi-instance string (`src/config/models.py:_stack_prior_server_urls`);
    this takes the FIRST concrete endpoint, exactly like
    `health.py:_first_backend_url`, and appends `/v1` since llama-server's
    OpenAI-compatible route lives at `/v1/chat/completions`
    (`src/backends/llama_server.py` calls it the same way) -- the bare
    host:port `get_config()` returns is not itself the base
    `http_chat_completer` needs. Falls back to `_ACTION_REPAIR_DEFAULT_ROLE`
    if `role` is not a configured key; raises if neither resolves (caught by
    `action_repair_completer`, never by this function's caller directly)."""
    urls = _get_orchestrator_config().server_urls.as_dict()
    raw = urls.get(role) or urls.get(_ACTION_REPAIR_DEFAULT_ROLE)
    if not raw:
        raise RuntimeError(
            f"no server URL resolved for role {role!r} or fallback role "
            f"{_ACTION_REPAIR_DEFAULT_ROLE!r} in ServerURLsConfig"
        )
    if raw.startswith("full:"):
        raw = raw[len("full:") :]
    base = raw.split(",")[0].rstrip("/")
    return f"{base}/v1"


def _local_planner_base_url() -> str:
    """Base URL for the TD-21.2/.3 repair completer.

    `AUTOPILOT_LOCAL_PLANNER_URL`, when set, is an explicit full override --
    used verbatim (with a trailing `/chat/completions` stripped, since
    `http_chat_completer` appends it) and role resolution is skipped
    entirely; this is for an operator pointing somewhere config resolution
    cannot reach (e.g. a server outside the registry). Otherwise resolves
    `AUTOPILOT_LOCAL_PLANNER_ROLE` (default `frontdoor`, the same env var and
    default `LocalPlannerProvider` already uses) to a llama-server base URL
    via `_resolve_role_server_base_url`."""
    explicit = os.environ.get(_ACTION_REPAIR_URL_ENV)
    if explicit:
        url = explicit.rstrip("/")
        suffix = "/chat/completions"
        if url.endswith(suffix):
            url = url[: -len(suffix)]
        return url
    role = os.environ.get(_ACTION_REPAIR_ROLE_ENV) or _ACTION_REPAIR_DEFAULT_ROLE
    return _resolve_role_server_base_url(role)


def action_repair_completer() -> CompleteFn:
    """Build the `CompleteFn` for the TD-21.2/.3 repair turn. Constructing
    this never makes a network call AND never raises by itself -- a base-URL
    resolution failure is deferred into the returned function, so it surfaces
    as a normal transport-style exception at the ONE point
    (`parse_with_repair`'s per-turn try/except) that already turns any
    `complete()` exception into a typed "failed" result, rather than crashing
    the whole planning cycle. llama-server serves one model per process and
    does not need a `model` field on `/v1/chat/completions`, so none is sent
    (unlike the orchestrator's own multi-model `/v1` compat layer, which
    TD-21.30(e) in `src/structured_output/repair.py` was written for)."""
    try:
        base_url = _local_planner_base_url()
    except Exception as exc:  # noqa: BLE001 - deferred to the repair turn's own try/except
        message = f"could not resolve a TD-21.2/.3 repair completer base URL: {exc}"

        def _unresolvable(messages: Any, schema: Any) -> str:
            raise RuntimeError(message)

        return _unresolvable
    return http_chat_completer(base_url)


def _action_type_json_schema(action_type: str, spec: dict[str, Any]) -> dict[str, Any]:
    """One `oneOf` branch of `autopilot_action_schema()`, derived from the
    SAME `_ACTION_SCHEMAS` entry `_validate_action_schema` already enforces
    downstream -- allowed keys become the closed property set, `required`
    keys become JSON Schema `required`, and `enums` become per-property
    `enum` constraints. Nothing here is invented beyond that existing
    contract; field TYPES beyond an enum's own value types are intentionally
    left unconstrained (`_ACTION_SCHEMAS` does not declare them either)."""
    allowed = sorted(spec.get("allowed", {"type"}) | {"type"})
    required = sorted({"type"} | set(spec.get("required", set())))
    enums = spec.get("enums", {})
    properties: dict[str, Any] = {"type": {"const": action_type}}
    for key in allowed:
        if key == "type":
            continue
        properties[key] = {"enum": sorted(enums[key], key=str)} if key in enums else {}
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def autopilot_action_schema() -> dict[str, Any]:
    """JSON schema for the fenced ``json:autopilot_actions`` object (TD-21.2).

    A `oneOf` over every `_ACTION_SCHEMAS` entry, so adding a new action type
    only requires editing `_ACTION_SCHEMAS`; this schema updates with it.
    `additionalProperties: True` at the TOP level is deliberate: with no
    top-level `properties`, `additionalProperties: False` here would forbid
    every property on the instance rather than deferring to the `oneOf`
    branches, which each already close themselves. Does not encode the
    deeper single-variable / semantic constraints in
    `validate_single_variable` (file naming, single-flag limits, numeric
    ranges) -- those already run on both a fished AND a repaired action via
    the existing `_draft_unusable_reason` path in `planner_coordinator.py`,
    so repeating them here would duplicate, not extend, the safety net.
    """
    return {
        "type": "object",
        "additionalProperties": True,
        "oneOf": [
            _action_type_json_schema(action_type, spec)
            for action_type, spec in _ACTION_SCHEMAS.items()
        ],
    }


def autopilot_rationale_schema() -> dict[str, Any]:
    """JSON schema for the fenced ``json:autopilot_rationale`` sidecar
    (TD-21.3). The optional `vidya_claim_ids` lists claims the planner says
    influenced its action; the archive validates IDs against the evidence
    actually shown. Rubric axes remain caller-defined."""
    return {
        "type": "object",
        "properties": {
            "falsifier": {"type": "string"},
            "rubric_scores": {"type": "object"},
            "vidya_claim_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": [],
        "additionalProperties": False,
    }


def extract_action_with_repair(
    text: str,
    *,
    complete: CompleteFn,
    site: str = _ACTION_REPAIR_SITE,
) -> RepairResult:
    """TD-21.2: `extract_action` first (unchanged, 0 calls on a clean draft);
    on a miss, ONE constrained completion turn against `autopilot_action_schema()`.

    Persisted-state semantics (before/after this conversion):
    BEFORE, a fish miss on the ACTION block made the draft "unusable" ->
    `planner_coordinator._mark_failure` opened the provider's circuit-breaker
    counter, ONE fallback provider was re-prompted FROM ZERO (a full new
    draft call), and if both providers missed, `autopilot.py`'s deterministic
    `seed_batch` action was the one dispatched and journaled.
    AFTER, a malformed-but-repairable ACTION block is recovered via one
    constrained completion and the REPAIRED action is what
    `planner_coordinator` treats as usable -- no fallback provider call, no
    `_mark_failure`, no seed_batch substitution for that draft. The caller is
    responsible for persisting `RepairResult.status` alongside the action
    (see `planner_coordinator.PlannerDecision.action_parse_status`) so a
    repaired record is distinguishable from a clean parse in the journal.
    An UNREPAIRABLE block still returns `status="failed"`, `value=None` --
    the exact same typed non-result `extract_action` returning `None` gave
    before; the existing fallback/seed_batch path is UNCHANGED for that case.
    Old journal rows are NOT backfilled with a parse_status -- readers must
    treat a missing field as "parsed" (the only state that existed before).

    `require_evidence=True` (2026-09-24, live-smoke finding): a grammar that
    forces a required field forces the model to fill it even when the draft
    never states one -- observed live, a `deep_eval` draft with no stated
    tier repaired to a fabricated `tier=2` that would have driven a real
    action. Action fields are operational parameters and MUST come from the
    draft, never be invented to satisfy the schema; see
    `src.structured_output.repair.parse_with_repair`'s `require_evidence`
    docstring for the exact leaf-evidence rule (numbers/short strings must
    appear in the raw text; the `type` discriminator is `const`-exempt).
    """
    fast = extract_action(text)
    if fast is not None:
        return RepairResult(fast, "parsed", "", site, 0)

    result = parse_with_repair(
        text,
        schema=autopilot_action_schema(),
        complete=complete,
        instruction=_ACTION_REPAIR_INSTRUCTION,
        site=site,
        kind="any",
        require_evidence=True,
    )
    if result.value is None:
        return result
    unwrapped = _unwrap_action(result.value)
    if unwrapped is None:
        # The repaired value validated against the oneOf action schema (so it
        # has a `type` const) but somehow doesn't satisfy `_unwrap_action`'s
        # own narrower contract -- treat as a failure rather than dispatch
        # something no other extract_action() caller would accept.
        return RepairResult(None, "failed", "repaired value missing usable type", site, result.repair_calls)
    return RepairResult(unwrapped, result.status, result.reason, site, result.repair_calls)


def extract_rationale_with_repair(
    text: str,
    *,
    complete: CompleteFn,
    site: str = _RATIONALE_REPAIR_SITE,
) -> RepairResult:
    """TD-21.3: same repair shape as TD-21.2, scoped to the
    ``json:autopilot_rationale`` sidecar.

    A rationale block that is simply ABSENT is not a parse failure --
    `extract_rationale`'s own docstring is explicit that omission is a
    legitimate, soft outcome ("rationale capture is observability, not a
    gate") -- so repair is attempted only when the marker IS present but the
    payload fails to parse/validate; an absent marker still returns
    `status="parsed"` with the existing empty default, unchanged from before
    this conversion.

    Persisted-state semantics: BEFORE, a malformed (but present) rationale
    block silently persisted `{"falsifier": "", "rubric_scores": {}}` with no
    signal that anything was lost. AFTER, the malformed block is repaired via
    one constrained completion and the repaired object is what gets
    persisted, `status="repaired"`; only a block that fails BOTH fishing and
    the repair turn falls back to the same empty default as before (now
    flagged `status="failed"` rather than being silently indistinguishable
    from a genuine omission). Old journal rows are NOT backfilled.

    `require_evidence=True` (2026-09-24, same live-smoke finding as
    `extract_action_with_repair`): `rubric_scores` values are numbers, so
    they are exactly the invented-leaf risk the evidence check targets;
    `falsifier` is ordinarily well over the short-string evidence threshold
    and so is exempt in practice anyway (a paraphrase there is legitimate),
    but turning the check on costs nothing and closes the same class of gap
    should a future short falsifier ever occur.
    """
    empty: dict[str, Any] = {"falsifier": "", "rubric_scores": {}}
    marker = "```json:autopilot_rationale"
    if marker not in text:
        return RepairResult(empty, "parsed", "", site, 0)

    start = text.index(marker) + len(marker)
    end = text.find("```", start)
    payload = text[start:end] if end != -1 else text[start:]

    result = parse_with_repair(
        payload,
        schema=autopilot_rationale_schema(),
        complete=complete,
        instruction=_RATIONALE_REPAIR_INSTRUCTION,
        site=site,
        require_evidence=True,
    )
    if result.value is None:
        return RepairResult(dict(empty), result.status, result.reason, site, result.repair_calls)
    falsifier = result.value.get("falsifier", "")
    rubric = result.value.get("rubric_scores", {})
    if not isinstance(falsifier, str):
        falsifier = str(falsifier)
    if not isinstance(rubric, dict):
        rubric = {}
    rationale = {"falsifier": falsifier, "rubric_scores": rubric}
    claim_ids = result.value.get("vidya_claim_ids")
    if isinstance(claim_ids, list) and all(isinstance(cid, str) for cid in claim_ids):
        rationale["vidya_claim_ids"] = claim_ids
    return RepairResult(rationale, result.status, result.reason, site, result.repair_calls)


def _validate_action_schema(action: dict[str, Any]) -> str | None:
    action_type = action.get("type", "")
    schema = _ACTION_SCHEMAS.get(action_type)
    if schema is None:
        return None

    allowed = schema.get("allowed", set())
    extra = sorted(set(action) - allowed)
    if extra:
        return f"{action_type} unsupported keys: {extra}; allowed keys: {sorted(allowed)}"

    missing = sorted(schema.get("required", set()) - set(action))
    if missing:
        if missing == ["file"] and action_type in {
            "prompt_mutation",
            "gepa_optimize",
            "code_mutation",
        }:
            return f"{action_type} must specify a single target file"
        return f"{action_type} missing required keys: {missing}"

    for key, values in schema.get("enums", {}).items():
        if key not in action:
            continue
        value = action[key]
        if isinstance(value, bool) or value not in values:
            return f"{action_type} {key} must be one of {sorted(values)}; got {value!r}"

    return None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_int_range(
    action: dict[str, Any],
    key: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> str | None:
    if key not in action:
        return None
    value = action[key]
    if not _is_int(value):
        return f"{action.get('type', 'action')} {key} must be an integer; got {value!r}"
    if min_value is not None and value < min_value:
        return f"{action.get('type', 'action')} {key} must be >= {min_value}; got {value!r}"
    if max_value is not None and value > max_value:
        return f"{action.get('type', 'action')} {key} must be <= {max_value}; got {value!r}"
    return None


def _validate_number_range(
    action: dict[str, Any],
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_exclusive: bool = False,
) -> str | None:
    if key not in action:
        return None
    value = action[key]
    action_type = action.get("type", "action")
    if not _is_number(value):
        return f"{action_type} {key} must be numeric; got {value!r}"
    if min_value is not None:
        below = value <= min_value if min_exclusive else value < min_value
        if below:
            op = ">" if min_exclusive else ">="
            return f"{action_type} {key} must be {op} {min_value}; got {value!r}"
    if max_value is not None and value > max_value:
        return f"{action_type} {key} must be <= {max_value}; got {value!r}"
    return None


def _validate_bool(action: dict[str, Any], key: str) -> str | None:
    if key not in action:
        return None
    value = action[key]
    if not isinstance(value, bool):
        return f"{action.get('type', 'action')} {key} must be a boolean; got {value!r}"
    return None


def _validate_str_list(action: dict[str, Any], key: str) -> str | None:
    if key not in action:
        return None
    value = action[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return f"{action.get('type', 'action')} {key} must be a list of strings"
    return None


def validate_single_variable(action: dict[str, Any]) -> str | None:
    """AP-9: Validate that an action proposes a single-variable change.

    Returns an error message if the action violates the single-variable
    constraint, or None if it passes.
    """
    action_type = action.get("type", "")

    schema_err = _validate_action_schema(action)
    if schema_err:
        return schema_err

    if action_type in ("prompt_mutation", "gepa_optimize"):
        target = action.get("file", "")
        if not target:
            return f"{action_type} must specify a single target file"
        if "," in target or ";" in target:
            return f"{action_type} targets multiple files: {target}"
        if action_type == "gepa_optimize":
            range_err = _validate_int_range(action, "max_evals", min_value=1, max_value=100)
            if range_err:
                return range_err

    elif action_type == "code_mutation":
        target = action.get("file", "")
        if not target:
            return "code_mutation must specify a single target file"
        if "," in target or ";" in target:
            return f"code_mutation targets multiple files: {target}"

    elif action_type == "structural_experiment":
        flags = action.get("flags", {})
        if not isinstance(flags, dict):
            return "structural_experiment flags must be an object"
        if len(flags) > 1:
            return (
                f"structural_experiment changes {len(flags)} flags at once "
                f"({list(flags.keys())}); limit to 1 for clean attribution"
            )
        for key, value in flags.items():
            if not isinstance(key, str) or not isinstance(value, bool):
                return "structural_experiment flags must map string names to booleans"

    elif action_type == "numeric_trial":
        params = action.get("params", {})
        if not isinstance(params, dict):
            return "numeric_trial params must be an object"
        # Optuna-suggested params are fine (controlled search), but explicit
        # multi-param overrides violate single-variable principle.
        if len(params) > 1:
            return (
                f"numeric_trial sets {len(params)} params explicitly; "
                "limit to 1 for clean attribution (Optuna suggestions exempt)"
            )

    elif action_type == "consult_gate_probe":
        range_err = _validate_int_range(action, "tier", min_value=1, max_value=3)
        if range_err:
            return range_err
        range_err = _validate_int_range(action, "turns", min_value=3, max_value=50)
        if range_err:
            return range_err

    elif action_type == "slot_compact":
        for key, min_value, max_value in (
            ("port", 1, 65535),
            ("slot_id", 0, None),
            ("keep_first", 0, None),
            ("n_future", 1, 8192),
        ):
            range_err = _validate_int_range(action, key, min_value=min_value, max_value=max_value)
            if range_err:
                return range_err
        for key in ("keep_ratio", "threshold"):
            range_err = _validate_number_range(
                action, key, min_value=0.0, max_value=1.0, min_exclusive=True
            )
            if range_err:
                return range_err
        bool_err = _validate_bool(action, "use_covariance")
        if bool_err:
            return bool_err
        if "layer_weights" in action:
            weights = action["layer_weights"]
            if (
                not isinstance(weights, list)
                or not weights
                or any(not _is_number(weight) for weight in weights)
            ):
                return "slot_compact layer_weights must be a non-empty numeric list"

    elif action_type == "seed_batch":
        range_err = _validate_int_range(action, "n_questions", min_value=1, max_value=50)
        if range_err:
            return range_err
        list_err = _validate_str_list(action, "suites")
        if list_err:
            return list_err

    elif action_type == "train_routing_models":
        range_err = _validate_int_range(action, "min_memories", min_value=1, max_value=100000)
        if range_err:
            return range_err

    elif action_type == "distill_skillbank":
        if "teacher" in action and not isinstance(action["teacher"], str):
            return "distill_skillbank teacher must be a string"
        list_err = _validate_str_list(action, "categories")
        if list_err:
            return list_err

    elif action_type == "reset_memories":
        for key in ("keep_seen", "keep_skills"):
            bool_err = _validate_bool(action, key)
            if bool_err:
                return bool_err

    elif action_type == "deep_eval":
        # Enum and required-key checks are covered by _ACTION_SCHEMAS.
        return None

    elif action_type == "distill_knowledge":
        range_err = _validate_int_range(action, "last_n", min_value=1, max_value=100)
        if range_err:
            return range_err

    return None
