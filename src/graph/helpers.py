"""Helper functions for orchestration graph nodes.

Shared utilities extracted from nodes.py to avoid code duplication across
node classes.  Includes REPL execution, error classification, escalation
logic, answer extraction, and state management.

Bug fixes included in this migration:
- ``state.escalation_count`` is incremented on every escalation.
- ``deps.failure_graph.record_failure()`` is called on every error.
- ``deps.hypothesis_graph.add_evidence()`` is called on task outcomes.
- Hardcoded ``EscalationPolicy()`` fallbacks are eliminated.
"""

from __future__ import annotations

import asyncio
import ast
import io
import logging
import os
import re
import tokenize
from pathlib import Path
from typing import Any
from pydantic_graph import GraphRunContext

from src.autopilot_core.measurement_guards import inband_error_text as _inband_error_text
from src.escalation import ErrorCategory
from src.exceptions import ContextOverflowError, InferenceError
from src.graph.error_classifier import classify_error as _classify_error_impl
from src.graph.escalation_helpers import detect_role_cycle as _detect_role_cycle_impl
from src.graph.repl_tap import tap_write_repl_exec as _tap_write_repl_exec_impl
from src.graph.repl_tap import tap_write_repl_result as _tap_write_repl_result_impl
from src.graph.answer_resolution import (  # noqa: F401
    _FINAL_RE,
    _extract_final_from_raw,
    _extract_prose_answer,
    _looks_like_prompt_echo,
    _rescue_from_last_output,
    _resolve_answer,
    _should_attempt_prose_rescue,
)
from src.graph.file_artifacts import (
    _persist_solution_file,
    _solution_file_path,
    _spill_if_truncated,
)
from src.graph.compaction import (  # noqa: F401
    _maybe_compact_context,
    _resolve_compaction_prompt,
)
from src.graph.budgets import (  # noqa: F401
    _REASONING_LENGTH_ALARM_MULTIPLIER,
    _budget_pressure_warnings,
    _check_budget_exceeded,
    _check_reasoning_length_alarm,
    _frontdoor_repl_non_tool_token_cap,
    _frontdoor_turn_token_cap,
    _repl_turn_token_cap,
)
from src.graph.session_summary import (
    _get_exploration_tool_calls,
    _init_session_log,
    _maybe_refresh_session_summary,
    _record_session_turn,
    _session_log_prompt_block,
)
from src.graph.workspace import (  # noqa: F401
    _select_and_broadcast_workspace_delta,
    _update_workspace_from_turn,
    _workspace_prompt_block,
)
from src.graph.task_ir_helpers import (
    _auto_gather_context,
    _auto_seed_tasks_from_task_ir,
    _check_anti_pattern,
    _extract_candidate_files_from_task_ir,
)
from src.graph.observability import (  # noqa: F401
    _add_evidence,
    _log_escalation,
    _record_failure,
    _record_mitigation,
)
from src.graph.decision_gates import (  # noqa: F401
    _check_approval_gate,
    _is_infra_failure,
    _make_end_result,
    _should_escalate,
    _should_retry,
    _timeout_skip,
)
from src.graph.think_harder import (  # noqa: F401
    _build_think_harder_config,
    _should_think_harder,
)
from src.roles import Role

from src.graph.state import (
    TaskDeps,
    TaskState,
)

log = logging.getLogger(__name__)

# Type aliases
Ctx = GraphRunContext[TaskState, TaskDeps]


def _use_inline_calls_in_tests() -> bool:
    """Return True when running under pytest to avoid threadpool teardown hangs."""
    return bool(os.getenv("PYTEST_CURRENT_TEST"))


def _frontdoor_trace_enabled() -> bool:
    raw = os.environ.get("ORCHESTRATOR_FRONTDOOR_TRACE", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _tap_write_repl_exec(code: str, turn: int) -> None:
    """Compatibility wrapper for REPL tap execution logging."""
    _tap_write_repl_exec_impl(code, turn)


def _tap_write_repl_result(
    output: str, error: str | None, is_final: bool, turn: int,
) -> None:
    """Compatibility wrapper for REPL tap result logging."""
    _tap_write_repl_result_impl(output, error, is_final, turn)


def _bep_turn_trace(turn: int, role: object, raw_output: str, code: str | None = None,
                    prompt: str | None = None, repeat_count: object = None) -> None:
    """Flag-gated per-turn observability for BEP OFF-arm (interleaved) debugging.

    Captures exactly what the model emits each turn + whether it calls the file-write tool vs
    ``open()`` vs neither — so the interleaved baseline is never debugged blind again (operator
    directive 2026-05-27: observability-first). Gated on ORCHESTRATOR_BEP_TURN_TRACE=1; fail-silent;
    default-off so production is unaffected. Writes JSONL to ``$tmp_dir/bep_turn_trace.jsonl``."""
    import os
    if os.environ.get("ORCHESTRATOR_BEP_TURN_TRACE") != "1":
        return
    try:
        import json as _json
        from datetime import datetime as _dt
        raw = raw_output or ""
        try:
            from src.config import get_config
            path = str(get_config().paths.tmp_dir / "bep_turn_trace.jsonl")
        except Exception:
            path = "/mnt/raid0/llm/tmp/bep_turn_trace.jsonl"
        _p = prompt or ""
        rec = {
            "ts": _dt.now().isoformat(), "turn": turn, "role": str(role),
            "calls_file_write_safe": "file_write_safe" in raw,
            "calls_open": "open(" in raw,
            "has_final": "FINAL(" in raw,
            "raw_output": raw[:4000],
            "extracted_code": (code[:2000] if code else None),
            # Proof instrumentation: what did the model actually RECEIVE this turn?
            "prompt_chars": (len(_p) if prompt is not None else None),
            "prompt_has_last_output": ("## Last Output" in _p) if prompt is not None else None,
            "prompt_has_rider": ("WRITE turns:" in _p) if prompt is not None else None,
            "prompt_has_loop_halt": ("LOOP HALTED" in _p) if prompt is not None else None,
            "prompt_tail": (_p[-700:] if prompt is not None else None),
            "repeat_count_seen": repeat_count,  # value the nudge saw this turn (proves accumulation)
        }
        with open(path, "a") as f:
            f.write(_json.dumps(rec) + "\n")
    except Exception:
        pass


def _repl_loop_guard_enabled() -> bool:
    """REPL loop-guard (default-off): fence-repair for FINAL/CALL-truncated code blocks +
    identical-non-advancing-turn breaker. Gated on ORCHESTRATOR_REPL_LOOP_GUARD=1 so the
    CRITICAL `_execute_turn` path is a true no-op in production until the BEP A/B validates it."""
    import os
    return os.environ.get("ORCHESTRATOR_REPL_LOOP_GUARD") == "1"


def _repair_unclosed_code_fence(text: str) -> str:
    """Fix A: the FINAL(/CALL( early-stop aborts streaming the instant it fires — which can
    happen *inside* an open ```fence, before the model emits the closing ```. That leaves an
    unclosed block, the extractor returns nothing, and the model re-emits the identical turn
    until timeout. An odd count of ``` markers means the last fence is unclosed → append one."""
    if text and text.count("```") % 2 == 1:
        return text.rstrip() + "\n```\n"
    return text


def _loop_guard_noprogress(made_progress: bool, prev_count: int) -> int:
    """Fix B counter (pure): count CONSECUTIVE non-advancing turns. A turn "made progress" iff it
    wrote a file (``file_write_safe``) or emitted ``FINAL(``. This is deliberately broader than the
    earlier identical-output check: the BEP read-first failures show up as re-reads, EMPTY outputs,
    and unknown-tool loops — all of which are "no progress" but not necessarily byte-identical, so
    an identical-match trigger missed them (proven: the hard intervention never fired in
    results-readfix3 because the model emptied out). Reset on progress; else increment."""
    return 0 if made_progress else prev_count + 1


_REFUSED_TOOL_CALL_LITERAL_RE = re.compile(
    r"\[ERROR: tool call .+ NOT executed: its JSON arguments "
    r"are malformed and could not be repaired\. Raw: .* -- re-emit "
    r"the call with valid JSON arguments\.\]",
    re.DOTALL,
)


def _loop_guard_classification_view(code: str) -> str:
    """Hide only native malformed-tool refusal literals from progress markers.

    Keep the original code for execution and diagnostics. The loop guard uses
    substring markers, so a raw refused-call echo containing ``FINAL(`` or
    ``file_write_safe`` must not masquerade as progress. Tokenize the source and
    blank only string-token spans whose decoded value matches the refusal
    envelope emitted by ``code_utils._render_call_code``. If tokenization fails,
    preserve the existing classification behavior.
    """
    spans: list[tuple[int, int]] = []
    line_starts = [0, *(match.end() for match in re.finditer("\n", code))]

    try:
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type != tokenize.STRING:
                continue
            try:
                value = ast.literal_eval(token.string)
            except (SyntaxError, ValueError):
                continue
            if not isinstance(value, str) or not _REFUSED_TOOL_CALL_LITERAL_RE.fullmatch(value):
                continue
            start = line_starts[token.start[0] - 1] + token.start[1]
            end = line_starts[token.end[0] - 1] + token.end[1]
            spans.append((start, end))
    except (IndentationError, SyntaxError, tokenize.TokenError):
        return code

    if not spans:
        return code

    view = list(code)
    for start, end in spans:
        for index in range(start, end):
            if view[index] not in "\r\n":
                view[index] = " "
    return "".join(view)


# ── Shared helpers ─────────────────────────────────────────────────────


def _classify_error(error_message: str) -> ErrorCategory:
    """Compatibility wrapper for extracted error classifier."""
    return _classify_error_impl(error_message)


def _maybe_compress_for_escalation(prompt: str, state: "TaskState") -> str:
    """Compress prompt when escalating to architect tier (WS3B).

    Only activates when:
    - escalation_count > 0 (we're in an escalated execution)
    - prompt > 16K chars (~4K tokens)
    - escalation_compression feature flag is enabled

    Uses LLMLingua-2 BERT for extractive token selection (~10-50ms on CPU).
    Preserves code structure tokens (def, class, import, FINAL).

    Saving: ~2K tokens at 1.2 t/s architect prefill = 1.67s per escalation.
    """
    from src.features import features as _get_features

    if not _get_features().escalation_compression:
        return prompt

    if state.escalation_count <= 0:
        return prompt

    # Only compress large prompts (>16K chars ≈ 4K tokens)
    if len(prompt) <= 16_000:
        return prompt

    try:
        from src.services.prompt_compressor import PromptCompressor

        compressor = PromptCompressor.get_instance()
        result = compressor.compress(
            prompt,
            target_ratio=0.5,
            force_tokens=["FINAL", "def ", "class ", "import "],
        )
        log.info(
            "Escalation compression: %d→%d chars (%.1f%% reduction, %.1fms)",
            result.original_chars,
            result.compressed_chars,
            (1 - result.actual_ratio) * 100,
            result.latency_ms,
        )
        return result.compressed_text
    except Exception as e:
        log.warning("Escalation compression failed, using uncompressed: %s", e)
        return prompt


def _clear_stale_tool_outputs(
    state: TaskState,
    keep_recent: int = 2,
    context_ratio_trigger: float = 0.4,
    max_context_tokens: int = 0,
) -> int:
    """Strip old <<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>> blocks from last_output.

    Keeps the last ``keep_recent`` blocks verbatim, replaces older ones
    with ``[Tool result cleared]`` placeholders.

    Args:
        state: Current task state (modifies ``state.last_output`` in place).
        keep_recent: Number of most-recent tool output blocks to preserve.
        context_ratio_trigger: Only clear when context exceeds this fraction
            of ``max_context_tokens``.  Set to 0 to always clear.
        max_context_tokens: Model's max context size in tokens.  When 0,
            uses a char-count heuristic (12000 chars ≈ 3000 tokens).

    Returns:
        Estimated tokens freed by clearing.
    """
    from src.features import features as _get_features

    if not _get_features().tool_result_clearing:
        return 0

    text = state.last_output
    if not text:
        return 0

    # Gate: only fire when context is large enough to matter
    if max_context_tokens > 0:
        ctx_tokens = len(state.context) // 4  # rough estimate
        if ctx_tokens < max_context_tokens * context_ratio_trigger:
            return 0
    else:
        if len(state.context) < 12000:
            return 0

    # Find all <<<TOOL_OUTPUT>>>...<<<END_TOOL_OUTPUT>>> blocks
    pattern = re.compile(
        r"<<<TOOL_OUTPUT>>>(.*?)<<<END_TOOL_OUTPUT>>>",
        re.DOTALL,
    )
    matches = list(pattern.finditer(text))
    if len(matches) <= keep_recent:
        return 0

    # Replace older blocks (all except last keep_recent)
    blocks_to_clear = matches[: -keep_recent] if keep_recent > 0 else matches
    tokens_freed = 0

    # Build replacement from end to start to preserve offsets
    new_text = text
    for match in reversed(blocks_to_clear):
        old_block = match.group(0)
        tokens_freed += len(old_block) // 4
        new_text = new_text[: match.start()] + "[Tool result cleared]" + new_text[match.end() :]

    state.last_output = new_text
    return tokens_freed


# Matches a completed CALL("tool_name", ...) invocation.  Used as an
# early-stop signal so the model pauses after writing a tool call and
# the REPL can execute it before the model continues reasoning.
_CALL_STOP_RE = re.compile(
    r'CALL\s*\(\s*"[^"]+"\s*(?:,\s*\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|\d+|True|False|None))*\s*\)',
)


def _is_comment_only(code: str) -> bool:
    """Return True if code has no executable lines (all comments/blank)."""
    for line in code.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return False
    return True


def _backend_infra_sentinel(raw_output: str) -> str | None:
    """Return ``raw_output`` when it IS an in-band backend/infra sentinel, else None.

    ``llm_call`` (``src.llm_primitives.primitives._llm_call_impl``) turns EVERY
    exception the backend call raises -- a placement/admission timeout
    (``ContentionDenied``, e.g. ``[ERROR: placement timeout role=... reason=
    race_lost holders=[] after 60.0s]``), a contention-gate denial, a
    circuit-open refusal, a queue-full rejection, a dead connection -- into a
    same-shaped ``f"[ERROR: {e}]"`` STRING return rather than raising. A REPL
    turn's raw LLM output IS that string exactly when the backend denied the
    call before generation ever started: the model never saw the prompt, so
    there is no "turn" to grade at all, and treating the string as a
    comment-only/unparseable model turn (and nudging toward "write executable
    code") mis-scores a host-resource denial as a model failure.

    Reuses the ONE canonical anchor (``INBAND_ERROR_PREFIX`` /
    ``inband_error_text``, ``src/autopilot_core/measurement_guards.py``)
    instead of a second copy of the prefix literal -- anchored to
    start-of-answer (after stripping leading whitespace) so a model
    legitimately discussing ``"[ERROR:"`` mid-answer is never mistaken for
    this.
    """
    return _inband_error_text(raw_output)


def _no_executable_code_nudge(state: TaskState, *, comment_ratio: float | None = None) -> str:
    if state.tool_required:
        prefix = (
            f"Your code is {int(comment_ratio * 100)}% comments — "
            if comment_ratio is not None
            else "Your output was all comments — no executable code ran. "
        )
        return (
            prefix
            + "This turn requires a tool call. Write executable Python now, call the required tool "
            'with TOOL("get_eval_secret", name="...") using the name from the prompt, and then '
            "call FINAL(secret). Do not explain or re-derive."
        )
    if comment_ratio is not None:
        return (
            f"Your code is {int(comment_ratio * 100)}% comments — you already reasoned through the problem. "
            'STOP re-deriving. Call FINAL now with the value you reached — e.g. FINAL("B") or FINAL(42). '
            "Do NOT start over. Do NOT re-explain."
        )
    return (
        "Your output was all comments — no executable code ran. "
        'You already reasoned through the problem. Call FINAL now with the actual value — e.g. FINAL("B") or FINAL(42).'
    )


def _log_state_snapshot(ctx: Ctx, role: str) -> None:
    """Persist a full state snapshot for the current turn (LangGraph pre-migration)."""
    try:
        import json as _json
        from src.graph.persistence import _state_to_dict

        state = ctx.state
        blob = {
            "type": "turn_snapshot",
            "turn": state.turns,
            "role": role,
            "state": _state_to_dict(state),
        }
        store = getattr(ctx.deps, "session_store", None)
        if store is not None:
            # D-f3: routed to the dedicated graph-snapshot writer (the old
            # positional save_checkpoint call never matched the store API).
            # Session-scoped only when the turn holds the session lease.
            store.save_graph_snapshot(
                state.task_id,
                _json.dumps(blob, default=str),
                "state_snapshot",
                session_id=getattr(ctx.deps, "session_id", None),
                fencing_token=getattr(ctx.deps, "session_fencing_token", None),
            )
        else:
            log.debug("State snapshot: turn=%d role=%s fields=%d", state.turns, role, len(blob["state"]))
    except Exception as exc:
        log.warning("State snapshot persist failed (turn snapshot dropped): %s", exc)


# ── BEP (J8): batched-edit parallel-apply turn divergence ────────────────────
# Flag-gated by features().batch_edit_mode (default OFF). When the root LM emits a
# fenced ```patchset block, apply it transactionally (sandbox → verify → promote)
# instead of running the interleaved REPL loop this turn. Zero behavior change when
# the flag is off OR no patchset block is present — _maybe_batch_edit_turn returns
# None and the caller falls through to the normal REPL path. Mirrors the existing
# file_mutation edit surface (same repo), so it adds no new write surface and is in
# fact safer than _file_write_safe (sandbox+verify before promote vs immediate write).
# Handoff: handoffs/active/batched-edit-parallel-apply.md (BEP-4/BEP-5).


def _batch_edit_repo_root() -> Path:
    """Apply target. Scratch task-root when ORCHESTRATOR_EDIT_ROOT is active (BEP A/B, #8); else
    the project root the interleaved REPL writes to — get_task_root() returns project_root when
    inactive, so this is default-off parity with the prior _get_project_root()."""
    from src.repl_environment.task_root import get_task_root
    return get_task_root()


def _batch_edit_verify_fn(sandbox_root: Path) -> bool:
    """Inference-free accept gate.

    Default behavior is unchanged: py_compile every staged .py file. When
    ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD is configured, the caller stages a tracked
    repo snapshot and this helper runs that command in the sandbox as the final
    accept gate.
    """
    import py_compile
    import subprocess

    root = Path(sandbox_root)
    for p in root.rglob("*.py"):
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception:
            return False
    verify_cmd = os.environ.get("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", "").strip()
    if not verify_cmd:
        return True
    try:
        timeout = float(os.environ.get("ORCHESTRATOR_BATCH_EDIT_VERIFY_TIMEOUT_SEC", "120"))
    except ValueError:
        log.warning("batch-edit verify timeout is not numeric")
        return False
    try:
        completed = subprocess.run(
            ["/bin/bash", "-lc", verify_cmd],
            cwd=root,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except Exception as exc:
        log.warning("batch-edit verify command raised: %s", exc)
        return False
    if completed.returncode != 0:
        log.warning(
            "batch-edit verify command failed rc=%s stdout=%r stderr=%r",
            completed.returncode,
            completed.stdout[-1000:],
            completed.stderr[-1000:],
        )
        return False
    return True


def _batch_edit_failure_summary(result: Any) -> str:
    parts = [
        f"{f.get('path', '?')}: {f.get('failure_type', '?')} {f.get('detail', '')}".strip()
        for f in (result.failed or [])
    ][:6]
    detail = "; ".join(parts) or "apply/verify failed"
    return (
        "[SYSTEM: batch-edit patchset could not be applied cleanly (" + detail + "). "
        "Re-emit a corrected ```patchset block, or proceed with normal step-by-step edits.]"
    )


def _finalize_batch_edit(
    state: Any, role: Role | str, result: Any, ps: Any
) -> tuple[str, str | None, bool, dict]:
    """Synthesize the terminal turn result for a successfully applied + promoted patch set."""
    files = ", ".join(sorted(result.diff_paths)) or "(none)"
    summary = (
        f"Batch edit applied {len(ps.files)} file change(s) [{files}] — "
        f"sandbox-verified (py_compile) and promoted transactionally."
    )
    artifacts = {
        "_batch_edit": {
            "files": sorted(result.diff_paths),
            "n_patches": len(ps.files),
            "verified": result.verify_passed,
        }
    }
    _record_session_turn(state, role=str(role), output=summary, is_final=True)
    return summary, None, True, artifacts


# BEP-1c telemetry: per-turn batch-edit outcome counts. The states are distinct (the hard gate
# requires malformed != absent): "absent" = no ```patchset block (model chose not to batch);
# "malformed" = block present but invalid JSON/schema (parse error); "applied" = sandbox-verified
# + promoted; "verify_failed" / "apply_failed" / "promote_failed" / "apply_error" = failure modes.
# The BEP-2 A/B reads these to separate parse-failure rate from "didn't batch".
_BATCH_EDIT_STATE_COUNTS: dict[str, int] = {}


def _record_batch_edit_state(state_name: str, *, turn: int | None = None, detail: str = "") -> None:
    _BATCH_EDIT_STATE_COUNTS[state_name] = _BATCH_EDIT_STATE_COUNTS.get(state_name, 0) + 1
    log.info("batch_edit_state=%s turn=%s %s", state_name, turn, detail.replace("\n", " ")[:160])


async def _maybe_batch_edit_turn(
    ctx: Ctx, role: Role | str, raw_llm_output: str
) -> tuple[str, str | None, bool, dict] | None:
    """BEP J8 divergence. Returns a 4-tuple to short-circuit the turn, or None to fall
    through to the normal REPL path. Default-off; only the flag-on + patchset-present
    path diverges, so flag-off behavior is provably unchanged."""
    from src.features import features as _get_features

    if not _get_features().batch_edit_mode:
        return None
    from src.repl_environment.task_root import request_scope as _request_scope

    _scope = _request_scope()
    if _scope is not None and not _scope.can_write:
        # INF-78 OAB-1: edit_mode='none' — a patchset may not be applied; the REPL path (whose
        # write tools refuse) handles the turn.
        return None
    state = ctx.state
    turn = getattr(state, "turns", None)
    # BEP-1c: "absent" (no patchset block) and "malformed" (block present but invalid) MUST be
    # distinguishable — both fall through to REPL, but they are different signals for the A/B.
    try:
        from src.batch_edit_parse import parse_patchset_from_model_output

        ps = parse_patchset_from_model_output(raw_llm_output)  # None if no fenced block
    except ValueError as exc:
        _record_batch_edit_state("malformed", turn=turn, detail=str(exc))
        return None  # present-but-malformed → fall back to REPL (model re-emits)
    if ps is None:
        _record_batch_edit_state("absent", turn=turn)
        return None

    from src.batch_edit_runner import (
        apply_patchset_sandboxed,
        cleanup_sandbox,
        promote_sandbox,
    )

    repo_root = _batch_edit_repo_root()
    full_tree_verify = bool(os.environ.get("ORCHESTRATOR_BATCH_EDIT_VERIFY_CMD", "").strip())
    try:
        result = await asyncio.to_thread(
            apply_patchset_sandboxed,
            ps,
            repo_root=repo_root,
            verify_fn=_batch_edit_verify_fn,
            full_tree=full_tree_verify,
        )
    except Exception as e:  # noqa: BLE001 — apply must never crash the turn
        _record_batch_edit_state("apply_error", turn=turn, detail=str(e))
        log.warning(
            "batch-edit apply raised (turn %d): %s — falling back to REPL", state.turns, e
        )
        return None

    if result.ok:
        promoted = False
        try:
            promoted = promote_sandbox(result, repo_root)
        finally:
            cleanup_sandbox(result)
        if promoted:
            _record_batch_edit_state("applied", turn=turn, detail=f"{len(result.diff_paths)} file(s)")
            log.info(
                "batch-edit (turn %d): applied+promoted %d file(s)",
                state.turns,
                len(result.diff_paths),
            )
            return _finalize_batch_edit(state, role, result, ps)
        _record_batch_edit_state("promote_failed", turn=turn)
        log.warning(
            "batch-edit verify passed but promotion failed (turn %d) — falling back to REPL",
            state.turns,
        )
        return None

    # apply/verify failed → never touch the live tree; nudge to re-emit or fall back
    cleanup_sandbox(result)
    _record_batch_edit_state(
        "verify_failed" if result.verify_passed is False else "apply_failed",
        turn=turn,
        detail=_batch_edit_failure_summary(result),
    )
    nudge = _batch_edit_failure_summary(result)
    _record_session_turn(state, role=str(role), nudge=nudge)
    return "", None, False, {"_nudge": nudge}


def _best_effort_last_inference_meta(primitives: Any) -> dict[str, Any]:
    """Read the calling THREAD's/context's own last-inference meta.

    Prefers the per-call-safe `get_last_inference_meta()` getter (TD-21.21/33) and falls
    back to the plain, possibly-shared `_last_inference_meta` attribute only for test
    doubles that predate the getter (e.g. hand-rolled fakes in older tests) -- identical
    to those doubles' pre-TD-21.33 behavior, never worse.

    This function is context-agnostic: it reports whatever the CURRENT thread/context sees.
    Callers that cross an `asyncio.to_thread` boundary between the call and the read MUST
    NOT call this directly from the parent side (a real `LLMPrimitives`'s ContextVar `.set()`
    made in the child thread's copied context is invisible to the parent once the `await`
    returns, so a parent-side call here would silently and always miss it in production and
    fall through to the racy shared attribute). Use `_call_llm_capturing_meta` below instead,
    which invokes this from INSIDE the same thread as the call (TD-21.33a).
    """
    getter = getattr(primitives, "get_last_inference_meta", None)
    if callable(getter):
        meta = getter()
        if meta:
            return dict(meta)
    return dict(getattr(primitives, "_last_inference_meta", {}) or {})


async def _call_llm_capturing_meta(
    llm_call_fn: Any, primitives: Any, *args: Any, **kwargs: Any
) -> tuple[str, dict[str, Any]]:
    """Call `llm_call_fn` and return `(result, this_calls_inference_meta)`.

    TD-21.33a: the real fix for the "documented residual race" TD-21.33 left in
    `_execute_turn` -- the model call runs via `await asyncio.to_thread(llm_call_fn, ...)`,
    and `asyncio.to_thread` copies the CALLING context (`contextvars.copy_context()`) into a
    new worker thread; a `.set()` made inside that thread (deep inside
    `_set_last_inference_meta`) is invisible to the PARENT once the `await` returns. Reading
    `get_last_inference_meta()` (or the plain `_last_inference_meta` attribute, which is
    shared across concurrent requests against the same `LLMPrimitives` instance -- the
    TD-21.21 hazard) back in the parent after the hop is therefore either always-None or
    racy, never both correct and race-free.

    The fix, mirroring `src/api/routes/chat.py`'s edit-transaction path: capture the meta
    INSIDE the same thread the call ran in, immediately after the call returns, and hand it
    back to the parent as part of the result -- never re-derived from ambient state after the
    hop. Whichever thread actually executes `_run` (the calling thread when
    `_use_inline_calls_in_tests()` is set or `llm_call_fn` is a `unittest.mock` double, a
    fresh worker thread otherwise), the getter read happens in *that* thread's own (copied)
    context, so it observes THIS call's `.set()` and never a concurrent call's -- closing the
    shared-attribute race, not just the None-in-production gap.

    For a test double without `get_last_inference_meta()` at all, `_best_effort_last_inference_meta`
    falls back to the plain attribute exactly as before -- unchanged behavior for those doubles.
    """

    def _run() -> tuple[str, dict[str, Any]]:
        result = llm_call_fn(*args, **kwargs)
        return result, _best_effort_last_inference_meta(primitives)

    # Unit tests often inject MagicMock llm_call; using to_thread on mocked
    # callables can deadlock event-loop teardown in pytest-asyncio.
    if _use_inline_calls_in_tests() or type(llm_call_fn).__module__.startswith("unittest.mock"):
        return _run()
    return await asyncio.to_thread(_run)


#: UFH14-B1 F2 (D4): share of the request's time budget after which a REPL turn
#: demands FINAL now (F12: 0.65 of the planner budget). Env-overridable.
ANSWER_FORCE_FRAC_ENV = "ORCHESTRATOR_REPL_ANSWER_FORCE_FRAC"
DEFAULT_ANSWER_FORCE_FRAC = 0.65
ANSWER_FORCE_MESSAGE = (
    "\n\n** TIME BUDGET: most of this request's time budget is spent. "
    "Do NOT explore further, do NOT delegate (no CALL), do NOT start over. "
    "Your code this turn MUST be FINAL(your best current answer). Submit what you have."
)


def _answer_force_frac() -> float:
    try:
        frac = float(os.environ.get(ANSWER_FORCE_FRAC_ENV, DEFAULT_ANSWER_FORCE_FRAC))
    except ValueError:
        frac = DEFAULT_ANSWER_FORCE_FRAC
    return frac if 0.0 < frac < 1.0 else DEFAULT_ANSWER_FORCE_FRAC


def answer_force_at(primitives: Any, now: float | None = None) -> float | None:
    """The perf_counter time after which REPL turns demand FINAL (F2), or None.

    None unless the ``repl_answer_force`` feature is on AND the request has a
    deadline (``primitives.get_request_deadline_s()``, an absolute perf_counter
    time): ``now + frac x (deadline - now)``. Stamped once on TaskState when the
    graph starts, so the fraction is of the budget left to the graph."""
    import time

    try:
        from src.features import features as _get_features

        if not _get_features().repl_answer_force:
            return None
        deadline = primitives.get_request_deadline_s() if primitives is not None else None
    except Exception:
        return None
    if not isinstance(deadline, (int, float)) or isinstance(deadline, bool):
        return None
    start = time.perf_counter() if now is None else float(now)
    if deadline <= start:
        return None
    return start + _answer_force_frac() * (deadline - start)


def _answer_force_due(state: Any) -> bool:
    import time

    at = getattr(state, "answer_force_at_s", None)
    if not isinstance(at, (int, float)) or isinstance(at, bool):
        return False
    return time.perf_counter() >= at


async def _execute_turn(ctx: Ctx, role: Role | str) -> tuple[str, str | None, bool, dict]:
    """Execute one LLM → REPL turn.

    Returns:
        (code_output, error_or_none, is_final, artifacts)
    """
    state = ctx.state
    deps = ctx.deps
    state.turns += 1
    log.debug("_execute_turn: turn=%d, role=%s", state.turns, role)

    # Full state snapshot (LangGraph pre-migration)
    from src.features import features as _get_features
    if _get_features().state_history_snapshots:
        _log_state_snapshot(ctx, str(role))

    # Generalized interrupt conditions (LangGraph pre-migration)
    if _get_features().generalized_interrupts:
        conditions = getattr(deps, "interrupt_conditions", None) or []
        if conditions:
            from src.graph.approval_gate import (
                check_interrupt_conditions,
                request_approval_for_interrupt,
                ApprovalDecision,
            )
            interrupt_desc = check_interrupt_conditions(conditions, state, state.artifacts)
            if interrupt_desc:
                decision = request_approval_for_interrupt(ctx, interrupt_desc)
                if decision == ApprovalDecision.REJECT:
                    _record_session_turn(state, role=str(role), error=f"Interrupted: {interrupt_desc}")
                    return "", f"Interrupted: {interrupt_desc}", False, {}

    # Initialize session log on first turn
    _init_session_log(state)

    # Clear stale tool outputs before compaction (C3)
    tool_tokens_freed = _clear_stale_tool_outputs(state)
    if tool_tokens_freed > 0:
        log.info("Cleared stale tool outputs: ~%d tokens freed", tool_tokens_freed)

    # UFH14-B1 F2: past the wall-time answer point, this turn demands FINAL now
    # and runs no session compaction (no compaction on the answer turn).
    force_answer = _answer_force_due(state)
    if force_answer:
        state.answer_forced_turns += 1
        log.info(
            "answer force: turn %d past %.0f%% of the request budget; demanding FINAL",
            state.turns, _answer_force_frac() * 100,
        )

    # Session compaction before execution
    if not force_answer:
        await _maybe_compact_context(ctx)

    if deps.primitives is None or deps.repl is None:
        _record_session_turn(state, role=str(role), error="No LLM primitives or REPL configured")
        return "", "No LLM primitives or REPL configured", False, {}

    # Attach per-request task tracking context for tool invocations.
    deps.repl._task_manager = state.task_manager  # noqa: SLF001
    deps.repl._task_type = state.task_type  # noqa: SLF001

    # Seed task manager from TaskIR and gather context before prompt build.
    if state.turns == 1:
        _auto_seed_tasks_from_task_ir(state)
    gathered_context = _auto_gather_context(ctx, _extract_candidate_files_from_task_ir(state))
    state.anti_pattern_warning = _check_anti_pattern(ctx) or ""

    # MF-FIN-1: set when THIS turn's prompt instructs the file-editing finish FINAL('done')
    # (interleaved-edit rider or loop-guard HALT) so the status-message guard accepts it.
    _edit_finish_sanctioned = False

    # Build prompt
    if state.escalation_prompt:
        prompt = state.escalation_prompt
        state.escalation_prompt = ""
    else:
        from src.prompt_builders.builder import PromptBuilder, build_corpus_context
        from src.prompt_builders.types import PromptConfig, PromptStyle

        repl_state = deps.repl.get_state()

        # Inject corpus context on first turn for prompt-lookup acceleration
        corpus_ctx = ""
        if state.turns == 1:
            corpus_ctx = build_corpus_context(
                role=str(role),
                task_description=state.prompt,
                task_id=state.task_id or None,
            )

        # Pass solution file path when there's an error so the model can patch
        sol_file = ""
        if state.last_error and state.last_code:
            import os
            candidate = _solution_file_path(state)
            if os.path.exists(candidate):
                sol_file = candidate

        builder = PromptBuilder(PromptConfig(style=PromptStyle.MINIMAL))
        _prompt_cfg = builder.config
        # INF-78 OAB-7: with a context bundle attached, the REPL's per-turn print cap is
        # THE bound on printed output reaching this prompt; widen the preview to it so a
        # capped turn is not re-truncated (and spilled) at the 1500-char default.
        _bundle_preview = getattr(deps.repl, "bundle_output_preview_chars", None)
        if isinstance(_bundle_preview, int) and _bundle_preview > _prompt_cfg.max_output_preview:
            _prompt_cfg.max_output_preview = _bundle_preview

        # Tool output compression (Phase 2 native): compress before spill
        _output_for_prompt = state.last_output
        _error_for_prompt = state.last_error
        if _get_features().tool_output_compression:
            try:
                import sys as _sys
                _root = "/mnt/raid0/llm/epyc-root/scripts/utils"
                if _root not in _sys.path:
                    _sys.path.insert(0, _root)
                from compress_tool_output import compress_tool_output as _compress
                _output_for_prompt = _compress(state.last_output, state.last_code)
                _error_for_prompt = _compress(state.last_error, state.last_code)
                _out_orig = len(state.last_output)
                _out_comp = len(_output_for_prompt)
                _err_orig = len(state.last_error)
                _err_comp = len(_error_for_prompt)
                state.compression_metrics = {
                    "output_original_chars": _out_orig,
                    "output_compressed_chars": _out_comp,
                    "output_ratio": round(_out_comp / max(_out_orig, 1), 3),
                    "error_original_chars": _err_orig,
                    "error_compressed_chars": _err_comp,
                    "command": (state.last_code or "")[:80],
                }
                if _out_orig > 0 and _out_comp / max(_out_orig, 1) < 0.7:
                    log.info(
                        "Tool output compressed: %d → %d chars (%.0f%%)",
                        _out_orig, _out_comp,
                        _out_comp / _out_orig * 100,
                    )
            except Exception:
                log.debug("Tool output compression failed, using raw output", exc_info=True)

        # CMV Action 11: spill long output/error to file with peek() pointer
        _spilled_output = _spill_if_truncated(
            _output_for_prompt, _prompt_cfg.max_output_preview, "output", state,
        )
        _spilled_error = _spill_if_truncated(
            _error_for_prompt, _prompt_cfg.max_error_preview, "error", state,
        )
        prompt = builder.build_root_lm_prompt(
            state=repl_state,
            original_prompt=state.prompt,
            last_output=_spilled_output,
            last_error=_spilled_error,
            turn=state.turns - 1,
            corpus_context=corpus_ctx,
            solution_file=sol_file,
        )
        if gathered_context:
            prompt += "\n\n[Auto Gathered Context]\n" + gathered_context
        prompt += "\n\n" + _workspace_prompt_block(state)

        # BEP (J8): batch-edit instructions rider — flag-gated (default off) + only for
        # file-editing roles (coder/architect). Tells the model it MAY emit one fenced
        # ```patchset block instead of interleaved edits; _maybe_batch_edit_turn applies it.
        if _get_features().batch_edit_mode and any(
            k in str(role).lower() for k in ("coder", "architect")
        ):
            from src.batch_edit_parse import build_batch_edit_instructions

            prompt += "\n\n" + build_batch_edit_instructions()
        # BEP-2 baseline (J8): symmetric interleaved-edit rider — flag-gated (default off).
        # Makes the OFF arm actually apply edits turn-by-turn so the A/B measures interleaved-vs-
        # batched edit latency, not "edits vs prose". Mutually exclusive with batch_edit_mode.
        elif _get_features().interleaved_edit_rider and any(
            k in str(role).lower() for k in ("coder", "architect")
        ):
            from src.batch_edit_parse import build_interleaved_edit_instructions

            prompt += "\n\n" + build_interleaved_edit_instructions()
            _edit_finish_sanctioned = True

    # Inject session log summary (processing history across turns)
    await _maybe_refresh_session_summary(state, deps)
    session_block = _session_log_prompt_block(state)
    if session_block:
        prompt += "\n\n" + session_block

    # Budget pressure warnings (Fast-RLM)
    budget_warnings = _budget_pressure_warnings(state)
    if budget_warnings:
        prompt += "\n\n" + budget_warnings

    # REPL loop-guard (Fix B, flag-gated default-off): an identical non-advancing turn means the
    # model is stuck (re-reading / re-writing) and the prior ADVISORY nudge did not break it under
    # greedy decode (proven: BEP read-first tasks looped 8 turns with the content already in-prompt).
    # HARD intervention: re-inject the content it already has + a forceful single-action directive,
    # appended LAST (recency) so it dominates the next generation.
    if _repl_loop_guard_enabled() and getattr(state, "repl_noprogress_count", 0) >= 2:
        _read = (getattr(state, "last_output", "") or "").strip()[:1500]
        _halt = (
            "\n\n** LOOP HALTED ** You just emitted the SAME action with no progress. STOP repeating it. "
            "Do NOT peek/read again — you already have what you need. THIS TURN emit a single closed "
            "```python block containing ONLY file_write_safe('<relative-path>', '''<full new file "
            "content>''') for each file you must change (relative paths resolve to the task workspace; "
            "do NOT use absolute paths). Do NOT call FINAL() in that block. If every required edit is "
            "already written, instead emit a block containing ONLY FINAL('done')."
        )
        if _read:
            _halt += "\n\nContent you already read (use it; do not re-read):\n```\n" + _read + "\n```"
        prompt += _halt
        _edit_finish_sanctioned = True

    # Graduated FINAL() nudge: midpoint soft reminder, then hard deadline.
    remaining = state.max_turns - state.turns
    if force_answer:
        # UFH14-B1 F2: the wall-time forced-answer turn (supersedes the
        # turn-count nudges below for this turn).
        prompt += ANSWER_FORCE_MESSAGE
    elif remaining <= 3:
        prompt += (
            f"\n\n** DEADLINE: {remaining} turn(s) remaining. "
            "You MUST call FINAL(your_computed_value) NOW with your best answer. "
            "Do NOT start over. Do NOT re-derive. Do NOT reason in comments. "
            "Submit what you have."
        )
    elif remaining == state.max_turns // 2 and state.turns > 1:
        prompt += (
            f"\n\n** REMINDER: {remaining} turn(s) remaining. "
            "Start converging on your answer. Call FINAL() when ready."
        )

    # Apply think-harder config override if set (same model, boosted params)
    llm_kwargs: dict = {}
    if state.think_harder_config:
        cot_prefix = state.think_harder_config.get("cot_prefix", "")
        if cot_prefix:
            prompt = cot_prefix + prompt
        n_tokens = state.think_harder_config.get("n_tokens")
        if n_tokens:
            llm_kwargs["n_tokens"] = n_tokens
        state.think_harder_config = None  # Clear after use

    # Tool-required turns can ramble until role timeout when left unlimited.
    # Apply a bounded per-turn token budget unless think-harder already set one.
    if state.tool_required and "n_tokens" not in llm_kwargs:
        llm_kwargs["n_tokens"] = _repl_turn_token_cap(state.difficulty_band)

    if (
        str(role) == str(Role.FRONTDOOR)
        and not state.tool_required
        and "n_tokens" not in llm_kwargs
    ):
        llm_kwargs["n_tokens"] = _frontdoor_repl_non_tool_token_cap()

    if str(role) == str(Role.FRONTDOOR) and "n_tokens" not in llm_kwargs:
        frontdoor_cap = _frontdoor_turn_token_cap()
        if frontdoor_cap > 0:
            llm_kwargs["n_tokens"] = frontdoor_cap

    # Compress prompt on escalation to reduce architect prefill time (WS3B)
    prompt = _maybe_compress_for_escalation(prompt, state)

    # Apply GBNF grammar on first turn when tool use is required
    if state.tool_required and state.turns == 1 and deps.repl is not None:
        try:
            tool_reg = deps.repl.tool_registry if hasattr(deps.repl, "tool_registry") else None
            if tool_reg is not None:
                grammar = tool_reg.generate_gbnf_grammar(str(role))
                if grammar:
                    llm_kwargs["grammar"] = grammar
                    state.grammar_enforced = True
        except Exception:
            pass  # Fall back to unconstrained generation

    # LLM call — stop at first code block close to prevent repetition loops.
    # REPL expects one action per turn; without this, the model can generate
    # FINAL("X") then repeat the same code block hundreds of tokens.
    # Early-stop streaming: abort generation the moment FINAL(...) or a
    # completed CALL(...) is detected.  For CALL, this lets the REPL
    # execute the tool and feed results back before the model continues.
    if deps.primitives is not None:
        deps.primitives._early_stop_check = lambda text: (
            bool(_FINAL_RE.search(text)) or bool(_CALL_STOP_RE.search(text))
        )
    llm_started = asyncio.get_event_loop().time()
    if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
        log.warning(
            "Frontdoor REPL turn start: task_id=%s turn=%d prompt_chars=%d n_tokens=%s tool_required=%s",
            state.task_id or "unknown",
            state.turns,
            len(prompt),
            llm_kwargs.get("n_tokens", "default"),
            state.tool_required,
        )
    _turn_call_meta: dict[str, Any] = {}
    try:
        llm_call_fn = deps.primitives.llm_call
        # TD-21.33a: capture this call's inference meta INSIDE the same thread the call
        # ran in (see `_call_llm_capturing_meta`) instead of reading ambient state back in
        # the parent after the `to_thread` hop.
        code, _turn_call_meta = await _call_llm_capturing_meta(
            llm_call_fn,
            deps.primitives,
            prompt,
            role=str(role),
            stop_sequences=["\n```\n"],
            skip_suffix=True,
            **llm_kwargs,
        )
    except ContextOverflowError as e:
        # The inference layer already did what it can without the graph
        # (bounded pool backoff, reroute to a larger-context role). Never retried
        # here and never swallowed — the turn fails with an explicit
        # context-overflow error. UFH14-B1: no forced session compaction here.
        # It compacted TaskState.context, which is never rendered into the turn
        # prompt, so it shrank nothing that overflowed while reporting "context
        # compacted; retry the turn" (and, with the LLM index, sent one more call
        # to the frontdoor server that had just overflowed). The next turn's
        # prompt drops the last output for this error; escalation to a
        # larger-context role is the error classifier's decision.
        log.warning(
            "Context overflow on turn %d (role=%s kind=%s n_prompt=%s n_ctx=%s)",
            state.turns, role, e.kind, e.n_prompt_tokens, e.n_ctx,
        )
        msg = f"LLM call failed: {e}"
        _record_session_turn(state, role=str(role), error=msg)
        return "", msg, False, {}
    except (InferenceError, ConnectionError, TimeoutError, OSError) as e:
        if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
            elapsed_ms = (asyncio.get_event_loop().time() - llm_started) * 1000
            log.warning(
                "Frontdoor REPL turn failure: task_id=%s turn=%d elapsed_ms=%.1f error=%s",
                state.task_id or "unknown",
                state.turns,
                elapsed_ms,
                e,
            )
        _record_session_turn(state, role=str(role), error=f"LLM call failed: {e}")
        return "", f"LLM call failed: {e}", False, {}
    except Exception as e:
        if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
            elapsed_ms = (asyncio.get_event_loop().time() - llm_started) * 1000
            log.warning(
                "Frontdoor REPL turn failure(unexpected): task_id=%s turn=%d elapsed_ms=%.1f error=%s",
                state.task_id or "unknown",
                state.turns,
                elapsed_ms,
                e,
            )
        _record_session_turn(state, role=str(role), error=f"LLM call failed (unexpected): {e}")
        return "", f"LLM call failed (unexpected): {e}", False, {}
    finally:
        if deps.primitives is not None:
            deps.primitives._early_stop_check = None

    if str(role) == str(Role.FRONTDOOR) and _frontdoor_trace_enabled():
        elapsed_ms = (asyncio.get_event_loop().time() - llm_started) * 1000
        log.warning(
            "Frontdoor REPL turn end: task_id=%s turn=%d elapsed_ms=%.1f raw_chars=%d infer_meta=%s",
            state.task_id or "unknown",
            state.turns,
            elapsed_ms,
            len(code),
            _turn_call_meta or "{}",
        )

    # Track aggregate completion tokens (Fast-RLM budget control)
    try:
        _meta = _turn_call_meta
        _completion_tokens = int(_meta.get("tokens", 0))
        _prompt_tokens = int(_meta.get("prompt_tokens", 0))
        if _completion_tokens > 0:
            state.aggregate_tokens += _completion_tokens
        # B5: Record tokens in session budget tracker if available
        if _get_features().session_token_budget:
            try:
                _stb = getattr(state, "_session_token_budget", None)
                if _stb is None:
                    from src.session_analytics import SessionTokenBudget
                    _stb = SessionTokenBudget.from_env()
                    state._session_token_budget = _stb  # type: ignore[attr-defined]
                _stb.record_tokens(prompt_tokens=_prompt_tokens, completion_tokens=_completion_tokens)
                _budget_status = _stb.check()
                if _budget_status.should_stop:
                    log.warning("B5 session token budget exhausted: %s", _budget_status.message)
            except Exception:
                log.debug("Session token budget check failed", exc_info=True)
    except Exception:
        log.debug("Token tracking failed", exc_info=True)

    # Save raw LLM output for FINAL() rescue before code extraction
    raw_llm_output = code
    # Observability-first (BEP OFF-arm debug): record exactly what the model emitted this turn,
    # before any extraction/rescue can alter it. Flag-gated default-off (no-op in production).
    _bep_turn_trace(state.turns, role, raw_llm_output, prompt=prompt,
                    repeat_count=getattr(state, "repl_noprogress_count", None))

    # Backend/infra denial masquerading as model output (INC-20260924-repl-
    # backend-error-masking): `llm_call` never raises for a placement/admission
    # timeout, a circuit-open breaker, or a dead backend connection -- it
    # returns the `[ERROR: ...]` sentinel as ordinary text (see
    # `_backend_infra_sentinel`). The model was never called, so this MUST end
    # the turn as an infrastructure failure here, before any of the "did the
    # model write code" heuristics below (comment-only nudge, prose rescue,
    # FINAL extraction) get a chance to read it as a turn the model produced.
    # No retry, no escalation to a different role: the denial is a fact about
    # a HOST resource (the CPU/GPU region another request holds), not a
    # capability gap a bigger model can out-think -- escalating just re-queues
    # the same 60s wait against the same held region (observed: escalation to
    # coder_escalation under GPU contention hit the identical wall). The
    # sentinel is returned verbatim as `error` (already `[ERROR: ...]`-prefixed)
    # so `_annotate_error` (`chat_pipeline/stages.py`) maps it to the correct
    # HTTP status (504 timeout, 503 unavailable, 429 admission, 502 backend)
    # instead of it being wrapped as `[FAILED: ...]` and read back as a generic
    # 500 -- see `_is_infra_failure` at each call site in `graph/nodes.py` /
    # `graph/langgraph/nodes.py`.
    _infra_sentinel = _backend_infra_sentinel(raw_llm_output)
    if _infra_sentinel is not None:
        log.warning(
            "REPL turn %d (role=%s) received an in-band backend/infra sentinel "
            "instead of model output -- ending as infra failure, no nudge: %s",
            state.turns, role, _infra_sentinel[:300],
        )
        _record_session_turn(state, role=str(role), error=_infra_sentinel)
        return "", _infra_sentinel, False, {"_infra_failure": True}

    # Reasoning length alarm (short-m@k Action 9): if <think> exceeds
    # 1.5× band budget, retry once with a conciseness nudge.
    if _check_reasoning_length_alarm(raw_llm_output, getattr(state, "difficulty_band", ""), _completion_tokens):
        if not getattr(state, "_alarm_retried", False):
            state._alarm_retried = True  # type: ignore[attr-defined]
            log.info(
                "Reasoning length alarm: completion_tokens=%d exceeds %.0f× budget for band=%s, retrying with conciseness nudge",
                _completion_tokens, _REASONING_LENGTH_ALARM_MULTIPLIER, state.difficulty_band,
            )
            _conciseness_nudge = (
                "\n\n[SYSTEM: Your reasoning was excessively long. "
                "Be concise — shorter reasoning chains are more accurate. "
                "Get to the answer directly.]"
            )
            _retry_prompt = prompt + _conciseness_nudge
            try:
                # TD-21.33a: same in-thread meta capture as the primary call above, so a
                # future reader of `_turn_call_meta` after the retry sees THIS retry's meta,
                # never the pre-retry call's (or nothing, or another request's).
                code, _turn_call_meta = await _call_llm_capturing_meta(
                    deps.primitives.llm_call, deps.primitives,
                    _retry_prompt, role=str(role), stop_sequences=["\n```\n"],
                    skip_suffix=True, **llm_kwargs,
                )
                raw_llm_output = code
            except Exception as e:
                log.warning("Reasoning length alarm retry failed: %s", e)

    # BEP (J8): flag-gated batched-edit divergence. No-op (returns None → fall through)
    # when batch_edit_mode is off or the model emitted no ```patchset block.
    _batch_edit_result = await _maybe_batch_edit_turn(ctx, role, raw_llm_output)
    if _batch_edit_result is not None:
        return _batch_edit_result

    # Extract and wrap code
    from src.prompt_builders import extract_code_from_response, auto_wrap_final

    # REPL loop-guard Fix A (flag-gated): close an unclosed ``` fence left by the FINAL/CALL
    # early-stop so the extractor can recover the block instead of returning nothing.
    if _repl_loop_guard_enabled():
        code = _repair_unclosed_code_fence(code)

    code = extract_code_from_response(code)
    code = auto_wrap_final(code)

    # REPL loop-guard Fix B: count consecutive no-progress turns (no file write / no FINAL).
    loop_guard_enabled = _repl_loop_guard_enabled()
    progress_code = _loop_guard_classification_view(code) if loop_guard_enabled else code
    if __import__("os").environ.get("ORCHESTRATOR_LOOPGUARD_PROBE") == "1":  # prod-safe diagnostic
        log.warning("LOOPGUARD-PROBE t=%s enabled=%s env=%r count=%s fws=%s final=%s",
                    getattr(state, "turns", "?"), loop_guard_enabled,
                    __import__("os").environ.get("ORCHESTRATOR_REPL_LOOP_GUARD"),
                    state.repl_noprogress_count, "file_write_safe" in progress_code,
                    "FINAL(" in progress_code)
    if loop_guard_enabled:
        _made_progress = ("file_write_safe" in progress_code) or ("FINAL(" in progress_code)
        state.repl_noprogress_count = _loop_guard_noprogress(_made_progress, state.repl_noprogress_count)

    # Persist extracted code for incremental editing on error/escalation
    state.last_code = code
    _persist_solution_file(state, code)

    _update_workspace_from_turn(state, role, raw_llm_output, None)

    # Prose answer rescue: model answered in prose (e.g. "The answer is D")
    # without producing FINAL() or code blocks.  Extract the answer from
    # the raw output and synthesize FINAL() to avoid an infinite REPL loop.
    if "FINAL(" not in code and _should_attempt_prose_rescue(raw_llm_output, code):
        prose_answer = _extract_prose_answer(raw_llm_output)
        if prose_answer is not None:
            log.info(
                "Prose answer rescue (turn %d): extracted %r from raw output",
                state.turns, prose_answer[:100],
            )
            code = f'FINAL("{prose_answer}")'

    # Comment-only guard: model reasoned in comments without executable code.
    # Try to rescue the answer from the comments before nudging.
    if _is_comment_only(code):
        comment_text = "\n".join(
            ln.strip().lstrip("#").strip()
            for ln in code.split("\n") if ln.strip().startswith("#")
        )
        prose_answer = _extract_prose_answer(comment_text)
        if prose_answer:
            log.info(
                "Comment-only rescue (turn %d): extracted %r from comments",
                state.turns, prose_answer[:50],
            )
            code = f'FINAL("{prose_answer}")'
            # Fall through to execute FINAL()
        else:
            log.info("Comment-only code detected (turn %d), nudging model", state.turns)
            nudge = _no_executable_code_nudge(state)
            _record_session_turn(state, role=str(role), code=code, nudge=nudge)
            return "", None, False, {"_nudge": nudge}

    # Comment-ratio guard: model is reasoning in comments with minimal
    # executable code (e.g. a bare `for` loop full of `# thinking...`).
    # This wastes turns without progress.  Nudge toward file-based workflow.
    code_lines = [ln for ln in code.split("\n") if ln.strip()]
    if code_lines:
        comment_lines = sum(1 for ln in code_lines if ln.strip().startswith("#"))
        ratio = comment_lines / len(code_lines)
        if ratio > 0.6 and len(code_lines) > 5 and "FINAL(" not in code:
            # Try to extract the answer the model already reasoned to
            comment_text = "\n".join(
                ln.strip().lstrip("#").strip()
                for ln in code_lines if ln.strip().startswith("#")
            )
            prose_answer = _extract_prose_answer(comment_text)
            if prose_answer:
                log.info(
                    "Comment-ratio rescue (turn %d, %.0f%% comments): extracted %r",
                    state.turns, ratio * 100, prose_answer[:50],
                )
                code = f'FINAL("{prose_answer}")'
                # Fall through to execute FINAL()
            else:
                log.info(
                    "High comment ratio (%.0f%%, turn %d), nudging to commit",
                    ratio * 100, state.turns,
                )
                nudge = _no_executable_code_nudge(state, comment_ratio=ratio)
                _record_session_turn(state, role=str(role), code=code, nudge=nudge)
                return "", None, False, {"_nudge": nudge}

    # Pre-REPL FINAL shortcut: if extracted code contains FINAL() mixed
    # with non-Python prose (common when code extraction pulls in markdown/
    # LaTeX), isolate just the FINAL line to avoid SyntaxError.
    if "FINAL(" in code:
        code_nontrivial = [
            ln for ln in code.split("\n")
            if ln.strip() and not ln.strip().startswith("#")
        ]
        final_lines = [ln for ln in code_nontrivial if "FINAL(" in ln]
        non_final_lines = [ln for ln in code_nontrivial if "FINAL(" not in ln]
        # If there are non-FINAL lines that look like prose (contain LaTeX
        # escapes, markdown bullets, or non-Python chars), discard them
        if final_lines and non_final_lines:
            suspect_count = sum(
                1 for ln in non_final_lines
                if ln.strip().startswith(("-", "*", ">"))
                or ln.strip().startswith("```")
                or "\\" in ln  # LaTeX escapes
                or any(c in ln for c in "λθπ≈∈∀∃")  # math Unicode
            )
            if suspect_count > 0 and suspect_count >= len(non_final_lines) * 0.5:
                log.info(
                    "Pre-REPL shortcut (turn %d): %d/%d non-FINAL lines look like prose, "
                    "isolating FINAL line",
                    state.turns, suspect_count, len(non_final_lines),
                )
                code = "\n".join(final_lines)

    # Capture exploration baseline for tool-call diffing in session log
    _exploration_baseline = 0
    try:
        if deps.repl is not None:
            elog = deps.repl.get_exploration_log()
            _exploration_baseline = len(elog.events)
    except Exception:
        pass

    # Write code to inference tap so the TUI shows what's being executed
    _tap_write_repl_exec(code, state.turns)

    # REPL execution
    try:
        repl_execute = deps.repl.execute
        if _use_inline_calls_in_tests() or type(repl_execute).__module__.startswith("unittest.mock"):
            result = repl_execute(code)
            if asyncio.iscoroutine(result):
                result = await result
        else:
            result = await asyncio.wait_for(
                asyncio.to_thread(repl_execute, code),
                timeout=deps.repl.config.timeout_seconds,
            )
    except asyncio.TimeoutError:
        from src.repl_environment.types import ExecutionResult

        result = ExecutionResult(
            output="",
            is_final=False,
            error=f"REPL execution timed out after {deps.repl.config.timeout_seconds}s",
        )

    # Track REPL execution count (Fast-RLM budget control)
    state.repl_executions += 1

    # Write execution result to inference tap
    _tap_write_repl_result(result.output, result.error, result.is_final, state.turns)

    # FINAL() rescue: if REPL execution failed but the model DID write
    # FINAL("answer") in its output, extract the answer directly.
    # This prevents escalation when code before FINAL() has errors.
    if not result.is_final and result.error:
        final_rescue = _extract_final_from_raw(raw_llm_output)
        if final_rescue is not None:
            log.info("FINAL() rescue: extracted %r from raw output (REPL error: %s)",
                     final_rescue[:100], result.error[:80])
            from src.repl_environment.types import ExecutionResult

            result = ExecutionResult(
                output="",
                is_final=True,
                final_answer=final_rescue,
            )

    # input() violation nudge: model wrote code using input() which is blocked.
    # The REPL error hint is too subtle — provide a concrete template so the
    # model knows exactly how to restructure for competitive programming tasks.
    if (
        not result.is_final
        and result.error
        and "input() is not available" in (result.error or "")
    ):
        nudge = (
            "STOP using input(). It is blocked in the REPL. For competitive programming:\n"
            '1. Put your ENTIRE solution in a triple-quoted string: solution = """\\nimport sys\\n'
            "input = sys.stdin.readline\\n...\\nprint(answer)\\n\"\"\"\n"
            '2. Test it: CALL("run_python_code", code=solution, stdin_data="<test input>")\n'
            "3. Submit it: FINAL(solution)\n"
            "Do NOT use bare input(). Wrap ALL code in a string variable."
        )
        _record_session_turn(state, role=str(role), code=code, error=result.error, nudge=nudge)
        return "", None, False, {"_nudge": nudge}

    # No-output guard: code ran successfully but produced no output, no error,
    # and no FINAL().  Typical case: model generated a class/function definition
    # that runs silently.  Without feedback the model repeats indefinitely.
    if not result.is_final and not result.error and not result.output:
        deferred_mode = bool(getattr(deps.repl, "_deferred_tool_results", False))
        tool_invocations = int(getattr(deps.repl, "_tool_invocations", 0))
        exploration_calls = int(getattr(deps.repl, "_exploration_calls", 0))
        tool_calls_observed = max(tool_invocations, exploration_calls)
        log.info("Silent execution detected (turn %d), nudging model", state.turns)
        if deferred_mode and tool_calls_observed > 0:
            nudge = (
                f"Your code called {tool_calls_observed} tool(s) but produced no output and did not call FINAL(). "
                "In deferred mode, tool results stay in variables unless you print them. "
                "Use print() to record key findings, then call FINAL() with the answer."
            )
        else:
            nudge = (
                "Your code ran but produced no output and did not call FINAL(). "
                "You must call FINAL with the actual computed value — e.g. FINAL(\"B\") or FINAL(42). "
                "If the task asks for code, call FINAL with the complete program text as a string."
            )
        _record_session_turn(
            state, role=str(role), code=code, nudge=nudge,
            tool_calls=_get_exploration_tool_calls(deps, _exploration_baseline),
        )
        return "", None, False, {"_nudge": nudge}

    # Status-message guard: model called FINAL() with a status phrase
    # instead of the actual answer (e.g. "Code execution complete.",
    # "Done", "Function implemented and tested successfully").  Reject and nudge.
    if result.is_final and hasattr(result, "final_answer") and result.final_answer:
        _fa = result.final_answer.strip().rstrip(".!").lower()
        _STATUS_PHRASES = {
            "code execution complete", "execution complete",
            "done", "complete", "completed", "implemented",
            "implementation complete", "finished", "success",
            "task complete", "task completed", "code complete",
            # Template placeholder echoes — model copied the example
            # instead of substituting the actual value
            "answer", "your answer", "your_answer",
            "your answer here", "your_answer_here",
            "result", "the answer", "the result",
            "your_computed_value", "your computed value",
            # Prompt-echo artifacts seen in seeding diagnostics
            "code", "explanation of code or reasoning",
            "code execution complete. check output",
        }
        # Keyword detection: catch longer status messages that aren't in the
        # exact set (e.g. "Function implemented and tested successfully").
        # Only flag if the answer has NO code-like content.
        _STATUS_KEYWORDS = {"implemented", "completed", "successfully", "finished", "executed"}
        _CODE_MARKERS = {"def ", "class ", "import ", "return ", "print(", "for ", "while ", "if ", "= "}
        _has_status_kw = any(kw in _fa for kw in _STATUS_KEYWORDS)
        _has_code = any(m in result.final_answer for m in _CODE_MARKERS)
        # MF-FIN-1: "done" is the sanctioned finish when this turn's prompt asked for it.
        _sanctioned_done = _edit_finish_sanctioned and _fa == "done"
        if not _sanctioned_done and (
            _fa in _STATUS_PHRASES or (_has_status_kw and not _has_code and len(_fa.split()) < 12)
        ):
            log.info("Status-message FINAL rejected (turn %d): %r", state.turns, result.final_answer)
            nudge = (
                f'FINAL("{result.final_answer}") is a status message, not an answer. '
                "FINAL must contain the actual answer or complete program text. "
                "If the task asks for code, call FINAL(your_code_as_string)."
            )
            _record_session_turn(state, role=str(role), code=code, nudge=nudge)
            return "", None, False, {"_nudge": nudge}

    artifacts = dict(deps.repl.artifacts) if hasattr(deps.repl, "artifacts") else {}
    # Prefer final_answer when is_final=True (FINAL() captures the answer
    # in final_answer, not in output)
    output = result.output
    if result.is_final and hasattr(result, "final_answer") and result.final_answer:
        output = result.final_answer
    log.debug(
        "_execute_turn: output=%r, error=%r, is_final=%s, code=%r",
        output[:200] if output else "",
        result.error[:200] if result.error else None,
        result.is_final,
        code[:200] if code else "",
    )
    _record_session_turn(
        state,
        role=str(role),
        code=code,
        output=output,
        error=result.error if result.error else None,
        is_final=result.is_final,
        tool_calls=_get_exploration_tool_calls(deps, _exploration_baseline),
    )
    return output, result.error if result.error else None, result.is_final, artifacts


MAX_CONSECUTIVE_NUDGES = 3
"""After this many nudges without progress, promote to a real error."""


def _repeated_nudge_failure(role: Any, nudge: Any) -> str:
    """Format a terminal failure after repeated corrective nudges."""
    detail = " ".join(str(nudge or "").split())
    if len(detail) > 300:
        detail = detail[:297] + "..."
    return f"[FAILED: repeated no-progress nudges at {role}: {detail}]"


def _detect_role_cycle(role_history: list[str]) -> bool:
    """Compatibility wrapper for extracted role-cycle detection."""
    return _detect_role_cycle_impl(role_history)


# ── FINAL() schema validation (Fast-RLM pattern) ─────────────────────────


def _render_schema_preamble(schema: dict) -> str:
    """Initial-prompt preamble shown to the agent when output_schema is set."""
    import json as _json
    return (
        "Your FINAL(...) call must pass a JSON-encoded string whose decoded value matches "
        "this JSON Schema:\n```json\n"
        + _json.dumps(schema, indent=2)
        + "\n```\nExample: FINAL(json.dumps(result_dict))"
    )


def _validate_final_answer(answer: str, schema: dict) -> tuple[bool, str | None, Any]:
    """Parse ``answer`` as JSON and validate against a JSON-Schema ``schema``.

    Accepts any JSON-Schema dict, including the output of
    ``pydantic.BaseModel.model_json_schema()`` and hand-written schemas.
    Returns (ok, error_message_or_None, parsed_value).
    """
    import json as _json
    try:
        parsed = _json.loads(answer)
    except (_json.JSONDecodeError, TypeError) as e:
        return False, f"FINAL value is not valid JSON: {e}", None
    try:
        import jsonschema  # type: ignore[import-untyped]
        jsonschema.validate(parsed, schema)
    except Exception as e:
        # jsonschema.ValidationError.message is the concise reason; .json_path
        # locates the offending field. Fall back to str(e) for other errors.
        path = getattr(e, "json_path", "") or ""
        msg = getattr(e, "message", "") or str(e)
        err = f"{msg} (at {path})" if path else msg
        return False, err, parsed
    return True, None, parsed


def _format_validation_failure_message(
    schema: dict, err: str, rejected: str, max_rejected_chars: int = 500
) -> str:
    """Retry-with-error message injected into the next-turn context."""
    import json as _json
    rejected_trunc = rejected[:max_rejected_chars]
    if len(rejected) > max_rejected_chars:
        rejected_trunc += f"...[truncated {len(rejected) - max_rejected_chars} chars]"
    return (
        "FINAL value failed schema validation.\n"
        f"Required JSON Schema:\n```json\n{_json.dumps(schema, indent=2)}\n```\n"
        f"Validation error: {err}\n"
        f"Rejected value: {rejected_trunc}\n"
        "Fix the value and call FINAL again. State is preserved."
    )
