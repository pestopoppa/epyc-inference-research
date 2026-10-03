"""Pipeline stage 10: REPL orchestration mode.

The largest pipeline stage — manages the multi-turn REPL loop with
escalation support, generation monitoring, two-stage summarization,
long-context exploration, and quality review gates.

The inner escalation loop is now driven by the pydantic-graph orchestration
graph (src.graph), replacing the manual for-loop. Bug fixes included:
- escalation_count is incremented on every escalation
- failure_graph.record_failure() is called on every error
- hypothesis_graph.add_evidence() is called on task outcomes
- Hardcoded EscalationPolicy() fallbacks are eliminated
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from datetime import datetime, timezone

from src.api.models import ChatRequest, ChatResponse
from src.api.routes.chat_pipeline.telemetry import (
    llm_completion_meta,
    work_completion_meta,
)
from src.api.services.memrl import failure_disposition_meta, score_completed_task
from src.graph import run_task, GraphConfig, TaskDeps, TaskState
from src.llm_primitives import LLMPrimitives
from src.constants import TOOL_OUTPUT_MATCH_LEN
from src.repl_environment import REPLEnvironment
from src.session.lease import HeldSessionLease, SessionLeaseManager
from src.session.models import Checkpoint
from src.session.protocol import normalize_checkpoint_for_repl_restore
from src.structured_output.repair import parse_with_repair, primitives_completer

from src.api.routes.chat_summarization import (
    _run_two_stage_summarization,
    _should_use_two_stage,
)
from src.api.routes.chat_utils import (
    LONG_CONTEXT_CONFIG,
    TWO_STAGE_CONFIG,
    RoutingResult,
    _strip_tool_outputs,
)
from src.api.structured_logging import task_extra

log = logging.getLogger(__name__)


# ── Stage 10: REPL orchestration mode ───────────────────────────────────


def _repl_work_meta(repl, answer: str) -> dict:
    """Build the M-11a2b `work` payload for a REPL-mode completion.

    Reads the two request-local trajectory records the REPL already keeps:

      * ``repl._invoked_tools`` — the request-local invocation records captured
        at the ``_invoke_tool`` chokepoint. Deliberately NOT
        ``repl.tool_registry.get_invocation_log()``, which is a process-global
        diagnostic ring shared by every concurrent request (and bounded, so it both
        leaks other requests' tools and silently evicts), see the note at the
        tools_called site below.
      * ``repl.get_code_log()`` — the bounded per-step code log
        (``REPLState.CODE_LOG_MAX_STEPS`` / ``CODE_LOG_MAX_CHARS``).

    Every read is guarded: work capture is observational and must never be the
    reason a completed request fails.
    """
    tool_calls: list = []
    repl_steps: list = []
    try:
        for inv in list(getattr(repl, "_invoked_tools", None) or []):
            tool_calls.append(
                {
                    # Same field names the tool_timings telemetry already uses,
                    # so there is one shape for "a tool ran", not two.
                    "tool_name": getattr(inv, "tool_name", None),
                    "elapsed_ms": getattr(inv, "elapsed_ms", None),
                    "success": getattr(inv, "success", None),
                }
            )
    except Exception:  # pragma: no cover - defensive
        tool_calls = []
    try:
        get_code_log = getattr(repl, "get_code_log", None)
        if callable(get_code_log):
            raw_steps = get_code_log()
            if isinstance(raw_steps, list):
                repl_steps = raw_steps
    except Exception:  # pragma: no cover - defensive
        repl_steps = []
    return work_completion_meta(
        answer=answer,
        tool_calls=tool_calls,
        repl_steps=repl_steps,
    )


def _tools_success(
    answer: str,
    tool_outputs: list,
    tool_invocations: int,
    invocation_log: list | None = None,
) -> bool | None:
    """Infer whether tool outputs influenced the final answer."""
    if tool_invocations <= 0:
        return None
    if not answer:
        return None

    if tool_outputs:
        answer_text = answer.lower()
        for output in tool_outputs:
            text = str(output).strip()
            if not text:
                continue
            if text[:TOOL_OUTPUT_MATCH_LEN].lower() in answer_text:
                return True
        return False

    # Deferred mode: _tool_outputs may be intentionally empty.
    if invocation_log:
        successes = [bool(getattr(inv, "success", False)) for inv in invocation_log]
        if successes:
            return any(successes)
    return False


def _build_delegation_diagnostics(
    role_history: list[str],
    delegation_events: list[dict],
    answer: str = "",
) -> dict:
    def _infer_break_reason() -> str:
        if delegation_events:
            for event in reversed(delegation_events):
                if not isinstance(event, dict):
                    continue
                reason = str(event.get("break_reason", "") or "").strip()
                if reason:
                    return reason
        text = (answer or "").lower()
        if not text:
            return ""
        if "before execution" in text or "pre-delegation" in text:
            return "pre_delegation_abort"
        if "lock timeout" in text:
            return "pre_delegation_lock_timeout"
        if "cancelled" in text or "canceled" in text:
            return "request_cancelled"
        if "deadline exceeded" in text:
            return "deadline_exceeded"
        if "timed out" in text or "timeout" in text:
            return "request_timeout"
        return ""

    edge_counts = Counter(
        (e.get("from_role", ""), e.get("to_role", ""))
        for e in delegation_events
        if isinstance(e, dict)
    )
    repeated_edges = {
        f"{src}->{dst}": n
        for (src, dst), n in edge_counts.items()
        if src and dst and n > 1
    }
    role_counts = Counter(role_history)
    repeated_roles = {r: n for r, n in role_counts.items() if n > 1}
    infer = [
        (e.get("inference_meta") or {})
        for e in delegation_events
        if isinstance(e, dict)
    ]
    infer = [m for m in infer if m]
    avg_prompt_ms = None
    avg_gen_ms = None
    if infer:
        p = [float(m.get("prompt_ms", 0) or 0) for m in infer]
        g = [float(m.get("gen_ms", 0) or 0) for m in infer]
        avg_prompt_ms = round(sum(p) / max(len(p), 1), 1)
        avg_gen_ms = round(sum(g) / max(len(g), 1), 1)
    loops = 0
    if delegation_events:
        try:
            loops = max(
                int(e.get("loop", 0) or 0)
                for e in delegation_events
                if isinstance(e, dict)
            )
        except (ValueError, TypeError):
            loops = len(delegation_events)
    return {
        "loops": int(loops),
        "phases_count": len(delegation_events),
        "break_reason": _infer_break_reason(),
        "cap_reached": False,
        "effective_max_loops": 0,
        "reentrant_depth": 0,
        "report_handles_count": 0,
        "report_handles": [],
        "delegation_inference_hops": len(infer),
        "avg_prompt_ms": avg_prompt_ms,
        "avg_gen_ms": avg_gen_ms,
        "role_chain_len": len(role_history),
        "delegation_events_count": len(delegation_events),
        "repeated_edges": repeated_edges,
        "repeated_roles": repeated_roles,
    }


def _build_tool_chain_summary(invocation_log: list, chain_exec_log: list[dict] | None = None) -> list[dict]:
    """Group chained tool invocations by chain_id for response diagnostics."""
    def _normalize_wave_timeline(entry: dict) -> list[dict]:
        """Return canonical wave-level chain telemetry for API responses."""
        raw = entry.get("wave_timeline", [])
        waves_list: list[dict] = []
        if isinstance(raw, list):
            for idx, item in enumerate(raw):
                if not isinstance(item, dict):
                    continue
                tools = item.get("tools", [])
                if not isinstance(tools, list):
                    tools = []
                waves_list.append(
                    {
                        "wave_index": int(item.get("wave_index", idx) or idx),
                        "tools": [str(t) for t in tools],
                        "mode_used": str(item.get("mode_used", entry.get("mode_used", "seq"))),
                        "elapsed_ms": item.get("elapsed_ms"),
                        "fallback_to_seq": bool(
                            item.get("fallback_to_seq", entry.get("fallback_to_seq", False))
                        ),
                        "parallel_mutations_enabled": bool(
                            item.get(
                                "parallel_mutations_enabled",
                                entry.get("parallel_mutations_enabled", False),
                            )
                        ),
                    }
                )

        if waves_list:
            return waves_list

        # Backward-compatible fallback when only wave count is available.
        if entry.get("waves", 0):
            return [
                {
                    "wave_index": 0,
                    "tools": [str(t) for t in (entry.get("tools", []) or [])],
                    "mode_used": str(entry.get("mode_used", "seq")),
                    "elapsed_ms": None,
                    "fallback_to_seq": bool(entry.get("fallback_to_seq", False)),
                    "parallel_mutations_enabled": bool(
                        entry.get("parallel_mutations_enabled", False)
                    ),
                }
            ]
        return []

    chains: dict[str, dict] = {}
    for inv in invocation_log:
        chain_id = getattr(inv, "chain_id", None)
        if not chain_id:
            continue
        entry = chains.setdefault(
            chain_id,
            {
                "chain_id": chain_id,
                "caller_type": getattr(inv, "caller_type", "chain"),
                "tools": [],
                "elapsed_ms": 0.0,
                "success": True,
            },
        )
        entry["tools"].append(getattr(inv, "tool_name", ""))
        entry["elapsed_ms"] += float(getattr(inv, "elapsed_ms", 0.0) or 0.0)
        entry["success"] = bool(entry["success"] and getattr(inv, "success", False))

    if chain_exec_log:
        for meta in chain_exec_log:
            if not isinstance(meta, dict):
                continue
            cid = str(meta.get("chain_id", "")).strip()
            if not cid:
                continue
            entry = chains.setdefault(
                cid,
                {
                    "chain_id": cid,
                    "caller_type": "chain",
                    "tools": [],
                    "elapsed_ms": 0.0,
                    "success": True,
                },
            )
            for k in (
                "mode_requested",
                "mode_used",
                "fallback_to_seq",
                "parallel_mutations_enabled",
                "waves",
                "steps",
                "wave_timeline",
            ):
                if k in meta:
                    entry[k] = meta[k]

    summaries = list(chains.values())
    for entry in summaries:
        entry["wave_timeline"] = _normalize_wave_timeline(entry)
    summaries.sort(key=lambda c: c["chain_id"])
    return summaries


async def _execute_repl(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
) -> ChatResponse:
    """Handle REPL orchestration mode (file exploration, tool access, large context).

    A request that carries ``session_id`` restores and then re-checkpoints that
    session's globals. uvicorn runs several worker processes, so two such
    requests can land on different workers; without cross-process ownership
    both restore the same checkpoint and the later save erases the earlier
    turn. The request therefore holds the session's SQLite lease (D-f) for its
    whole duration and fences its checkpoint write with the lease token.
    """
    session_id = getattr(request, "session_id", None)
    session_store = getattr(state, "session_store", None)
    manager = getattr(session_store, "leases", None) if session_id else None
    if not isinstance(manager, SessionLeaseManager):
        return await _execute_repl_body(
            request, routing, primitives, state, start_time, initial_role, None
        )
    async with HeldSessionLease(
        manager, session_id, label=f"chat:{routing.task_id}"
    ) as held:
        return await _execute_repl_body(
            request, routing, primitives, state, start_time, initial_role, held
        )


async def _execute_repl_body(
    request: ChatRequest,
    routing: RoutingResult,
    primitives: LLMPrimitives,
    state,
    start_time: float,
    initial_role,
    session_lease: HeldSessionLease | None,
) -> ChatResponse:
    """Body of :func:`_execute_repl`; ``session_lease`` is None when unleased."""
    task_id = routing.task_id
    routing_strategy = routing.routing_strategy
    formalization_applied = routing.formalization_applied

    # INF-78 OAB-7: a request-carried context bundle becomes the REPL variable `context`;
    # the ROOT prompt carries the caller's prompt plus the bundle's index only. The
    # ChatRequest validator already refused a malformed bundle (422).
    context_bundle = None
    root_prompt = request.prompt
    if getattr(request, "context_bundle", None) is not None:
        from src.repl_environment.context_bundle import ContextBundle

        context_bundle = ContextBundle.from_payload(
            request.context_bundle,
            print_cap_bytes=request.context_print_cap_bytes,
            pull_budget_bytes=request.context_pull_budget_bytes,
            search_enabled=bool(getattr(request, "context_search", False)),
        )
        root_prompt = f"{request.prompt}\n\n{context_bundle.render_root_block()}"

    # Create REPL environment — use DocumentREPLEnvironment when document
    # preprocessing produced structured results (sections, figures, etc.)
    combined_context = root_prompt
    if request.context:
        combined_context += f"\n\nContext:\n{request.context}"

    # Inject SkillBank skill context if available (SkillRL §3.2)
    if routing.skill_context:
        combined_context = f"{routing.skill_context}\n\n{combined_context}"

    # Derive tool context for cascading policy (WS-3: deny web for reasoning domains)
    from src.tool_policy import NO_WEB_TASK_TYPES

    _task_type = str(routing.task_ir.get("task_type", "chat"))
    _tool_context = {"no_web": True} if _task_type in NO_WEB_TASK_TYPES else {}

    if routing.document_result and routing.document_result.document_result:
        from src.repl_document import DocumentREPLEnvironment, DocumentContext

        doc_result = routing.document_result.document_result
        doc_context = DocumentContext.from_document_result(doc_result)

        # Use searchable text as REPL context, prepend the user's question
        doc_text = doc_result.to_searchable_text()
        combined_context = f"Question: {request.prompt}\n\n{doc_text}"

        repl = DocumentREPLEnvironment(
            context=combined_context,
            document_context=doc_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=initial_role,
            progress_logger=state.progress_logger,
            task_id=task_id,
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
            tool_context=_tool_context,
        )
        log.info(
            "Using DocumentREPLEnvironment: %d sections, %d figures",
            len(doc_context.sections),
            len(doc_context.figures),
            extra=task_extra(task_id=task_id, stage="execute", mode="repl_document"),
        )
    else:
        repl = REPLEnvironment(
            context=combined_context,
            llm_primitives=primitives,
            tool_registry=state.tool_registry,
            script_registry=state.script_registry,
            role=initial_role,
            progress_logger=state.progress_logger,
            task_id=task_id,
            retriever=state.hybrid_router.retriever if state.hybrid_router else None,
            hybrid_router=state.hybrid_router,
            tool_context=_tool_context,
        )
    if context_bundle is not None:
        repl.attach_context_bundle(context_bundle)
        log.info(
            "Context bundle attached: %d sections, %d bytes (%d in the prompt), print cap %d B",
            len(context_bundle.sections),
            context_bundle.total_bytes,
            sum(s.nbytes for s in context_bundle.sections if s.inline),
            context_bundle.print_cap_bytes,
            extra=task_extra(task_id=task_id, stage="execute", mode="repl_context_bundle"),
        )

    # Phase 3: cross-request globals restore (opt-in via session_id).
    session_id = getattr(request, "session_id", None)
    session_store = state.session_store if hasattr(state, "session_store") else None
    session_persistence: dict[str, object] = {
        "session_id": session_id,
        "restore_attempted": bool(session_id and session_store),
        "restore_found_checkpoint": False,
        "restore_success": False,
        "restored_globals": 0,
        "claimed_globals": 0,
        "unavailable_globals": {},
        "skipped_globals": [],
        "restore_error": None,
        "checkpoint_saved": False,
        "checkpoint_id": None,
        "saved_globals": 0,
        "save_error": None,
        "restore_protocol": {},
        "save_protocol_version": None,
        "lease_token": None,
        "lease_error": None,
        "lease_lost": False,
    }
    lease_blocked = False
    if session_lease is not None:
        session_persistence["lease_token"] = session_lease.token
        if session_lease.error:
            # Another live owner kept the session past the wait budget. Neither
            # restore (its checkpoint is about to be superseded) nor save (the
            # write would be refused) is honest, so the turn runs stateless and
            # says so.
            lease_blocked = True
            session_persistence["lease_error"] = session_lease.error
            session_persistence["restore_error"] = "session_lease_unavailable"
            session_persistence["save_error"] = "session_lease_unavailable"
    if session_id and session_store and not lease_blocked:
        try:
            checkpoint = session_store.get_latest_checkpoint(session_id)
            if checkpoint:
                session_persistence["restore_found_checkpoint"] = True
            if checkpoint:
                restore_payload, protocol_diag = normalize_checkpoint_for_repl_restore(
                    checkpoint.to_dict()
                )
                session_persistence["restore_protocol"] = protocol_diag
                reconciliation = repl.restore(restore_payload) or {}
                session_persistence["restore_success"] = True
                # Report what ACTUALLY landed in the live namespace, not what the
                # payload claimed — a name can be dropped at restore (builtin
                # collision) and the claimed count would never show it.
                session_persistence["restored_globals"] = len(
                    reconciliation.get("restored", [])
                )
                session_persistence["claimed_globals"] = reconciliation.get(
                    "claimed", len(restore_payload.get("user_globals", {}))
                )
                session_persistence["unavailable_globals"] = dict(
                    reconciliation.get("unavailable", {})
                )
                session_persistence["skipped_globals"] = list(
                    restore_payload.get("skipped_user_globals", []) or []
                )
                log.info(
                    "Restored %d globals from session %s",
                    session_persistence.get("restored_globals", 0),
                    session_id[:8],
                    extra=task_extra(task_id=task_id, stage="execute", mode="repl_restore"),
                )
        except Exception as e:
            session_persistence["restore_error"] = str(e)
            log.warning(
                "Session globals restore failed for %s: %s",
                session_id[:8],
                e,
                extra=task_extra(task_id=task_id, stage="execute", mode="repl_restore"),
            )

    # Check for two-stage summarization opportunity (never for a bundle: its REPL is the
    # point, and two-stage would answer without it)
    if context_bundle is None and request.real_mode and _should_use_two_stage(
        prompt=request.prompt,
        context=request.context,
    ):
        try:
            answer, two_stage_stats = await _run_two_stage_summarization(
                prompt=request.prompt,
                context=request.context or "",
                primitives=primitives,
                state=state,
                task_id=task_id,
            )

            elapsed = time.perf_counter() - start_time
            state.increment_request(mock_mode=False, turns=2)
            actual_role_history = [
                str(role_name)
                for role_name in two_stage_stats.get("role_history", [])
                if role_name
            ]
            producer_role = str(
                two_stage_stats.get("producer_role")
                or (
                    actual_role_history[-1]
                    if actual_role_history
                    else TWO_STAGE_CONFIG["stage1_role"]
                )
            )
            reported_role_history = []
            for role_name in [str(initial_role), *actual_role_history]:
                if not reported_role_history or reported_role_history[-1] != role_name:
                    reported_role_history.append(role_name)

            if state.progress_logger:
                cache_info = "cache hit" if two_stage_stats.get("cache_hit") else "cache miss"
                state.progress_logger.log_task_completed(
                    task_id=task_id,
                    success=True,
                    details=f"Two-stage summarization ({cache_info}), {elapsed:.3f}s",
                    completion_meta={
                        "producer_role": producer_role,
                        "delegation_lineage": reported_role_history,
                        "final_answer_role": producer_role,
                        **llm_completion_meta(primitives),
                    },
                )
                score_completed_task(
                    state,
                    task_id,
                    force_role=request.force_role,
                    real_mode=request.real_mode,
                )

            cache_stats = primitives.get_cache_stats() if primitives._backends else None

            return ChatResponse(
                answer=answer,
                turns=2,
                tokens_used=primitives.total_tokens_generated,
                elapsed_seconds=elapsed,
                mock_mode=False,
                real_mode=True,
                cache_stats=cache_stats,
                routed_to=str(initial_role),
                role_history=reported_role_history,
                routing_strategy=routing_strategy,
                mode="repl",
                tokens_generated=primitives.total_tokens_generated,
                formalization_applied=formalization_applied,
                tools_used=0,
                prompt_eval_ms=primitives.total_prompt_eval_ms,
                generation_ms=primitives.total_generation_ms,
                predicted_tps=primitives._last_predicted_tps,
                http_overhead_ms=primitives.total_http_overhead_ms,
                skills_retrieved=len(routing.skill_ids) if routing.skill_ids else None,
                skill_ids=routing.skill_ids,
            )
        except Exception as e:
            logging.warning(f"Two-stage summarization failed: {type(e).__name__}: {e}")

    # Detect long context → use REPL-based exploration strategy
    context_chars = len(combined_context)
    use_long_context_exploration = (
        LONG_CONTEXT_CONFIG["enabled"]
        and request.real_mode
        and context_chars > LONG_CONTEXT_CONFIG["threshold_chars"]
    )

    if use_long_context_exploration:
        logging.info(
            f"Long context detected ({context_chars:,} chars). Using REPL exploration strategy."
        )

    max_turns = (
        LONG_CONTEXT_CONFIG["max_turns"] if use_long_context_exploration else request.max_turns
    )
    if getattr(request, "task_root", None):
        # INF-78 OAB-1 / R3: a task_root-scoped agentic caller declares its own turn budget
        # (up to ChatRequest.MAX_TURNS_SCOPED). Its 35-75k-char prompt always crosses the
        # long-context threshold, whose 8-turn cap would otherwise silently replace it.
        max_turns = request.max_turns

    # ── Run orchestration graph ──────────────────────────────────────
    # The graph replaces the manual for-loop with typed node transitions.
    # Bug fixes: escalation_count incremented, failure_graph.record_failure()
    # called, hypothesis_graph.add_evidence() called.

    graph_config = GraphConfig(max_turns=max_turns)
    try:
        graph_config = GraphConfig.from_config()
        graph_config.max_turns = max_turns
    except Exception:
        pass  # Use defaults if config unavailable

    # FINAL() schema validation (Fast-RLM pattern). Inject schema preamble into
    # the initial REPL context when the feature is enabled AND caller provided
    # an output_schema. Validation happens after run_task returns below.
    from src.features import features as _get_features
    from src.graph.helpers import (
        _render_schema_preamble,
        _validate_final_answer,
        _format_validation_failure_message,
    )
    _schema = (
        request.output_schema
        if _get_features().final_schema_validation
        else None
    )
    if _schema:
        _schema_preamble = _render_schema_preamble(_schema)
        combined_context = f"{_schema_preamble}\n\n{combined_context}"
        # The turn prompt is rebuilt from TaskState.PROMPT every turn
        # (graph/helpers._execute_turn); TaskState.context is never rendered into
        # it, so a preamble placed only there never reached the model.
        root_prompt = f"{root_prompt}\n\n{_schema_preamble}"

    task_state = TaskState(
        task_id=task_id,
        prompt=root_prompt,
        context=combined_context,
        task_ir=routing.task_ir,
        task_type=str(routing.task_ir.get("task_type", "chat")),
        current_role=initial_role,
        role_history=[str(initial_role)],
        max_turns=max_turns,
        tool_required=routing.tool_required,
        tool_hint=routing.tool_hint,
        difficulty_band=routing.difficulty_band,
    )
    # UFH14-B1 F2: wall-time forced-answer point (None unless repl_answer_force
    # is on and the request has a deadline).
    from src.graph.helpers import answer_force_at as _answer_force_at

    task_state.answer_force_at_s = _answer_force_at(primitives)

    task_deps = TaskDeps(
        primitives=primitives,
        repl=repl,
        failure_graph=state.failure_graph if hasattr(state, "failure_graph") else None,
        hypothesis_graph=state.hypothesis_graph if hasattr(state, "hypothesis_graph") else None,
        config=graph_config,
        progress_logger=state.progress_logger,
        session_store=state.session_store if hasattr(state, "session_store") else None,
        session_id=session_id if (session_lease is not None and not lease_blocked) else None,
        session_fencing_token=(
            session_lease.token if (session_lease is not None and not lease_blocked) else None
        ),
    )

    # Up to 2 attempts: initial run + one retry-with-error if schema validation
    # (and the TD-21.1 repair turn below) both fail. Shared REPLEnvironment via
    # task_deps preserves agent state across retries; repl_executions budget
    # naturally bounds total work.
    #
    # TD-21.1 (typed-decision-plane.md): before paying for a full graph
    # re-run (turns=0, role reset, the whole run_task graph re-entered), spend
    # ONE constrained repair turn on the model's own FINAL text via the
    # shared src/structured_output/repair.py helper. fish_json alone often
    # recovers a value wrapped in stray prose/fences with ZERO extra calls;
    # only a genuine miss spends the one extraction turn. Only if that repair
    # also fails do we fall back to the pre-existing retry-from-zero path.
    # `_schema_valid`/`_schema_invalid_reason` are carried past the loop so
    # the final response can be flagged instead of silently served as valid
    # (see the terminal check right before the ChatResponse is built).
    _max_validation_attempts = 2 if _schema else 1
    graph_result = None
    _schema_valid: bool | None = None
    _schema_invalid_reason: str | None = None
    # Each schema retry re-enters the graph with task_state.turns reset to 0 (a fresh turn
    # budget), so graph_result.turns only counts the LAST attempt. Sum them for the reported
    # `turns` (INF-78 OAB-2 found retried calls undercounted).
    _turns_all_attempts = 0
    for _attempt in range(_max_validation_attempts):
        graph_result = await run_task(task_state, task_deps, start_role=initial_role)
        _turns_all_attempts += int(getattr(graph_result, "turns", 0) or 0)
        if not _schema or not graph_result.answer:
            break
        _ok, _err, _ = _validate_final_answer(graph_result.answer, _schema)
        if not _ok:
            _producing_role = (
                str(graph_result.role_history[-1])
                if graph_result.role_history
                else str(initial_role)
            )
            _repair = parse_with_repair(
                graph_result.answer,
                schema=_schema,
                complete=primitives_completer(primitives, _producing_role),
                site="repl_final",
                # TD-21.34: `_schema` is the CALLER's own output_schema for the
                # FINAL answer -- a missing required number/short string must
                # not be invented to satisfy the grammar (the exact live-smoke
                # failure mode this program hardens against).
                require_evidence=True,
            )
            if _repair.status in ("parsed", "repaired"):
                graph_result.answer = json.dumps(_repair.value)
                _ok, _err = True, None
            else:
                _err = _repair.reason or _err
        if _ok:
            _schema_valid = True
            break
        _schema_valid = False
        _schema_invalid_reason = _err or "(unknown)"
        if _attempt + 1 < _max_validation_attempts:
            _failure_msg = _format_validation_failure_message(
                _schema, _schema_invalid_reason, graph_result.answer
            )
            task_state.context = f"{combined_context}\n\n{_failure_msg}"
            # ...and the retry must SEE why the last FINAL failed (TaskState.prompt is
            # what the next turn renders; see the preamble note above).
            task_state.prompt = f"{root_prompt}\n\n{_failure_msg}"
            task_state.turns = 0
            task_state.current_role = initial_role
            task_state.role_history = [str(initial_role)]

    answer = graph_result.answer
    turns = max(int(graph_result.turns or 0), _turns_all_attempts)
    role_history = graph_result.role_history or [str(initial_role)]
    current_role = role_history[-1] if role_history else str(initial_role)
    delegation_events = graph_result.delegation_events

    # ── Post-graph processing ────────────────────────────────────────

    # If max turns reached without FINAL() and graph returned empty
    if not answer:
        # Try rescue: extract answer from last LLM output before generating error
        last_output = task_state.last_output
        if last_output:
            from src.graph.helpers import _rescue_from_last_output

            rescued = _rescue_from_last_output(last_output)
            if rescued:
                log.info("Post-graph rescue: %r", rescued[:100])
                answer = rescued

    if not answer:
        last_output = task_state.last_output
        tool_outputs = repl.artifacts.get("_tool_outputs", [])
        cleaned_output = _strip_tool_outputs(last_output, tool_outputs) if last_output else ""

        if cleaned_output and len(cleaned_output) > 20:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]\n\n{cleaned_output}"
        elif last_output:
            answer = (
                f"[Max turns ({max_turns}) reached without FINAL()]\n\nLast output:\n{last_output}"
            )
        else:
            answer = f"[Max turns ({max_turns}) reached without FINAL()]"

    elapsed = time.perf_counter() - start_time
    state.increment_request(mock_mode=False, turns=turns)

    cache_stats = None
    if request.real_mode and primitives._backends:
        cache_stats = primitives.get_cache_stats()

    success = not answer.startswith("[ERROR") and not answer.startswith("[Max turns")
    repl.log_exploration_completed(success=success, result=answer)

    if state.progress_logger:
        role_info = f", roles: {' -> '.join(role_history)}" if len(role_history) > 1 else ""
        state.progress_logger.log_task_completed(
            task_id=task_id,
            success=success,
            details=f"Real inference: {turns} turns, {elapsed:.3f}s{role_info}",
            completion_meta={
                "producer_role": current_role,
                "delegation_lineage": role_history,
                "final_answer_role": current_role,
                "workspace_version": (task_state.workspace_state or {}).get("version", 0),
                "workspace_decisions": len((task_state.workspace_state or {}).get("decisions", [])),
                **llm_completion_meta(primitives),
                # M-11a2b. This is the richest work the pipeline produces — the
                # trajectory a distilled skill would have to be learned FROM —
                # and none of it reached the episodic store before.
                **_repl_work_meta(repl, answer),
                # `success` here is False for an in-band `[ERROR: ...]` (infra)
                # AND for `[Max turns ...]` (a genuine failure to solve the
                # task). Only the first is stamped, so the router keeps learning
                # from the second.
                **failure_disposition_meta(
                    answer=answer,
                    tokens_generated=primitives.total_tokens_generated,
                ),
            },
        )
        score_completed_task(
            state,
            task_id,
            force_role=request.force_role,
            real_mode=request.real_mode,
        )

    # REQUEST-LOCAL invocation records captured at the _invoke_tool chokepoint
    # (context.py). We must NOT use repl.tool_registry.get_invocation_log() here:
    # that registry is process-global (one instance, src/api/__init__.py) and its
    # ring is shared by every concurrently served request, so this request would
    # report other requests' tools (and the ring is bounded, so it also evicts). These records expose the same attribute interface as
    # ToolInvocation (tool_name/elapsed_ms/success/chain_id/caller_type/result),
    # so every consumer below is unchanged.
    invocation_log = list(getattr(repl, "_invoked_tools", None) or [])
    tool_outputs = repl.artifacts.get("_tool_outputs", [])
    tools_success = _tools_success(
        answer,
        tool_outputs,
        repl._tool_invocations,
        invocation_log=invocation_log,
    )
    delegation_success = None
    if delegation_events:
        delegation_success = any(e.get("success") for e in delegation_events)
    delegation_diag = _build_delegation_diagnostics(
        role_history,
        delegation_events,
        answer=answer,
    )

    # invocation_log is already request-local (see above), so tools_called/
    # tool_timings/tools_used are too — a no-tool request reports nothing.
    tools_called = [inv.tool_name for inv in invocation_log]
    tool_timings = [
        {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success}
        for inv in invocation_log
    ]
    chain_exec_log = []
    if hasattr(repl, "get_chain_execution_log"):
        try:
            chain_exec_log = repl.get_chain_execution_log()
        except Exception:
            chain_exec_log = []
    tool_chains = _build_tool_chain_summary(invocation_log, chain_exec_log=chain_exec_log)
    tools_used = max(repl._tool_invocations, len(tools_called), len(tool_timings))

    # Detect parallel tool usage from invocation log
    parallel_tools = False
    if len(tools_called) >= 2:
        read_only = set()
        if hasattr(repl, "_get_read_only_tools"):
            try:
                read_only = repl._get_read_only_tools()
            except Exception:
                read_only = set()
        if not read_only and repl.tool_registry and hasattr(repl.tool_registry, "get_read_only_tools"):
            try:
                read_only = set(repl.tool_registry.get_read_only_tools())
            except Exception:
                read_only = set()
        parallel_tools = all(t in read_only for t in tools_called) and len(tools_called) >= 2

    if session_id and session_store and not lease_blocked:
        try:
            session = session_store.get_session(session_id)
            if session:
                repl_checkpoint = repl.checkpoint()
                checkpoint = Checkpoint(
                    id=str(uuid.uuid4()),
                    session_id=session_id,
                    created_at=datetime.now(timezone.utc),
                    context_hash="sha256:chat_pipeline",
                    artifacts=repl_checkpoint.get("artifacts", {}),
                    execution_count=repl_checkpoint.get("execution_count", 0),
                    exploration_calls=repl_checkpoint.get("exploration_calls", 0),
                    message_count=turns,
                    trigger="chat_request",
                    user_globals=repl_checkpoint.get("user_globals", {}),
                    variable_lineage=repl_checkpoint.get("variable_lineage", {}),
                    skipped_user_globals=repl_checkpoint.get("skipped_user_globals", []),
                    protocol_version=1,
                )
                if session_lease is not None:
                    session_store.save_checkpoint(
                        checkpoint, fencing_token=session_lease.token
                    )
                else:
                    session_store.save_checkpoint(checkpoint)
                session_persistence["checkpoint_saved"] = True
                session_persistence["checkpoint_id"] = checkpoint.id
                session_persistence["saved_globals"] = len(checkpoint.user_globals)
                session_persistence["save_protocol_version"] = checkpoint.protocol_version
            else:
                session_persistence["save_error"] = "session_not_found"
        except Exception as e:
            session_persistence["save_error"] = str(e)
            if session_lease is not None:
                session_persistence["lease_lost"] = session_lease.lost
            log.warning(
                "Session checkpoint save failed for %s: %s",
                session_id[:8],
                e,
                extra=task_extra(task_id=task_id, stage="execute", mode="repl_checkpoint"),
            )

    # Extract web_research tool results for Search-R1 reward pipeline
    web_research_results = []
    for inv in invocation_log:
        if inv.tool_name == "web_research" and inv.success and isinstance(getattr(inv, "result", None), dict):
            wr = inv.result
            web_research_results.append({
                "success": wr.get("success", True),
                "error": wr.get("error", ""),
                "no_results_reason": wr.get("no_results_reason", ""),
                "query": wr.get("query", ""),
                "search_backend": wr.get("search_backend", ""),
                "search_result_count": wr.get("search_result_count", 0),
                "pages_fetched": wr.get("pages_fetched", 0),
                "pages_synthesized": wr.get("pages_synthesized", 0),
                "pages_irrelevant": wr.get("pages_irrelevant", 0),
                "irrelevant_rate": wr.get("irrelevant_rate", 0.0),
                "total_elapsed_ms": wr.get("total_elapsed_ms", 0.0),
                "sources": [
                    {
                        "url": s.get("url", ""),
                        "title": s.get("title", ""),
                        **({"relevant": s["relevant"]} if "relevant" in s else {}),
                    }
                    for s in wr.get("sources", [])
                    if isinstance(s, dict)
                ],
            })

    # Extract scratchpad insights from task_state (Search-R1 Step 5)
    scratchpad_insights = [
        {"turn": e.turn, "category": e.category, "insight": e.insight, "confidence": e.confidence}
        for e in task_state.scratchpad_entries
        if hasattr(e, "category")
    ]

    # TD-21.1 terminal fix: retries (plus the repair turn above) are exhausted
    # and the FINAL() value STILL does not validate against request.output_schema.
    # Before this fix the invalid value was returned as ChatResponse with no
    # signal at all (error_code/error_detail both None => HTTP 200, caller has
    # no way to tell the answer never validated). Now it is flagged the same
    # way every other structured-validation failure in this pipeline is
    # flagged (see chat.py's plan_review_drop, error_code=422): the answer
    # text is unchanged (still the model's best/last attempt, useful to a
    # caller doing its own recovery) but error_code=422 + error_detail carry
    # the schema-validation failure so the HTTP layer (chat.py's `_finalize`/
    # `response.error_code` branch) returns a non-200 status instead of a
    # silent success.
    _schema_error_code = 422 if (_schema and _schema_valid is False) else None

    # INF-78 OAB-7 / OAB-12: exact pull accounting for the bundle (echo + one log line).
    context_pulls = None
    if context_bundle is not None:
        context_pulls = context_bundle.accounting()
        _pt = context_pulls["totals"]
        log.info(
            "Context bundle pulls: %d calls, %d/%d bytes pulled/offered (%d unique), "
            "%d printed -> %d shown, %d capped turns, %d refused",
            _pt["pull_calls"],
            _pt["bytes_pulled"],
            context_pulls["offered"]["bytes"],
            _pt["unique_bytes"],
            _pt["printed_bytes"],
            _pt["shown_bytes"],
            _pt["turns_capped"],
            _pt["refused"],
            extra=task_extra(task_id=task_id, stage="execute", mode="repl_context_bundle"),
        )
    _schema_error_detail = (
        f"FINAL() value failed output_schema validation after retry and repair: "
        f"{_schema_invalid_reason}"
        if _schema_error_code
        else None
    )

    return ChatResponse(
        answer=answer,
        turns=turns,
        tokens_used=primitives.total_tokens_generated,
        elapsed_seconds=elapsed,
        mock_mode=False,
        real_mode=request.real_mode,
        cache_stats=cache_stats,
        routed_to=str(current_role),
        role_history=[str(r) for r in role_history],
        routing_strategy=routing_strategy,
        mode="repl",
        tokens_generated=primitives.total_tokens_generated,
        formalization_applied=formalization_applied,
        error_code=_schema_error_code,
        error_detail=_schema_error_detail,
        tools_used=tools_used,
        tools_called=tools_called,
        tool_timings=tool_timings,
        tool_chains=tool_chains,
        delegation_events=delegation_events,
        delegation_diagnostics=delegation_diag,
        tools_success=tools_success,
        delegation_success=delegation_success,
        prompt_eval_ms=primitives.total_prompt_eval_ms,
        generation_ms=primitives.total_generation_ms,
        predicted_tps=primitives._last_predicted_tps,
        http_overhead_ms=primitives.total_http_overhead_ms,
        # Orchestrator intelligence diagnostics
        think_harder_attempted=task_state.think_harder_attempted,
        think_harder_succeeded=task_state.think_harder_succeeded,
        think_harder_expected_roi=float(task_state.artifacts.get("think_harder_expected_roi", 0.0)),
        grammar_enforced=task_state.grammar_enforced,
        parallel_tools_used=parallel_tools,
        cache_affinity_bonus=task_state.cache_affinity_bonus,
        # SkillBank integration
        skills_retrieved=len(routing.skill_ids) if routing.skill_ids else None,
        skill_ids=routing.skill_ids,
        session_persistence=session_persistence,
        # Context window management (C1/C3)
        compaction_triggered=task_state.compaction_count > 0,
        compaction_tokens_saved=task_state.compaction_tokens_saved,
        # Tool output compression metrics
        compression_metrics=task_state.compression_metrics,
        # Web research telemetry (Search-R1)
        web_research_results=web_research_results,
        # Scratchpad insights (Search-R1 Step 5)
        scratchpad_insights=scratchpad_insights,
        # INF-78 OAB-7: bundle pull accounting (None without a bundle)
        context_pulls=context_pulls,
    )
