"""Context management, completion signals, and tool dispatch.

Provides mixin with: context_len, chunk_context, summarize_chunks,
FINAL/FINAL_VAR, mark_finding, list_findings, get/clear_findings,
tracked LLM calls, and tool/script invocation.
"""

from __future__ import annotations

import logging
from typing import Any

from src.repl_environment.types import FinalSignal

log = logging.getLogger(__name__)

_MISSING = object()


class _ContextMixin:
    """Mixin providing context management and tool dispatch.

    Includes: _context_len, _chunk_context, _summarize_chunks, _final, _final_var,
    _mark_finding, _list_findings, get_findings, clear_findings, _tracked_llm_call,
    _tracked_llm_batch, _invoke_tool, _call_tool, _list_tools, _invoke_script, _find_scripts.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _execution_count: int — number of execute() calls
        _findings_buffer: list — key findings buffer for session persistence
        _tool_invocations: int — tool invocation counter
        llm_primitives: Any | None — LLM primitives for delegation
        tool_registry: Any | None — registry of available tools
        script_registry: Any | None — registry of prepared scripts
        role: str — current agent role
        _globals: dict — restricted globals for REPL execution
    """

    # ------------------------------------------------------------------
    # Long context exploration tools
    # ------------------------------------------------------------------

    def _context_len(self) -> int:
        """Return the character length of the context.

        Returns:
            Number of characters in the context string.
        """
        bundle = getattr(self, "_context_bundle", None)
        if bundle is not None:  # INF-78 OAB-7: the bundle's size (metadata, not a pull)
            return bundle.total_chars
        return len(self.context)

    def _chunk_context(self, n_chunks: int = 4, overlap: int = 200) -> list[dict]:
        """Split context into roughly equal chunks with metadata.

        Args:
            n_chunks: Number of chunks to split into (default 4).
            overlap: Characters of overlap between chunks (default 200).

        Returns:
            List of dicts with index, start, end, text, char_count.
        """
        self._exploration_calls += 1
        bundle = getattr(self, "_context_bundle", None)
        text = bundle.full_text() if bundle is not None else self.context
        total = len(text)
        if total == 0:
            return []

        n_chunks = max(1, min(n_chunks, 20))  # Cap at 20
        chunk_size = total // n_chunks
        chunks = []

        for i in range(n_chunks):
            start = max(0, i * chunk_size - (overlap if i > 0 else 0))
            end = min(total, (i + 1) * chunk_size + (overlap if i < n_chunks - 1 else 0))
            # Avoid splitting mid-word: extend to next whitespace
            if end < total:
                ws = text.find("\n", end)
                if ws != -1 and ws - end < 200:
                    end = ws

            chunk_text = text[start:end]
            if bundle is not None:  # INF-78 OAB-7: every chunk is a counted pull
                bundle.span_pull(start, end, op="chunk_context")
            chunks.append(
                {
                    "index": i,
                    "start": start,
                    "end": end,
                    "text": chunk_text,
                    "char_count": len(chunk_text),
                }
            )

        self._exploration_log.add_event(
            "chunk_context",
            {"n_chunks": n_chunks, "overlap": overlap, "total_chars": total},
            f"{len(chunks)} chunks",
        )
        return chunks

    def _summarize_chunks(
        self,
        task: str = "Summarize the key content",
        n_chunks: int = 4,
        role: str = "worker_general",
    ) -> list[dict]:
        """Chunk context and summarize each chunk in parallel using workers.

        Args:
            task: The task/question to apply to each chunk.
            n_chunks: Number of chunks (default 4).
            role: Worker role to use (default "worker_general").

        Returns:
            List of dicts with index, chunk_start, chunk_end, chunk_chars, summary.
        """
        self._exploration_calls += 1

        if self.llm_primitives is None:
            return [{"error": "LLM primitives not available"}]

        chunks = self._chunk_context(n_chunks)
        if not chunks:
            return [{"error": "Empty context"}]

        # Build prompts for each chunk
        prompts = []
        for chunk in chunks:
            prompt = (
                f"You are analyzing section {chunk['index'] + 1} of {len(chunks)} "
                f"from a larger document (chars {chunk['start']}-{chunk['end']}).\n\n"
                f"## Task\n{task}\n\n"
                f"## Section Content\n{chunk['text'][:15000]}\n\n"  # Cap per-chunk
                f"## Instructions\nAnalyze this section for the task above. "
                f"Be specific about what you find. Note any references to other "
                f"sections or information that may require cross-referencing."
            )
            prompts.append(prompt)

        # Dispatch to workers in batch
        try:
            summaries = self.llm_primitives.llm_batch(
                prompts,
                role=role,
                n_tokens=512,
            )
        except Exception as e:
            # This previously swallowed the exception with no logging at all
            # (not even a fallback to sequential calls, unlike
            # chat_summarization.py's analogous site) -- a real batch
            # failure was invisible outside the returned error dict.
            log.warning(
                "summarize_chunks: llm_batch failed for role=%s (%d chunks): %s",
                role,
                len(prompts),
                e,
            )
            return [{"error": f"Batch call failed: {e}"}]

        results = []
        for i, (chunk, summary) in enumerate(zip(chunks, summaries)):
            results.append(
                {
                    "index": i,
                    "chunk_start": chunk["start"],
                    "chunk_end": chunk["end"],
                    "chunk_chars": chunk["char_count"],
                    "summary": summary,
                }
            )

        self._exploration_log.add_event(
            "summarize_chunks",
            {"task": task[:100], "n_chunks": n_chunks, "role": role},
            f"{len(results)} summaries",
        )
        return results

    # ------------------------------------------------------------------
    # Completion signals
    # ------------------------------------------------------------------

    def _final(self, answer: Any = _MISSING, **kwargs: Any) -> None:
        """Signal completion with final answer.

        Args:
            answer: The final answer to return. Keyword aliases accepted for
                model compatibility: answer, result, secret, value, response.

        Raises:
            FinalSignal: Raised to terminate execution (after validation).
            ValueError: If exploration requirement not met.
        """
        if answer is _MISSING:
            for key in ("answer", "result", "secret", "value", "response"):
                if key in kwargs:
                    answer = kwargs.pop(key)
                    break
        if answer is _MISSING:
            raise ValueError(
                "FINAL() requires an answer. Use FINAL(answer) or FINAL(result=answer)."
            )
        if kwargs:
            keys = ", ".join(sorted(kwargs))
            raise ValueError(f"FINAL() got unsupported keyword argument(s): {keys}")

        # Check forced exploration validation
        if self.config.require_exploration_before_final:
            if self._exploration_calls < self.config.min_exploration_calls:
                raise ValueError(
                    f"Premature FINAL: Must call at least {self.config.min_exploration_calls} "
                    f"exploration function(s) (peek, grep, llm_call, llm_batch) before FINAL(). "
                    f"Current exploration calls: {self._exploration_calls}. "
                    "Use peek() or grep() to examine the context first."
                )
        # Guard: model passed a function/class object instead of source text.
        # str() on callables produces '<function foo at 0x...>' which is useless.
        if callable(answer):
            raise ValueError(
                f"FINAL() received a {type(answer).__name__} object, not a string. "
                "Pass the source code as a string: FINAL('''def foo(): ...''')"
            )
        raise FinalSignal(str(answer))

    def _final_var(self, var_name: str) -> None:
        """Signal completion, returning contents of a variable.

        Args:
            var_name: Name of variable in artifacts dict to return.

        Raises:
            FinalSignal: Raised to terminate execution (after validation).
            KeyError: If variable not found in artifacts.
            ValueError: If exploration requirement not met.
        """
        # Check forced exploration validation
        if self.config.require_exploration_before_final:
            if self._exploration_calls < self.config.min_exploration_calls:
                raise ValueError(
                    f"Premature FINAL_VAR: Must call at least {self.config.min_exploration_calls} "
                    f"exploration function(s) (peek, grep, llm_call, llm_batch) before FINAL_VAR(). "
                    f"Current exploration calls: {self._exploration_calls}. "
                    "Use peek() or grep() to examine the context first."
                )
        if var_name not in self.artifacts:
            raise KeyError(f"Variable '{var_name}' not found in artifacts")
        raise FinalSignal(str(self.artifacts[var_name]))

    # ------------------------------------------------------------------
    # Stuck signal (non-terminating recovery aid)
    # ------------------------------------------------------------------

    def _stuck(self, reason: str) -> str:
        """Signal that the REPL is stuck and request recovery guidance.

        Unlike FINAL() which terminates execution, STUCK() is a non-terminating
        signal that:
        1. Logs the stuck reason to the exploration log
        2. Queries episodic memory for similar stuck situations (if available)
        3. Suggests recovery actions based on tool co-occurrence patterns
        4. Partially resets the turn counter to allow more exploration

        Args:
            reason: Why the REPL is stuck (free-form description).

        Returns:
            Recovery guidance string with suggested next actions.
        """
        self._exploration_calls += 1

        # Build exploration summary for context
        strategy_summary = self._exploration_log.get_strategy_summary()
        last_tools = [
            e.function for e in self._exploration_log.events[-5:]
        ] if self._exploration_log.events else []

        # Query episodic memory for similar stuck situations
        memory_guidance = ""
        if hasattr(self, "_recall") and callable(getattr(self, "_recall", None)):
            try:
                recall_result = self._recall(f"stuck: {reason}", limit=3)
                # Skip empty and "unavailable" (broken recall) payloads; they
                # are not "similar past situations".
                if (
                    recall_result
                    and "No memories" not in recall_result
                    and '"status": "unavailable"' not in recall_result
                    and '"results": []' not in recall_result
                ):
                    memory_guidance = f"\n## Similar past situations\n{recall_result}"
            except Exception:
                pass  # Recall is best-effort

        # Query tool co-occurrence for recovery suggestions
        tool_suggestions = ""
        if last_tools:
            last_tool = last_tools[-1]
            try:
                from src.repl_environment.suggestions import generate_suggestions
                suggestions = generate_suggestions(last_tool, max_suggestions=3)
                if suggestions:
                    tool_suggestions = f"\n## Suggested next tools\n{suggestions}"
            except Exception:
                pass  # Suggestions are best-effort

        # Build recovery guidance
        lines = [
            f"## Stuck: {reason}",
            f"Exploration so far: {strategy_summary.get('total_events', 0)} calls, "
            f"strategy: {strategy_summary.get('strategy_type', 'none')}",
        ]
        if last_tools:
            lines.append(f"Recent tools: {', '.join(last_tools)}")

        lines.append("\n## Recovery options")
        lines.append("1. Try a different search query or approach angle")
        lines.append("2. Use `recall('your question')` to check episodic memory")
        lines.append("3. Use `escalate('reason')` to hand off to a specialist model")
        lines.append("4. Use `FINAL('partial answer')` if you have a partial result")

        if memory_guidance:
            lines.append(memory_guidance)
        if tool_suggestions:
            lines.append(tool_suggestions)

        guidance = "\n".join(lines)

        # Log the stuck event
        self._exploration_log.add_event(
            "stuck",
            {
                "reason": reason,
                "exploration_calls": self._exploration_calls,
                "last_tools": last_tools,
                "strategy": strategy_summary.get("strategy_type", "none"),
            },
            {"guidance_length": len(guidance)},
        )

        return guidance

    # ------------------------------------------------------------------
    # Session persistence (findings)
    # ------------------------------------------------------------------

    def _mark_finding(
        self,
        content: str,
        tags: list[str] | None = None,
        source_file: str | None = None,
        source_page: int | None = None,
        source_section: str | None = None,
    ) -> dict[str, Any]:
        """Mark a key finding for session persistence.

        Args:
            content: The finding text (required).
            tags: Optional tags for categorization.
            source_file: Optional source file path.
            source_page: Optional page number (for PDFs).
            source_section: Optional section ID or title.

        Returns:
            Dict with finding info including ID.
        """
        import time
        import uuid

        finding = {
            "id": str(uuid.uuid4()),
            "content": content,
            "tags": tags or [],
            "source": {
                "file": source_file,
                "page": source_page,
                "section": source_section,
            },
            "turn": self._execution_count,
            "timestamp": time.time(),
        }

        self._findings_buffer.append(finding)

        return {
            "id": finding["id"],
            "content": content[:100] + "..." if len(content) > 100 else content,
            "tags": finding["tags"],
            "status": "marked",
        }

    def _list_findings(self) -> list[dict[str, Any]]:
        """List all findings marked in this session.

        Returns:
            List of finding summaries with id, content preview, tags, and turn.
        """
        return [
            {
                "id": f["id"],
                "content": f["content"][:80] + "..." if len(f["content"]) > 80 else f["content"],
                "tags": f["tags"],
                "turn": f["turn"],
            }
            for f in self._findings_buffer
        ]

    def get_findings(self) -> list[dict[str, Any]]:
        """Get all findings (full content) for external access.

        Returns:
            List of full finding dicts.
        """
        return self._findings_buffer.copy()

    def clear_findings(self) -> int:
        """Clear the findings buffer after syncing.

        Returns:
            Number of findings cleared.
        """
        count = len(self._findings_buffer)
        self._findings_buffer = []
        return count

    # ------------------------------------------------------------------
    # LLM call wrappers (exploration tracked)
    # ------------------------------------------------------------------

    def _tracked_llm_call(self, *args, **kwargs) -> str:
        """Wrapper for llm_call that tracks exploration.

        Returns:
            llm_call result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_call(*args, **kwargs)
        self._exploration_log.add_event("llm_call", {"args": args, "kwargs": kwargs}, result)
        return result

    def _tracked_llm_batch(self, *args, **kwargs) -> list[str]:
        """Wrapper for llm_batch that tracks exploration.

        Returns:
            llm_batch result.
        """
        self._exploration_calls += 1
        result = self.llm_primitives.llm_batch(*args, **kwargs)
        self._exploration_log.add_event("llm_batch", {"args": args, "kwargs": kwargs}, result)
        return result

    # ------------------------------------------------------------------
    # Tool / script dispatch
    # ------------------------------------------------------------------

    def _invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """Invoke a registered tool and capture request-local telemetry.

        Thin wrapper around _dispatch_tool that records ONE per-request
        invocation (name, elapsed_ms, success, chain_id, caller_type, result)
        into self._invoked_tools. repl_executor reads ONLY this list for
        per-request tool telemetry — NEVER ToolRegistry.get_invocation_log(),
        which is a process-global, bounded diagnostic ring: reading it leaks other
        concurrent requests' tools into this one (tools_called/tools_used).
        The records expose the same attribute interface as ToolInvocation so the
        downstream consumers are unchanged. Covers every dispatch path below.
        """
        if self.tool_registry is None:
            raise RuntimeError("No tool registry configured")

        import time as _time
        from types import SimpleNamespace as _NS

        self._tool_invocations += 1
        _t0 = _time.perf_counter()
        _ok = True
        _result: Any = None
        try:
            _result = self._dispatch_tool(tool_name, **kwargs)
            return _result
        except Exception:
            _ok = False
            raise
        finally:
            _records = getattr(self, "_invoked_tools", None)
            if _records is not None:
                # Mirror the registry's ToolInvocation semantics regardless of the
                # structured-output flag. Under ORCHESTRATOR_STRUCTURED_TOOL_OUTPUT
                # ToolRegistry.invoke() RETURNS a ToolOutput envelope and converts
                # a handler failure to ToolOutput(ok=False) WITHOUT raising — so
                # "no exception" is NOT success, and the envelope is NOT the raw
                # result. Unwrap so success := ok and result := raw output, which
                # keeps consumers like web_research extraction (isinstance(result,
                # dict)) and tools_success correct.
                _succ, _res = _ok, _result
                try:
                    from src.registry.tool_registry import ToolOutput as _ToolOutput
                    _is_envelope = isinstance(_result, _ToolOutput)
                except Exception:  # pragma: no cover - import guard
                    _is_envelope = (
                        hasattr(_result, "ok")
                        and hasattr(_result, "output")
                        and hasattr(_result, "status")
                    )
                if _is_envelope:
                    _succ = bool(getattr(_result, "ok", _ok))
                    _res = getattr(_result, "output", _result)
                _chain_id = getattr(self, "_active_tool_chain_id", None)
                _records.append(
                    _NS(
                        tool_name=tool_name,
                        elapsed_ms=(_time.perf_counter() - _t0) * 1000.0,
                        success=_succ,
                        chain_id=_chain_id,
                        caller_type="chain" if _chain_id else "direct",
                        result=_res,
                    )
                )

    def _dispatch_tool(self, tool_name: str, **kwargs) -> Any:
        """Dispatch a tool call. Telemetry is captured by _invoke_tool."""
        import time

        # Task-manager-specialized tools (stateful, request-local).
        if tool_name in {"task_create", "task_update", "task_list", "budget_override"}:
            from orchestration.tools.task_management import set_active_task_manager
            from orchestration.tools.task_management import (
                tool_budget_override,
                tool_task_create,
                tool_task_list,
                tool_task_update,
            )

            manager = getattr(self, "_task_manager", None)
            if manager is None:
                raise RuntimeError("Task manager not attached")
            set_active_task_manager(manager)
            if tool_name == "task_create":
                return tool_task_create(**kwargs)
            if tool_name == "task_update":
                return tool_task_update(**kwargs)
            if tool_name == "task_list":
                return tool_task_list()
            return tool_budget_override(**kwargs)

        # Optional tool-call budget enforcement when task manager is attached.
        manager = getattr(self, "_task_manager", None)
        if manager is not None and hasattr(self.tool_registry, "_tools"):
            tool = self.tool_registry._tools.get(tool_name)  # noqa: SLF001
            if tool is not None:
                budget = manager.current_task_budget()
                if budget is not None:
                    bucket = self._tool_budget_bucket(tool)
                    if bucket and not budget.can_use(bucket):
                        raise RuntimeError(
                            f"Budget exhausted for {bucket}. Remaining: {budget.remaining_summary()}"
                        )
                    if bucket:
                        budget.consume(bucket)

        chain_id = getattr(self, "_active_tool_chain_id", None)
        chain_index = int(getattr(self, "_active_tool_chain_index", 0))
        caller_type = "chain" if chain_id else "direct"

        # TD-4: closed-set typed argument selection (flag
        # ``typed_decisions_tool_args``, default off). Fail-open: None keeps
        # the model-provided kwargs and the existing validation path.
        from src.features import features

        decision_started = time.perf_counter()
        prepared = None
        typed_enabled = features().typed_decisions_tool_args
        if typed_enabled:
            from src.typed_decisions.prepared_action import prepare_action

            catalog, read_set, revision = self._tool_decision_snapshot()
            offered = self._eligible_tool_choices(catalog, caller_type)
            if offered:
                prepared = prepare_action(
                    task_revision=revision,
                    catalog=catalog,
                    read_set=read_set,
                    expires_at_ns=time.time_ns() + 60_000_000_000,
                    offered_choices=offered,
                    requested_model=f"role:{self.role}",
                )
        # Only a prepared action consumes typed arguments; with the flag off
        # (or nothing offered) the result would be discarded, so skip the call.
        typed_arguments = self._typed_tool_arguments(tool_name) if prepared is not None else None
        selection_ms = (time.perf_counter() - decision_started) * 1000
        validation = None
        authorized = False
        validation_started = time.perf_counter()
        if prepared is not None:
            from dataclasses import replace

            from src.typed_decisions.prepared_action import validate_prepared_action

            primitives = getattr(self, "llm_primitives", None)
            get_meta = getattr(primitives, "get_last_inference_meta", None)
            try:
                meta = get_meta() if callable(get_meta) else None
            except Exception:
                meta = None
            if typed_arguments is not None and isinstance(meta, dict):
                model_name = meta.get("model")
                prepared = replace(
                    prepared,
                    resolved_model=model_name if isinstance(model_name, str) else None,
                )
            catalog, read_set, revision = self._tool_decision_snapshot()
            current_offered = self._eligible_tool_choices(catalog, caller_type)
            authorized = tool_name in current_offered
            validation = validate_prepared_action(
                prepared,
                selected_id=tool_name if typed_arguments is not None else None,
                current_task_revision=revision,
                current_catalog=catalog,
                current_read_set=read_set,
                current_offered_choices=current_offered,
                authorized=authorized,
                model_available=self._selector_model_available(),
            )
            if validation.status == "accepted":
                kwargs = typed_arguments
        validation_ms = (time.perf_counter() - validation_started) * 1000

        # Fall back to REPL globals for tools like run_python_code that are
        # registered as direct REPL functions, not in the tool registry.
        dispatch_started = time.perf_counter()
        outcome = {"status": "error", "detail": "not_dispatched"}
        try:
            try:
                result = self.tool_registry.invoke(
                    tool_name,
                    self.role,
                    caller_type=caller_type,
                    chain_id=chain_id,
                    chain_index=chain_index,
                    context=getattr(self, "tool_context", None),
                    **kwargs,
                )
            except ValueError:
                repl_globals = getattr(self, "_globals", None)
                if repl_globals and tool_name in repl_globals and callable(repl_globals[tool_name]):
                    result = repl_globals[tool_name](**kwargs)
                else:
                    raise
            if prepared is not None:
                from src.registry.tool_registry import ToolOutput

                if isinstance(result, ToolOutput):
                    outcome = {"status": result.status, "detail": "ToolOutput"}
                else:
                    outcome = {"status": "returned", "detail": type(result).__name__}
        except Exception as exc:
            outcome = {"status": "error", "detail": type(exc).__name__}
            raise
        finally:
            if prepared is not None and validation is not None:
                from src.config import get_config
                from src.typed_decisions.prepared_action import append_receipt, make_receipt

                receipt = make_receipt(
                    prepared,
                    validation,
                    authorization_result="allowed" if authorized else "denied",
                    component_timing_ms={
                        "selection": selection_ms,
                        "validation": validation_ms,
                        "dispatch": (time.perf_counter() - dispatch_started) * 1000,
                    },
                    total_timing_ms=(time.perf_counter() - decision_started) * 1000,
                    observed_downstream_outcome=outcome,
                )
                self._last_decision_receipt = receipt
                try:
                    append_receipt(
                        get_config().paths.artifacts_dir
                        / "typed_decisions"
                        / "decision_receipts.jsonl",
                        receipt,
                    )
                except Exception:
                    log.exception("could not persist decision receipt")
        if chain_id:
            self._active_tool_chain_index = chain_index + 1

        # Track in research context
        if hasattr(self, "_research_context"):
            try:
                node_id = self._research_context.add(
                    tool="TOOL",
                    query=f"{tool_name}({', '.join(f'{k}={v!r}' for k, v in list(kwargs.items())[:3])})",
                    content=str(result)[:8000],
                    parent_id=getattr(self, "_last_research_node", None),
                )
                self._last_research_node = node_id
            except Exception:
                pass  # Silently ignore research tracking failures

        return result

    def _tool_decision_snapshot(self) -> tuple[dict[str, Any], dict[str, str], str]:
        """Capture the live tool catalogue and task inputs for dispatch recheck."""
        from src.typed_decisions.prepared_action import fingerprint

        registry = getattr(self, "tool_registry", None)
        tools = getattr(registry, "_tools", None)
        catalog = {
            name: {
                "parameters": getattr(tool, "parameters", None),
                "allowed_callers": getattr(tool, "allowed_callers", None),
                "code_hash": getattr(tool, "code_hash", None),
                "handler_identity": id(getattr(tool, "handler", None)),
                "side_effects": getattr(tool, "side_effects", None),
                "destructive": getattr(tool, "destructive", False),
            }
            for name, tool in (tools.items() if isinstance(tools, dict) else ())
        }
        manager = getattr(self, "_task_manager", None)
        task_id = getattr(manager, "current_task_id", None)
        try:
            task = manager.get(task_id) if task_id is not None else None
        except (AttributeError, KeyError):
            task = None
        revision = fingerprint(
            {
                "context": str(getattr(self, "context", "")),
                "task_revision": str(getattr(self, "_task_revision", "")),
                "task_id": task_id,
                "task_status": getattr(task, "status", None),
                "task_subject": getattr(task, "subject", None),
                "task_description": getattr(task, "description", None),
            }
        )
        budget = manager.current_task_budget() if manager is not None else None
        primitives = getattr(self, "llm_primitives", None)
        urls = getattr(primitives, "server_urls", None)
        read_set = {
            "role": str(getattr(self, "role", "")),
            "tool_context": repr(getattr(self, "tool_context", None)),
            "selector_instance": str(id(primitives)),
            "selector_backend_url": str(urls.get(str(self.role))) if isinstance(urls, dict) else "",
            "budget_remaining": budget.remaining_summary() if budget is not None else "",
        }
        return catalog, read_set, revision

    def _eligible_tool_choices(self, catalog: dict[str, Any], caller_type: str) -> tuple[str, ...]:
        """Read current tool existence and policy; never infer permission from selection."""
        registry = getattr(self, "tool_registry", None)
        can_use = getattr(registry, "can_use_tool", None)
        if not callable(can_use):
            return ()
        choices = []
        for name in catalog:
            tool = registry._tools[name]
            if caller_type not in (getattr(tool, "allowed_callers", None) or [caller_type]):
                continue
            try:
                if can_use(self.role, name, context=getattr(self, "tool_context", None)):
                    choices.append(name)
            except Exception:
                continue
        return tuple(sorted(choices))

    def _selector_model_available(self) -> bool:
        """Recheck the selector seam and any configured backend circuit."""
        primitives = getattr(self, "llm_primitives", None)
        if not callable(getattr(primitives, "llm_call", None)):
            return False
        urls = getattr(primitives, "server_urls", None)
        backend_url = urls.get(str(self.role)) if isinstance(urls, dict) else None
        tracker = getattr(primitives, "health_tracker", None)
        if backend_url and tracker is not None:
            try:
                return bool(tracker.is_available(backend_url))
            except Exception:
                return False
        return True

    def _typed_tool_arguments(self, tool_name: str) -> dict[str, Any] | None:
        """Closed-set typed arguments for one tool call (TD-4), or ``None``.

        Delegates to ``src.typed_decisions.tool_args_integration``, which owns
        the default-off flag and the fail-open contract. The import is lazy so
        the REPL package does not depend on jsonschema at import time, and the
        call never raises: any failure returns ``None`` and the caller keeps
        the model-provided arguments.
        """
        registry = getattr(self, "tool_registry", None)
        tools = getattr(registry, "_tools", None)
        tool = tools.get(tool_name) if isinstance(tools, dict) else None
        if tool is None:
            return None
        try:
            from src.typed_decisions.tool_args_integration import (
                maybe_typed_arguments,
                registry_parameters_to_schema,
            )

            return maybe_typed_arguments(
                tool_name=tool_name,
                parameters=registry_parameters_to_schema(getattr(tool, "parameters", None)),
                state=getattr(self, "context", None),
                primitives=getattr(self, "llm_primitives", None),
                role=getattr(self, "role", ""),
            )
        except Exception:  # pragma: no cover - defensive: never break dispatch
            return None

    def _tool_budget_bucket(self, tool: Any) -> str | None:
        """Map tool metadata to a coarse budget bucket."""
        side_effects = set(getattr(tool, "side_effects", []) or [])
        if "calls_llm" in side_effects:
            return "llm"
        if "local_exec" in side_effects:
            return "exec"
        if "modifies_files" in side_effects or "system_state" in side_effects:
            return "write"
        if "read_only" in side_effects:
            return "read"

        category = str(getattr(getattr(tool, "category", None), "value", ""))
        if category in {"code", "system"}:
            return "exec"
        if category in {"file", "web", "data", "math", "llm", "specialized"}:
            return "read"
        return None

    def _call_tool(self, tool_name: str, **kwargs) -> str:
        """Invoke a registered tool and return JSON-serialized result.

        Args:
            tool_name: Name of the tool to invoke.
            **kwargs: Tool arguments.

        Returns:
            JSON-serialized string of the tool result.
        """
        import json as _json

        result = self._invoke_tool(tool_name, **kwargs)
        try:
            return _json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            return str(result)

    def _list_tools(self) -> list[dict[str, Any]]:
        """List available tools for the current role.

        Returns:
            List of tool info dicts.
        """
        if self.tool_registry is None:
            return []

        return self.tool_registry.list_tools(
            role=self.role, context=getattr(self, "tool_context", None),
        )

    def _invoke_script(self, script_id: str, **kwargs) -> Any:
        """Invoke a prepared script by ID.

        Args:
            script_id: Script identifier.
            **kwargs: Script arguments.

        Returns:
            Script result.
        """
        # INF-78 OAB-1: unavailable in a task_root-scoped request (no-op otherwise).
        from src.repl_environment.task_root import scope_refusal as _scope_refusal

        _scope_denial = _scope_refusal('SCRIPT')
        if _scope_denial is not None:
            raise PermissionError(_scope_denial)
        if self.script_registry is None:
            raise RuntimeError("No script registry configured")

        # Pass sandbox globals for code execution
        return self.script_registry.invoke(
            script_id,
            sandbox_globals=self._globals,
            **kwargs,
        )

    def _find_scripts(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Find scripts matching a natural language query.

        Args:
            query: Search query.
            limit: Maximum results to return.

        Returns:
            List of matching script info dicts.
        """
        if self.script_registry is None:
            return []

        matches = self.script_registry.find_scripts(query, limit=limit)
        return [
            {
                "id": m.script.id,
                "description": m.script.description,
                "score": round(m.score, 2),
                "matched_on": m.matched_on,
            }
            for m in matches
        ]
