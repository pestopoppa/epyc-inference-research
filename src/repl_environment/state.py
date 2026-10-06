"""State inspection, exploration tracking, and checkpoint/restore.

Provides mixin with: get_state, exploration log access, checkpoint, restore, reset.
"""

from __future__ import annotations

import json
import logging
import time
import types
from typing import Any, TYPE_CHECKING

from src.repl_environment import safe_pickle
from src.repl_environment.types import ExplorationEvent, ExplorationLog, REPLTimeout

if TYPE_CHECKING:
    from orchestration.repl_memory.retriever import TwoPhaseRetriever

logger = logging.getLogger(__name__)


def _is_json_serializable(value: Any) -> bool:
    """Return True when value is safe to store in JSON checkpoints."""
    if isinstance(value, (types.ModuleType, types.FunctionType, types.MethodType, type)):
        return False
    if callable(value):
        return False
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


class _StateMixin:
    """Mixin providing state inspection and persistence.

    Includes: get_state, get_exploration_log, get_grep_history, clear_grep_history,
    get_exploration_strategy, log_exploration_completed, suggest_exploration,
    checkpoint, restore, get_checkpoint_metadata, reset.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _execution_count: int — number of execute() calls
        _final_answer: str | None — final answer if set
        _grep_hits_buffer: list — grep results buffer
        _findings_buffer: list — key findings buffer
        _globals: dict — restricted globals for REPL execution
        _restore_reconciliation: dict | None — set by restore(); what actually
            landed in the live namespace vs what the checkpoint claimed
        progress_logger: Any | None — progress tracking service
        task_id: str — task identifier for logging
        _build_globals: Callable[[], dict] — method to rebuild globals dict
    """

    def get_state(self) -> str:
        """Get a summary of current REPL state for the Root LM.

        Returns:
            String describing available variables and artifacts.
        """
        deferred_mode = bool(getattr(self, "_deferred_tool_results", False))
        artifact_keys = list(self.artifacts.keys()) if self.artifacts else []
        if deferred_mode and "_tool_outputs" in artifact_keys:
            artifact_keys = [k for k in artifact_keys if k != "_tool_outputs"]

        bundle = getattr(self, "_context_bundle", None)
        if bundle is not None:
            # INF-78 OAB-7: `context` is the request's bundle, not a string.
            context_line = (
                f"context: bundle ({len(bundle.sections)} sections, {bundle.total_chars} chars; "
                "context.index() lists them; nothing from it is in this prompt until you print it)"
            )
        else:
            context_line = f"context: str ({len(self.context)} chars)"
        state_lines = [
            context_line,
            f"artifacts: {artifact_keys if artifact_keys else '{}'}",
        ]

        # Show artifact previews
        for key, value in self.artifacts.items():
            if deferred_mode and key == "_tool_outputs":
                continue
            preview = str(value)[:100]
            if len(str(value)) > 100:
                preview += "..."
            state_lines.append(f"  artifacts['{key}']: {preview}")

        # Show user-defined variables that persist across turns.
        globals_dict = getattr(self, "_globals", {})
        builtin_keys = getattr(self, "_builtin_global_keys", frozenset())
        user_vars: list[str] = []
        for key, value in globals_dict.items():
            if key in builtin_keys or key.startswith("_"):
                continue
            type_name = type(value).__name__
            preview = repr(value)
            if len(preview) > 80:
                preview = preview[:80] + "..."
            user_vars.append(f"  {key} ({type_name}) = {preview}")

        if bundle is not None:
            # INF-78 OAB-7: a variable's repr preview is the second route (after print)
            # from a pulled value into the root prompt; count exactly what is shown.
            bundle.note_state_preview(sum(len(v.encode("utf-8")) for v in user_vars[:20]))
        if user_vars:
            state_lines.append("")
            state_lines.append("## Available Variables (from previous turns)")
            state_lines.extend(user_vars[:20])
            if len(user_vars) > 20:
                state_lines.append(f"  ... and {len(user_vars) - 20} more")

        # Names a restore expected to bring back but could not. The live listing
        # above is ground truth for what EXISTS; this is ground truth for what is
        # MISSING, so the model does not silently reference a dead name.
        reconciliation = getattr(self, "_restore_reconciliation", None)
        unavailable = (reconciliation or {}).get("unavailable") or {}
        if unavailable:
            state_lines.append("")
            state_lines.append("## Not Restored (do not reference these — rebuild if needed)")
            for key in sorted(unavailable)[:20]:
                state_lines.append(f"  {key}: {unavailable[key]}")
            if len(unavailable) > 20:
                state_lines.append(f"  ... and {len(unavailable) - 20} more")

        # Include research context if sufficient nodes exist
        if hasattr(self, "_research_context") and len(self._research_context.nodes) >= 3:
            state_lines.append("")
            state_lines.append(self._research_context.render())

        return "\n".join(state_lines)

    # ---- curation layer (D-d) ----

    def remember(self, name: str, note: str | None = None) -> str:
        """Agent-facing: annotate a variable so it survives as curated state.

        Auto-save is the safety net; ``remember`` is curation. Marking a variable
        records the agent's own description of WHY it matters, which is the part a
        type-and-repr inventory cannot reconstruct, and pins it ahead of the
        elision cap in resume summaries.

        Args:
            name: Name of a variable in the REPL namespace.
            note: Optional short description of what the variable is for.

        Returns:
            A confirmation string (the REPL surfaces return values to the model).
        """
        if not isinstance(name, str) or not name:
            return "[ERROR: remember(name) expects a variable name as a string]"

        globals_dict = getattr(self, "_globals", {})
        if name not in globals_dict:
            available = [
                k
                for k in globals_dict
                if not k.startswith("_")
                and k not in getattr(self, "_builtin_global_keys", frozenset())
            ]
            return (
                f"[ERROR: no variable named '{name}' in the REPL. "
                f"Available: {', '.join(sorted(available)[:15]) or '(none)'}]"
            )

        if note is not None and not isinstance(note, str):
            return "[ERROR: remember(name, note=...) expects note to be a string]"

        marks = getattr(self, "_curated", None)
        if marks is None:
            marks = self._curated = {}
        marks[name] = (note or "").strip()[:500]

        value = globals_dict[name]
        return (
            f"Remembered '{name}' ({type(value).__name__})"
            + (f": {marks[name]}" if marks[name] else "")
        )

    def get_curated(self) -> dict[str, str]:
        """Variables the agent explicitly marked, name -> note."""
        return dict(getattr(self, "_curated", {}) or {})

    # ---- code log (D-c1 measurement input; NOT injected into any prompt) ----

    #: Cap on retained code-log steps. Bounded so a long session cannot grow the
    #: checkpoint without limit; the counterfactual sizing below stays honest by
    #: reporting how many steps were elided.
    CODE_LOG_MAX_STEPS = 200
    #: Per-step source cap. Failed steps are compressed to their first line.
    CODE_LOG_MAX_CHARS = 4000

    def _record_code_log(self, code: str, ok: bool) -> None:
        """Append one executed step to the bounded code log.

        Recording only. Whether this belongs in the resume preamble is the open
        question D-c/D-c1 exists to answer — do not wire it into a prompt without
        that evidence.
        """
        log = getattr(self, "_code_log", None)
        if log is None:
            log = self._code_log = []

        source = (code or "").strip()
        if not ok:
            # A failed step's value is "this was tried and did not work", which
            # its first line carries; the body is noise.
            first = source.split("\n")[0] if source else ""
            source = first[:200]
        elif len(source) > self.CODE_LOG_MAX_CHARS:
            source = source[: self.CODE_LOG_MAX_CHARS] + "\n# ... [truncated]"

        log.append(
            {
                "step": self._execution_count,
                "ok": bool(ok),
                "code": source,
                "chars": len(code or ""),
            }
        )
        if len(log) > self.CODE_LOG_MAX_STEPS:
            elided = len(log) - self.CODE_LOG_MAX_STEPS
            del log[:elided]
            self._code_log_elided = getattr(self, "_code_log_elided", 0) + elided

    def get_code_log(self) -> list[dict[str, Any]]:
        """The bounded record of executed steps this session."""
        return list(getattr(self, "_code_log", []) or [])

    def code_log_metrics(self) -> dict[str, Any]:
        """Counterfactual sizing for the resume code log (D-c1).

        Answers "what WOULD a code-log preamble have cost?" without changing any
        prompt. Chars are converted to a rough token estimate with the same //4
        heuristic used elsewhere in this module.
        """
        log = getattr(self, "_code_log", []) or []
        ok_steps = [e for e in log if e.get("ok")]
        failed_steps = [e for e in log if not e.get("ok")]
        # What a preamble would actually render: successful steps in full, failed
        # steps compressed to a one-line comment (the fast-rlm shape).
        rendered = sum(len(e.get("code", "")) for e in ok_steps)
        rendered += sum(len(e.get("code", "")) + 24 for e in failed_steps)
        return {
            "steps": len(log),
            "steps_ok": len(ok_steps),
            "steps_failed": len(failed_steps),
            "steps_elided": getattr(self, "_code_log_elided", 0),
            "raw_chars": sum(e.get("chars", 0) for e in log),
            "rendered_chars": rendered,
            "rendered_tokens_est": rendered // 4,
        }

    def get_exploration_log(self) -> ExplorationLog:
        """Get the detailed exploration log.

        Returns:
            ExplorationLog containing all exploration events.
        """
        return self._exploration_log

    def get_grep_history(self) -> list[dict[str, Any]]:
        """Get grep hits buffer for two-stage summarization.

        Returns:
            List of grep hit records.
        """
        return self._grep_hits_buffer

    def clear_grep_history(self) -> None:
        """Clear the grep hits buffer.

        Call this when starting a new summarization task to avoid
        mixing grep hits from different documents.
        """
        self._grep_hits_buffer = []

    def get_exploration_strategy(self) -> dict[str, Any]:
        """Get a summary of the exploration strategy used.

        Returns:
            Dictionary with strategy summary including event counts and type.
        """
        return self._exploration_log.get_strategy_summary()

    def log_exploration_completed(
        self,
        success: bool,
        result: str = "",
    ) -> dict[str, Any]:
        """Log exploration completion to ProgressLogger.

        Args:
            success: Whether the task completed successfully.
            result: The final result (used for token efficiency calculation).

        Returns:
            Dictionary with the logged exploration data.
        """
        strategy = self.get_exploration_strategy()
        result_tokens = len(result) // 4  # Rough token estimate
        efficiency = self._exploration_log.get_token_efficiency(result_tokens)

        exploration_data = {
            "strategy": strategy,
            "efficiency": efficiency,
            "success": success,
        }

        # Log to ProgressLogger if available
        if self.progress_logger is not None:
            query_preview = self.context[:100] if self.context else ""
            self.progress_logger.log_exploration(
                task_id=self.task_id,
                query=query_preview,
                strategy_used=strategy.get("strategy_type", "unknown"),
                tokens_spent=strategy.get("total_tokens", 0),
                success=success,
                function_counts=strategy.get("function_counts"),
            )

        return exploration_data

    def suggest_exploration(
        self,
        task_description: str,
        retriever: TwoPhaseRetriever | None = None,
    ) -> list[str]:
        """Suggest exploration strategies based on similar past tasks.

        Args:
            task_description: Description of the current task.
            retriever: TwoPhaseRetriever from orchestration.repl_memory (optional).

        Returns:
            List of suggested exploration function calls as strings.
        """
        suggestions = []
        episodic_suggestions = []

        # If retriever available, query for similar successful exploration tasks
        if retriever is not None:
            try:
                context_preview = self.context[:500] if self.context else ""
                results = retriever.retrieve_for_exploration(
                    query=task_description,
                    context_preview=context_preview,
                )

                if results:
                    # Extract suggestions from successful similar tasks
                    for r in results[:3]:
                        # Only use high-quality memories (Q > 0.6, successful)
                        if r.q_value < 0.6 or r.memory.outcome != "success":
                            continue

                        context = r.memory.context or {}
                        strategy = context.get("exploration_strategy", {})
                        function_counts = strategy.get("function_counts", {})
                        strategy_type = strategy.get("strategy_type", "")

                        # Generate specific suggestions based on what worked
                        if function_counts.get("grep", 0) > 0:
                            episodic_suggestions.append(
                                f"grep('pattern')  # Similar task (q={r.q_value:.2f}) used grep"
                            )
                        if function_counts.get("llm_call", 0) > 0:
                            episodic_suggestions.append(
                                "llm_call('summarize key points')  # Similar task delegated effectively"
                            )
                        if strategy_type == "scan" and function_counts.get("peek", 0) > 0:
                            peek_count = function_counts["peek"]
                            episodic_suggestions.append(
                                f"# Scan strategy worked: {peek_count} peek() calls"
                            )

            except Exception:
                pass  # Silently ignore retrieval errors

        # Default suggestions based on context characteristics
        context_len = len(self.context)

        if context_len < 500:
            suggestions.append("peek(500)  # Context is short, read it all")
        elif context_len < 2000:
            suggestions.append("peek(1000)  # Scan the beginning")
        else:
            suggestions.append("peek(500)  # Preview context")
            suggestions.append("grep('keyword')  # Search for specific patterns")

        # Prepend episodic suggestions (learned patterns first)
        return episodic_suggestions + suggestions

    # =========================================================================
    # Checkpoint & Restore (for session persistence)
    # =========================================================================

    def checkpoint(self) -> dict[str, Any]:
        """Create a checkpoint of the current REPL state.

        Returns:
            Dict suitable for JSON serialization and later restore().
        """
        if getattr(self, "timed_out", False) is True:
            timeout_error = getattr(self, "timeout_error", None)
            if not isinstance(timeout_error, str):
                timeout_error = "REPLTimeout: Execution timed out"
            raise REPLTimeout(timeout_error.removeprefix("REPLTimeout: "))

        def sanitize_value(value: Any) -> Any:
            """Sanitize a value for JSON serialization."""
            if _is_json_serializable(value):
                return value
            # Mark as unserializable with type info
            return {
                "__unserializable__": True,
                "type": type(value).__name__,
                "repr": repr(value)[:100],  # Truncated repr for debugging
            }

        def sanitize_artifacts(artifacts: dict[str, Any]) -> dict[str, Any]:
            """Sanitize artifacts dict, marking non-serializable values."""
            sanitized = {}
            for key, value in artifacts.items():
                if isinstance(value, dict):
                    # Recursively sanitize nested dicts
                    sanitized[key] = sanitize_artifacts(value)
                elif isinstance(value, list):
                    # Sanitize list items
                    sanitized[key] = [sanitize_value(item) for item in value]
                else:
                    sanitized[key] = sanitize_value(value)
            return sanitized

        # Sanitize exploration log events for serialization
        exploration_events = []
        for event in self._exploration_log.events:
            exploration_events.append(
                {
                    "function": event.function,
                    "args": sanitize_artifacts(event.args) if isinstance(event.args, dict) else {},
                    "result_size": event.result_size,
                    "timestamp": event.timestamp,
                    "token_estimate": event.token_estimate,
                }
            )

        # Serialize research context if available
        research_context_data = None
        if hasattr(self, "_research_context"):
            research_context_data = self._research_context.to_dict()

        globals_dict = getattr(self, "_globals", {})
        builtin_keys = getattr(self, "_builtin_global_keys", frozenset())
        user_globals: dict[str, Any] = {}
        pickled_globals: dict[str, Any] = {}
        variable_lineage: dict[str, dict[str, Any]] = {}
        skipped_user_globals: list[str] = []
        skip_reasons: dict[str, str] = {}

        curated = self.get_curated()

        def _lineage(value: Any, tier: str, key: str) -> dict[str, Any]:
            entry = {
                "role": getattr(self, "role", "unknown"),
                "saved_at_execution_count": self._execution_count,
                "saved_at_ts": time.time(),
                "value_type": type(value).__name__,
                "tier": tier,
            }
            if key in curated:
                entry["curated"] = True
                if curated[key]:
                    entry["note"] = curated[key]
            return entry

        for key, value in globals_dict.items():
            if key in builtin_keys or key.startswith("_"):
                continue
            if _is_json_serializable(value):
                user_globals[key] = value
                variable_lineage[key] = _lineage(value, "json", key)
                continue
            # JSON cannot hold it. Try the hardened pickle boundary (D-a):
            # signed, size-capped, and only loadable under an allowlist of
            # inert data types. Anything else is reported, never silently lost.
            try:
                pickled_globals[key] = safe_pickle.dumps(value)
                variable_lineage[key] = _lineage(value, "pickle", key)
            except Exception as e:
                skipped_user_globals.append(key)
                skip_reasons[key] = str(e)[:200]

        curated_lost = [k for k in skipped_user_globals if k in curated]
        if curated_lost:
            logger.warning(
                "Checkpoint could not save %d CURATED variables the agent explicitly marked: %s",
                len(curated_lost),
                ", ".join(curated_lost),
            )

        if skipped_user_globals:
            logger.warning(
                "Checkpoint skipped %d globals that neither JSON nor the pickle allowlist could "
                "hold: %s",
                len(skipped_user_globals),
                ", ".join(skipped_user_globals[:10]),
            )

        return {
            "version": 1,  # Schema version for future compatibility
            "artifacts": sanitize_artifacts(self.artifacts),
            "execution_count": self._execution_count,
            "exploration_calls": self._exploration_calls,
            "exploration_tokens": self._exploration_log.total_exploration_tokens,
            "exploration_events": exploration_events,
            "grep_hits_buffer": self._grep_hits_buffer,
            "findings_buffer": self._findings_buffer,  # Key findings
            "context_length": len(self.context),  # For verification, not full context
            "task_id": self.task_id,
            "research_context": research_context_data,
            "curated": curated,
            "code_log": self.get_code_log(),
            "code_log_metrics": self.code_log_metrics(),
            "user_globals": user_globals,
            "pickled_globals": pickled_globals,
            "variable_lineage": variable_lineage,
            "skipped_user_globals": skipped_user_globals,
            "skip_reasons": skip_reasons,
        }

    def restore(self, checkpoint: dict[str, Any]) -> dict[str, Any]:
        """Restore REPL state from a checkpoint.

        Note: Non-serializable artifacts remain as marker dicts.
        The context is NOT restored - it should be passed to __init__.

        Args:
            checkpoint: Dict from a previous checkpoint() call.

        Returns:
            A reconciliation dict describing what ACTUALLY landed in the live
            namespace, as opposed to what the checkpoint claimed. Also stored on
            ``self._restore_reconciliation`` for later readers. Keys:
                restored: list[str] — names now present in _globals
                unavailable: dict[str, str] — name -> reason it is not present
                claimed: int — count of user_globals the checkpoint carried
                dropped_at_save: list[str] — skipped_user_globals from save time

        Raises:
            ValueError: If checkpoint format is invalid.
        """
        version = checkpoint.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported checkpoint version: {version}")

        # Restore artifacts
        self.artifacts = checkpoint.get("artifacts", {})

        # Restore execution state
        self._execution_count = checkpoint.get("execution_count", 0)
        self._exploration_calls = checkpoint.get("exploration_calls", 0)

        # Restore exploration log
        self._exploration_log = ExplorationLog()
        self._exploration_log.total_exploration_tokens = checkpoint.get("exploration_tokens", 0)
        for event_data in checkpoint.get("exploration_events", []):
            event = ExplorationEvent(
                function=event_data.get("function", ""),
                args=event_data.get("args", {}),
                result_size=event_data.get("result_size", 0),
                timestamp=event_data.get("timestamp", 0.0),
                token_estimate=event_data.get("token_estimate", 0),
            )
            self._exploration_log.events.append(event)

        # Restore grep hits buffer
        self._grep_hits_buffer = checkpoint.get("grep_hits_buffer", [])

        # Restore the code log (measurement input only — never injected into a
        # prompt; see D-c/D-c1).
        self._code_log = list(checkpoint.get("code_log", []) or [])
        self._curated = dict(checkpoint.get("curated", {}) or {})

        # Restore findings buffer
        self._findings_buffer = checkpoint.get("findings_buffer", [])

        # Restore research context if available
        if hasattr(self, "_research_context") and checkpoint.get("research_context"):
            from src.research_context import ResearchContext

            self._research_context = ResearchContext.from_dict(
                checkpoint["research_context"],
                use_semantic=self._research_context.use_semantic,
            )
            # Restore last node pointer if any nodes exist
            if self._research_context.nodes:
                # Use most recent node as last_research_node
                most_recent = max(
                    self._research_context.nodes.values(),
                    key=lambda n: n.timestamp,
                )
                self._last_research_node = most_recent.id

        # Rebuild globals with restored artifacts
        self._globals = self._build_globals()
        builtin_keys = set(getattr(self, "_builtin_global_keys", frozenset()))
        user_globals = checkpoint.get("user_globals", {}) or {}
        unavailable: dict[str, str] = {}
        for key, value in user_globals.items():
            if key in builtin_keys:
                unavailable[key] = "name collides with an engine-owned builtin; not restored"
                continue
            try:
                self._globals[key] = value
            except Exception as e:  # pragma: no cover - defensive
                unavailable[key] = f"restore failed: {type(e).__name__}: {e}"[:200]

        # Values JSON could not hold, carried through the hardened pickle
        # boundary. Every failure mode here (HMAC mismatch, non-allowlisted
        # global, oversize, corrupt) fails CLOSED: the name is reported
        # unavailable, never partially applied.
        pickled = checkpoint.get("pickled_globals", {}) or {}
        for key, envelope in pickled.items():
            if key in builtin_keys:
                unavailable[key] = "name collides with an engine-owned builtin; not restored"
                continue
            try:
                self._globals[key] = safe_pickle.loads(envelope)
            except Exception as e:
                unavailable[key] = f"{type(e).__name__}: {e}"[:200]

        claimed_names = list(user_globals) + [k for k in pickled if k not in user_globals]

        # Reconcile against the LIVE namespace rather than trusting the payload:
        # a name the checkpoint claimed is only "restored" if it is actually here.
        restored_names = [k for k in claimed_names if k in self._globals and k not in unavailable]
        for key in claimed_names:
            if key not in restored_names and key not in unavailable:
                unavailable[key] = "absent from the live namespace after restore"

        # Variables that never made it into the checkpoint in the first place.
        skip_reasons = checkpoint.get("skip_reasons", {}) or {}
        for key in checkpoint.get("skipped_user_globals", []) or []:
            unavailable.setdefault(
                key,
                skip_reasons.get(key, "not storable at save time; never checkpointed"),
            )

        reconciliation = {
            "restored": restored_names,
            "unavailable": unavailable,
            "claimed": len(claimed_names),
            "dropped_at_save": list(checkpoint.get("skipped_user_globals", []) or []),
        }
        self._restore_reconciliation = reconciliation

        if restored_names or unavailable:
            logger.info(
                "Restored %d/%d globals from checkpoint (%d unavailable: %s)",
                len(restored_names),
                len(claimed_names),
                len(unavailable),
                ", ".join(sorted(unavailable)[:10]) or "none",
            )
        return reconciliation

    def get_checkpoint_metadata(self) -> dict[str, Any]:
        """Get metadata about current state for checkpoint decision.

        Returns:
            Dict with state metadata.
        """
        return {
            "execution_count": self._execution_count,
            "exploration_calls": self._exploration_calls,
            "artifact_count": len(self.artifacts),
            "context_length": len(self.context),
            "grep_hits_count": len(self._grep_hits_buffer),
            "findings_count": len(self._findings_buffer),
        }

    def reset(self) -> None:
        """Reset the REPL state (clear artifacts, keep context)."""
        self.artifacts.clear()
        self._final_answer = None
        self._execution_count = 0
        self._exploration_calls = 0
        self._exploration_log = ExplorationLog()  # Reset exploration log
        self._grep_hits_buffer = []  # Clear grep history for two-stage pipeline
        self._findings_buffer = []  # Clear findings buffer
        # Reset research context
        if hasattr(self, "_research_context"):
            self._research_context.clear()
            self._last_research_node = None
        self._globals = self._build_globals()
