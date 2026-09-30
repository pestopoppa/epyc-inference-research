"""PromptBuilder class and module-level convenience functions.

Contains:
- PromptBuilder class (Root LM, escalation, step, stage2 review, system prompts)
- Module-level backward-compat functions
- build_routing_context() for MemRL integration
- build_long_context_exploration_prompt() for large document processing
"""

from __future__ import annotations

from dataclasses import asdict
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.constants import TASK_IR_OBJECTIVE_LEN
from src.prompt_builders.types import (
    EscalationPrompt,
    PromptConfig,
    PromptStyle,
    RootLMPrompt,
    StepPrompt,
)
from src.prompt_builders.constants import (
    COMPACT_ROOT_LM_TOOLS,
    DEFAULT_ROOT_LM_RULES,
    DEFAULT_ROOT_LM_TOOLS,
)
from src.prompt_builders.resolver import resolve_prompt
from src.roles import Role, chain_name_to_role, get_tier
from src.runtime import routing_stage_timing
from src.task_ir import canonicalize_task_ir
from src.features import features as _get_features

_log = logging.getLogger(__name__)
_CORPUS_REGISTRY: Any | None = None
_DEFAULT_CORPUS_SLOW_QUERY_MS = 5000.0
_CORPUS_TASK_SCOPED_ROLES_ENV = "ORCHESTRATOR_CORPUS_TASK_SCOPED_ROLES"
_CODE_CORPUS_TASK_MARKERS = (
    "```",
    ".py",
    ".js",
    ".ts",
    ".go",
    ".rs",
    ".cpp",
    ".java",
    "def ",
    "class ",
    "function ",
    "import ",
    "return ",
    "traceback",
    "stack trace",
    "implement",
    "refactor",
    "debug",
    "fix bug",
    "write code",
    "unit test",
)

# ── Fallback Constants ──────────────────────────────────────────────────────

_ROOT_LM_SYSTEM_FALLBACK = (
    "You have access to a Python REPL with tools. "
    "If you know the answer, call FINAL() immediately — no code needed. "
    "If the task benefits from computation or tools, write Python. "
    "Always end with FINAL() or FINAL_VAR()."
)

_TASK_DECOMPOSITION_FALLBACK = """Decompose this task into 2-5 parallel-executable steps.
Return ONLY a JSON array, no markdown fences, no explanation.

Each step: {{"id":"S1","actor":"worker"|"coder"|"architect","action":"what to do","depends_on":[],"parallel_group":"group_name","outputs":["result"]}}

Rules:
- Independent steps share a parallel_group so they run simultaneously
- Use depends_on only when a step needs another step's output
- actor: "worker" for exploration/summarization, "coder" for code, "architect" for design
- Keep actions concise (1-2 sentences)

Task: {objective}{context_note}

JSON:"""

_ROLE_PROMPT_FALLBACKS = {
    Role.FRONTDOOR: (
        "You are an orchestrator AI. Your job is to understand user requests, "
        "break them into tasks, and generate Python code that executes in a "
        "sandboxed REPL environment. You can call sub-LMs for complex subtasks."
    ),
    Role.CODER_ESCALATION: (
        "You are a software architect. Design robust systems and APIs. "
        "Consider scalability, security, and long-term maintainability."
    ),
    Role.ARCHITECT_GENERAL: (
        "You are a system architect. You handle complex problems that require "
        "deep reasoning and coordination across multiple domains."
    ),
    Role.INGEST_LONG_CONTEXT: (
        "You are a document analysis specialist. Process and synthesize "
        "information from long documents while maintaining accuracy."
    ),
    Role.WORKER_GENERAL: (
        "You are a general-purpose assistant. Complete tasks efficiently "
        "and accurately. Focus on the specific task at hand."
    ),
    Role.WORKER_MATH: (
        "You are a mathematical reasoning specialist. Solve math problems "
        "step by step, showing your work clearly."
    ),
    Role.WORKER_VISION: (
        "You are a vision-language specialist. Analyze images and provide "
        "accurate descriptions and interpretations."
    ),
}


if TYPE_CHECKING:
    from src.context_manager import ContextManager
    from src.escalation import EscalationContext, EscalationDecision


class PromptBuilder:
    """Builder for all prompt types in the orchestration system.

    This class provides a unified interface for building prompts for:
    - Root LM (orchestrator) prompts
    - Escalation prompts
    - Step execution prompts
    - Role-specific system prompts

    Example:
        builder = PromptBuilder()

        # Build Root LM prompt
        prompt = builder.build_root_lm_prompt(
            state="artifacts = {}",
            original_prompt="Summarize the document",
            turn=0,
        )

        # Build escalation prompt
        esc_prompt = builder.build_escalation_prompt(
            original_prompt="...",
            state="...",
            failure_context=context,
            decision=decision,
        )
    """

    def __init__(self, config: PromptConfig | None = None):
        """Initialize the prompt builder.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or PromptConfig()

    def _resolve_tools(self) -> str:
        """Resolve tools prompt text from file, style, or default.

        Priority: tools_file > TOOL_DEFINITION_VARIANT env > style-based > DEFAULT.
        File read is ~1ms — no caching needed (enables hot-swap).

        P3d A/B harness: set TOOL_DEFINITION_VARIANT=verbose|default|compact
        to override style-based selection for A/B testing.
        """
        if self.config.tools_file:
            try:
                return Path(self.config.tools_file).read_text().strip()
            except OSError as exc:
                _log.warning("Failed to read tools_file %s: %s", self.config.tools_file, exc)

        # P3d: A/B variant override via env var
        import os
        variant = os.environ.get("TOOL_DEFINITION_VARIANT", "").strip().lower()
        if variant == "verbose":
            from src.prompt_builders.constants import VERBOSE_ROOT_LM_TOOLS
            return VERBOSE_ROOT_LM_TOOLS
        elif variant == "compact":
            return COMPACT_ROOT_LM_TOOLS
        elif variant == "default":
            return DEFAULT_ROOT_LM_TOOLS

        if self.config.style == PromptStyle.MINIMAL:
            return COMPACT_ROOT_LM_TOOLS
        return DEFAULT_ROOT_LM_TOOLS

    def _resolve_rules(self) -> str:
        """Resolve rules prompt text from file or default.

        Priority: rules_file > DEFAULT_ROOT_LM_RULES.
        """
        if self.config.rules_file:
            try:
                return Path(self.config.rules_file).read_text().strip()
            except OSError as exc:
                _log.warning("Failed to read rules_file %s: %s", self.config.rules_file, exc)
        return DEFAULT_ROOT_LM_RULES

    def build_root_lm_prompt(
        self,
        state: str,
        original_prompt: str,
        last_output: str = "",
        last_error: str = "",
        turn: int = 0,
        routing_context: str = "",
        corpus_context: str = "",
        *,
        as_structured: bool = False,
        solution_file: str = "",
    ) -> str | RootLMPrompt:
        """Build the prompt for the Root LM (frontdoor).

        The Root LM generates Python code that executes in a sandboxed REPL.
        It has access to the context and can call sub-LMs for complex tasks.

        Args:
            state: Current REPL state from repl.get_state()
            original_prompt: The user's original prompt
            last_output: Output from the last code execution
            last_error: Error from the last code execution (if any)
            turn: Current turn number (0-indexed)
            as_structured: Return RootLMPrompt instead of string

        Returns:
            Prompt string or RootLMPrompt if as_structured=True
        """
        prompt = RootLMPrompt(
            system=resolve_prompt("root_lm_system", _ROOT_LM_SYSTEM_FALLBACK),
            tools=self._resolve_tools(),
            rules=self._resolve_rules(),
            state=f"Turn {turn + 1}\n{state}",
            task=original_prompt,
            instruction="Answer or write code. End with FINAL().\n\n",
        )

        # Build context section based on last output/error
        context_parts = []
        if last_error:
            error_preview = last_error[: self.config.max_error_preview]
            if len(last_error) > self.config.max_error_preview:
                error_preview += "..."
            context_parts.extend(
                [
                    "## Last Error",
                    "```",
                    error_preview,
                    "```",
                ]
            )
            if solution_file:
                context_parts.append(
                    f"Your previous code is saved at `{solution_file}`. "
                    "Read it with `peek(99999, file_path=\"{}\")`, fix ONLY the broken part, ".format(solution_file)
                    + "and rewrite the corrected version with `file_write_safe`. "
                    "Do NOT rewrite from scratch — make targeted fixes to the existing code."
                )
            else:
                context_parts.append(
                    "Fix ONLY the broken part of your previous code. "
                    "Do NOT rewrite from scratch — make targeted fixes."
                )
        elif last_output:
            output_preview = last_output[: self.config.max_output_preview]
            if len(last_output) > self.config.max_output_preview:
                output_preview += "..."
            context_parts.extend(
                [
                    "## Last Output",
                    "```",
                    output_preview,
                    "```",
                ]
            )

        if _get_features().deferred_tool_results:
            context_parts.append(
                "Tool results are deferred to in-turn variables by default. "
                "Use print() for any result you need in subsequent turns."
            )

        if routing_context:
            context_parts.append(f"## Routing Intelligence\n{routing_context}")

        if corpus_context:
            prompt.reference_code = corpus_context

        if context_parts:
            prompt.context = "\n".join(context_parts)

        if as_structured:
            return prompt
        # RTE-Prefix: when enabled, fixed sections render before per-turn ones
        # so REPL turns share a longer KV-cache prefix (see to_string docs).
        return prompt.to_string(prefix_stable=_get_features().prefix_stable_order)

    def build_escalation_prompt(
        self,
        original_prompt: str,
        state: str,
        failure_context: "EscalationContext",
        decision: "EscalationDecision",
        *,
        as_structured: bool = False,
        use_toon: bool = False,
    ) -> str | EscalationPrompt:
        """Build a prompt for an escalated role.

        Includes failure context to help the higher-tier model understand
        what went wrong and what was tried.

        Uses EscalationContext/EscalationDecision types for escalation info.

        Args:
            original_prompt: The user's original prompt.
            state: Current REPL state.
            failure_context: Context about the failure.
            decision: The routing decision with escalation reason.
            as_structured: Return EscalationPrompt instead of string.
            use_toon: Use TOON encoding for architect-tier escalations.

        Returns:
            Prompt string or EscalationPrompt if as_structured=True
        """
        # Handle both legacy (role) and new (current_role) attribute names
        role = getattr(failure_context, "current_role", None) or getattr(
            failure_context, "role", "unknown"
        )

        # TOON format for architect escalations (compact, structured)
        if use_toon:
            return self._build_escalation_toon(
                original_prompt, state, failure_context, decision, role
            )

        # Check if previous role left a solution file
        sol_file = getattr(failure_context, "solution_file", "") or ""
        if sol_file:
            instructions = (
                f"The previous role's code is saved at `{sol_file}`. "
                f"Read it with `peek(99999, file_path=\"{sol_file}\")`, "
                "fix ONLY the broken part, and rewrite the corrected version "
                f"with `file_write_safe(\"{sol_file}\", corrected_code)`. "
                "Do NOT rewrite from scratch — make targeted fixes.\n"
                "Output Python code that will execute in the REPL environment."
            )
        else:
            instructions = (
                "Fix the issue and complete the task. "
                "You have more capability than the previous role.\n"
                "Output Python code that will execute in the REPL environment."
            )

        prompt = EscalationPrompt(
            header=f"# Escalation from {role}",
            failure_info=(
                f"The {role} failed after "
                f"{failure_context.failure_count} attempts.\n"
                f"Reason: {decision.reason}"
            ),
            state=state,
            task=original_prompt,
            instructions=instructions,
        )

        # Build error details
        error_parts = [f"Category: {failure_context.error_category}"]
        if failure_context.gate_name:
            error_parts.append(f"Gate: {failure_context.gate_name}")
        if failure_context.error_message:
            error_preview = failure_context.error_message[: self.config.max_error_preview]
            error_parts.extend(
                [
                    "",
                    "Error message:",
                    "```",
                    error_preview,
                    "```",
                ]
            )
        prompt.error_details = "\n".join(error_parts)

        # Inject scratchpad insights from previous role
        scratchpad = getattr(failure_context, "scratchpad_entries", None) or []
        if scratchpad:
            insight_lines = "\n".join(
                e.to_bullet() if hasattr(e, "to_bullet") else f"- {e}"
                for e in scratchpad
            )
            prompt.failure_info += f"\n\n## Previous Insights\n{insight_lines}"

        if as_structured:
            return prompt
        return prompt.to_string()

    def _build_escalation_toon(
        self,
        original_prompt: str,
        state: str,
        failure_context: "EscalationContext",
        decision: "EscalationDecision",
        role: str,
    ) -> str:
        """Build TOON-encoded escalation prompt for architect tier.

        Compact format optimized for slow models (~7 t/s).
        """
        from src.services.toon_encoder import encode, is_available

        # Build failures list
        failures = [{
            "tier": str(role),
            "try": failure_context.failure_count,
            "err": str(failure_context.error_category),
        }]
        if failure_context.gate_name:
            failures[0]["gate"] = failure_context.gate_name
        if failure_context.error_message:
            # Truncate error to ~200 chars for TOON compactness
            failures[0]["msg"] = failure_context.error_message[:200]

        toon_data = {
            "task": original_prompt[:500],  # Cap task length
            "failures": failures,
        }

        # Include state summary if present (truncated)
        if state and len(state) > 10:
            toon_data["state"] = state[:300]

        if is_available():
            return f"```toon\n{encode(toon_data)}\n```"
        else:
            # Fallback: compact JSON-like format
            import json
            return f"```toon\n{json.dumps(toon_data, separators=(',', ':'))}\n```"

    def build_step_prompt(
        self,
        action: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        context: ContextManager | None = None,
        constraints: str = "",
        *,
        as_structured: bool = False,
    ) -> str | StepPrompt:
        """Build a prompt for step execution.

        Uses ContextManager for rich context formatting if provided.

        Args:
            action: The action/task to perform.
            inputs: Input keys to include from context.
            outputs: Expected output keys.
            context: Optional ContextManager for input formatting.
            constraints: Additional constraints or requirements.
            as_structured: Return StepPrompt instead of string

        Returns:
            Prompt string or StepPrompt if as_structured=True
        """
        prompt = StepPrompt(action=action, constraints=constraints)

        # Build inputs section
        if inputs:
            if context:
                context_str = context.build_prompt_context(
                    input_keys=inputs,
                    max_chars=self.config.max_context_chars,
                )
                if context_str:
                    prompt.inputs = context_str
                else:
                    missing = [k for k in inputs if not context.has(k)]
                    if missing:
                        prompt.inputs = f"\n(Not available: {', '.join(missing)})"
            else:
                prompt.inputs = f"\n{', '.join(inputs)}"

        # Build outputs section
        if outputs:
            prompt.outputs = ", ".join(outputs)

        if as_structured:
            return prompt
        return prompt.to_string()

    def build_stage2_review_prompt(
        self,
        draft_summary: str,
        grep_hits: list[dict[str, Any]],
        figures: list[dict[str, Any]],
        original_task: str = "",
    ) -> str:
        """Build a prompt for Stage 2 of two-stage summarization.

        Stage 2 receives the frontdoor's draft summary along with key excerpts
        (grep hits) and figure descriptions. The large model refines the summary
        without reading the full document.

        Args:
            draft_summary: The draft summary from Stage 1 (frontdoor).
            grep_hits: List of grep hit records from REPL's get_grep_history().
                       Each record contains pattern, source, match_count, hits.
            figures: List of figure descriptions from document processing.
            original_task: Optional original summarization task.

        Returns:
            Prompt string for Stage 2 model.
        """
        parts = [
            "# Document Summary Review",
            "",
            "You are reviewing a draft summary produced by a fast model. Your task is to:",
            "1. Verify accuracy against the provided excerpts",
            "2. Add important details that may be missing",
            "3. Improve clarity and completeness",
            "4. Correct any inaccuracies",
            "",
        ]

        # Add original task if provided
        if original_task:
            parts.extend(
                [
                    "## Original Request",
                    original_task,
                    "",
                ]
            )

        # Add draft summary
        parts.extend(
            [
                "## Draft Summary (to review and refine)",
                "```",
                draft_summary[:8000],  # Cap draft length
                "```",
                "",
            ]
        )

        # Add grep excerpts (key source material)
        if grep_hits:
            parts.extend(
                [
                    "## Key Excerpts from Source Document",
                    "These are search results the draft model found relevant:",
                    "",
                ]
            )

            # NOTE: TOON encoding was tested for grep hits but found to be LESS
            # efficient than Markdown due to pattern repetition. Keeping Markdown.
            total_chars = 0
            max_excerpt_chars = 6000  # Cap total excerpt size

            for hit_record in grep_hits:
                if total_chars >= max_excerpt_chars:
                    parts.append(
                        f"[... additional excerpts truncated at {max_excerpt_chars} chars]"
                    )
                    break

                pattern = hit_record.get("pattern", "")
                hits = hit_record.get("hits", [])

                if hits:
                    parts.append(f"### Search: `{pattern}`")
                    for hit in hits[:5]:  # Cap at 5 hits per pattern
                        context = hit.get("context", hit.get("match", ""))
                        line_num = hit.get("line_num", "?")
                        excerpt = f"Line {line_num}: {context[:500]}"
                        total_chars += len(excerpt)
                        parts.append(excerpt)
                        if total_chars >= max_excerpt_chars:
                            break
                    parts.append("")

        # Add figure descriptions
        if figures:
            parts.extend(
                [
                    "## Figures in Document",
                    "",
                ]
            )

            for i, fig in enumerate(figures[:10], 1):  # Cap at 10 figures
                desc = fig.get("description", fig.get("text", "No description"))
                page = fig.get("page", "?")
                parts.append(f"**Figure {i}** (Page {page}): {desc[:500]}")
            parts.append("")

        # Final instruction
        parts.extend(
            [
                "## Your Task",
                "Write a refined, accurate executive summary based on the draft and excerpts above.",
                "",
                "IMPORTANT OUTPUT FORMAT:",
                "- Write the summary directly - no thinking, analysis, or meta-commentary",
                "- Do not explain your reasoning or discuss terminology",
                "- Do not include phrases like 'here is' or 'based on the draft'",
                "- Start directly with the summary content",
                "- Keep the same structure as the draft but improve accuracy and completeness",
            ]
        )

        return "\n".join(parts)

    def get_system_prompt(self, role: Role) -> str:
        """Get the system prompt for a specific role.

        Hot-swappable: reads from orchestration/prompts/roles/{role.value}.md
        if the file exists, otherwise uses the fallback constant.

        Args:
            role: The role to get the system prompt for.

        Returns:
            System prompt string.
        """
        tier = get_tier(role)
        role_name = role.value

        fallback = _ROLE_PROMPT_FALLBACKS.get(
            role,
            f"You are a {tier.name.lower()}-tier {role_name} assistant. "
            f"Complete tasks efficiently and accurately.",
        )

        prompt = resolve_prompt(role_name, fallback, subdir="roles")

        # B1: Inject frozen user profile into system prompt (prefix cache stable)
        from src.features import features as _get_features
        if _get_features().user_modeling:
            try:
                from src.user_modeling.profile_store import get_profile_store
                _snapshot = get_profile_store().frozen_snapshot("default")
                if _snapshot:
                    prompt = f"{prompt}\n\n## User Profile\n{_snapshot}"
            except Exception:
                pass  # fail-silent: profile unavailable should not break prompting

        return prompt


# Module-level functions for backwards compatibility
_default_builder: PromptBuilder | None = None


def _get_builder() -> PromptBuilder:
    """Get the default prompt builder."""
    global _default_builder
    if _default_builder is None:
        _default_builder = PromptBuilder()
    return _default_builder


def build_root_lm_prompt(
    state: str,
    original_prompt: str,
    last_output: str = "",
    last_error: str = "",
    turn: int = 0,
    routing_context: str = "",
    corpus_context: str = "",
) -> str:
    """Build the prompt for the Root LM (frontdoor).

    Module-level function for backwards compatibility.
    See PromptBuilder.build_root_lm_prompt() for full documentation.
    """
    result = _get_builder().build_root_lm_prompt(
        state=state,
        original_prompt=original_prompt,
        last_output=last_output,
        last_error=last_error,
        turn=turn,
        routing_context=routing_context,
        corpus_context=corpus_context,
    )
    return result if isinstance(result, str) else result.to_string()


def build_escalation_prompt(
    original_prompt: str,
    state: str,
    failure_context: "EscalationContext",
    decision: "EscalationDecision",
) -> str:
    """Build a prompt for an escalated role.

    Module-level function for backwards compatibility.
    See PromptBuilder.build_escalation_prompt() for full documentation.
    """
    result = _get_builder().build_escalation_prompt(
        original_prompt=original_prompt,
        state=state,
        failure_context=failure_context,
        decision=decision,
    )
    return result if isinstance(result, str) else result.to_string()


def build_step_prompt(
    action: str,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    context: ContextManager | None = None,
) -> str:
    """Build a prompt for step execution.

    Module-level function for backwards compatibility.
    See PromptBuilder.build_step_prompt() for full documentation.
    """
    result = _get_builder().build_step_prompt(
        action=action,
        inputs=inputs,
        outputs=outputs,
        context=context,
    )
    return result if isinstance(result, str) else result.to_string()


def build_stage2_review_prompt(
    draft_summary: str,
    grep_hits: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    original_task: str = "",
) -> str:
    """Build a prompt for Stage 2 of two-stage summarization.

    Module-level function for backwards compatibility.
    See PromptBuilder.build_stage2_review_prompt() for full documentation.
    """
    return _get_builder().build_stage2_review_prompt(
        draft_summary=draft_summary,
        grep_hits=grep_hits,
        figures=figures,
        original_task=original_task,
    )


@routing_stage_timing.timed_stage("routing_context")
def build_routing_context(
    role: str,
    hybrid_router: Any,
    task_description: str,
    max_chars: int = 300,
) -> str:
    """Build compact routing context for injection into Root LM prompt.

    Called on turn 0 only to give the model initial routing intelligence
    from MemRL. Kept very compact to minimize token overhead (~75 tokens).

    Args:
        role: Current model's role.
        hybrid_router: HybridRouter instance (may be None).
        task_description: The user's task.
        max_chars: Maximum characters for routing context.

    Returns:
        Compact routing context string, or "" if unavailable.
    """
    if hybrid_router is None:
        return ""

    from src.roles import get_tier

    try:
        task_ir = canonicalize_task_ir({
            "task_type": "chat",
            "objective": task_description[:TASK_IR_OBJECTIVE_LEN],
        })
        results = hybrid_router.retriever.retrieve_for_routing(task_ir)

        tier = get_tier(role)

        if not results:
            return f"Role: {role} (Tier {tier.value})\nNo similar past tasks found."[:max_chars]

        # Build structured routing data for TOON encoding
        similar = []
        for r in results[:3]:
            ctx = r.memory.context or {}
            similar.append(
                {
                    "role": ctx.get("role", r.memory.action),
                    "Q": round(r.q_value, 2),
                    "outcome": r.memory.outcome or "?",
                }
            )

        best = hybrid_router.retriever.get_best_action(results)
        routing_data = {
            "self": {"role": role, "tier": tier.value},
            "similar": similar,
        }
        if best:
            routing_data["suggested"] = {"role": best[0], "conf": round(best[1], 2)}

        # TOON encoding: ~50% reduction on uniform similar-tasks array
        try:
            from src.services.toon_encoder import is_available, encode

            if is_available() and len(similar) >= 2:
                return encode(routing_data)[:max_chars]
        except Exception:
            pass

        # Fallback: compact text format
        lines = [f"Role: {role} (Tier {tier.value})"]
        for s in similar:
            lines.append(f"  Similar → {s['role']}: Q={s['Q']:.2f} ({s['outcome']})")
        if best:
            lines.append(f"Suggested: {best[0]} (conf={best[1]:.2f})")
        return "\n".join(lines)[:max_chars]

    except Exception:
        return ""


def _role_name_for_registry(role: str) -> str:
    """Normalize enum/generic role names before registry lookup."""
    role_text = str(role or "").strip()
    normalized = Role.from_string(role_text)
    if normalized is not None:
        return normalized.value
    chained = chain_name_to_role(role_text)
    if chained is not None:
        return chained.value
    return role_text


def _get_corpus_registry() -> Any:
    """Load the default registry once for corpus retrieval metadata."""
    global _CORPUS_REGISTRY
    if _CORPUS_REGISTRY is None:
        from src.registry.registry_loader import RegistryLoader

        _CORPUS_REGISTRY = RegistryLoader(validate_paths=False)
    return _CORPUS_REGISTRY


def _corpus_config_from_mapping(config_data: dict[str, Any], corpus_config_cls: Any) -> Any:
    cfg = corpus_config_cls(
        enabled=bool(config_data.get("enabled", False)),
        index_path=str(config_data.get("index_path", "") or ""),
        max_snippets=int(config_data.get("max_snippets", 3) or 3),
        max_chars=int(config_data.get("max_chars", 3000) or 3000),
    )
    for key in (
        "min_score",
        "exact_only",
        "rag_enabled",
        "rag_max_snippets",
        "rag_max_chars",
        "rag_min_score",
        "rag_roles",
    ):
        if key in config_data:
            setattr(cfg, key, config_data[key])
    return cfg


def _same_corpus_config(current: Any, desired: Any) -> bool:
    return (
        bool(getattr(current, "enabled", False)) == bool(getattr(desired, "enabled", False))
        and str(getattr(current, "index_path", "")) == str(getattr(desired, "index_path", ""))
        and int(getattr(current, "max_snippets", 0) or 0)
        == int(getattr(desired, "max_snippets", 0) or 0)
        and int(getattr(current, "max_chars", 0) or 0)
        == int(getattr(desired, "max_chars", 0) or 0)
    )


def _configured_corpus_retriever(retriever_cls: Any, desired_config: Any) -> Any:
    retriever = retriever_cls.get_instance()
    if not _same_corpus_config(retriever.config, desired_config):
        retriever_cls.reset_instance()
        retriever = retriever_cls.get_instance(desired_config)
    else:
        retriever.config = desired_config
    return retriever


def _split_role_list(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        items = value
    else:
        return set()
    return {str(item).strip() for item in items if str(item).strip()}


def _task_scoped_corpus_roles(config_data: dict[str, Any]) -> set[str]:
    """Roles allowed to use corpus retrieval only for code-looking tasks."""
    roles = _split_role_list(config_data.get("task_scoped_roles"))
    roles.update(_split_role_list(config_data.get("code_task_roles")))
    try:
        import os

        roles.update(_split_role_list(os.environ.get(_CORPUS_TASK_SCOPED_ROLES_ENV)))
    except Exception:
        pass
    return roles


def _looks_like_code_corpus_task(task_description: str) -> bool:
    text = str(task_description or "").lower()
    return any(marker in text for marker in _CODE_CORPUS_TASK_MARKERS)


def _emit_corpus_context_event(event: dict[str, Any]) -> None:
    status = str(event.get("status") or "")
    reason = str(event.get("reason") or "")
    msg = (
        "Corpus retrieval %s role=%s reason=%s snippets=%s chars=%s latency_ms=%s"
    )
    args = (
        status,
        event.get("role"),
        reason,
        event.get("snippet_count", 0),
        event.get("context_chars", 0),
        event.get("retrieval_latency_ms", 0.0),
    )
    extra = {"corpus_retrieval": event}
    if status in {"injected"}:
        _log.info(msg, *args, extra=extra)
    elif status in {"error", "slow_query"}:
        _log.warning(msg, *args, extra=extra)
    else:
        _log.debug(msg, *args, extra=extra)


def build_corpus_context(
    role: str,
    task_description: str,
    config: Any | None = None,
    *,
    task_id: str | None = None,
    request_id: str | None = None,
) -> str:
    """Build corpus context for prompt-lookup acceleration.

    Retrieves code snippets from the corpus index and formats them for
    injection into the prompt. Only activates for lookup-enabled roles.

    Args:
        role: Current model's role name.
        task_description: The user's task description.
        config: Optional CorpusConfig override.
        task_id: Optional orchestrator task id for retrieval telemetry.
        request_id: Optional caller request id for retrieval telemetry.

    Returns:
        Formatted reference code string, or "" if not applicable.
    """
    role_name = _role_name_for_registry(role)
    event: dict[str, Any] = {
        "role": role_name,
        "task_id": task_id,
        "request_id": request_id,
        "status": "skipped",
        "reason": "",
        "snippet_count": 0,
        "context_chars": 0,
        "retrieval_latency_ms": 0.0,
    }
    try:
        from src.services.corpus_retrieval import (
            CorpusConfig,
            CorpusRetriever,
            extract_code_query,
        )
    except ImportError as exc:
        event.update(
            status="error",
            reason="import_error",
            failure_detail=f"{type(exc).__name__}: {exc}",
        )
        _emit_corpus_context_event(event)
        return ""

    try:
        desired_config = config
        slow_query_ms = _DEFAULT_CORPUS_SLOW_QUERY_MS
        if desired_config is None:
            try:
                registry = _get_corpus_registry()
                runtime_cfg = registry.get_corpus_config()
                role_cfg = registry.get_role(role_name)
                role_enabled = bool(role_cfg.acceleration.corpus_retrieval)
                task_scoped_roles = _task_scoped_corpus_roles(runtime_cfg)
                task_scope_enabled = False
                event.update(
                    index_path=runtime_cfg.get("index_path", ""),
                    role_corpus_enabled=role_enabled,
                    task_scoped_roles=sorted(task_scoped_roles),
                    runtime_corpus_enabled=bool(runtime_cfg.get("enabled", False)),
                )
                if not runtime_cfg.get("enabled", False):
                    event.update(status="skipped", reason="config_disabled")
                    _emit_corpus_context_event(event)
                    return ""
                if not role_enabled:
                    if role_name in task_scoped_roles:
                        if not _looks_like_code_corpus_task(task_description):
                            event.update(
                                status="skipped",
                                reason="task_scope_disabled",
                                corpus_task_scope="code",
                            )
                            _emit_corpus_context_event(event)
                            return ""
                        role_enabled = True
                        task_scope_enabled = True
                        event.update(role_corpus_enabled=True, role_corpus_scope="code_task")
                    else:
                        event.update(status="skipped", reason="role_disabled")
                        _emit_corpus_context_event(event)
                        return ""
                desired_config = _corpus_config_from_mapping(runtime_cfg, CorpusConfig)
                slow_query_ms = float(
                    runtime_cfg.get(
                        "slow_query_ms",
                        runtime_cfg.get("timeout_ms", _DEFAULT_CORPUS_SLOW_QUERY_MS),
                    )
                    or _DEFAULT_CORPUS_SLOW_QUERY_MS
                )
                if task_scope_enabled:
                    event.update(corpus_task_scope="code")
            except Exception as exc:
                event.update(
                    status="error",
                    reason="registry_error",
                    failure_detail=f"{type(exc).__name__}: {exc}",
                )
                _emit_corpus_context_event(event)
                return ""

        retriever = _configured_corpus_retriever(CorpusRetriever, desired_config)
        event.update(
            index_path=getattr(retriever.config, "index_path", ""),
            runtime_corpus_enabled=bool(getattr(retriever.config, "enabled", False)),
        )

        if not retriever.config.enabled:
            event.update(status="skipped", reason="config_disabled")
            _emit_corpus_context_event(event)
            return ""

        query = extract_code_query(task_description)
        snippets = retriever.retrieve(query)
        diagnostics = retriever.last_diagnostics
        if diagnostics is not None:
            diag_payload = asdict(diagnostics)
            event.update(
                query_ngrams=diag_payload.get("query_ngrams", 0),
                shards_queried=diag_payload.get("shards_queried", 0),
                shards_failed=diag_payload.get("shards_failed", 0),
                shards_unavailable=diag_payload.get("shards_unavailable", 0),
                candidates_found=diag_payload.get("candidates_found", 0),
                retrieval_latency_ms=round(float(diag_payload.get("elapsed_ms") or 0.0), 3),
                failure_reason=diag_payload.get("failure_reason", ""),
                failure_detail=diag_payload.get("failure_detail", ""),
            )
        elapsed_ms = float(event.get("retrieval_latency_ms") or 0.0)
        if slow_query_ms > 0 and elapsed_ms > slow_query_ms:
            event.update(
                status="slow_query",
                reason="retrieval_latency_exceeded",
                slow_query_ms=slow_query_ms,
                snippet_count=len(snippets),
            )
            _emit_corpus_context_event(event)
            return ""
        context = retriever.format_for_prompt(snippets)
        event.update(
            status="injected" if context else "skipped",
            reason="" if context else "no_snippets",
            snippet_count=len(snippets),
            context_chars=len(context),
        )
        _emit_corpus_context_event(event)
        return context
    except Exception as exc:
        event.update(
            status="error",
            reason="retrieval_error",
            failure_detail=f"{type(exc).__name__}: {exc}",
        )
        _emit_corpus_context_event(event)
        return ""


def build_long_context_exploration_prompt(
    original_prompt: str,
    context_chars: int,
    state: str = "",
) -> str:
    """Build a prompt that instructs the model to explore large context via REPL tools.

    Instead of dumping the full context into the LLM's input window, this prompt
    tells the model to use REPL exploration tools (peek, grep, chunk_context,
    summarize_chunks) to process it piece by piece.

    Args:
        original_prompt: The user's original task/question.
        context_chars: Character count of the full context.
        state: Current REPL state.

    Returns:
        Prompt string for the Root LM.
    """
    est_tokens = context_chars // 4
    n_chunks = max(2, min(8, est_tokens // 4000))  # ~4K tokens per chunk

    # Detect search/needle tasks
    search_keywords = ["find", "search", "locate", "identify", "extract", "detect", "where"]
    is_search = any(kw in original_prompt.lower().split() for kw in search_keywords)

    if is_search:
        first_step = """# Step 1: Search for relevant items
matches = grep(r"key|secret|password|token|credential|api_key")
for m in matches:
    print(f"Line {m['line']}: {m['text']}")"""
    else:
        first_step = """# Step 1: Inspect document structure
header = peek(2000)
print(header)"""

    return f"""You are an orchestrator processing a large document ({context_chars:,} chars, ~{est_tokens:,} tokens).

## Task
{original_prompt}

## Strategy
The full document is stored in the `context` variable.
These functions are ALREADY AVAILABLE — call them directly:

```python
{first_step}

# Step 2: Search for specific terms
matches = grep(r"pattern_here")  # returns list of {{"line": int, "text": str, "position": int}}

# Step 3: Parallel analysis (dispatches to workers)
summaries = summarize_chunks(
    task="{original_prompt[:80]}",
    n_chunks={n_chunks}
)  # returns list of {{"index": int, "summary": str}}

# Step 4: Synthesize and return
FINAL("your answer here")
```

**DO NOT** define your own functions. `peek`, `grep`, `summarize_chunks`, `FINAL` are built-ins.
**DO NOT** try to read files from disk — the context is already loaded in memory.

{f"## Current State{chr(10)}{state}" if state else ""}

## Rules
- NEVER import anything — all tools are pre-loaded
- NEVER send full context to llm_call() (too large)
- Use grep() to find specific needles, summarize_chunks() for broad analysis
- Workers return section summaries — you synthesize the final answer
- Output only valid Python code — no markdown, no explanations

Code:"""


def build_task_decomposition_prompt(objective: str, context: str = "") -> str:
    """Prompt for architect to decompose a task into parallel-executable steps.

    Asks for a JSON array of plan steps with TaskIR-compatible fields.
    Uses abbreviated instruction to minimize tokens at 6.75 t/s.

    Args:
        objective: The user's task/question.
        context: Optional context text.

    Returns:
        Prompt string for the architect model.
    """
    context_note = f"\nContext ({len(context)} chars): {context[:200]}..." if context else ""
    return resolve_prompt(
        "task_decomposition", _TASK_DECOMPOSITION_FALLBACK,
        objective=objective[:500],
        context_note=context_note,
    )
