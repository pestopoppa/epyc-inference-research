"""Intent classification and mode selection for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: direct mode detection, mode selection (REPL as unified default),
MemRL-informed mode selection, and proactive intent classification.

NOTE (2026-02-04): React mode has been unified into REPL with structured_mode=True.
The _select_mode() function now only returns "direct" or "repl".
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.constants import TASK_IR_OBJECTIVE_LEN
from src.registry.stack_priors import live_stack_role_records
from src.runtime import routing_stage_timing
from src.task_ir import canonicalize_task_ir

log = logging.getLogger(__name__)


# ── Direct-mode heuristic ─────────────────────────────────────────────

# MCQ answer patterns: "(A)", "A)", "A.", etc. — at least 3 choices
_MCQ_CHOICE_RE = re.compile(
    r"(?:^|\n)\s*(?:\(?[A-E]\)|[A-E]\.|[A-E]\))\s+\S",
)

# Question-word prefixes for short factual questions
_QUESTION_WORDS = (
    "what", "which", "who", "whom", "where", "when",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "will", "would", "should", "has", "have", "had",
    "name", "list", "define",
)

# Keywords that indicate the prompt needs REPL (code execution, tools, etc.)
_REPL_KEYWORDS = (
    "implement", "code", "function", "class ", "method", "debug",
    "refactor", "write a program", "write a script", "write code",
    "algorithm", "data structure", "fix the bug", "compile",
    "step by step", "step-by-step", "multi-step", "research",
    "analyze", "investigate", "explore", "search for",
    "read the file", "run the", "execute", "calculate",
    "```",  # code blocks in prompt
)

# Keywords that strongly indicate tool use is required (not just REPL mode).
# Used for forced tool use: if matched, model must invoke a tool on first turn.
_TOOL_REQUIRED_KEYWORDS = {
    "search for": "grep",
    "find in": "grep",
    "grep": "grep",
    "look up": "peek",
    "read the file": "peek",
    "list the files": "list_dir",
    "calculate": None,  # Need tools but no specific hint
    "compute": None,
    "run the": "run_shell",
}
_EXPLICIT_TOOL_CALL_RE = re.compile(
    r"\b(?:TOOL|CALL)\s*\(\s*['\"]([A-Za-z_][A-Za-z0-9_.:-]*)['\"]"
)
_EXPLICIT_TOOL_FORCE_CONTEXT_RE = re.compile(
    r"\b("
    r"return executable python only|"
    r"use exactly (?:these )?(?:two )?lines|"
    r"must (?:call|invoke|use)|"
    r"call the (?:required )?tool|"
    r"only available from the tool"
    r")\b",
    re.IGNORECASE,
)

_EMERGENCY_HEURISTIC_PRIOR_ROLES = ("frontdoor",)


def detect_tool_requirement(prompt: str) -> tuple[bool, str | None]:
    """Detect if a prompt strongly requires tool use.

    Returns:
        (tool_required, tool_hint) — tool_hint is a specific tool name or None.
    """
    explicit_tool = _EXPLICIT_TOOL_CALL_RE.search(prompt)
    if explicit_tool and _EXPLICIT_TOOL_FORCE_CONTEXT_RE.search(prompt):
        return True, explicit_tool.group(1)

    prompt_lower = prompt.lower()
    for keyword, hint in _TOOL_REQUIRED_KEYWORDS.items():
        if keyword in prompt_lower:
            return True, hint
    return False, None


def _should_use_direct(prompt: str, context: str | None) -> bool:
    """Heuristic: should this prompt bypass REPL and use direct mode?

    Conservative — only short-circuits obvious simple questions:
    - MCQ with choices in prompt (< 2000 chars)
    - Short factual questions (< 300 chars, starts with question word)

    Always returns False for coding tasks, long contexts, or multi-step
    indicators, letting MemRL/REPL handle those.

    Args:
        prompt: The user's prompt.
        context: Optional context text.

    Returns:
        True if direct mode is appropriate.
    """
    prompt_len = len(prompt)
    context_len = len(context) if context else 0

    # Never short-circuit long contexts — REPL needed for exploration
    if context_len > 8000 or prompt_len > 4000:
        return False

    prompt_lower = prompt.lower()

    # Never short-circuit coding tasks or multi-step indicators
    if any(kw in prompt_lower for kw in _REPL_KEYWORDS):
        return False

    # MCQ pattern: at least 3 answer choices + prompt under 2000 chars
    if prompt_len < 2000:
        choices = _MCQ_CHOICE_RE.findall(prompt)
        if len(choices) >= 3:
            log.debug("Direct-mode heuristic: MCQ detected (%d choices)", len(choices))
            return True

    # Short factual question: < 300 chars, starts with question word
    if prompt_len < 300:
        first_word = prompt_lower.lstrip().split()[0] if prompt_lower.strip() else ""
        if first_word in _QUESTION_WORDS:
            log.debug("Direct-mode heuristic: short factual question (%r...)", prompt[:50])
            return True

    return False


@routing_stage_timing.timed_stage("mode")
def _select_mode(
    prompt: str,
    context: str,
    state: "Any",
) -> str:
    """Select execution mode: direct or repl.

    React mode has been unified into REPL with structured_mode=True.
    This function now only returns "direct" or "repl".

    Uses a conservative heuristic for obvious simple questions (MCQ, short
    factual), then MemRL route_with_mode() if available, falls back to REPL.
    REPL is the universal superset: models can FINAL() immediately for simple
    questions, or use tools/escalate/delegate for complex ones.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        state: Application state (may have hybrid_router).

    Returns:
        One of "direct" or "repl".
    """
    # Heuristic short-circuit for obviously simple questions
    if _should_use_direct(prompt, context):
        log.debug("Heuristic: direct mode for simple question")
        return "direct"

    # Try MemRL-based mode selection if available
    if hasattr(state, "hybrid_router") and state.hybrid_router is not None:
        try:
            task_ir = {
                "task_type": "chat",
                "objective": prompt[:TASK_IR_OBJECTIVE_LEN],
                "priority": "interactive",
                "context_length": len(context) if context else 0,
            }
            task_ir = canonicalize_task_ir(task_ir)
            _routing, _strategy, mode = state.hybrid_router.route_with_mode(task_ir)
            # Map legacy "react" to "repl" (React is now unified)
            if mode == "react":
                mode = "repl"
            if mode in ("direct", "repl"):
                return mode
        except Exception as exc:
            log.debug("MemRL route_with_mode failed, using heuristic: %s", exc)

    # Heuristic fallback: REPL is the default (superset of direct + react)
    # Model can FINAL("answer") immediately for simple questions, or use
    # tools/escalate/delegate for complex ones. The REPL can also operate
    # in structured_mode=True for React-style one-tool-per-turn execution.
    #
    # Direct mode is only used when explicitly forced via request.force_mode.
    # The model-initiated mode selection defaults to REPL for MemRL exposure.
    return "repl"


def _classify_and_route(
    prompt: str,
    context: str = "",
    has_image: bool = False,
    binding_router: "Any | None" = None,
) -> tuple[str, str]:
    """Classify prompt intent and proactively route to the best specialist.

    Zero-latency keyword heuristic. Returns (role, strategy).
    Falls back to frontdoor if no strong signal.

    If binding_routing is enabled and a binding_router is provided,
    checks for higher-priority overrides after classification.

    Args:
        prompt: The user's prompt.
        context: Optional context text.
        has_image: Whether the request includes an image.
        binding_router: Optional BindingRouter for priority overrides.

    Returns:
        Tuple of (role_name, routing_strategy).
    """
    from src.classifiers import classify_and_route

    result = classify_and_route(prompt, context, has_image)
    role, strategy = result.role, result.strategy

    # Check binding overrides (OpenClaw pattern)
    if binding_router is not None:
        from src.features import features as _get_features

        if _get_features().binding_routing:
            # Map role to task_type for binding lookup
            task_type = _role_to_task_type(role)
            override = binding_router.resolve(task_type)
            if override is not None and override != role:
                log.info(
                    "Binding override: %s → %s (task_type=%s)",
                    role, override, task_type,
                )
                role = override
                strategy = f"binding:{strategy}"

    return role, strategy


def _role_to_task_type(role: str) -> str:
    """Map a role name to a task type for binding lookup."""
    if "coder" in role:
        return "code"
    if "ingest" in role:
        return "ingest"
    if "architect" in role:
        return "reasoning"
    if "worker_math" in role:
        return "math"
    if "worker_vision" in role:
        return "vision"
    if "worker" in role:
        return "explore"
    return "general"


def _role_record_float(
    record: dict[str, Any],
    section: str,
    field: str,
    default: float = 0.0,
) -> float:
    block = record.get(section)
    if not isinstance(block, dict):
        return default
    value = block.get(field, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _role_residency_rank(record: dict[str, Any]) -> int:
    serving = record.get("serving")
    tier = str(serving.get("tier", "") if isinstance(serving, dict) else "").lower()
    return {"hot": 0, "warm": 1, "cold": 2}.get(tier, 3)


def _heuristic_prior_role_sort_key(item: tuple[str, dict[str, Any]]) -> tuple[int, int, float, str]:
    role, record = item
    return (
        0 if role == "frontdoor" else 1,
        _role_residency_rank(record),
        -_role_record_float(record, "model", "mem_gb"),
        role,
    )


def _degraded_heuristic_prior_roles() -> tuple[str, ...]:
    """Return the emergency heuristic-prior fallback without live topology facts."""
    return _EMERGENCY_HEURISTIC_PRIOR_ROLES


def _live_heuristic_prior_roles() -> tuple[str, ...]:
    live_records = live_stack_role_records()
    if not live_records:
        return _degraded_heuristic_prior_roles()
    role_ids = tuple(
        role
        for role, _record in sorted(
            live_records.items(),
            key=_heuristic_prior_role_sort_key,
        )
    )
    return role_ids or _degraded_heuristic_prior_roles()


def _heuristic_role_priors(
    prompt: str,
    context: str = "",
    has_image: bool = False,
) -> dict[str, float]:
    """Build lightweight heuristic priors over routing roles.

    Priors are advisory only and should be combined with learned evidence.
    """
    role, _ = _classify_and_route(prompt, context, has_image=has_image)
    prior_roles = _live_heuristic_prior_roles()
    role_id = str(role)
    baseline_denominator = (
        len(prior_roles)
        if role_id not in prior_roles
        else max(len(prior_roles) - 1, 1)
    )
    baseline_prior = 0.45 / baseline_denominator if baseline_denominator else 0.15
    priors: dict[str, float] = {candidate: baseline_prior for candidate in prior_roles}
    priors[role_id] = max(priors.get(role_id, 0.0), 0.55)
    if _should_use_direct(prompt, context):
        priors["frontdoor"] = max(priors.get("frontdoor", 0.0), 0.7)
    total = sum(priors.values())
    if total > 0:
        priors = {k: v / total for k, v in priors.items()}
    return priors


def _select_role_from_prior(priors: dict[str, float]) -> str:
    """Select the highest-probability role from priors."""
    if not priors:
        return "frontdoor"
    return max(priors.items(), key=lambda kv: kv[1])[0]
