"""Architect-facing prompts: plan review and delegation.

The answer-review verdict and revision prompts were removed by RI-18c (the RI-18
review gate DROP).
"""

from __future__ import annotations

from typing import Any

from src.prompt_builders.resolver import resolve_prompt
from src.registry.stack_priors import live_stack_role_records


# ── Fallback Constants ──────────────────────────────────────────────────────

_PLAN_REVIEW_FALLBACK = """Review plan. Reply JSON ONLY:
{{"d":"ok|reorder|drop|add|reroute","s":0.0-1.0,"f":"<15 words","p":[]}}

d=decision, s=confidence, f=feedback, p=patches (optional)
drop=discard this plan and continue with the normal no-plan route; never refuse the task
Patch format: {{"step":"S1","op":"reroute|drop|add|reorder","v":"new_value"}}

Task: {objective}
Type: {task_type}
Plan:
{steps_section}

Verdict:"""

_ARCHITECT_INVESTIGATE_FALLBACK = """You are a software architect. This pass is for discovery, not for prescribing a fix.

REPL: available for quick math/estimation ONLY. Never write full programs.

OUTPUT FORMAT: Reply with EXACTLY ONE decision line. No explanation before or after.
- Direct answer: D|<answer>
- Delegate to specialist: I|brief:<spec>|to:<role>

Rules:
- For factual/reasoning/multiple-choice: D|answer IMMEDIATELY. No elaboration.
  Never delegate factual questions to any role — specialists cannot look up facts. Answer directly.
- For quick math: compute in REPL, then D|answer
- For code/algorithms/implementation: I|brief:identify the files, call paths, tests, and unknowns that matter; report file relationships and missing evidence only|to:coder_escalation
- For parallel coding subtasks or file-split implementation: I|brief:map the independent file relationships and unknowns first; do not propose a solution|to:coder_escalation
- For debugging/fixing buggy code: ask for the smallest evidence-backed correction, not a rewrite.
- For investigation/search: I|brief:map the open questions, relevant files, and missing evidence; do not propose a solution|to:worker_general
- For long-context reading comprehension: D|answer from the supplied evidence
- Valid roles: {valid_roles_section}

Use the supplied context as an evidence bundle, not as a solution plan. If delegating, the brief should name why each relevant file or slice matters and what remains uncertain. Do not assume omitted files are irrelevant.

CRITICAL: Output the decision line ONLY. Stop generating after D|answer or I|brief:...|to:role. Do NOT explain your reasoning, justify your choice, or add any text after the decision.
{context_section}
Question: {question}

Decision:"""

_ARCHITECT_INVESTIGATE_ROLE_ORDER = (
    "coder_escalation",
    "worker_general",
    "worker_math",
    "worker_summarize",
    "worker_vision",
    "vision_escalation",
)


def _architect_investigate_valid_roles_section() -> str:
    """Return the architect investigate allowlist from live stack truth."""
    live_roles = live_stack_role_records()
    allowed = [role for role in _ARCHITECT_INVESTIGATE_ROLE_ORDER if role in live_roles]
    roles = allowed or list(_ARCHITECT_INVESTIGATE_ROLE_ORDER)
    return ", ".join(roles)


_ARCHITECT_SYNTHESIS_FALLBACK = """The specialist has investigated and reported back. Extract the answer from their report.

Respond ONLY with:
D|<the answer>
{investigate_option}
If the report contains a complete correct answer, respond with:
D|Approved

If the report contains a clear answer (e.g. a single letter or short value), use it — respond D| followed by that answer. For code solutions, programs, or computed values, trust the specialist's result.

For factual/science/MCQ questions: if the specialist's answer contradicts your own strong knowledge, use YOUR answer instead. The specialist may lack domain expertise.

If the report is truly empty or an error, use your own reasoning to answer. Do NOT re-delegate.

Question: {question}

Specialist Report:
{report}

Decision:"""


# ── Plan Review Prompts ───────────────────────────────────────────────────


def build_plan_review_prompt(
    objective: str,
    task_type: str,
    plan_steps: list[dict[str, Any]],
) -> str:
    """Build architect plan review prompt — forces hyper-concise JSON output.

    The architect reviews the frontdoor's tentative plan and can confirm,
    reroute steps, reorder, drop, or add missing steps.

    Args:
        objective: Task objective (truncated to 200 chars).
        task_type: Task type (e.g., "code", "chat").
        plan_steps: List of plan step dicts with id, actor/role, action, deps.

    Returns:
        Prompt string for architect plan review (~100-120 tokens input).
    """
    # Format steps compactly: S1:coder:Implement handler->output.py
    step_lines = []
    for step in plan_steps[:8]:  # Cap at 8 steps
        step_id = step.get("id", "S?")
        actor = step.get("actor", step.get("role", "worker"))
        action = step.get("action", "")[:50]
        outputs = step.get("outputs", step.get("out", []))
        out_str = ",".join(str(o) for o in outputs[:2]) if outputs else ""
        deps = step.get("deps", step.get("inputs", []))
        dep_str = f"({','.join(str(d) for d in deps[:2])})" if deps else ""

        line = f"{step_id}:{actor}:{action}"
        if out_str:
            line += f"->{out_str}"
        if dep_str:
            line += dep_str
        step_lines.append(line)

    steps_block = "\n".join(step_lines)

    return resolve_prompt(
        "plan_review",
        _PLAN_REVIEW_FALLBACK,
        objective=objective[:200],
        task_type=task_type,
        steps_section=steps_block,
    )


# ── Architect Delegation Prompts ─────────────────────────────────────────


def build_architect_investigate_prompt(
    question: str,
    context: str = "",
) -> str:
    """Build prompt asking architect to decide: answer directly or delegate investigation.

    The architect emits TOON-encoded decisions:
    - Direct: ``D|<answer>``
    - Investigate (ReAct): ``I|brief:<text>|to:<role>``
    - Investigate (REPL/draft): ``I|brief:<text>|to:<role>|mode:repl``
    Falls back to JSON if architect ignores TOON instruction.

    Args:
        question: The user's question.
        context: Optional TOON-encoded or plain-text context.

    Returns:
        Prompt string for architect.
    """
    context_section = ""
    if context:
        context_section = (
            f"\nEvidence bundle (for discovery; not a solution plan):\n{context[:3000]}\n"
        )

    return resolve_prompt(
        "architect_investigate",
        _ARCHITECT_INVESTIGATE_FALLBACK,
        context_section=context_section,
        valid_roles_section=_architect_investigate_valid_roles_section(),
        question=question[:2000],
    )


def _specialist_clearly_failed(report: str) -> bool:
    """Return True if the specialist report is empty or an explicit error."""
    if not report or not report.strip():
        return True
    prefixes = ("[ERROR", "[FAILED", "[Investigation failed", "[Delegation failed")
    return report.strip().startswith(prefixes)


def build_architect_synthesis_prompt(
    question: str,
    report: str,
    loop_num: int,
    max_loops: int,
) -> str:
    """Build prompt for architect to synthesize from investigation report.

    The architect extracts/approves the specialist's answer.  Re-delegation
    is only offered when the specialist clearly failed (empty output or
    error prefix).  This prevents pointless loops where the architect
    repeatedly delegates without progress.

    Args:
        question: The original user question.
        report: The specialist's investigation report.
        loop_num: Current loop number (1-indexed).
        max_loops: Maximum allowed loops.

    Returns:
        Prompt string for architect synthesis.
    """
    # Only offer re-delegation if specialist clearly failed AND loops remain
    can_reinvestigate = loop_num < max_loops and _specialist_clearly_failed(report)
    if can_reinvestigate:
        investigate_option = (
            f"\nThe specialist FAILED. If you need to retry (loop {loop_num}/{max_loops}), respond with:\n"
            "I|brief:<what to investigate differently>|to:coder_escalation\n"
        )
    else:
        investigate_option = "\nDo NOT delegate or request further investigation. Answer now.\n"

    return resolve_prompt(
        "architect_synthesis",
        _ARCHITECT_SYNTHESIS_FALLBACK,
        investigate_option=investigate_option,
        question=question[:2000],
        report=report[:6000],
    )
