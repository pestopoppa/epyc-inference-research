"""Specialist-report compression + handle storage helpers.

Extracted from src/api/routes/chat_delegation.py during the 2026-05-22 Task-C
Phase 2 refactor. chat_delegation.py re-exports every public name here.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.delegation_reports import store_report
from src.roles import Role

from .chat_delegation_config import (
    DelegationConfig,
    _delegation_config,
    _normalize_delegate_role,
)

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives


def _trim_block(text: str, max_chars: int) -> str:
    body = (text or "").strip()
    if max_chars <= 0:
        return ""
    if len(body) <= max_chars:
        return body
    return body[:max_chars].rstrip() + "..."



def _store_report_handle(
    report: str,
    delegate_to: str,
    cfg: DelegationConfig | None = None,
) -> dict[str, str] | None:
    cfg = cfg or _delegation_config()
    text = (report or "").strip()
    if not text:
        return None
    try:
        return store_report(text, delegate_to)
    except Exception as exc:
        if cfg.trace_enabled:
            log.warning("Failed to persist delegation report handle: %s", exc)
        return None



def _to_report_handle_text(handle: dict[str, str], summary: str) -> str:
    return (
        f"[REPORT_HANDLE id={handle.get('id')} chars={handle.get('chars')} "
        f"sha16={handle.get('sha16')}]\n"
        f"Summary:\n{(summary or '').strip()}"
    )


_CODER_ROLES = frozenset({str(Role.CODER_ESCALATION)})
_SEARCH_ROLES = frozenset({str(Role.WORKER_GENERAL)})

_CODER_PREAMBLE = "You are {role}. Execute the delegated coding task quickly.\n\n"

_SEARCH_PREAMBLE = (
    "You are {role}. Execute the delegated task.\n\n"
    "You have a Python REPL with these tools:\n"
    "  web_search(query)        — search the web, returns results\n"
    "  web_fetch(url)           — fetch a URL's content\n"
    "  CALL(\"web_research\", query=\"...\") — deep web research with synthesis\n"
    "For factual questions, ALWAYS use web_search() before answering.\n"
    "For math/computation, write Python code — do NOT compute in your head.\n\n"
)

_DEFAULT_PREAMBLE = "You are {role}. Execute the delegated task.\n\n"



def _build_compact_specialist_prompt(
    delegate_to: str,
    question: str,
    brief: str,
    turn: int,
    last_output: str,
    last_error: str,
) -> str:
    """Compact specialist prompt for delegated mode to reduce prefill cost."""
    delegate_role = _normalize_delegate_role(delegate_to)
    # Load role-specific instructions if available (hot-swap)
    role_instructions = ""
    try:
        from src.prompt_builders.resolver import resolve_prompt
        role_text = resolve_prompt(delegate_role, "", subdir="roles")
        if role_text:
            role_instructions = role_text.strip() + "\n\n"
    except Exception:
        pass

    # Role-appropriate preamble
    if delegate_role in _CODER_ROLES:
        preamble = _CODER_PREAMBLE.format(role=delegate_role)
    elif delegate_role in _SEARCH_ROLES:
        preamble = _SEARCH_PREAMBLE.format(role=delegate_role)
    else:
        preamble = _DEFAULT_PREAMBLE.format(role=delegate_role)

    prompt = (
        f"{preamble}"
        f"{role_instructions}"
        f"User question:\n{question}\n\n"
        f"Architect guidance:\n{brief}\n\n"
        "Output Python code only when computation is required. "
        "If you already have a complete implementation/report, output it directly. "
        "If executing in REPL, end with FINAL(answer) when possible.\n"
    )
    if turn > 0 and (last_output or last_error):
        prompt += "\nPrevious turn signals:\n"
        if last_output:
            prompt += f"- output: {_trim_block(last_output, 600)}\n"
        if last_error:
            prompt += f"- error: {_trim_block(last_error, 400)}\n"
    return prompt



def _maybe_summarize_specialist_report(
    report: str,
    question: str,
    primitives: "LLMPrimitives",
    *,
    force: bool = False,
) -> str:
    """Summarize oversized specialist reports via worker_summarize."""
    cfg = _delegation_config()
    text = (report or "").strip()
    if not text:
        return report
    if not cfg.summarize_long_reports:
        return report
    if not force and len(text) < cfg.summarize_report_chars:
        return report
    prompt = (
        "Summarize the specialist report for the architect. Keep only actionable "
        "implementation details and final recommendation. Max 12 bullets, no fluff.\n\n"
        f"Question:\n{question[:1200]}\n\n"
        f"Specialist report:\n{text[:12000]}"
    )
    try:
        summarized = primitives.llm_call(
            prompt,
            role="worker_summarize",
            skip_suffix=True,
            n_tokens=cfg.summarize_n_tokens,
        )
        summarized = (summarized or "").strip()
        if summarized:
            if cfg.trace_enabled:
                log.warning(
                    "Delegation summarize: report_chars=%d -> summary_chars=%d",
                    len(text),
                    len(summarized),
                )
            return summarized
    except Exception as exc:
        if cfg.trace_enabled:
            log.warning("Delegation summarize failed, keeping original report: %s", exc)
    return report



def _compress_report_for_loop(
    report: str,
    question: str,
    primitives: "LLMPrimitives",
    delegate_to: str,
    *,
    skip_summary: bool = False,
) -> tuple[str, dict[str, str] | None]:
    """Persist long reports and return compact handle+summary text.

    skip_summary: the caller uses the full report (rescued), so any summary
    would be discarded — persist the handle as usual but skip the
    worker_summarize LLM call and return the report unchanged.
    """
    cfg = _delegation_config()
    text = (report or "").strip()
    if not text:
        return report, None
    if skip_summary:
        handle = None
        if cfg.report_handles and len(text) >= cfg.report_handle_chars:
            handle = _store_report_handle(text, delegate_to, cfg=cfg)
        return report, handle
    if not cfg.report_handles:
        return _maybe_summarize_specialist_report(text, question, primitives), None
    if len(text) < cfg.report_handle_chars:
        return _maybe_summarize_specialist_report(text, question, primitives), None
    handle = _store_report_handle(text, delegate_to, cfg=cfg)
    if handle is None:
        return _maybe_summarize_specialist_report(text, question, primitives), None
    summary = _maybe_summarize_specialist_report(text, question, primitives, force=True)
    return _to_report_handle_text(handle, summary), handle
