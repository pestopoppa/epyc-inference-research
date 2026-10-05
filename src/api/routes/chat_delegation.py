"""Architect delegation for chat endpoints.

Extracted from chat.py during Phase 1 decomposition.
Contains: architect decision parsing (TOON/JSON/text),
multi-loop delegation pipeline where architect formulates
investigation briefs and specialists execute via ReAct or REPL.
"""

# ruff: noqa: F401

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import hashlib as _hashlib
import threading

from src.constants import (
    DELEGATION_BRIEF_KEY_LEN,
    DELEGATION_MAX_SAME_TARGET,
    DELEGATION_MAX_TOTAL_TOKENS,
)
from src.delegation_reports import store_report
from src.env_parsing import env_int as _env_int
from src.exceptions import InferenceError
from src.repl_environment import REPLEnvironment

# Phase-1/2/3 extractions (2026-05-22 Task C). chat_delegation.py keeps the
# loop functions in place (Phase 4 deferred to keep test patches working at
# src.api.routes.chat_delegation.{_run_architect_decision,_run_specialist_loop,
# REPLEnvironment}). The extracted helpers below are re-exported so any future
# patches against e.g. chat_delegation._parse_architect_decision keep working.
from .chat_delegation_config import (
    DelegationConfig,
    _delegation_config,
    _delegation_local,
    _delegation_specialist_turn_token_cap,
    _get_delegation_depth,
    _normalize_delegate_role,
    _valid_delegate_roles,
)
from .chat_delegation_decision import (
    _apply_decision_guards,
    _architect_compute_token_budget,
    _architect_decision_token_budget,
    _classify_failure_reason,
    _extract_toon_decision,
    _parse_architect_decision,
    _strip_think,
    resolve_architect_decision,
)
from .chat_delegation_reports import (
    _build_compact_specialist_prompt,
    _compress_report_for_loop,
    _maybe_summarize_specialist_report,
    _store_report_handle,
    _to_report_handle_text,
    _trim_block,
)


log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives


# Valid delegation targets for architect briefs
_VALID_DELEGATE_ROLES = _valid_delegate_roles()

















def _run_architect_decision(
    prompt_text: str,
    question: str,
    primitives: "LLMPrimitives",
    architect_role: str,
    tool_registry: "Any | None",
) -> tuple[str | None, int, int]:
    """Run architect's mini-REPL decision loop (Phase A).

    Returns:
        (arch_response, computation_turns, tool_invocations, failure_reason)
    """
    from src.graph.helpers import _extract_prose_answer
    from src.prompt_builders import extract_code_from_response

    # Mini-REPL loop: architect can compute before deciding
    arch_repl = REPLEnvironment(
        context=question,
        llm_primitives=primitives,
        tool_registry=tool_registry,
        role=architect_role,
    )
    arch_response = None
    arch_last_output = ""
    raw = ""

    for _aturn in range(4):  # Max 4 computation turns
        if _aturn == 0:
            full_prompt = prompt_text
        else:
            full_prompt = (
                f"{prompt_text}\n\n"
                f"Computation result:\n{arch_last_output}\n\n"
                f"Now make your decision (D|answer or I|brief:...|to:role):"
            )
        # Turn 0 = routing decision (tight budget).
        # Turns 1+ = computation follow-up (full budget).
        n_tok = (
            _architect_decision_token_budget(architect_role)
            if _aturn == 0
            else _architect_compute_token_budget(architect_role)
        )
        # Early-stop streaming: abort generation once a complete TOON
        # decision line is detected.  Saves 3000+ tokens of post-decision
        # rambling on the architect model.
        # Uses _strip_think to also handle incomplete <think> blocks so
        # that deliberation *about* delegating isn't mistaken for a real
        # TOON decision.
        # IMPORTANT: Free-form D| answers must wait for a trailing newline
        # before firing.  The old pattern `D\|.+` fired after a single
        # token (e.g. "D|The" → answer="The"), truncating multi-word
        # answers.  MCQ answers (D|A through D|D) still fire immediately
        # via the lookahead shortcut.
        _toon_re = re.compile(r"D\|[A-D](?=[^a-zA-Z]|$)|D\|[^\n]+\n|I\|.+\|to:\w+")
        # Qwen3 CoT is strictly one <think>...</think> then the answer.
        # A second <think> after </think> is always a degenerate loop —
        # kill generation and keep whatever answer appeared between them.
        _think_reentry_re = re.compile(r"</think>.*<think>", re.DOTALL)

        def _architect_early_stop(text: str) -> bool:
            if _toon_re.search(_strip_think(text)):
                return True
            if _think_reentry_re.search(text):
                return True
            return False

        primitives._early_stop_check = _architect_early_stop
        try:
            raw = primitives.llm_call(
                full_prompt,
                role=architect_role,
                skip_suffix=True,
                n_tokens=n_tok,
            )
        except (InferenceError, ConnectionError, TimeoutError, OSError) as e:
            log.warning(f"Architect call failed (turn {_aturn}): {e}")
            return None, _aturn + 1, arch_repl._tool_invocations, _classify_failure_reason(e)
        except Exception as e:
            log.warning(f"Architect call failed unexpectedly (turn {_aturn}): {e}")
            return None, _aturn + 1, arch_repl._tool_invocations, _classify_failure_reason(e)
        finally:
            primitives._early_stop_check = None

        # Strip <think>...</think> tags (reasoning models), including
        # incomplete trailing blocks so mid-thought I|/D| aren't parsed.
        stripped = _strip_think(raw).strip()
        # llm_call may return "[ERROR: ...]" strings instead of raising.
        # Treat these as Phase-A failures so break_reason is populated.
        if stripped.startswith("[ERROR:"):
            return (
                None,
                _aturn + 1,
                arch_repl._tool_invocations,
                _classify_failure_reason(RuntimeError(stripped)),
            )
        # Extract TOON decision from anywhere in the response.
        arch_response = _extract_toon_decision(stripped)
        if arch_response:
            break
        # Check start-of-response for JSON (fallback TOON format).
        # Require minimum length to avoid accepting truncated fragments
        # like '{"' that the model emits when confused by the prompt.
        if (stripped.startswith("{") or stripped.startswith("```")) and len(stripped) > 5:
            arch_response = stripped
            break

        # Turn 0 must be a routing decision (D| or I|).  If the
        # architect wrote prose or code instead, treat it as a direct
        # answer.  DO NOT enter the computation loop — the architect
        # models are slow (6-10 t/s) and 3 computation turns of 3375-
        # 5150 tokens each can burn 10-26 minutes on a single question.
        if _aturn == 0:
            # Prose answer rescue: extract from "The answer is X"
            prose = _extract_prose_answer(stripped)
            if prose:
                arch_response = f"D|{prose}"
                log.info("Architect turn 0: no TOON, prose rescue -> D|%s", prose[:50])
            else:
                # Avoid double-wrapping: if stripped already starts with
                # D| or I| (but _extract_toon_decision couldn't parse it,
                # e.g. bare "D|\n<reasoning>"), pass it through directly
                # so _parse_architect_decision's long-answer rescue can
                # extract the MCQ letter from the reasoning body.
                if stripped and re.match(r"^[DI]\|", stripped):
                    arch_response = stripped
                else:
                    arch_response = f"D|{stripped}" if stripped else raw
                log.info("Architect turn 0: no TOON, treating raw output as D|answer (%d chars)", len(stripped or raw))
            break

        # Turns 1+: treat as Python code — execute in mini-REPL
        code = extract_code_from_response(raw)
        if not code.strip():
            # No code extracted, treat raw output as direct answer
            arch_response = raw
            break
        # Comment-only guard: architect is reasoning in comments, not computing
        if all(
            not ln.strip() or ln.strip().startswith("#")
            for ln in code.split("\n")
        ):
            log.info("Architect computation turn %d: comment-only, treating as direct answer", _aturn)
            arch_response = raw
            break
        result = arch_repl.execute(code)
        if result.is_final:
            arch_response = result.final_answer or ""
            break
        arch_last_output = result.output or result.error or ""
        log.info(f"Architect computation turn {_aturn}: {len(arch_last_output)} chars output")
    else:
        # Exhausted computation turns, use last raw response
        arch_response = raw if raw else f"D|{arch_last_output}"

    return arch_response, _aturn + 1, arch_repl._tool_invocations, None




def _maybe_dcp_seed_context(
    query: str,
    *,
    code_search_fn,
    base_ctx: str,
    budget: int = 2000,
) -> str:
    """DCP-4 (J7): advisory delegation context pre-assembly (flag-gated, default off).

    Returns `base_ctx` augmented with a budget-bounded, rendered bundle of code relevant to
    `query` (so the specialist sees it upfront → fewer reactive top-ups), or `base_ctx`
    unchanged when the flag is off OR assembly fails. Advisory only: reactive discovery in
    the specialist REPL stays fully enabled; this is a seed, not a context firewall.
    Handoff: handoffs/active/delegation-context-preassembly.md (DCP-4).
    """
    from src.features import features as _get_features

    if not _get_features().dcp_pre_assembly:
        return base_ctx
    try:
        from pathlib import Path as _Path

        from src.context_discovery import assemble_delegation_bundle, render_bundle
        from src.repl_environment.task_root import get_task_root

        # BEP/DCP harness (Phase 1, #10): DCP bundle bodies come from the scratch task-root when
        # ORCHESTRATOR_EDIT_ROOT is active; else project_root (get_task_root() == project_root when
        # inactive → default-off parity).
        _root = _Path(get_task_root())

        def _file_reader_fn(path: str):
            try:
                return (_root / path).read_text(encoding="utf-8")
            except Exception:
                return None

        bundle = assemble_delegation_bundle(
            query,
            budget=budget,
            code_search_fn=code_search_fn,
            file_reader_fn=_file_reader_fn,
        )
        seed = render_bundle(bundle, file_reader_fn=_file_reader_fn)
        if seed:
            return (
                f"{base_ctx}\n\n[DCP pre-assembled context]\n{seed}".strip()
                if base_ctx
                else seed
            )
    except Exception as e:  # noqa: BLE001 — advisory; never block delegation
        log.debug("DCP-4 pre-assembly skipped: %s", e)
    return base_ctx


def _run_specialist_loop(
    question: str,
    context: str,
    brief: str,
    delegate_to: str,
    delegate_mode: str,
    primitives: "LLMPrimitives",
    tool_registry: "Any | None",
    time_budget_s: float | None = None,
) -> tuple[str, int, list[str], list[dict], bool, bool, dict[str, Any], list[dict]]:
    """Run specialist delegation execution loop (Phase B).

    Returns:
        (report, tool_invocations, tools_called, tool_timings, timed_out,
        report_rescued, inference_metadata, repl_turn_errors)
    """
    from src.prompt_builders import (
        build_root_lm_prompt,
        extract_code_from_response,
        auto_wrap_final,
    )
    from src.prompt_builders.builder import build_corpus_context
    import hashlib

    tools_called: list[str] = []
    phase_tool_timings: list[dict] = []
    timed_out = False
    report_rescued = False
    infer_meta_last: dict[str, Any] = {}
    repl_turn_errors: list[dict] = []  # Per-turn error tracking for observability
    cfg = _delegation_config()

    # Both react and repl modes use the same REPL loop
    # (react = structured_mode=True, repl = structured_mode=False)
    structured = delegate_mode == "react"
    max_delegate_turns = (
        cfg.specialist_max_turns_react
        if delegate_mode == "react"
        else cfg.specialist_max_turns_repl
    )
    specialist_turn_cap = _delegation_specialist_turn_token_cap(
        delegate_mode=delegate_mode,
        question=question,
        brief=brief,
        delegate_to=delegate_to,
        cfg=cfg,
    )
    # Coding tasks (USACO, competitive programming) need more specialist time
    _q_lower = f"{question}\n{brief}".lower()
    _coding_time_signals = (
        "usaco", "codeforces", "leetcode", "sample input", "input format",
        "output format", "stdin", "write a python", "implement",
        "algorithm", "write code",
    )
    _is_coding = any(s in _q_lower for s in _coding_time_signals)
    _specialist_cap = cfg.specialist_max_seconds * 2.5 if _is_coding else cfg.specialist_max_seconds
    specialist_time_budget_s = (
        min(_specialist_cap, float(time_budget_s))
        if time_budget_s is not None
        else _specialist_cap
    )
    q_for_specialist = _trim_block(question, cfg.specialist_question_chars)
    brief_for_specialist = _trim_block(brief, cfg.specialist_brief_chars)
    ctx_for_specialist = _trim_block(context, cfg.specialist_context_chars)
    specialist_started = time.perf_counter()
    try:
        deleg_repl = REPLEnvironment(
            context=(
                f"{brief_for_specialist}\n\nContext:\n{ctx_for_specialist}"
                if ctx_for_specialist
                else brief_for_specialist
            ),
            llm_primitives=primitives,
            tool_registry=tool_registry,
            role=delegate_to,
            structured_mode=structured,
        )
        deleg_last_output = ""
        deleg_last_error = ""
        _prev_code_hash = ""  # dedup guard
        # Build specialist task: full question + architect's brief as
        # design guidance.  Previously only the brief was passed, so the
        # specialist never saw the actual problem (constraints, examples,
        # method signatures).
        specialist_task = (
            f"{q_for_specialist}\n\n"
            f"## Architect guidance\n{brief_for_specialist}"
        )
        # Retrieve corpus context once for the delegation (turn 0 only)
        corpus_ctx = (
            build_corpus_context(
                role=delegate_to,
                task_description=q_for_specialist,
                task_id=primitives.get_request_task_id(),
                request_id=primitives.get_request_id(),
            )
            if cfg.specialist_corpus_context
            else ""
        )
        # DCP-4 (J7): advisory pre-assembled code bundle (flag-gated, default off).
        # No-op (returns corpus_ctx unchanged) when dcp_pre_assembly is off; otherwise seeds
        # the specialist with relevant code via the same ColGREP the REPL uses reactively.
        corpus_ctx = _maybe_dcp_seed_context(
            q_for_specialist,
            code_search_fn=lambda q, limit=20: deleg_repl._code_search(q, limit=limit),  # noqa: SLF001
            base_ctx=corpus_ctx,
        )

        for _turn in range(max_delegate_turns):
            elapsed_s = time.perf_counter() - specialist_started
            if elapsed_s >= specialist_time_budget_s:
                log.warning(
                    "Specialist loop timeout: role=%s mode=%s turns=%d elapsed=%.1fs budget=%.1fs",
                    delegate_to, delegate_mode, _turn, elapsed_s, specialist_time_budget_s,
                )
                timed_out = True
                report = (deleg_last_output or "").strip()
                if report.lower() in {"", "[no output]", "observation: [no output]"}:
                    report = (
                        f"[Delegation timeout after {_turn} turn(s), {elapsed_s:.1f}s. "
                        f"Specialist role={delegate_to}, mode={delegate_mode}. "
                        f"Brief={brief[:160]}]"
                    )
                break
            if cfg.compact_specialist_prompt:
                deleg_prompt = _build_compact_specialist_prompt(
                    delegate_to=delegate_to,
                    question=q_for_specialist,
                    brief=brief_for_specialist,
                    turn=_turn,
                    last_output=deleg_last_output,
                    last_error=deleg_last_error,
                )
            else:
                repl_state = deleg_repl.get_state()
                deleg_prompt = build_root_lm_prompt(
                    state=repl_state,
                    original_prompt=specialist_task,
                    last_output=deleg_last_output,
                    last_error=deleg_last_error,
                    turn=_turn,
                    corpus_context=corpus_ctx if _turn == 0 else "",
                )
            _final_re = re.compile(
                r"""FINAL\(\s*(?:'{3}(.+?)'{3}|"{3}(.+?)"{3}|["'](.+?)["']|(\S+?))\s*\)""",
                re.DOTALL,
            )
            _call_re = re.compile(
                r'CALL\s*\(\s*"[^"]+"\s*(?:,\s*\w+\s*=\s*(?:"[^"]*"|\'[^\']*\'|\d+|True|False|None))*\s*\)',
            )
            primitives._early_stop_check = lambda text: (
                bool(_final_re.search(text)) or bool(_call_re.search(text))
            )
            llm_started = time.perf_counter()
            try:
                code = primitives.llm_call(
                    deleg_prompt,
                    role=delegate_to,
                    stop_sequences=["\n```\n"],
                    n_tokens=specialist_turn_cap,
                )
            finally:
                primitives._early_stop_check = None
            llm_elapsed_ms = (time.perf_counter() - llm_started) * 1000
            # TD-21.33: `primitives.llm_call` above and this read are in the SAME synchronous
            # call (no thread/task boundary crossed), so the per-call-safe ContextVar getter
            # applies cleanly -- unlike `_last_inference_meta`, a concurrent request against
            # this same (possibly shared) instance can never clobber it between the two lines.
            # Runtime `isinstance` (not the `TYPE_CHECKING`-only import above) so a test double
            # that is not a real `LLMPrimitives` keeps its pre-existing exact behavior.
            from src.llm_primitives import LLMPrimitives as _LLMPrimitivesRT

            _meta_val = (
                primitives.get_last_inference_meta()
                if isinstance(primitives, _LLMPrimitivesRT)
                else getattr(primitives, "_last_inference_meta", None)
            )
            infer_meta_last = dict(_meta_val or {})
            if infer_meta_last:
                infer_meta_last["llm_elapsed_ms"] = round(llm_elapsed_ms, 1)
            raw_deleg_output = code
            if cfg.trace_enabled:
                log.warning(
                    "Delegation trace turn=%d role=%s mode=%s prompt_chars=%d raw_chars=%d llm_ms=%.1f infer=%s",
                    _turn,
                    delegate_to,
                    delegate_mode,
                    len(deleg_prompt),
                    len(raw_deleg_output or ""),
                    llm_elapsed_ms,
                    infer_meta_last,
                )
            code = extract_code_from_response(code)
            code = auto_wrap_final(code)
            if cfg.trace_enabled:
                log.warning(
                    "Delegation trace turn=%d extracted_code_chars=%d has_final=%s",
                    _turn,
                    len(code or ""),
                    "FINAL(" in (code or ""),
                )
            # Dedup guard: if coder generates identical code twice in a
            # row, inject an error to break the loop instead of wasting
            # another turn on the same silent execution.
            _code_hash = hashlib.sha256(code.encode()).hexdigest()
            if _code_hash == _prev_code_hash:
                # Identical code won't produce a different result — stop looping.
                # Return the raw output as a rescued report instead of burning
                # another LLM call that will likely repeat the same code again.
                repl_turn_errors.append({"turn": _turn, "error": "dedup_identical_code"})
                report = (raw_deleg_output or deleg_last_output or "").strip()
                if report:
                    report_rescued = True
                    log.info(
                        "Dedup guard: identical code on turn %d, returning existing output as report (%d chars)",
                        _turn, len(report),
                    )
                break
            _prev_code_hash = _code_hash
            # Prose report rescue: specialist answered in prose without
            # code or FINAL().  In delegation, the specialist's prose IS
            # the investigation report — return it to the architect for
            # synthesis instead of discarding it or looping.
            if "FINAL(" not in code and not any(
                ln.strip() and not ln.strip().startswith("#")
                and any(ln.strip().startswith(kw) for kw in (
                    "def ", "class ", "for ", "while ", "if ", "try:",
                    "print(", "result =", "answer =",
                ))
                for ln in code.split("\n")
            ):
                # No executable code found — treat raw prose as report
                report = raw_deleg_output.strip()
                if report:
                    report_rescued = True
                    log.info(
                        "Delegation prose report (turn %d): %d chars returned to architect",
                        _turn, len(report),
                    )
                    break
            # Code-report rescue: specialist returned substantial code/design
            # text but omitted FINAL(). For delegation we need a report, not
            # necessarily an executed terminal value. Accept it directly to
            # avoid repeated 30s generation loops that only attempt FINAL().
            # Exception: don't rescue code containing input() — it's a
            # competitive programming solution that needs to be rewritten
            # using CALL("run_python_code", ...) before it can work.
            _has_blocked_input = "input()" in code and "CALL(" not in code
            if "FINAL(" not in code and not _has_blocked_input:
                non_comment_lines = [
                    ln for ln in code.split("\n")
                    if ln.strip() and not ln.strip().startswith("#")
                ]
                if len(non_comment_lines) >= 6 or len(code.strip()) >= 400:
                    report = raw_deleg_output.strip() or code.strip()
                    if report:
                        report_rescued = True
                        log.info(
                            "Delegation code report rescue (turn %d): %d chars returned to architect",
                            _turn, len(report),
                        )
                        break
            # Comment-only guard
            if all(
                not ln.strip() or ln.strip().startswith("#")
                for ln in code.split("\n")
            ):
                deleg_last_error = (
                    "Your output was all comments — no executable code ran. "
                    "Write Python code that computes the answer and call FINAL(answer)."
                )
                repl_turn_errors.append({"turn": _turn, "error": "comment_only"})
                deleg_last_output = ""
                continue
            exec_started = time.perf_counter()
            result = deleg_repl.execute(code)
            exec_elapsed_ms = (time.perf_counter() - exec_started) * 1000
            if cfg.trace_enabled:
                log.warning(
                    "Delegation trace turn=%d exec_ms=%.1f is_final=%s output_chars=%d error_chars=%d",
                    _turn,
                    exec_elapsed_ms,
                    bool(getattr(result, "is_final", False)),
                    len((getattr(result, "output", "") or "")),
                    len((getattr(result, "error", "") or "")),
                )
            if result.is_final:
                report = result.final_answer or ""
                # Specialist already reached a terminal answer; return directly
                # instead of spending another architect synthesis turn.
                report_rescued = True
                break
            deleg_last_output = result.output or ""
            deleg_last_error = result.error or ""
            if deleg_last_error:
                # Classify the error for observability
                _err = deleg_last_error
                if "NameError" in _err:
                    _err_type = "name_error"
                elif "Unknown tool" in _err or "ValueError" in _err:
                    _err_type = "tool_error"
                elif "SyntaxError" in _err:
                    _err_type = "syntax_error"
                elif "Timeout" in _err or "timeout" in _err:
                    _err_type = "timeout"
                else:
                    _err_type = "runtime_error"
                repl_turn_errors.append({
                    "turn": _turn,
                    "error": _err_type,
                    "detail": _err[:200],
                })
        else:
            log.warning(
                "Specialist loop turn cap reached: role=%s mode=%s turns=%d",
                delegate_to, delegate_mode, max_delegate_turns,
            )
            report = deleg_repl.get_state()
        # REQUEST-LOCAL invocation records captured at the _invoke_tool chokepoint
        # (src/repl_environment/context.py). Deliberately NOT
        # deleg_repl.tool_registry.get_invocation_log(): the registry is
        # process-global (one instance per API process), so its ring interleaves
        # every concurrently served request's calls and this specialist phase was
        # attributing other requests' tools — including their web_research results,
        # which feed the Search-R1 reward pipeline below.
        for inv in list(getattr(deleg_repl, "_invoked_tools", None) or []):
            tools_called.append(inv.tool_name)
            # Estimate tool output tokens (~4 chars per token)
            _output_tokens = 0
            if inv.success and inv.result is not None:
                if isinstance(inv.result, str):
                    _output_tokens = len(inv.result) // 4
                elif isinstance(inv.result, dict):
                    _output_tokens = len(str(inv.result)) // 4
            phase_tool_timings.append(
                {"tool_name": inv.tool_name, "elapsed_ms": inv.elapsed_ms, "success": inv.success,
                 "output_tokens": _output_tokens}
            )
            # Capture web_research results for Search-R1 reward pipeline
            if inv.tool_name == "web_research" and inv.success and isinstance(getattr(inv, "result", None), dict):
                wr = inv.result
                phase_tool_timings.append({
                    "tool_name": "_web_research_result",
                    "web_research_data": {
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
                    },
                })
    except (InferenceError, ConnectionError, TimeoutError, OSError) as e:
        report = f"[Delegation failed: {e}]"
        err_text = str(e).lower()
        timed_out = any(
            marker in err_text
            for marker in ("timeout", "timed out", "deadline", "lock timeout", "cancel")
        )
        return report, 0, tools_called, phase_tool_timings, timed_out, report_rescued, infer_meta_last, repl_turn_errors
    except Exception as e:
        report = f"[Delegation failed (unexpected): {e}]"
        return report, 0, tools_called, phase_tool_timings, timed_out, report_rescued, infer_meta_last, repl_turn_errors

    return (
        report,
        deleg_repl._tool_invocations,
        tools_called,
        phase_tool_timings,
        timed_out,
        report_rescued,
        infer_meta_last,
        repl_turn_errors,
    )


def _architect_delegated_answer(
    question: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "Any",
    architect_role: str = "architect_general",
    max_loops: int = 3,
    force_response_on_cap: bool = True,
) -> tuple[str, dict]:
    """Multi-loop architect delegation: architect decides, specialist investigates.

    Loop flow:
      1. Architect decides (answer directly or delegate)
      2. If delegate: specialist runs ReAct or REPL loop
      3. Architect synthesizes from report (or requests more)
      4. Repeat up to max_loops

    Args:
        question: The user's question.
        context: Optional context text.
        primitives: LLM primitives for inference.
        state: Application state (has tool_registry, etc.).
        architect_role: Architect role to use.
        max_loops: Maximum delegation loops.
        force_response_on_cap: If True, force architect to answer when cap reached.
            If False, return partial with needs_input flag (production mode).

    Returns:
        Tuple of (answer_text, stats_dict).
    """
    from src.orchestration.interaction import (
        Interaction,
        InteractionTelemetry,
        SchedulerPolicy,
    )

    total_tools = 0
    all_tools_called: list[str] = []
    interaction = Interaction(
        kind="delegate",
        owner_role=architect_role,
        callee_role=architect_role,
        skill="architect_delegation",
        scheduler_policy=SchedulerPolicy.from_primitives(primitives),
        telemetry=InteractionTelemetry(
            interaction_type="delegate",
            skill="architect_delegation",
        ),
    )
    interaction.start()
    stats: dict = {
        "loops": 0,
        "phases": [],
        "specialist_output": "",
        "needs_input": False,
        "tools_used": 0,
        "delegation_events": [],
        "tool_timings": [],
        "cap_reached": False,
        "break_reason": "",
        "effective_max_loops": max_loops,
        "reentrant_depth": 0,
        "report_handles": [],
        "delegation_cache_lookups": 0,
        "delegation_cache_hits": 0,
        "delegation_cache_misses": 0,
    }

    # Optional TOON encoding for context
    toon_context = context
    if context:
        try:
            from src.services.toon_encoder import encode, is_available

            if is_available() and len(context) > 100:
                toon_context = encode({"ctx": context[:3000]})
        except Exception as exc:
            log.debug("TOON encoding failed, using raw context: %s", exc)

    tool_registry = getattr(state, "tool_registry", None)
    reports: list[str] = []
    previous_brief_keys: set[str] = set()  # Semantic dedup: hash(brief + delegate_to)
    delegate_history: list[str] = []  # Track delegation targets for repetition guard

    # Re-entrance guard: if specialist escalates back to architect, reduce max_loops
    depth = _get_delegation_depth()
    if depth > 0:
        effective_max_loops = min(max_loops, 1)
        log.warning(
            "Re-entrant delegation detected (depth=%d), reducing max_loops %d -> %d",
            depth, max_loops, effective_max_loops,
        )
    else:
        effective_max_loops = max_loops
    stats["effective_max_loops"] = effective_max_loops
    stats["reentrant_depth"] = depth

    _delegation_local.depth = depth + 1
    try:
        answer, result_stats = _architect_delegated_answer_inner(
            question, context, primitives, state, architect_role,
            effective_max_loops, force_response_on_cap, toon_context,
            tool_registry, reports, previous_brief_keys, delegate_history,
            total_tools, all_tools_called, stats, interaction,
        )
        interaction.complete()
        return answer, result_stats
    except Exception:
        interaction.fail()
        raise
    finally:
        _delegation_local.depth = depth


def _architect_delegated_answer_inner(
    question: str,
    context: str,
    primitives: "LLMPrimitives",
    state: "Any",
    architect_role: str,
    max_loops: int,
    force_response_on_cap: bool,
    toon_context: str,
    tool_registry: "Any | None",
    reports: list[str],
    previous_brief_keys: set[str],
    delegate_history: list[str],
    total_tools: int,
    all_tools_called: list[str],
    stats: dict,
    interaction: "Any",
) -> tuple[str, dict]:
    """Inner loop extracted to keep try/finally clean in the outer function."""
    from src.prompt_builders import (
        build_architect_investigate_prompt,
        build_architect_synthesis_prompt,
    )
    cfg = _delegation_config()
    cumulative_delegate_tokens = 0
    orchestration_started = time.perf_counter()
    # Coding tasks need more total delegation budget to fit longer specialist runs
    _q_lower = question.lower()
    _coding_budget_signals = (
        "usaco", "codeforces", "leetcode", "sample input", "input format",
        "output format", "stdin", "implement", "algorithm",
    )
    _is_coding_task = any(s in _q_lower for s in _coding_budget_signals)
    total_budget_s = cfg.total_max_seconds * 2.0 if _is_coding_task else cfg.total_max_seconds
    stats.setdefault("report_handles", [])
    # User-facing counterpart of `reports`: `reports` may hold compact
    # handle+summary text for the architect loop prompt, but every return that
    # hands a specialist report to the user must use the full report (DCP-13).
    full_reports: list[str] = []

    for loop in range(max_loops + 1):  # +1 for initial decision
        total_elapsed_s = time.perf_counter() - orchestration_started
        if total_elapsed_s >= total_budget_s:
            log.warning(
                "Delegation total timeout: elapsed=%.1fs budget=%.1fs",
                total_elapsed_s, total_budget_s,
            )
            stats["break_reason"] = "wall_clock_budget"
            break
        phase_start = time.perf_counter()

        # ── Phase A: Architect call with computation REPL ──
        if loop == 0:
            prompt_text = build_architect_investigate_prompt(question, toon_context)
        else:
            report = reports[-1] if reports else ""
            prompt_text = build_architect_synthesis_prompt(
                question,
                report,
                loop,
                max_loops,
            )

        decision_result = _run_architect_decision(
            prompt_text, question, primitives, architect_role, tool_registry,
        )
        if len(decision_result) == 3:
            # Backward-compatible path for tests/patches that still mock the legacy tuple.
            arch_response, computation_turns, arch_tools = decision_result
            phase_a_failure_reason = None
        else:
            arch_response, computation_turns, arch_tools, phase_a_failure_reason = decision_result
        total_tools += arch_tools

        if arch_response is None:
            # Early error return
            stats["break_reason"] = phase_a_failure_reason or "pre_delegation_architect_error"
            stats["phases"].append(
                {
                    "loop": loop,
                    "phase": "A",
                    "ms": round((time.perf_counter() - phase_start) * 1000),
                    "decision": "error",
                    "computation_turns": computation_turns,
                }
            )
            if full_reports:
                return full_reports[-1], stats
            return "[ERROR: Architect delegation failed]", stats

        phase_a_ms = (time.perf_counter() - phase_start) * 1000
        # TD-21.4: fish first (unchanged behaviour on a clean parse); repair
        # on a miss instead of ever serving unparsed prose to the user or
        # silently clamping an unrecognized delegate_to/delegate_mode.
        decision = resolve_architect_decision(
            arch_response, primitives=primitives, architect_role=architect_role,
        )
        decision_answer = str(decision.get("answer", "") or "").strip()
        if decision["mode"] == "direct" and decision_answer.startswith("[ERROR:"):
            stats["break_reason"] = _classify_failure_reason(RuntimeError(decision_answer))
            stats["phases"].append(
                {
                    "loop": loop,
                    "phase": "A",
                    "ms": round(phase_a_ms),
                    "decision": "error",
                    "computation_turns": computation_turns,
                }
            )
            if full_reports:
                return full_reports[-1], stats
            return "[ERROR: Architect delegation failed]", stats
        stats["phases"].append(
            {
                "loop": loop,
                "phase": "A",
                "ms": round(phase_a_ms),
                "decision": decision["mode"],
                "computation_turns": computation_turns,
            }
        )

        log.info(f"Architect loop {loop}: {decision['mode']} ({phase_a_ms:.0f}ms, {computation_turns} turns)")

        decision = _apply_decision_guards(decision, question, loop, primitives, architect_role)

        # ── Direct/final answer ──
        if decision["mode"] == "direct":
            answer = decision["answer"]
            # If architect just says "Approved" and we have specialist output,
            # the specialist's document IS the final answer
            if answer.lower().strip().startswith("approved") and stats["specialist_output"]:
                answer = stats["specialist_output"]
            stats["loops"] = loop
            stats["tools_used"] = max(
                total_tools,
                len(all_tools_called),
                len(stats.get("tool_timings", [])),
            )
            stats["tools_called"] = all_tools_called
            return answer, stats

        # ── Last-loop guard: skip specialist, go to forced synthesis ──
        # Running the specialist on the final iteration wastes hundreds of
        # seconds and the result is thrown away by forced synthesis anyway.
        if loop >= max_loops:
            log.info("Architect still wants to investigate on last loop (%d), forcing synthesis", loop)
            stats["break_reason"] = "max_loops"
            break

        # ── Phase B: Specialist execution ──
        brief = decision["brief"]
        delegate_to = decision["delegate_to"]
        delegate_mode = decision["delegate_mode"]

        # ── Loop guard 1: Semantic dedup ──
        # Hash brief + delegate_to as combined key. Architect rephrasing
        # the same brief to a different target still gets caught because
        # we also check target repetition separately (guard 3).
        brief_key = _hashlib.sha256(
            f"{brief.strip().lower()[:DELEGATION_BRIEF_KEY_LEN]}|{delegate_to}".encode()
        ).hexdigest()
        if brief_key in previous_brief_keys:
            log.warning(
                "Semantic dedup: duplicate brief+target (loop %d, target=%s), forcing synthesis",
                loop, delegate_to,
            )
            stats["break_reason"] = "semantic_dedup"
            break
        previous_brief_keys.add(brief_key)

        # ── Loop guard 2: Max cumulative tokens ──
        if cumulative_delegate_tokens > DELEGATION_MAX_TOTAL_TOKENS:
            log.warning(
                "Token budget exceeded: %d > %d tokens across delegation loops, forcing synthesis",
                cumulative_delegate_tokens, DELEGATION_MAX_TOTAL_TOKENS,
            )
            stats["break_reason"] = "token_budget"
            break

        # ── Loop guard 3: Role repetition ──
        # If architect delegates to the same role N consecutive times, force synthesis.
        delegate_history.append(delegate_to)
        if len(delegate_history) >= DELEGATION_MAX_SAME_TARGET:
            recent = delegate_history[-DELEGATION_MAX_SAME_TARGET:]
            if all(r == delegate_to for r in recent):
                log.warning(
                    "Role repetition guard: %s delegated %d consecutive times, forcing synthesis",
                    delegate_to, DELEGATION_MAX_SAME_TARGET,
                )
                stats["break_reason"] = "role_repetition"
                break

        log.info(f"Delegating to {delegate_to} (mode={delegate_mode}): {brief[:100]}...")

        phase_b_start = time.perf_counter()
        tokens_before = primitives.total_tokens_generated
        remaining_budget_s = total_budget_s - (time.perf_counter() - orchestration_started)
        specialist_budget_s = max(10.0, remaining_budget_s - 5.0)

        # ── Delegation result cache: check before running specialist ──
        from src.delegation_cache import get_delegation_cache as _get_deleg_cache
        _deleg_cache = _get_deleg_cache()
        _cache_key = _deleg_cache.make_key(brief, delegate_to)
        phase_tool_timings: list[dict] = []
        specialist_repl_errors: list[dict] = []
        stats["delegation_cache_lookups"] = (
            int(stats.get("delegation_cache_lookups", 0) or 0) + 1
        )
        # Legacy cache entries contain only the loop-compressed text. They
        # cannot safely stand in for the original user-facing report.
        _cached = _deleg_cache.get(_cache_key, require_full_report=True)
        if _cached is not None and not getattr(_cached, "full_report", None):
            _cached = None
        if _cached is not None:
            report = _cached.report
            report_for_loop = report
            final_report = report
            full_report = _cached.full_report
            report_handle = _cached.report_handle
            specialist_timed_out = False
            report_rescued = False
            specialist_infer_meta: dict = {}
            if report_handle:
                stats["report_handles"].append(report_handle)
            stats["delegation_cache_hits"] = (
                int(stats.get("delegation_cache_hits", 0) or 0) + 1
            )
            log.info(
                "Delegation cache hit (loop %d, target=%s, age=%.0fs)",
                loop, delegate_to, _cached.age_seconds,
            )
        else:
            stats["delegation_cache_misses"] = (
                int(stats.get("delegation_cache_misses", 0) or 0) + 1
            )
            # Give specialist its full role timeout instead of squeezing it
            # by the parent's remaining deadline.  The specialist loop already
            # enforces wall-clock limits via elapsed checks and specialist_budget_s.
            from src.config import get_config as _get_config
            _role_timeout_s = float(_get_config().timeouts.for_role(delegate_to))
            _specialist_deadline_s = time.perf_counter() + max(_role_timeout_s, specialist_budget_s)
            _get_workload_class = getattr(primitives, "get_request_workload_class", None)
            _workload_class = _get_workload_class() if callable(_get_workload_class) else None

            with primitives.request_context(
                cancel_check=primitives.get_request_cancel_check(),
                deadline_s=_specialist_deadline_s,
                task_id=primitives.get_request_task_id(),
                priority=primitives.get_request_priority(),
                workload_class=_workload_class,
            ):
                report, deleg_tools, deleg_tools_called, phase_tool_timings, specialist_timed_out, report_rescued, specialist_infer_meta, specialist_repl_errors = _run_specialist_loop(
                    question,
                    context,
                    brief,
                    delegate_to,
                    delegate_mode,
                    primitives,
                    tool_registry,
                    time_budget_s=specialist_budget_s,
                )
            total_tools += deleg_tools
            all_tools_called.extend(deleg_tools_called)
            full_report = report
            # Rescued reports are returned in full, so skip the (discarded)
            # summarization call; the report handle is still persisted.
            report_for_loop, report_handle = _compress_report_for_loop(
                report, question, primitives, delegate_to,
                skip_summary=report_rescued,
            )
            final_report = full_report if report_rescued else report_for_loop
            if report_handle:
                stats["report_handles"].append(report_handle)

            # Store in cache for future reuse (skip failed/error reports)
            cache_report = final_report
            if cache_report and not cache_report.startswith(("[ERROR", "[Delegation failed")):
                delegate_tokens_for_cache = primitives.total_tokens_generated - tokens_before
                _deleg_cache.put(
                    _cache_key, cache_report, delegate_to,
                    tokens_used=delegate_tokens_for_cache,
                    report_handle=report_handle,
                    full_report=full_report,
                )

        phase_b_ms = (time.perf_counter() - phase_b_start) * 1000
        delegate_tokens = primitives.total_tokens_generated - tokens_before
        cumulative_delegate_tokens += delegate_tokens
        reports.append(report_for_loop)
        full_reports.append(full_report)
        stats["specialist_output"] = full_report
        stats["phases"].append(
            {
                "loop": loop,
                "phase": "B",
                "ms": round(phase_b_ms),
                "delegate_to": delegate_to,
                "delegate_mode": delegate_mode,
            }
        )
        stats["tool_timings"].extend(phase_tool_timings)
        # Delegation telemetry
        report_text = final_report or ""
        failed_prefixes = (
            "[ERROR",
            "[Delegation failed",
            "[Delegation timeout",
            "[Investigation failed",
            "[REPL delegation failed",
        )
        success = bool(report_text) and not report_text.startswith(failed_prefixes)
        stats["delegation_events"].append(
            interaction.emit_event(
                to_role=delegate_to,
                task_summary=brief[:DELEGATION_BRIEF_KEY_LEN],
                success=success,
                elapsed_ms=round(phase_b_ms),
                tokens_generated=delegate_tokens,
                metadata={
                    "inference_meta": {
                    "transport": specialist_infer_meta.get("transport"),
                    "completion_reason": specialist_infer_meta.get("completion_reason"),
                    "prompt_ms": specialist_infer_meta.get("prompt_ms"),
                    "gen_ms": specialist_infer_meta.get("gen_ms"),
                    "first_token_ms": specialist_infer_meta.get("first_token_ms"),
                    "chunks": specialist_infer_meta.get("stream_chunks"),
                    "tokens": specialist_infer_meta.get("tokens"),
                    "llm_elapsed_ms": specialist_infer_meta.get("llm_elapsed_ms"),
                    },
                    "repl_turn_errors": specialist_repl_errors,
                },
            ).to_delegation_event_payload()
        )

        log.info(f"Specialist {delegate_to} done ({phase_b_ms:.0f}ms, {len(final_report)} chars)")

        # Specialist timeout is a strong signal of lock pressure or decode stall.
        # Force synthesis immediately instead of re-delegating into another stall.
        if specialist_timed_out:
            stats["break_reason"] = "specialist_timeout"
            break
        # If specialist already produced a substantial report (prose/code rescue),
        # return it directly to avoid an expensive architect synthesis hop.
        if report_rescued:
            stats["break_reason"] = "specialist_report"
            stats["loops"] = loop + 1
            stats["tools_used"] = max(
                total_tools,
                len(all_tools_called),
                len(stats.get("tool_timings", [])),
            )
            stats["tools_called"] = all_tools_called
            return final_report, stats

    # ── Cap reached ──
    stats["loops"] = max_loops
    stats["tools_used"] = max(
        total_tools,
        len(all_tools_called),
        len(stats.get("tool_timings", [])),
    )
    stats["tools_called"] = all_tools_called

    if force_response_on_cap:
        stats["cap_reached"] = True
        if not stats.get("break_reason"):
            stats["break_reason"] = "forced_synthesis"
        if (
            cfg.skip_synthesis_on_timeout
            and stats.get("break_reason") in {"specialist_timeout", "wall_clock_budget"}
            and full_reports
        ):
            # Timeout-triggered synthesis can itself stall on a saturated specialist/
            # architect path. Returning the latest report prevents request timeouts.
            log.warning(
                "Skipping forced synthesis due to timeout break_reason=%s, returning latest report",
                stats.get("break_reason"),
            )
            return full_reports[-1], stats
        # Force architect to synthesize with whatever we have
        log.warning(f"Architect delegation capped at {max_loops} loops, forcing synthesis")
        forced_prompt = (
            f"You MUST answer now. Synthesize from all available information.\n\n"
            f"Question: {question[:2000]}\n\n"
            f"Investigation reports:\n" + "\n---\n".join(reports[-3:])  # Last 3 reports
        )
        try:
            answer = primitives.llm_call(
                forced_prompt,
                role=architect_role,
                skip_suffix=True,
                n_tokens=cfg.forced_synthesis_n_tokens,
            )
            return answer.strip(), stats
        except Exception as exc:
            log.debug("Forced synthesis failed, returning last report: %s", exc)
            # Last resort: return the latest specialist report
            return full_reports[-1] if full_reports else "[ERROR: Delegation exhausted]", stats
    else:
        # Production mode: signal that user input needed
        stats["needs_input"] = True
        partial = full_reports[-1] if full_reports else ""
        return partial, stats
