"""Architect-decision parsing + decision-guard helpers.

Extracted from src/api/routes/chat_delegation.py during the 2026-05-22 Task-C
Phase 3 refactor. Handles TOON/JSON-ish/text response parsing, token-budget
math for architect roles, failure-reason classification, and decision-guard
enforcement. chat_delegation.py re-exports every public name here.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.llm_primitives import LLMPrimitives

from src.registry.stack_priors import DEFAULT_OUTPUT as DEFAULT_STACK_PRIORS
from src.structured_output.repair import parse_with_repair, primitives_completer

from .chat_delegation_config import (
    _delegation_config,
    _normalize_delegate_role,
    _valid_delegate_roles,
)

log = logging.getLogger(__name__)


def _strip_think(text: str) -> str:
    """Strip complete and incomplete <think> blocks.

    During streaming, models may produce ``<think>I should delegate with
    I|brief:...`` without closing the tag.  The incomplete block must be
    stripped so that deliberation about delegation isn't mistaken for an
    actual TOON decision.
    """
    # 1. Complete blocks
    result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. Trailing incomplete block (opened but never closed)
    result = re.sub(r"<think>.*$", "", result, flags=re.DOTALL)
    return result



def _extract_toon_decision(text: str) -> str | None:
    """Extract D|answer or I|brief:...|to:... from anywhere in model output.

    The model often embeds TOON decisions mid-sentence after reasoning:
      "The answer is C. Decision: D|CTo confirm this..."
    This function extracts "D|C" from that mess.

    Strategy:
      1. MCQ shortcut: D| followed by single letter [A-D] not followed by alpha
      2. Own-line: D|... on its own line
      3. General: D| followed by text until newline (take first sentence)
      4. I| delegation patterns
    """
    # Template-echo blocklist: model echoed the placeholder instead of
    # substituting an actual answer.  Return None so the caller falls
    # through to prose rescue or raw-output handling.
    _TEMPLATE_ECHOES = {"answer", "<answer>", "the answer", "your answer"}

    # 1. MCQ: D|X where X is A-D, followed by non-alpha or end
    mcq = re.search(r"D\|([A-D])(?=[^a-zA-Z]|$)", text)
    if mcq:
        return "D|" + mcq.group(1)

    # 1b. D|I| hybrid: architect emits "D|I|brief:...|to:role" — strip
    # the leading D| and treat as delegation.  4+ sightings across batches.
    hybrid = re.search(r"D\|(I\|.+?)(?:\n|$)", text)
    if hybrid:
        return hybrid.group(1).strip()

    # 2. Own line: D|... on its own line
    own_line = re.search(r"^D\|(.+)$", text, re.MULTILINE)
    if own_line:
        val = own_line.group(1).strip()
        if val.lower() in _TEMPLATE_ECHOES:
            return None
        return "D|" + val

    # 3. General D|: take text until period+space, newline, or next D|
    general = re.search(r"D\|(.+?)(?:\.\s|\n|D\||$)", text)
    if general:
        answer = general.group(1).strip().rstrip(".")
        if answer and answer.lower() not in _TEMPLATE_ECHOES:
            return "D|" + answer

    # 4. I| delegation — accept with or without "brief:" prefix.
    # Models sometimes emit I|description|to:role without "brief:".
    invest = re.search(r"I\|(brief:.+?)(?:\n|$)", text, re.IGNORECASE)
    if invest:
        return "I|" + invest.group(1).strip()

    # 4b. Lenient I|: no "brief:" prefix but has "|to:" somewhere
    invest_lenient = re.search(r"I\|(.+?\|to:\w+)", text)
    if invest_lenient:
        raw_val = invest_lenient.group(1).strip()
        # Normalize: prepend "brief:" if missing
        if not raw_val.lower().startswith("brief:"):
            raw_val = "brief:" + raw_val
        return "I|" + raw_val

    return None



def _parse_architect_decision(response: str, *, strict: bool = False) -> dict | None:
    """Parse architect's TOON-encoded decision.

    Handles:
    - TOON direct: ``D|<answer>``
    - TOON investigate: ``I|brief:<text>|to:<role>`` (default mode=react)
    - TOON investigate+mode: ``I|brief:<text>|to:<role>|mode:repl``
    - JSON: ``{"mode":"direct","answer":"..."}`` or ``{"mode":"investigate",...}``
    - Markdown-wrapped JSON: ```json {...} ```
    - Bare text fallback: treated as direct answer

    Args:
        response: Raw architect response text.
        strict: TD-21.4. When ``False`` (the default, used by every existing
            call site and preserved byte-for-byte), every branch below always
            returns a dict -- the long-D|-answer "best effort" keep-as-is, the
            invalid-role/mode clamp, and the bare-text fallback all still fire
            exactly as before. When ``True`` (used only by
            :func:`resolve_architect_decision`), those same three low-confidence
            branches return ``None`` instead of guessing, so the caller can
            tell "cleanly parsed" apart from "guessed" and spend a repair turn
            only on the latter. Every OTHER branch (clean D|<=50 chars, clean
            D| with a successful MCQ-letter rescue, I|/JSON with an
            already-valid delegate_to/delegate_mode) returns the identical
            dict regardless of ``strict`` -- the happy path never changes.

    Returns:
        Dict with keys: mode ("direct"/"investigate"), answer, brief,
        delegate_to, delegate_mode ("react"/"repl"); or ``None`` when
        ``strict=True`` and the response did not parse with confidence.
    """
    text = response.strip()

    # ── Strip leading prose/thinking before D|/I| ──
    # Models sometimes emit reasoning or <think> tags before the protocol prefix.
    # Search for D| or I| on its own line and strip everything before it.
    if not text.startswith(("D|", "I|")):
        # Try to find D| or I| at the start of any line
        toon_match = re.search(r"^([DI]\|.*)$", text, re.MULTILINE)
        if toon_match:
            log.info(
                "[architect-parse] recovered D|/I| from mid-response (stripped %d chars of preamble)",
                toon_match.start(),
            )
            text = toon_match.group(0).strip()

    # ── TOON: D|<answer> ──
    if text.startswith("D|"):
        raw_answer = text[2:].strip()
        # Guard: model emitted D| then started reasoning instead of answering.
        # If the "answer" is suspiciously long, try to rescue an MCQ letter
        # from the first line or from the reasoning body.
        if len(raw_answer) > 50:
            # Try MCQ letter on the same line as D|
            first_line = raw_answer.split("\n", 1)[0].strip()
            mcq_match = re.match(r"^([A-D])(?:[^a-zA-Z]|$)", first_line)
            if mcq_match:
                raw_answer = mcq_match.group(1)
            else:
                # Try to find a clear MCQ answer in the reasoning.
                # Patterns (checked in priority order):
                #   "Answer: B", "Correct Answer: B"
                #   "answer is A", "answer would be A", "answer should be A"
                #   "option A seems", "option A with"
                rescue = re.search(
                    r"(?:the\s+)?(?:correct\s+)?answer\s*(?:is|would\s+be|should\s+be|:)\s*([A-D])(?=[^a-zA-Z]|$)",
                    raw_answer,
                    re.IGNORECASE,
                )
                if not rescue:
                    rescue = re.search(
                        r"\boption\s+([A-D])(?:\s+(?:seems|with|is|looks)\b|[^a-zA-Z]|$)",
                        raw_answer,
                        re.IGNORECASE,
                    )
                if not rescue:
                    # Last resort: find the last D|X (MCQ) in the reasoning.
                    # Handles models that emit empty D| first, then reason,
                    # then conclude with D|B at the end.
                    last_toon = list(re.finditer(
                        r"D\|([A-D])(?=[^a-zA-Z]|$)", raw_answer
                    ))
                    if last_toon:
                        rescue = last_toon[-1]
                if rescue:
                    raw_answer = rescue.group(1).upper()
                elif strict:
                    # TD-21.4: no MCQ rescue found and the "answer" is long
                    # enough to plausibly be un-rescued reasoning/prose, not
                    # a genuine short answer -- let the caller repair it
                    # instead of guessing.
                    return None
                # else: keep raw_answer as-is (best effort, strict=False)
        return {
            "mode": "direct",
            "answer": raw_answer,
            "brief": "",
            "delegate_to": "",
            "delegate_mode": "react",
        }

    # ── TOON: I|brief:...|to:...[|mode:...] ──
    if text.startswith("I|"):
        parts_str = text[2:]
        fields: dict[str, str] = {}
        for segment in parts_str.split("|"):
            if ":" in segment:
                key, _, val = segment.partition(":")
                fields[key.strip().lower()] = val.strip()

        brief = fields.get("brief", parts_str)
        delegate_to = _normalize_delegate_role(fields.get("to", "coder_escalation"))
        delegate_mode = fields.get("mode", "react")

        # TD-21.4: an unrecognized role/mode is a parse failure, not a value
        # to silently substitute a default for -- let the caller repair it.
        if strict and delegate_to not in _valid_delegate_roles():
            return None
        if strict and delegate_mode not in ("react", "repl"):
            return None
        # Clamp to valid role (strict=False only -- legacy best-effort path)
        if delegate_to not in _valid_delegate_roles():
            delegate_to = "coder_escalation"
        # Clamp to valid mode (strict=False only -- legacy best-effort path)
        if delegate_mode not in ("react", "repl"):
            delegate_mode = "react"

        return {
            "mode": "investigate",
            "answer": "",
            "brief": brief,
            "delegate_to": delegate_to,
            "delegate_mode": delegate_mode,
        }

    # ── JSON (possibly markdown-wrapped) ──
    import re as _re

    json_match = _re.search(r"```(?:json)?\s*\n?(.*?)```", text, _re.DOTALL)
    json_text = json_match.group(1).strip() if json_match else text

    # Try JSON parse
    try:
        obj = json.loads(json_text)
        if isinstance(obj, dict):
            mode = obj.get("mode", "direct")
            if mode == "investigate":
                delegate_to = _normalize_delegate_role(
                    obj.get("delegate_to", obj.get("to", "coder_escalation"))
                )
                delegate_mode = obj.get("delegate_mode", obj.get("mode_detail", "react"))
                # TD-21.4: same "repair, don't clamp" rule as the I| branch.
                if strict and delegate_to not in _valid_delegate_roles():
                    return None
                if strict and delegate_mode not in ("react", "repl"):
                    return None
                if delegate_to not in _valid_delegate_roles():
                    delegate_to = "coder_escalation"
                if delegate_mode not in ("react", "repl"):
                    delegate_mode = "react"
                return {
                    "mode": "investigate",
                    "answer": "",
                    "brief": obj.get("brief", ""),
                    "delegate_to": delegate_to,
                    "delegate_mode": delegate_mode,
                }
            return {
                "mode": "direct",
                "answer": obj.get("answer", json_text),
                "brief": "",
                "delegate_to": "",
                "delegate_mode": "react",
            }
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # ── Bare text fallback — treat as direct answer (strict=False only) ──
    # TD-21.4: this is the worst of the silent defaults -- unrecognized
    # prose wrapped as a direct answer with no failure mode at all. When
    # strict, tell the caller so it can repair instead.
    if strict:
        return None
    return {
        "mode": "direct",
        "answer": text,
        "brief": "",
        "delegate_to": "",
        "delegate_mode": "react",
    }


# TD-21.4: architect answers that plausibly ARE a genuine short answer (an
# MCQ letter, a short factual value) rather than un-rescued reasoning. This
# is the exact same threshold `_parse_architect_decision` already uses above
# (`len(raw_answer) > 50` marks a D| answer as "suspiciously long" reasoning
# rather than a real answer) -- reused here, not invented, so the failure
# path applies the identical judgment the parser already makes today.
_PLAUSIBLE_DIRECT_ANSWER_MAX_CHARS = 50

_DECISION_REPAIR_INSTRUCTION = (
    "Convert the architect's reply, given as the user message, into the "
    "routing decision it made. mode 'direct' means it gave (or was trying "
    "to give) a final answer to the question -- put that answer, copied "
    "verbatim, in `answer`. mode 'investigate' means it wants a specialist "
    "to do further work -- put the task description in `brief`, the "
    "specialist role it named in `delegate_to`, and the execution mode it "
    "named (or 'react' if none) in `delegate_mode`. Use an empty string for "
    "any field that does not apply to the mode you chose. Never invent an "
    "answer, brief, or role that is not already present in the reply."
)


def _decision_repair_schema() -> dict:
    """TD-21.4 repair-turn schema. `delegate_to`/`delegate_mode` enums are
    derived from the SAME live allow-lists `_parse_architect_decision` already
    clamps against (`_valid_delegate_roles()`, and the `("react", "repl")`
    literal checked throughout this module) -- never invented values. Computed
    fresh per call since `_valid_delegate_roles()` reads the live stack
    registry and can change between calls."""
    delegate_roles = sorted(_valid_delegate_roles())
    return {
        "type": "object",
        "properties": {
            "mode": {"type": "string", "enum": ["direct", "investigate"]},
            "answer": {"type": "string"},
            "brief": {"type": "string"},
            "delegate_to": {"type": "string", "enum": ["", *delegate_roles]},
            "delegate_mode": {"type": "string", "enum": ["react", "repl"]},
        },
        "required": ["mode", "answer", "brief", "delegate_to", "delegate_mode"],
        "additionalProperties": False,
    }


def resolve_architect_decision(
    response: str,
    *,
    primitives: "LLMPrimitives",
    architect_role: str,
    site: str = "chat_delegation.architect_decision",
) -> dict:
    """TD-21.4: parse the architect's TOON/JSON control decision, repairing
    on a miss instead of ever serving unparsed prose as a user-visible answer
    or silently clamping an unrecognized role/mode.

    1. ``_parse_architect_decision(response, strict=True)`` -- the exact same
       fish the module always used (``~8`` stacked regexes plus the JSON
       branch), just refusing to guess. When this parses cleanly, its dict is
       returned UNCHANGED -- the happy path costs zero extra calls and is
       byte-identical to the pre-TD-21.4 behaviour.
    2. On a miss: ONE constrained extraction turn via
       ``parse_with_repair`` against ``_decision_repair_schema()``, on the
       SAME ``architect_role`` (no new server/role dependency; already on the
       ``/completion`` lane per the audit). A schema-invalid role or mode
       cannot come back from this turn: json-schema ``enum`` -- not a Python
       clamp -- is what makes an unrecognized value impossible to receive as
       "repaired".
    3. On repair failure: a typed failure the caller must handle explicitly.
       This function does not invent one on its own -- it maps the failure to
       whichever of TWO existing, already-wired sentinels the call sites
       understand, chosen by whether the raw text is plausibly a short direct
       answer:
         - short (``<= 50`` chars, the same threshold `_parse_architect_decision`
           already uses to judge a D| answer "not suspiciously long"): treat
           it AS a direct answer, verbatim -- no worse than what the pre-fix
           code already did for genuinely short unprefixed answers.
         - otherwise (long prose, i.e. exactly the case that used to leak to
           the user unexamined): return the existing ``"[ERROR: ...]"``
           answer sentinel, which `_architect_delegated_answer_inner`
           (`chat_delegation.py`) ALREADY special-cases
           (``decision_answer.startswith("[ERROR:")``) to end the loop and
           surface/escalate the failure -- no new plumbing needed downstream.
    """
    strict = _parse_architect_decision(response, strict=True)
    if strict is not None:
        return strict

    schema = _decision_repair_schema()
    complete = primitives_completer(primitives, architect_role)
    result = parse_with_repair(
        response,
        schema=schema,
        complete=complete,
        instruction=_DECISION_REPAIR_INSTRUCTION,
        site=site,
        # TD-21.34: `answer`/`brief` are content the architect must have
        # already produced -- an invented one would serve fabricated prose
        # (or a fabricated investigation brief) as if the model had said it.
        # `mode`/`delegate_to`/`delegate_mode` are exempt: they are
        # CLASSIFICATIONS this schema's `enum`s ask the model to map its own
        # free-form decision onto, and the raw reply essentially never
        # spells "investigate" or a role name like "coder_escalation"
        # verbatim, so evidence-checking them would spuriously fail every
        # legitimate repair.
        require_evidence=True,
        evidence_exempt={"mode", "delegate_to", "delegate_mode"},
    )
    if result.status in ("parsed", "repaired"):
        value = result.value
        mode = value["mode"]
        if mode == "investigate":
            return {
                "mode": "investigate",
                "answer": "",
                "brief": value.get("brief", ""),
                "delegate_to": value.get("delegate_to") or "coder_escalation",
                "delegate_mode": value.get("delegate_mode") or "react",
            }
        return {
            "mode": "direct",
            "answer": value.get("answer", ""),
            "brief": "",
            "delegate_to": "",
            "delegate_mode": "react",
        }

    # Repair failed (transport error, or the extraction turn itself did not
    # validate). `decline_question` is never passed above, so `"declined"`
    # cannot occur here.
    raw = response.strip()
    if raw and len(raw) <= _PLAUSIBLE_DIRECT_ANSWER_MAX_CHARS:
        return {
            "mode": "direct",
            "answer": raw,
            "brief": "",
            "delegate_to": "",
            "delegate_mode": "react",
        }
    log.warning(
        "[architect-parse] TD-21.4 repair failed (site=%s, reason=%r); "
        "surfacing as an error instead of serving unparsed prose",
        site, result.reason,
    )
    return {
        "mode": "direct",
        "answer": f"[ERROR: architect decision unparseable: {result.reason}]",
        "brief": "",
        "delegate_to": "",
        "delegate_mode": "react",
    }


_DEFAULT_ARCHITECT_COMPUTE_N_TOKENS = 768
_DEFAULT_ARCHITECT_DECISION_N_TOKENS = 512
_DEFAULT_NON_ARCHITECT_COMPUTE_N_TOKENS = 512
_DEFAULT_NON_ARCHITECT_DECISION_N_TOKENS = 256
_DEGRADED_ARCHITECT_BUDGET_ROLES = frozenset({"architect_general"})
_STACK_PRIOR_ARCHITECT_BUDGET_ROLES_CACHE: frozenset[str] | None = None


def _architect_budget_roles_from_stack_priors(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> frozenset[str] | None:
    """Return live architect roles from generated stack priors."""
    try:
        from src.registry.stack_priors import load_stack_priors_artifact, live_stack_role_records
    except Exception:
        return None

    if load_stack_priors_artifact(stack_priors_path) is None:
        return None
    return frozenset(
        role for role in live_stack_role_records(stack_priors_path) if role.startswith("architect_")
    )


def _architect_budget_roles(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> frozenset[str]:
    """Return roles that receive architect delegation budgets."""
    global _STACK_PRIOR_ARCHITECT_BUDGET_ROLES_CACHE
    if stack_priors_path == DEFAULT_STACK_PRIORS:
        if _STACK_PRIOR_ARCHITECT_BUDGET_ROLES_CACHE is None:
            _STACK_PRIOR_ARCHITECT_BUDGET_ROLES_CACHE = _architect_budget_roles_from_stack_priors(
                stack_priors_path
            )
        derived = _STACK_PRIOR_ARCHITECT_BUDGET_ROLES_CACHE
    else:
        derived = _architect_budget_roles_from_stack_priors(stack_priors_path)
    return derived if derived is not None else _DEGRADED_ARCHITECT_BUDGET_ROLES


def _architect_compute_budget_map(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> dict[str, int]:
    """Return live architect compute budgets keyed by generated role."""
    return {
        role: _DEFAULT_ARCHITECT_COMPUTE_N_TOKENS
        for role in sorted(_architect_budget_roles(stack_priors_path))
    }


def _architect_decision_budget_map(
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> dict[str, int]:
    """Return live architect decision budgets keyed by generated role."""
    return {
        role: _DEFAULT_ARCHITECT_DECISION_N_TOKENS
        for role in sorted(_architect_budget_roles(stack_priors_path))
    }



def _architect_decision_token_budget(
    role: str,
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> int:
    """Token budget for architect routing decision (turn 0)."""
    cfg = _delegation_config()
    default = _architect_decision_budget_map(stack_priors_path).get(
        role,
        _DEFAULT_NON_ARCHITECT_DECISION_N_TOKENS,
    )
    if cfg.architect_decision_n_tokens_override > 0:
        return max(64, cfg.architect_decision_n_tokens_override)
    return max(64, default)



def _architect_compute_token_budget(
    role: str,
    *,
    stack_priors_path: Path = DEFAULT_STACK_PRIORS,
) -> int:
    """Token budget for architect computation follow-up turns."""
    cfg = _delegation_config()
    default = _architect_compute_budget_map(stack_priors_path).get(
        role,
        _DEFAULT_NON_ARCHITECT_COMPUTE_N_TOKENS,
    )
    if cfg.architect_compute_n_tokens_override > 0:
        return max(128, cfg.architect_compute_n_tokens_override)
    return max(128, default)


def __getattr__(name: str) -> object:
    """Preserve legacy budget-map imports without freezing live role tables."""
    if name == "_ARCHITECT_TOKEN_BUDGET":
        return _architect_compute_budget_map()
    if name == "_ARCHITECT_DECISION_BUDGET":
        return _architect_decision_budget_map()
    raise AttributeError(name)



def _classify_failure_reason(exc: Exception) -> str:
    """Map inference failure text to a stable delegated break_reason."""
    text = str(exc).lower()
    if "lock timeout" in text:
        return "pre_delegation_lock_timeout"
    if "deadline exceeded" in text:
        return "deadline_exceeded"
    if "cancelled" in text or "canceled" in text:
        return "request_cancelled"
    if "timed out" in text or "timeout" in text:
        return "request_timeout"
    return "pre_delegation_architect_error"


# TD-21.5: the MCQ-misroute re-prompt's letter recovery. A closed set of
# exactly 4 values -- schema `enum`, not a regex, is what makes an
# off-set value structurally impossible to receive as "repaired".
_MCQ_LETTER_REPAIR_SCHEMA: dict = {
    "type": "object",
    "properties": {"letter": {"type": "string", "enum": ["A", "B", "C", "D"]}},
    "required": ["letter"],
    "additionalProperties": False,
}

_MCQ_LETTER_REPAIR_INSTRUCTION = (
    "The reply, given as the user message, was asked to answer a "
    "multiple-choice question with exactly one letter A-D. Read the reply "
    "and report which letter it gave (or clearly meant to give) in "
    "`letter`. Never guess a letter the reply does not itself support."
)


def _apply_decision_guards(
    decision: dict,
    question: str,
    loop: int,
    primitives: "LLMPrimitives",
    architect_role: str,
) -> dict:
    """Apply guard clauses to architect decision (MCQ misroute, short-answer, coding task).

    Returns:
        Potentially modified decision dict.
    """
    # ── MCQ misroute guard ──
    # If the question is multiple-choice (has A/B/C/D options) and the
    # architect tries to delegate, force a direct answer.  Specialists
    # cannot reason about factual/science MCQ — delegation just wastes
    # 50-300s and usually returns a wrong answer.
    if decision["mode"] == "investigate" and loop == 0:
        _mcq_re = re.compile(
            r"(?:^|\n)\s*[A-D]\s*[).\]]",  # A) or A. or A]
            re.MULTILINE,
        )
        if _mcq_re.search(question):
            log.warning(
                "MCQ misroute blocked: architect tried to delegate factual MCQ "
                "(brief=%s), forcing direct answer",
                decision["brief"][:80],
            )
            # Re-prompt the architect with a forced direct-answer instruction
            force_prompt = (
                f"This is a multiple-choice question. You MUST answer directly.\n"
                f"Respond with D| followed by the letter (A, B, C, or D). No delegation.\n"
                f"Do NOT explain your reasoning. Output ONLY the decision line.\n\n"
                f"Question: {question[:2000]}\n\n"
                f"Answer with the letter only (A, B, C, or D).\n\nDecision:"
            )
            try:
                from src.chat_completions_roles import thinking_off

                # RI-23: a 128-token letter answer — thinking OFF on the chat lane
                # (no-op while thinking_roles_chat_lane is off).
                with thinking_off():
                    forced_raw = primitives.llm_call(
                        force_prompt,
                        role=architect_role,
                        skip_suffix=True,
                        n_tokens=128,
                    )
                forced_stripped = _strip_think(forced_raw).strip()
                forced_decision = _extract_toon_decision(forced_stripped)
                if forced_decision and forced_decision.startswith("D|"):
                    decision = _parse_architect_decision(forced_decision)
                    log.info("MCQ misroute recovered: architect answered D|%s", decision["answer"])
                else:
                    # Cheap fish first (unchanged): extract any single
                    # letter A-D. Strip D|/I| prefix first to avoid matching
                    # the protocol marker as an MCQ letter.
                    _cleaned = re.sub(r"^[DI]\|", "", forced_stripped).strip()
                    letter_match = re.search(r"\b([A-D])\b", _cleaned)
                    if letter_match:
                        decision = {"mode": "direct", "answer": letter_match.group(1),
                                    "brief": "", "delegate_to": "", "delegate_mode": "react"}
                        log.info("MCQ misroute recovered (letter extract): D|%s", decision["answer"])
                    else:
                        # TD-21.5: the fish missed -- one native-enum repair
                        # turn on the SAME closed set (A-D), same
                        # architect_role, instead of leaving the previous
                        # mis-routed "investigate" decision silently in place
                        # with no attempt at all.
                        letter_result = parse_with_repair(
                            forced_stripped,
                            schema=_MCQ_LETTER_REPAIR_SCHEMA,
                            complete=primitives_completer(primitives, architect_role),
                            instruction=_MCQ_LETTER_REPAIR_INSTRUCTION,
                            site="chat_delegation_decision.mcq_misroute_letter",
                            # TD-21.34: the instruction already says "never guess a
                            # letter the reply does not itself support" -- the
                            # letter is an extracted fact (which letter the reply
                            # gave), not a classification, so require_evidence
                            # enforces the same contract the docstring promises.
                            require_evidence=True,
                        )
                        if letter_result.status in ("parsed", "repaired"):
                            decision = {
                                "mode": "direct",
                                "answer": letter_result.value["letter"],
                                "brief": "", "delegate_to": "", "delegate_mode": "react",
                            }
                            log.info(
                                "MCQ misroute recovered (repair): D|%s", decision["answer"]
                            )
                        else:
                            # On a miss, keep the previous decision as-is.
                            # It is already a schema-valid "investigate"
                            # decision (not fabricated prose or a clamped
                            # role) -- falling back to it is a safe no-op,
                            # unlike TD-21.4's bare-text fallback.
                            log.warning(
                                "MCQ misroute letter repair failed (reason=%r); "
                                "keeping prior decision", letter_result.reason,
                            )
            except Exception as exc:
                log.warning("MCQ misroute re-prompt failed: %s", exc)

    # ── Short-answer delegation guard ──
    # If the architect wants to delegate but the brief is essentially a
    # computed answer (short, numeric, or a factual statement), force
    # direct answer.  This catches: architect solves "soda bottle costs
    # $1.50" in <think>, then delegates "compute the cost" to coder who
    # has nothing to add.  The coder burns 50-300s round-tripping the
    # answer the architect already has.
    if decision["mode"] == "investigate" and loop == 0:
        brief = decision["brief"]
        _code_delegate = decision["delegate_to"] == "coder_escalation"
        _code_signals_in_q = any(
            sig in question for sig in (
                "INPUT FORMAT", "OUTPUT FORMAT", "SAMPLE INPUT",
                "USACO", "Codeforces", "Write a Python", "def ",
                "```python",
            )
        )
        # If delegating to coder but the question is NOT a coding task,
        # the architect is misrouting a factual/math question.
        if _code_delegate and not _code_signals_in_q:
            # Check if the brief looks like a computed answer rather
            # than a genuine implementation task.
            brief_words = brief.split()
            brief_is_short = len(brief_words) < 15
            brief_has_number = bool(re.search(r"\d+\.?\d*", brief))
            if brief_is_short and brief_has_number:
                log.warning(
                    "Short-answer delegation blocked: architect delegated "
                    "D|%s to %s for non-code question, forcing direct. "
                    "Brief: %s",
                    brief[:30],
                    decision["delegate_to"],
                    brief[:80],
                )
                # Extract the numeric answer from the brief
                number_match = re.search(r"[\d]+\.?\d*", brief)
                forced_answer = number_match.group(0) if number_match else brief
                decision = {
                    "mode": "direct",
                    "answer": forced_answer,
                    "brief": "",
                    "delegate_to": "",
                    "delegate_mode": "react",
                }

    # ── Coding task direct-answer guard ──
    # If the question asks for code (CP, LeetCode, implementation tasks)
    # and the architect gives a short direct answer instead of delegating,
    # force delegation to coder.  The scorer expects runnable code, not a
    # numeric value like "4" or "-1".
    if decision["mode"] == "direct" and loop == 0:
        _code_signals = (
            "INPUT FORMAT", "OUTPUT FORMAT", "SAMPLE INPUT", "SAMPLE OUTPUT",
            "reads from stdin", "writes to stdout", "USACO", "Codeforces",
            "Write a Python solution",
            "Write a Python function",
            "def ", "```python",
            "Include proper type hints",
            "handle edge cases",
        )
        if any(sig in question for sig in _code_signals):
            short_answer = decision["answer"].strip()
            # Only intercept short answers (not full programs)
            if len(short_answer) < 50 and not short_answer.startswith(
                ("import", "def ", "class ")
            ):
                log.warning(
                    "Code direct-answer blocked: architect answered D|%s for coding "
                    "question, forcing delegation to coder_escalation",
                    short_answer[:30],
                )
                # Don't leak the architect's numeric guess to the
                # coder — it causes hardcoded FINAL(N) instead of
                # a general solution.
                hint = "" if re.fullmatch(r"-?\d+\.?\d*", short_answer.strip()) else f" {short_answer}"
                decision = {
                    "mode": "investigate",
                    "answer": "",
                    "brief": f"Implement a complete Python solution that reads from stdin and writes to stdout.{hint}",
                    "delegate_to": "coder_escalation",
                    "delegate_mode": "repl",
                }

    return decision
