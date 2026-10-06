"""One-pass JSON-mode runner for the typed decision plane (TD-1 core).

``run_typed_decisions`` asks a single ``LLMPrimitives.llm_call`` for one JSON
object answering every ``Question`` in the catalogue, validates it against the
generated Draft 2020-12 response schema, and returns a ``DecisionResult`` of
typed decisions plus typed parse failures. No live route calls this yet —
TD-5 wires it in. ``mode="native"`` dispatches to
``src/typed_decisions/native.py`` (TD-1a), which scores the declared
candidates from one constrained generation instead.

Why one pass / one prompt:
    * One prefill instead of N, and the prompt is built so the instructions +
      schema + state prefix is byte-identical across calls with the same
      inputs, letting the backend default ``cache_prompt=True`` reuse KV state.
    * A single emission keeps the questions mutually contextual and makes the
      whole batch one validation unit.

Failure contract:
    * ``llm_call`` returns ``"[ERROR: ...]"`` strings when the backend raises.
      Those are detected and become a ``transport_error`` with NO retry — a
      corrective prompt cannot fix a dead transport.
    * A response that extracts to JSON but fails schema/value validation gets
      exactly ONE corrective retry (``max_retries=1``) appending a short
      ``CORRECTION`` message that names the error. Every failed attempt is
      recorded in ``DecisionResult.failures`` in attempt order, so a retry
      that recovers still leaves its first-attempt failure visible as
      diagnostics.
    * A question that cannot be answered yields a ``ParseFailure``; it never
      silently receives a default value. ``failures`` is empty only when the
      first attempt succeeded outright.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from jsonschema import Draft202012Validator

from src.typed_decisions.confidence import (
    choice_confidence,
    normalize_probabilities,
    score_confidence,
)
from src.typed_decisions.schema import _labels as _schema_labels
from src.typed_decisions.schema import build_response_schema
from src.typed_decisions.types import (
    Decision,
    DecisionResult,
    ParseFailure,
    Question,
    QuestionKind,
)

if TYPE_CHECKING:
    from src.typed_decisions.native import CueStyle, TokenizeFn

logger = logging.getLogger(__name__)

# Deterministic decode: temperature 0.0 plus a pinned seed. Retries still
# change the prompt (the correction block), so pinned sampling does not make
# a corrective retry a no-op.
_DECODE_SEED = 0

# Output budget when the caller does not pin n_tokens: one answer object per
# question plus room for the JSON envelope; never below the role floor.
_MIN_N_TOKENS = 256
# Per-answer-object structural tokens (quoted question id, value key/value,
# the "probabilities" key, the "confidence" key/value, and the surrounding
# braces/colons/commas) — everything in an answer entry EXCEPT its
# probability-label entries, which scale with the question's candidate count
# (2 for noul, len(options) for choice, len(levels) for score) and are
# costed separately below.
_TOKENS_PER_QUESTION_BASE = 30
# Tokens per probability-label entry ("<label>": <number>,) — quoted label
# (a candidate string can be multiple subword tokens), colon, a decimal
# probability, and a comma.
_TOKENS_PER_LABEL = 9
# Safety multiplier over the structural estimate. TD-1d.2 window-diag
# (2026-09-24): the old flat `64 tokens/question` heuristic undershot a real
# 24-question decision_set_v1 catalogue -- the MINIMAL valid answer for that
# exact catalogue measured ~999 tokens compact / ~1621 tokens pretty-printed
# (Qwen3-family tokenizer, offline proxy for frontdoor's Qwen3.6 tokenizer;
# see tests/unit/test_typed_decisions.py::TestDefaultNTokens), against the old 1600-token
# TOTAL budget -- i.e. pretty-printed JSON alone (a formatting choice models
# commonly make even under schema-constrained decoding) already exceeded the
# whole budget with zero margin for anything else. 1.6x covers
# pretty-printing whitespace, longer-than-minimal numeric/label formatting,
# and a bounded amount of preamble before the constrained JSON.
_TOKENS_SAFETY_MULTIPLIER = 1.6
# Flat top-level envelope overhead ({"answers": {...}}) plus rounding slack.
_TOKENS_OVERHEAD = 64

_CORRECTION_HEADER = "CORRECTION"

REASON_NO_JSON = "no_json"
REASON_SCHEMA_VIOLATION = "schema_violation"
REASON_INVALID_VALUE = "invalid_value"
REASON_TRANSPORT_ERROR = "transport_error"

_INSTRUCTIONS = """\
Answer every question in the catalogue below in ONE pass.

- choice questions: return the chosen option label under the "choice" key.
- score questions: return the chosen integer level under the "score" key.
- noul questions: return true or false under the "noul" key.

Return EXACTLY ONE JSON object and nothing else (no prose, no markdown fences).
The object has a single key "answers"; each answer is keyed by the question id
and contains:
  * the value key for the question kind ("choice" | "score" | "noul");
  * "probabilities": one probability for EVERY candidate listed for the
    question, summing to 1.0;
  * "confidence": your confidence in the answer, in [0, 1].

Rules:
1. Every question id must appear exactly once in "answers".
2. Use candidate labels exactly as written; never invent a label.
3. Probabilities must be non-negative and sum to 1.0 per question.
"""


def run_typed_decisions(
    primitives: Any,
    *,
    state: str,
    questions: Sequence[Question],
    role: str,
    mode: str = "json",
    max_retries: int = 1,
    n_tokens: int | None = None,
    cue_style: CueStyle | str = "id_only",
    tokenize_fn: TokenizeFn | None = None,
) -> DecisionResult:
    """Run one typed-decision pass and return typed decisions / failures.

    Args:
        primitives: The ``LLMPrimitives`` seam (anything exposing
            ``llm_call(prompt, role=..., n_tokens=..., json_schema=...,
            temperature=..., seed=...)``).
        state: Task/context state injected after the stable prefix.
        questions: The catalogue; ids must be unique and non-empty.
        role: Registry role the call is charged to.
        mode: ``"json"`` (this function) or ``"native"`` (dispatches to
            ``native.run_typed_decisions_native``: one constrained token per
            question with candidate-probability slicing). Any other value is
            rejected.
        max_retries: Corrective retries after the first attempt
            (default 1 -> at most 2 calls). IGNORED in native mode: a
            grammar-constrained decode has no free-form text to correct.
        n_tokens: Output budget; a per-question default is computed when
            ``None``. Forwarded to the native runner in native mode (where
            the default is exactly one token per native-capable question).
        cue_style: Native-mode cue style (TD-1d): ``"id_only"`` (the default
            since TD-6; cue sweep 11.98x at 15/16 agreement, n=1; 9.60x at
            n=4, contested pending TD-1d.0), ``"full"`` or
            ``"short"``, or the ``native.CueStyle`` enum. Forwarded to the
            native runner, which replays that cue before each answer token.
            IGNORED by the JSON arm, which replays no cue. The default is the
            enum's value string because importing ``CueStyle`` at module scope
            here would close an import cycle (``native`` imports this module's
            validation contract).
        tokenize_fn: Text -> token ids seam for native-mode candidate binding
            (TD-1b). IGNORED by the JSON arm, which never tokenizes
            candidates; forwarded to the native runner, which resolves a
            default from ``primitives`` when ``None`` and fails every
            question closed with ``native_tokenizer_unavailable`` when no
            tokenizer can be resolved.

    Returns:
        ``DecisionResult``. ``prompt_sha256`` hashes the canonical
        first-attempt prompt, excluding any correction blocks, so it is a
        stable identity for identical inputs.
    """
    if mode == "native":
        # Lazy import: native.py imports this module's validation contract and
        # transport reason, so a module-level import here would be circular.
        from src.typed_decisions.native import run_typed_decisions_native

        return run_typed_decisions_native(
            primitives,
            state=state,
            questions=questions,
            role=role,
            n_tokens=n_tokens,
            cue_style=cue_style,
            tokenize_fn=tokenize_fn,
        )
    if mode != "json":
        raise ValueError(f"unknown typed-decisions mode: {mode!r}")
    catalogue = _validated_catalogue(questions)

    schema = build_response_schema(catalogue)
    prompt = _build_prompt(state, catalogue, schema)
    prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if n_tokens is None:
        n_tokens = _default_n_tokens(catalogue)

    validator = Draft202012Validator(schema)
    attempts = 1 + max(0, int(max_retries))
    failures: list[ParseFailure] = []
    raw_text = ""
    parsed: dict[str, Any] | None = None
    failure_context: tuple[str, str] | None = None
    started = time.perf_counter()

    for attempt in range(1, attempts + 1):
        call_prompt = prompt
        if failure_context is not None:
            call_prompt = prompt + _correction_message(*failure_context)
        raw_text = str(
            primitives.llm_call(
                call_prompt,
                role=role,
                n_tokens=n_tokens,
                json_schema=schema,
                temperature=0.0,
                seed=_DECODE_SEED,
            )
            or ""
        )
        failure, obj = _parse_attempt(raw_text, validator, attempt, attempts)
        if failure is None:
            parsed = obj
            break
        failures.append(failure)
        failure_context = (failure.reason, failure.detail)
        logger.warning(
            "typed_decisions: attempt %d/%d failed (%s): %s",
            attempt,
            attempts,
            failure.reason,
            failure.detail,
        )
        if failure.reason == REASON_TRANSPORT_ERROR:
            break

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if parsed is None:
        return DecisionResult(
            decisions=(),
            failures=tuple(failures),
            raw_text=raw_text,
            mode=mode,
            elapsed_ms=elapsed_ms,
            prompt_sha256=prompt_sha256,
        )

    decisions, answer_failures = _decisions_from_answers(parsed, catalogue, mode)
    return DecisionResult(
        decisions=tuple(decisions),
        failures=tuple(failures + answer_failures),
        raw_text=raw_text,
        mode=mode,
        elapsed_ms=elapsed_ms,
        prompt_sha256=prompt_sha256,
    )


def _validated_catalogue(questions: Sequence[Question]) -> list[Question]:
    catalogue = list(questions)
    if not catalogue:
        raise ValueError("run_typed_decisions requires at least one Question")
    seen: set[str] = set()
    for question in catalogue:
        if question.id in seen:
            raise ValueError(f"duplicate question id: {question.id!r}")
        seen.add(question.id)
    return catalogue


def _default_n_tokens(questions: Sequence[Question]) -> int:
    """Schema-shape-aware output budget for the JSON arm's one-pass answer.

    Scales with each question's actual candidate-label count (schema.py's
    ``_labels``: 2 for noul, ``len(options)`` for choice, ``len(levels)`` for
    score) rather than a flat per-question constant -- a catalogue skewed
    toward 4-way choice/score questions needs a materially bigger
    "probabilities" object per answer than one that is all noul. See the
    module-level constants' docstrings for the calibration evidence.
    """
    structural = sum(
        _TOKENS_PER_QUESTION_BASE + _TOKENS_PER_LABEL * len(_schema_labels(question))
        for question in questions
    )
    return max(
        _MIN_N_TOKENS,
        int(structural * _TOKENS_SAFETY_MULTIPLIER) + _TOKENS_OVERHEAD,
    )


def _build_prompt(state: str, questions: Sequence[Question], schema: dict[str, Any]) -> str:
    """Build the canonical prompt: stable prefix + state + question catalogue.

    The instruction block and the schema are identical for identical
    question catalogues, and the state/catalogue sections are deterministic
    in their inputs, so two calls with the same inputs produce byte-identical
    prompts (asserted by the prompt-stability test).
    """
    schema_text = json.dumps(schema, indent=2, sort_keys=True)
    return (
        f"{_INSTRUCTIONS}\n"
        f"RESPONSE JSON SCHEMA (authoritative):\n{schema_text}\n\n"
        f"STATE:\n{state}\n\n"
        f"QUESTION CATALOG:\n{_build_catalog(questions)}\n"
    )


def _build_catalog(questions: Sequence[Question]) -> str:
    lines: list[str] = []
    for index, question in enumerate(questions, start=1):
        labels = _candidate_labels(question)
        lines.append(f"Q{index} id={question.id} kind={question.kind.value}")
        lines.append(f"  question: {question.text}")
        lines.append(f"  candidates: {' | '.join(labels)}")
        for label, description in zip(labels, question.option_descriptions):
            if description is not None:
                lines.append(f"  candidate description [{label}]: {description}")
        for criterion in question.criteria:
            lines.append(f"  criterion: {criterion}")
        lines.append("")
    return "\n".join(lines).rstrip()


def _candidate_labels(question: Question) -> list[str]:
    if question.kind is QuestionKind.CHOICE:
        return list(question.options)
    if question.kind is QuestionKind.SCORE:
        return [str(level) for level in question.levels]
    return ["true", "false"]


def _correction_message(reason: str, detail: str) -> str:
    if reason == REASON_SCHEMA_VIOLATION:
        problem = f"the response did not match the schema ({detail})"
    elif reason == REASON_NO_JSON:
        problem = f"no single JSON object could be extracted ({detail})"
    else:
        problem = f"{reason}: {detail}"
    return (
        f"\n\n{_CORRECTION_HEADER}: your previous reply was unusable — "
        f"{problem}. Return exactly one JSON object matching the RESPONSE JSON "
        "SCHEMA above, with every question id present exactly once and every "
        "probabilities object summing to 1.0."
    )


def _extract_json_object(text: str) -> str | None:
    """Return the first balanced top-level ``{...}`` substring, or ``None``.

    Brace-depth, string and escape aware; tolerates fenced code blocks and
    prose around the JSON. Same algorithm as
    ``src/proactive_delegation/review_grammar._extract_json_object`` — kept
    local so this package owns its parse contract end to end.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_attempt(
    raw_text: str,
    validator: Draft202012Validator,
    attempt: int,
    attempts: int,
) -> tuple[ParseFailure | None, dict[str, Any] | None]:
    """Extract + schema-validate one emission into ``(failure, object)``."""
    prefix = f"attempt {attempt}/{attempts}"
    if raw_text.strip().startswith("[ERROR:"):
        return (
            ParseFailure(REASON_TRANSPORT_ERROR, f"{prefix}: {raw_text.strip()}"),
            None,
        )
    candidate = _extract_json_object(raw_text)
    if candidate is None:
        return ParseFailure(REASON_NO_JSON, f"{prefix}: no balanced JSON object found"), None
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParseFailure(REASON_NO_JSON, f"{prefix}: JSON decode error: {exc}"), None
    if not isinstance(obj, dict):
        return (
            ParseFailure(REASON_NO_JSON, f"{prefix}: top-level JSON is {type(obj).__name__}"),
            None,
        )
    errors = sorted(validator.iter_errors(obj), key=lambda error: list(error.absolute_path))
    if not errors:
        return None, obj
    summaries = []
    for error in errors[:5]:
        path = "$" + "".join(f".{part}" for part in error.absolute_path)
        summaries.append(f"{path}: {error.message}")
    return (
        ParseFailure(
            REASON_SCHEMA_VIOLATION,
            f"{prefix}: {len(errors)} violation(s): " + "; ".join(summaries),
        ),
        None,
    )


def _decisions_from_answers(
    obj: dict[str, Any],
    questions: Sequence[Question],
    mode: str,
) -> tuple[list[Decision], list[ParseFailure]]:
    """Turn a schema-valid response into decisions plus per-question failures.

    The schema already guarantees every answer is present and well-typed;
    the checks here are the defensive value-level pass (a choice value still
    outside the option set, a score outside the levels, a non-boolean noul,
    an unknown question id) so a violation can never become a silent default.
    """
    answers = obj.get("answers")
    if not isinstance(answers, dict):
        return [], [ParseFailure(REASON_SCHEMA_VIOLATION, "'answers' is not an object")]
    decisions: list[Decision] = []
    failures: list[ParseFailure] = []
    for question in questions:
        entry = answers.get(question.id)
        if not isinstance(entry, dict):
            failures.append(
                ParseFailure(REASON_INVALID_VALUE, f"question {question.id!r}: missing answer")
            )
            continue
        try:
            decisions.append(_decision_from_entry(entry, question, mode))
        except ValueError as exc:
            failures.append(ParseFailure(REASON_INVALID_VALUE, f"question {question.id!r}: {exc}"))
    known_ids = {question.id for question in questions}
    for extra in sorted(set(answers) - known_ids):
        failures.append(ParseFailure(REASON_INVALID_VALUE, f"unknown question id {extra!r}"))
    return decisions, failures


def _decision_from_entry(entry: Mapping[str, Any], question: Question, mode: str) -> Decision:
    raw_probabilities = entry.get("probabilities")
    if not isinstance(raw_probabilities, Mapping):
        raise ValueError("'probabilities' is not an object")

    if question.kind is QuestionKind.CHOICE:
        value = entry.get("choice")
        if value not in question.options:
            raise ValueError(f"choice {value!r} not in options {list(question.options)}")
        probabilities = normalize_probabilities(
            _candidate_probabilities(raw_probabilities, list(question.options))
        )
        confidence = choice_confidence(probabilities)
    elif question.kind is QuestionKind.SCORE:
        value = entry.get("score")
        if value not in question.levels:
            raise ValueError(f"score {value!r} not in levels {list(question.levels)}")
        label_to_level = {str(level): level for level in question.levels}
        probabilities = normalize_probabilities(
            {
                label_to_level[label]: probability
                for label, probability in _candidate_probabilities(
                    raw_probabilities, list(label_to_level)
                ).items()
            }
        )
        confidence = score_confidence(probabilities)
    else:
        value = entry.get("noul")
        if not isinstance(value, bool):
            raise ValueError(f"noul value {value!r} is not a boolean")
        probabilities = normalize_probabilities(
            _candidate_probabilities(raw_probabilities, ["true", "false"])
        )
        confidence = choice_confidence(probabilities)

    return Decision(
        question_id=question.id,
        kind=question.kind,
        value=value,
        probabilities=probabilities,
        confidence=confidence,
        mode=mode,
        token_logprob=None,
    )


def _candidate_probabilities(
    raw_probabilities: Mapping[str, Any],
    labels: Sequence[str],
) -> dict[str, float]:
    missing = [label for label in labels if label not in raw_probabilities]
    if missing:
        raise ValueError(f"probabilities missing candidates {missing}")
    return {label: raw_probabilities[label] for label in labels}
