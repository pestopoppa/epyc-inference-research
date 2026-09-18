#!/usr/bin/env python3
"""BEAM scoring primitives: the per-nugget judge contract and the FOLD (CME-2).

Source: "Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in
LLMs" (arXiv 2510.27246, ICLR 2026). Code MIT
(github.com/mohammadtavakoli78/BEAM); data CC BY-SA 4.0 (HF ``Mohammadta/BEAM``).

Nothing in this module opens a socket or loads a model. The judge is a callable
the caller injects; every function here is pure over stored verdicts.

THE FOLD (CME-2, ``intake-1337#record``, read off BEAM
``src/evaluation/report_results.py:37-66`` @ ``b2da22ea``)
------------------------------------------------------------------------------
1. **Per question** = mean of that question's three-valued nugget verdicts
   (each in {0, 0.5, 1}) — EXCEPT ``event_ordering``, whose reference fold uses
   ``tau_norm`` (normalised Kendall tau-b over LLM-aligned sequences). When a
   record carries ``tau_norm`` it is used; otherwise the nugget mean is used and
   the fallback is REPORTED as ``event_ordering_basis``, never silent.
2. **Per ability** = mean over that ability's questions.
3. **Headline** = the UNWEIGHTED mean of the reported ability columns.

Secondary diagnostics — clearly labelled, never the headline:

* ``rubric_item_micro_average`` — mean of every nugget verdict pooled across
  questions (rubric-item weighted). Event ordering, summarisation and
  contradiction resolution hold ~54% of the rubric items but 30% of the macro
  weight, so this fold is not a monotone transform of the headline.
* ``binarised_pass_count`` / ``binarised_total_checks`` / ``binarised_pass_rate``
  — verdicts counted as passed iff ``>= 0.5``. This is the MemPalace #125 fold
  (``beam_100k_bench.py:625-627``, 49.0 vs 55.7 on the same run).

The mutation this contract exists to catch: a ``>= 0.5`` binarisation creeping
into the headline turns a synthetic all-0.5 run from 0.500 into 1.000.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable, Iterable, Mapping

#: The ten BEAM memory abilities, spelled as the ``probing_questions`` keys.
BEAM_ABILITIES: tuple[str, ...] = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
EVENT_ORDERING = "event_ordering"

#: The judge's three-valued scale (BEAM ``unified_llm_judge_base_prompt``).
NUGGET_VERDICTS: tuple[float, ...] = (0.0, 0.5, 1.0)

#: Bumped whenever the fold semantics change. Consumers compare it FIRST.
#:   1  2026-09-15 (CME-2): per-question nugget mean (tau_norm for event_ordering
#:      when present), per-ability mean, unweighted ability headline.
FOLD_VERSION = 1
FOLD_NAME = "beam_macro"
BINARISATION_THRESHOLD = 0.5

#: The judge prompt is BEAM's unified nugget prompt with the probing QUESTION
#: added to the inputs. BEAM's own ten ``evaluate_*`` functions accept
#: ``probing_question`` and discard it, so its RESPONSIVENESS REQUIREMENT can
#: never fire (CME-3); intake-1337's harness carries the same repair.
JUDGE_PROMPT_VERSION = "beam-unified-b2da22ea+question/v1"
QUESTION_IN_JUDGE_PROMPT = True

_JUDGE_PROMPT_TEMPLATE = """
You are an expert evaluator tasked with judging whether the LLM's response demonstrates compliance with the specified RUBRIC CRITERION.

## EVALUATION INPUTS
- QUESTION (what the user asked): <question>
- RUBRIC CRITERION (what to check): <rubric_item>
- RESPONSE TO EVALUATE: <llm_response>

## EVALUATION RUBRIC:
The rubric defines a specific requirement, constraint, or expected behavior that the LLM response should demonstrate.

**IMPORTANT**: Pay careful attention to whether the rubric specifies:
- **Positive requirements** (things the response SHOULD include/do)
- **Negative constraints** (things the response SHOULD NOT include/do, often indicated by "no", "not", "avoid", "absent")

## RESPONSIVENESS REQUIREMENT
A compliant response must be **on-topic** relative to the QUESTION and attempt to answer it.
- If the response does not address the QUESTION, score **0.0** and stop.
- For negative constraints, both must hold: (a) the response is responsive to the QUESTION, and (b) the prohibited element is absent.

## SEMANTIC TOLERANCE RULES:
Judge by meaning, not exact wording.
- Accept **paraphrases** and **synonyms** that preserve intent.
- **Case/punctuation/whitespace** differences must be ignored.
- **Numbers/currencies/dates** may appear in equivalent forms (e.g., "$68,000", "68k", "68,000 USD", or "sixty-eight thousand dollars"). Treat them as equal when numerically equivalent.
- If the rubric expects a number or duration, prefer **normalized comparison** (extract and compare values) over string matching.

## STYLE NEUTRALITY (prevents style contamination):
Ignore tone, politeness, length, and flourish unless the rubric explicitly requires a format/structure (e.g., "itemized list", "no citations", "one sentence").
- Do **not** penalize hedging, voice, or verbosity if content satisfies the rubric.
- Only evaluate format when the rubric **explicitly** mandates it.

## SCORING SCALE:
- **1.0 (Complete Compliance)**: Fully complies with the rubric criterion.
  - Positive: required element present, accurate, properly executed (allowing semantic equivalents).
  - Negative: prohibited element **absent** AND response is **responsive**.

- **0.5 (Partial Compliance)**: Partially complies.
  - Positive: element present but minor inaccuracies/incomplete execution.
  - Negative: generally responsive and mostly avoids the prohibited element but with minor/edge violations.

- **0.0 (No Compliance)**: Fails to comply.
  - Positive: required element missing or incorrect.
  - Negative: prohibited element present **or** response is non-responsive/evasive even if the element is absent.

## EVALUATION INSTRUCTIONS:
1. **Understand the Requirement**: Determine if the rubric is asking for something to be present (positive) or absent (negative/constraint).

2. **Parse Compound Statements**: If the rubric contains multiple elements connected by "and" or commas, evaluate whether:
   - **All elements** must be present for full compliance (1.0)
   - **Some elements** present indicates partial compliance (0.5)
   - **No elements** present indicates no compliance (0.0)

3. **Check Compliance**:
   - For positive requirements: Look for the presence and quality of the required element
   - For negative constraints: Look for the absence of the prohibited element

4. **Assign Score**: Based on compliance with the specific rubric criterion according to the scoring scale above.

5. **Provide Reasoning**: Explain whether the rubric criterion was satisfied and justify the score.

## OUTPUT FORMAT:
Return your evaluation in JSON format with two fields:

{
   "score": [your score: 1.0, 0.5, or 0.0],
   "reason": "[detailed explanation of whether the rubric criterion was satisfied and why this justified the assigned score]"
}

NOTE: ONLY output the json object, without any explanation before or after that
"""


class BEAMFoldError(ValueError):
    """A record the fold refuses to score, rather than scoring it wrong."""


class NuggetVerdictError(ValueError):
    """A judge output that is not a three-valued verdict.

    Raised, never coerced to 0.0: a decode failure scored as "no compliance" is
    indistinguishable from a real 0.0 and silently deflates the run (CJ-10).
    """


# ── judge contract ───────────────────────────────────────────────────────────


def build_nugget_judge_prompt(question: str, rubric_item: str, response: str) -> str:
    """BEAM's unified nugget prompt, with the probing question actually passed."""
    return (_JUDGE_PROMPT_TEMPLATE
            .replace("<question>", str(question))
            .replace("<rubric_item>", str(rubric_item))
            .replace("<llm_response>", str(response)))


def coerce_nugget_verdict(value: Any) -> float:
    """Return a verdict in {0.0, 0.5, 1.0} or raise :class:`BEAMFoldError`."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BEAMFoldError(f"nugget verdict must be a number in {NUGGET_VERDICTS}, got {value!r}")
    as_float = float(value)
    if not math.isfinite(as_float) or as_float not in NUGGET_VERDICTS:
        raise BEAMFoldError(f"nugget verdict must be one of {NUGGET_VERDICTS}, got {value!r}")
    return as_float


def parse_nugget_verdict(text: str) -> float:
    """Parse one judge completion into a three-valued verdict, or raise.

    Accepts a bare JSON object or one wrapped in a code fence. Anything else —
    no JSON, no ``score``, a score off the scale — raises
    :class:`NuggetVerdictError`.
    """
    if not isinstance(text, str) or not text.strip():
        raise NuggetVerdictError("empty judge output")
    body = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.DOTALL)
    if fenced:
        body = fenced.group(1)
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", body, re.DOTALL)
        if not match:
            raise NuggetVerdictError("judge output carries no JSON object") from None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise NuggetVerdictError(f"judge output JSON does not parse: {exc}") from None
    if not isinstance(obj, dict) or "score" not in obj:
        raise NuggetVerdictError("judge output has no 'score' field")
    raw = obj["score"]
    if isinstance(raw, str):
        try:
            raw = float(raw.strip())
        except ValueError:
            raise NuggetVerdictError(f"judge score is not numeric: {obj['score']!r}") from None
    try:
        return coerce_nugget_verdict(raw)
    except BEAMFoldError as exc:
        raise NuggetVerdictError(str(exc)) from None


def judge_question(prompt_dict: Mapping[str, Any], response: str,
                   judge: Callable[[str], str]) -> dict:
    """Score one response per nugget through an injected ``judge(prompt) -> text``.

    Returns the fold record for this question. A verdict that does not parse
    raises; the caller decides whether that question is an ERROR row.
    """
    config = prompt_dict.get("scoring_config", {})
    nuggets = list(config.get("nuggets") or [])
    if not nuggets:
        raise BEAMFoldError(f"{prompt_dict.get('id')}: scoring_config carries no nuggets")
    question = config.get("probing_question", "")
    verdicts = [parse_nugget_verdict(judge(build_nugget_judge_prompt(question, item, response)))
                for item in nuggets]
    return {
        "question_id": prompt_dict.get("id"),
        "ability": config.get("ability"),
        "nugget_verdicts": verdicts,
    }


# ── the fold ─────────────────────────────────────────────────────────────────


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def question_score(record: Mapping[str, Any]) -> tuple[float, str]:
    """Per-question score and the basis that produced it (``tau_norm`` | ``nugget_mean``)."""
    ability = record.get("ability")
    if ability not in BEAM_ABILITIES:
        raise BEAMFoldError(f"unknown BEAM ability {ability!r}")
    raw = record.get("nugget_verdicts")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise BEAMFoldError(f"{record.get('question_id')}: nugget_verdicts must be a non-empty list")
    verdicts = [coerce_nugget_verdict(v) for v in raw]
    tau = record.get("tau_norm")
    if ability == EVENT_ORDERING and tau is not None:
        if isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= tau <= 1.0:
            raise BEAMFoldError(f"{record.get('question_id')}: tau_norm must be in [0, 1]")
        return float(tau), "tau_norm"
    return _mean(verdicts), "nugget_mean"


def _event_ordering_basis(bases: list[str]) -> str:
    if not bases:
        return "none"
    have = sum(1 for b in bases if b == "tau_norm")
    if have == len(bases):
        return "tau_norm"
    if have == 0:
        return "nugget_mean_fallback"
    return f"mixed({have}/{len(bases)} tau_norm)"


def fold_beam(records: Iterable[Mapping[str, Any]]) -> dict:
    """Apply the BEAM fold to per-question records; return headline + diagnostics.

    Each record: ``question_id`` (unique), ``ability``, ``nugget_verdicts``
    (non-empty, each in {0, 0.5, 1}), optional ``tau_norm`` for event ordering.
    """
    per_question: list[dict] = []
    by_ability: dict[str, list[float]] = {a: [] for a in BEAM_ABILITIES}
    eo_bases: list[str] = []
    all_verdicts: list[float] = []
    seen: set[str] = set()

    for record in records:
        qid = record.get("question_id")
        if not isinstance(qid, str) or not qid:
            raise BEAMFoldError("every record needs a non-empty question_id")
        if qid in seen:
            raise BEAMFoldError(f"duplicate question_id {qid!r}")
        seen.add(qid)
        score, basis = question_score(record)
        verdicts = [coerce_nugget_verdict(v) for v in record["nugget_verdicts"]]
        all_verdicts.extend(verdicts)
        by_ability[record["ability"]].append(score)
        if record["ability"] == EVENT_ORDERING:
            eo_bases.append(basis)
        per_question.append({
            "question_id": qid, "ability": record["ability"], "score": score,
            "basis": basis, "n_nuggets": len(verdicts),
        })

    per_ability = {
        ability: {"questions": len(scores), "score": _mean(scores)}
        for ability, scores in by_ability.items() if scores
    }
    reported = sorted(per_ability)
    headline = _mean([per_ability[a]["score"] for a in reported]) if reported else None
    passed = sum(1 for v in all_verdicts if v >= BINARISATION_THRESHOLD)
    total = len(all_verdicts)

    return {
        "fold": FOLD_NAME,
        "fold_version": FOLD_VERSION,
        "headline": headline,
        "per_ability": per_ability,
        "abilities_reported": reported,
        "abilities_missing": [a for a in BEAM_ABILITIES if a not in per_ability],
        "n_questions": len(per_question),
        "n_nuggets": total,
        "event_ordering_basis": _event_ordering_basis(eo_bases),
        "secondary_diagnostics": {
            "label": "SECONDARY DIAGNOSTICS - not the BEAM fold; never quote as a BEAM score",
            "rubric_item_micro_average": _mean(all_verdicts) if all_verdicts else None,
            "binarised_threshold": BINARISATION_THRESHOLD,
            "binarised_pass_count": passed,
            "binarised_total_checks": total,
            "binarised_pass_rate": passed / total if total else None,
        },
        "per_question": per_question,
    }


__all__ = [
    "BEAM_ABILITIES", "EVENT_ORDERING", "NUGGET_VERDICTS", "FOLD_VERSION", "FOLD_NAME",
    "BINARISATION_THRESHOLD", "JUDGE_PROMPT_VERSION", "QUESTION_IN_JUDGE_PROMPT",
    "BEAMFoldError", "NuggetVerdictError", "build_nugget_judge_prompt",
    "coerce_nugget_verdict", "parse_nugget_verdict", "judge_question", "question_score",
    "fold_beam",
]
