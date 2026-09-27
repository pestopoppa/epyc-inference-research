#!/usr/bin/env python3
"""Deterministic scorer for debug benchmark suite.

Scores model outputs against ground-truth answers using methods from
public benchmarks (exact_match, multiple_choice, code_execution,
programmatic, substring, f1, f1_list, llm_judge, math_verify,
structural_exact_match). No heuristics, no Claude-as-Judge needed
(except the explicit llm_judge method).

Usage:
    from scripts.benchmark.debug_scorer import score_answer

    result = score_answer(
        answer="The answer is 42",
        expected="42",
        scoring_method="exact_match",
        scoring_config={"extract_pattern": r"#### (\\d+)"},
    )
    print(result)  # True/False
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any


log = logging.getLogger(__name__)

_SCORER_TMP_ROOT = Path("/mnt/raid0/llm/tmp")
# 2026-09-27 ARCHITECT SWAP (operator-decided): the default judge follows the
# PROCESS — the Qwen3.8-27B Q8 on the MI210 (:8083), which now serves
# architect_critic. Keeping the label on architect_general would silently rebind the
# judge MODEL to Flash-Next on the serial whole-machine CPU instance (:8074): a
# measurement-instrument change (CJ-11 judge binding) plus a scorer tail sized for
# the GPU lane (30 s judge timeout). Rebinding the judge MODEL is an operator/CJ-11
# decision, not a side effect of a role relabel.
DEFAULT_LLM_JUDGE_ROLE = "architect_critic"

# ``start_new_session`` keeps an untrusted scorer's descendants out of the
# caller's process group, but it also means an externally interrupted caller
# can leave the direct scorer child alive.  Set PR_SET_PDEATHSIG *inside* that
# child rather than via ``preexec_fn``: this scorer is used from threaded
# callers, where preexec_fn can deadlock before exec.
_PARENT_DEATH_GUARD = """\
import ctypes as _epyc_ctypes
import os as _epyc_os
import signal as _epyc_signal

_epyc_libc = _epyc_ctypes.CDLL(None, use_errno=True)
if _epyc_libc.prctl(1, _epyc_signal.SIGKILL, 0, 0, 0) != 0:
    raise RuntimeError("unable to install scorer parent-death guard")
if _epyc_os.getppid() == 1:
    raise SystemExit("scorer parent exited before parent-death guard installed")
del _epyc_ctypes, _epyc_libc, _epyc_os, _epyc_signal
"""


class ScoringUnavailableError(RuntimeError):
    """Raised when a requested scorer cannot run.

    Signals a scorer-infrastructure defect — a missing dependency, an
    unreachable judge, or an unparseable GOLD/expected answer — as distinct
    from the model merely producing a wrong answer. Callers MUST surface this
    as an eval ERROR (an item that could not be scored) and NEVER fold it into
    a ``False`` (wrong-answer) result. Silently swapping in a different scorer
    (e.g. exact_match / substring) on such a failure is exactly how threaded
    ``math_verify`` parses were mis-scored en masse; this exception makes the
    failure loud instead of catastrophic-and-quiet.
    """


class _JudgeResponseError(ValueError):
    """Internal, non-sensitive classification of an unusable judge response."""

    def __init__(self, category: str, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.detail = detail


class AnswerParseError(ScoringUnavailableError):
    """A MODEL-SIDE answer could not be parsed into the required shape.

    TD-21.11..21.14 (``handoffs/active/typed-decision-plane.md``): distinct
    from ``ScoringUnavailableError``'s original scope (a missing dependency,
    an unreachable judge, or an unparseable GOLD/expected value) — here the
    scoring INSTRUMENT works fine, but the model's own answer does not fit
    the shape it needs (no option letter, no recognisable list, no
    ``solution =`` marker, no extractable final answer). Subclassing
    ``ScoringUnavailableError`` means every existing
    ``except ScoringUnavailableError`` catch — notably
    ``seeding_scoring.score_answer_or_error`` — already routes it through
    the EXCLUDED / ``scoring_failed`` path with no changes there: that is
    what "route through the existing exclusion path" means operationally.

    Raised ONLY when ``EXCLUDE_UNPARSEABLE_ANSWERS`` is True. This is an
    eval-quality-denominator semantics change (a row that used to be
    ``False``/WRONG/IN the denominator becomes EXCLUDED/OUT of it), so it is
    gated behind that default-OFF flag pending an operator-ratified
    ``eval_quality`` era row (see the flag's own docstring, and the
    TD-21.11..21.14 commit message, for the proposed era text). Default
    OFF: every site below instead calls ``_record_parse_failure`` and
    returns ``False`` — today's scoring behaviour, unchanged byte-for-byte.
    """


# TD-21.11..21.14 / proposed era "EQ-1" (eval_quality; NOT YET RATIFIED —
# instrument_eras.yaml is human-only and this commit does not touch it).
#
# Flipping this to True changes live eval-quality numbers: a model-side
# parse failure at one of the four sites below (multiple_choice,
# f1_list, structural_exact_match, exact_match's last-resort fallback)
# currently scores `False` — WRONG, IN the quality denominator. With this
# flag True it instead raises `AnswerParseError`, which
# `seeding_scoring.score_answer_or_error` turns into an EXCLUDED
# `scoring_failed` row (OUT of the quality denominator) via the same path
# already used for an unreachable judge or a malformed gold
# (`artifacts/audits/td-json-consumer-audit-20260924.md` J-03/J-05/J-06/J-07).
#
# This is deliberately independent of E17-eval-task-failed-scores-zero-quality
# (ETR-1, `orchestration/instrument_eras.yaml`): E17 governs rows that already
# carry a structural `error` (an agent/config-caused `task_failed` row scores
# 0 but STAYS IN the denominator). A model-side parse failure carries no
# `error` at all today — nothing currently routes it through disposition
# classification — so this flag is a NEW boundary, not a re-application of
# E17's. It also runs the opposite direction from E17's own fix (which
# pulled a class of failure INTO the denominator to stop a broken config
# from shrinking its own denominator); before ratifying, confirm this
# doesn't reopen that same hole for a model that produces unparseable output
# instead of an error.
#
# ONE-LINE FLIP to ratify: set this to True (only after the EQ-1 row is
# added to `orchestration/instrument_eras.yaml` by the operator/human-only
# path — see the proposed row text in the TD-21.11..21.14 commit message).
#
# RE-ENABLED 2026-09-24 (operator, on fix/e19-wrong-not-unparseable): was
# SUSPENDED to False (commit fcbb705f) because the TD-21.12/21.14
# classification treated "no structured marker AND the raw last-line/
# line-split fallback doesn't match" as a parse failure, excluding WRONG
# answers (e.g. "Final answer: 999" vs 25) from the quality denominator
# instead of scoring them wrong. `_score_exact_match` and `_score_f1_list`
# below are corrected: a candidate extracted by ANY means — including the
# raw-last-line / line-split fallback — that simply disagrees with gold is
# now always WRONG, never excluded; only a genuinely empty extraction
# (nothing at all, not even fallback content) raises `AnswerParseError`.
# `tests/unit/test_b7_golden_corpus_pin.py` (the 5 *_cot_wrongfinal /
# cotwrong sentinel/audit cases) is the regression proof.
EXCLUDE_UNPARSEABLE_ANSWERS = True


# TD-21.9/21.10/21.15 (judge OUTPUT SHAPE; NOT YET RATIFIED — see the proposed
# "E19" era text in the ratification script). TD-21.32 (handoff): judge
# OUTPUT SHAPE (this flag) is orthogonal to judge BINDING (WHICH model —
# CJ-11, `canonical-judge-suite-revamp.md`, `src/llm.py:61-64`). This flag
# touches neither role resolution nor endpoint precedence — every site below
# still resolves the judge via the SAME existing seam
# (`scoring_config["judge_role"]` > `LLM_JUDGE_ROLE` env >
# `architect_critic` [the :8083 27B; `architect_general` before the 2026-09-27
# ARCHITECT SWAP], `_llm_judge_force_role`) and only constrains + parses
# whatever judge that seam already points at. Rebinding the judge later
# (CJ-11/CJ-13/CJ-14) changes nothing here.
#
# Flipping this to True changes THREE sites:
#
#   TD-21.9 (`_score_llm_judge`/`_parse_judge_boolean_verdict`, orchestrator
#   `/chat` branch): the wire is UNCHANGED — `output_schema={"type":
#   "boolean"}` was already sent unconditionally before this flag existed —
#   but the verdict is now parsed STRICTLY (exact `"true"`/`"false"` after
#   stripping) instead of `.lower().startswith("true")`, which is a
#   prefix-match artifact (e.g. a truncated `"tru"` or a judge that prefaces
#   its verdict with "truely not equivalent" both score `True` today). A
#   verdict that still isn't exactly `"true"`/`"false"` under this flag is a
#   SCORER-side failure and raises `ScoringUnavailableError` DIRECTLY — never
#   `AnswerParseError`. `AnswerParseError` (TD-21.11..21.14) is reserved for
#   the MODEL's own answer failing to fit its own required shape; here the
#   scoring INSTRUMENT (the judge) produced unusable output, the same class
#   of defect this module already raises `ScoringUnavailableError` for on an
#   unreachable judge or a malformed backend envelope a few lines below.
#   Already routes through `score_answer_or_error`'s EXCLUDED/
#   `scoring_failed` path with no further change — same as every other
#   `ScoringUnavailableError` in this module, and unconditional on
#   `EXCLUDE_UNPARSEABLE_ANSWERS` (that flag governs the DIFFERENT,
#   model-side `AnswerParseError` boundary only).
#
#   TD-21.10 (`request_llm_judge_text`, raw llama-server override branch):
#   this branch sends NO schema at all today (the defect the handoff names) —
#   flipping this flag sends the SAME `output_schema` as an OpenAI
#   `response_format` envelope on `/v1/chat/completions`, matching the
#   protocol this branch already speaks. Strict parsing then applies
#   identically to both branches.
#
#   TD-21.15 (`scripts/autopilot/eval_tower.py::_rubric_scores_for_answer`,
#   read via this SAME flag through `_load_orchestrator_debug_scorer()` so
#   ONE flag governs both files, never two copies to drift): sends
#   `rubric_scoring.RUBRIC_JUDGE_SCHEMA` as `output_schema` on the rubric
#   judge's `call_orchestrator_forced` turn; a reply the existing lenient
#   fisher (`_parse_rubric_judge_scores`) cannot parse at all then gets ONE
#   `parse_with_repair` extraction turn back to the SAME judge role before
#   that judge is dropped. The `deterministic_rubric_fallback`/
#   `rubric_source="heuristic_fallback"` path taken when EVERY configured
#   judge is still unparseable is UNCHANGED by this flag either way: it is
#   already clearly marked (`rubric_source`) and already counted per-arm by
#   the pre-existing SCORE-08 `rubric_source_counts` rollup, so converting it
#   to `scoring_failed` would be a second, larger denominator change riding
#   on a flag meant to fix judge-output PARSING, not judge-fallback policy.
#
# Judge-parse-outcome counters (`judge_parse_stats`, below) are ALWAYS on,
# independent of this flag — the observability this flag needs before
# ratification, mirroring `_PARSE_FAILURE_COUNTS`/EQ-1's own precedent.
#
# Default OFF: every site above is byte-identical to pre-flag behaviour.
CONSTRAIN_JUDGE_OUTPUT = True

_PARSE_FAILURE_LOCK = threading.Lock()
# Keyed by (arm_key, scoring_method), NOT scoring_method alone. `arm_key` is
# whatever the caller put in `scoring_config["_eval_batch_id"]` (eval_tower.py
# stamps a fresh `uuid4` per `_eval_batch` call onto every question dispatched
# in that batch; the seeding harness leaves it unset). Keying by arm_key —
# rather than a bare global counter, and rather than a thread-local/contextvar
# — is what makes two arms scoring CONCURRENTLY in the same process safe:
# eval_tower's scoring pool is a `ThreadPoolExecutor`
# (`scripts/autopilot/eval_tower.py::_eval_batch`), and a plain
# `ThreadPoolExecutor.submit` does NOT propagate the caller's contextvars into
# worker threads, so a contextvar- or threading.local-based scope would
# silently fall back to one shared bucket across arms whenever scoring runs
# off the main thread — exactly the mixing this must prevent. The key travels
# WITH the data (through `scoring_config`, already threaded to every scorer
# call), so it survives any thread hop intact. `arm_key=None` (no
# `_eval_batch_id` set — the seeding harness's own call sites, and any
# standalone script) is its own bucket, matching the single-arm-per-process
# usage those callers already have.
_PARSE_FAILURE_COUNTS: dict[tuple[Any, str], int] = {}


def _record_parse_failure(scoring_method: str, arm_key: Any = None) -> None:
    """Count one model-side parse failure, independent of the exclusion flag.

    The ONLY always-on signal for the standing per-arm parse-failure-rate
    rule (2026-07-20, ``architect-model-selection-bench.md``: "report a
    per-arm parse-failure rate next to every accuracy number; any cross-arm
    difference is a scoring bug until proven otherwise"). This lets the rate
    be measured and reported even while ``EXCLUDE_UNPARSEABLE_ANSWERS`` stays
    False and live scores are unaffected — which is precisely the evidence
    an operator needs before ratifying the flag.

    Bucketed by ``arm_key`` (see the module-level comment above
    ``_PARSE_FAILURE_COUNTS``) so concurrent arms/trials in one process never
    mix counts.
    """
    with _PARSE_FAILURE_LOCK:
        key = (arm_key, scoring_method)
        _PARSE_FAILURE_COUNTS[key] = _PARSE_FAILURE_COUNTS.get(key, 0) + 1


def parse_failure_stats(arm_key: Any = None) -> dict[str, int]:
    """Per-scoring-method count of model-side parse failures for one ``arm_key``
    (``None`` — the default — is its own bucket, not "all arms")."""
    with _PARSE_FAILURE_LOCK:
        return {
            method: count
            for (key, method), count in _PARSE_FAILURE_COUNTS.items()
            if key == arm_key
        }


def reset_parse_failure_stats(arm_key: Any = None) -> None:
    """Zero the parse-failure counters for one ``arm_key``. Call once per
    reporting scope (an arm/batch) to bound the dict's lifetime — a
    long-running process (autopilot) would otherwise accumulate one entry
    per distinct arm_key forever."""
    with _PARSE_FAILURE_LOCK:
        for key in [k for k in _PARSE_FAILURE_COUNTS if k[0] == arm_key]:
            del _PARSE_FAILURE_COUNTS[key]


# TD-21.9/21.10/21.15: judge-parse-outcome counters. Distinct from
# `_PARSE_FAILURE_COUNTS` above — those count a MODEL-side answer failing to
# fit ITS OWN required shape (TD-21.11..21.14); these count a JUDGE reply
# failing to fit the shape the SCORER asked the judge for. ALWAYS recorded,
# independent of `CONSTRAIN_JUDGE_OUTPUT`, so the rate is observable before
# that flag is ratified — the same observability contract
# `_record_parse_failure` gave EQ-1. Keyed by (arm_key, site, outcome):
# `site` is `"llm_judge_boolean"` (TD-21.9/21.10) or `"rubric_judge"`
# (TD-21.15); `outcome` is one of `"parsed"`, `"repaired"`, `"unparseable"`
# (`"repaired"` is only ever recorded when `CONSTRAIN_JUDGE_OUTPUT` is True —
# it is the one outcome that requires the extra repair turn the flag gates).
_JUDGE_PARSE_LOCK = threading.Lock()
_JUDGE_PARSE_COUNTS: dict[tuple[Any, str, str], int] = {}


def _record_judge_parse_outcome(site: str, outcome: str, arm_key: Any = None) -> None:
    with _JUDGE_PARSE_LOCK:
        key = (arm_key, site, outcome)
        _JUDGE_PARSE_COUNTS[key] = _JUDGE_PARSE_COUNTS.get(key, 0) + 1


def judge_parse_stats(arm_key: Any = None) -> dict[str, dict[str, int]]:
    """Per-site outcome counts of judge parse attempts for one ``arm_key``
    (``None`` — the default — is its own bucket, not "all arms"), shaped
    ``{site: {outcome: count}}``."""
    with _JUDGE_PARSE_LOCK:
        stats: dict[str, dict[str, int]] = {}
        for (key, site, outcome), count in _JUDGE_PARSE_COUNTS.items():
            if key != arm_key:
                continue
            stats.setdefault(site, {})[outcome] = count
        return stats


def reset_judge_parse_stats(arm_key: Any = None) -> None:
    """Zero the judge-parse-outcome counters for one ``arm_key``."""
    with _JUDGE_PARSE_LOCK:
        for key in [k for k in _JUDGE_PARSE_COUNTS if k[0] == arm_key]:
            del _JUDGE_PARSE_COUNTS[key]


def _unparseable_answer(
    scoring_method: str, detail: str, config: dict[str, Any] | None = None
) -> bool:
    """A model-side parse failure at one of the TD-21.11..21.14 sites.

    Always records the failure via ``_record_parse_failure`` so the rate is
    visible regardless of the flag, bucketed by ``config["_eval_batch_id"]``
    when the caller set one (see the module-level comment above
    ``_PARSE_FAILURE_COUNTS``). When ``EXCLUDE_UNPARSEABLE_ANSWERS`` is True,
    raises ``AnswerParseError`` so the row is routed through
    ``score_answer_or_error``'s existing EXCLUDED/``scoring_failed`` path
    instead of being scored wrong. Default False: returns ``False``,
    identical to every prior release.
    """
    arm_key = (config or {}).get("_eval_batch_id")
    _record_parse_failure(scoring_method, arm_key)
    if EXCLUDE_UNPARSEABLE_ANSWERS:
        raise AnswerParseError(f"answer_parse_failed[{scoring_method}]: {detail}")
    return False


def score_answer(
    answer: str,
    expected: Any,
    scoring_method: str,
    scoring_config: dict[str, Any] | None = None,
) -> bool:
    """Score a model answer against expected ground truth.

    Args:
        answer: The model's raw output.
        expected: The expected correct answer.
        scoring_method: One of: exact_match, multiple_choice,
            code_execution, programmatic, substring.
        scoring_config: Method-specific configuration.

    Returns:
        True if the answer is correct, False otherwise.
    """
    if not answer or not answer.strip():
        return False

    # Strip <think>...</think> blocks before scoring (architect models produce these)
    answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
    if not answer:
        return False

    expected = "" if expected is None else str(expected)
    config = scoring_config or {}

    scorers = {
        "exact_match": _score_exact_match,
        "multiple_choice": _score_multiple_choice,
        "code_execution": _score_code_execution,
        "programmatic": _score_programmatic,
        "substring": _score_substring,
        "f1": _score_f1,
        "f1_list": _score_f1_list,
        "llm_judge": _score_llm_judge,
        "math_verify": _score_math_verify,
        "structural_exact_match": _score_structural_exact_match,
    }

    scorer = scorers.get(scoring_method)
    if scorer is None:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    return scorer(answer, expected, config)


def _score_exact_match(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Extract answer via regex, compare to expected.

    Used for: GSM8K, MATH — where the answer is a number or expression.

    Config:
        extract_pattern: Regex with one capture group to extract the answer.
            Default: ``<answer>...</answer>`` tag extraction.
            Legacy fallback: ``#### (\\S+)`` (GSM8K standard).
        normalize: If True, strip whitespace and lowercase both sides.
    """
    pattern = config.get("extract_pattern", r"<answer>(.*?)</answer>")
    normalize = config.get("normalize", True)

    # Try to extract via pattern first
    extracted = _extract_answer(answer, pattern)
    if extracted is None:
        # Legacy fallback: try #### pattern for backward compatibility
        extracted = _extract_answer(answer, r"####[ \t]*\n?(\S+)")
    if extracted is None:
        boxed = _extract_boxed_answer(answer)
        if boxed is not None:
            extracted = boxed
    if extracted is None:
        # Last resort: try to find the expected value anywhere in the last line
        last_line = answer.strip().split("\n")[-1]
        extracted = last_line.strip()
    # TD-21.14 (corrected 2026-09-24): a candidate was extracted the moment
    # `extracted` is non-empty — whether via a structured <answer>/####/
    # \boxed{} marker OR the raw-last-line fallback. A candidate that simply
    # doesn't match gold is a genuine WRONG answer, never a parse failure:
    # the fallback still picked SOMETHING comparable out of the model's own
    # output. Only when `extracted` is empty (the model's answer was
    # empty/whitespace-only, so even the blind last-line guess found
    # nothing) is there truly no candidate to compare — that is the one
    # real model-side parse failure this site can detect.
    candidate_found = bool(extracted)

    if normalize:
        extracted = extracted.strip().lower().rstrip(".")
        expected_norm = expected.strip().lower().rstrip(".")
    else:
        expected_norm = expected.strip()

    # Numeric comparison for numbers (including word forms like "three" vs "3")
    _NUMBER_WORDS = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
        "thirteen": 13,
        "fourteen": 14,
        "fifteen": 15,
        "sixteen": 16,
        "seventeen": 17,
        "eighteen": 18,
        "nineteen": 19,
        "twenty": 20,
    }

    def _to_number(s: str) -> float | None:
        try:
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return _NUMBER_WORDS.get(s.lower()) if isinstance(s, str) else None

    ext_num = _to_number(extracted)
    exp_num = _to_number(expected_norm)
    if ext_num is not None and exp_num is not None:
        matched = abs(ext_num - exp_num) < 1e-6
    else:
        matched = extracted == expected_norm

    # Fallback: vision models wrap OCR results in prose like
    #   'The text in the image is "iRaeenlc".' or 'The image contains the text: iRaeenlc'
    # Try extracting quoted text or text after colon from the full answer.
    if not matched and normalize:
        answer_lower = _final_answer_region(answer).lower()
        # Check quoted: "answer" or 'answer'
        for q in ('"', "'", "\u201c"):
            q_end = "\u201d" if q == "\u201c" else q
            idx = answer_lower.find(q)
            if idx >= 0:
                end = answer_lower.find(q_end, idx + 1)
                if end > idx:
                    candidate = answer_lower[idx + 1 : end].strip().rstrip(".")
                    if candidate == expected_norm:
                        matched = True
                        break
        # Check after colon on last meaningful line
        if not matched:
            for line in reversed(_final_answer_region(answer).split("\n")):
                if ":" in line:
                    candidate = line.split(":", 1)[1].strip().lower().rstrip(".")
                    if candidate == expected_norm:
                        matched = True
                        break

    if matched:
        return True
    if candidate_found:
        # A candidate was extracted (structured marker or the raw-last-line
        # fallback) and it simply disagrees with gold: a genuine wrong
        # answer, never excluded.
        return False
    # TD-21.14: the model's answer was empty/whitespace-only, so even the
    # blind last-line guess found nothing to compare. This is a real
    # model-side parse failure.
    return _unparseable_answer(
        "exact_match",
        "the model's answer was empty/whitespace-only; no <answer>/#### "
        "/\\boxed{} marker and no last-line candidate could be extracted",
        config=config,
    )


def _score_multiple_choice(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Parse A/B/C/D or configured choice text from output.

    Used for: ARC-Challenge, MMLU, HellaSwag.

    Config:
        choices: Optional list of choice texts.
        choice_labels: Optional contiguous label range starting at ``A``
            (e.g. ``"ABCDEFGHIJ"`` for MMLU-Pro's 10 options). Default ``A-H``;
            rows that do not set it keep the historical range byte-for-byte.
    """
    choices = config.get("choices")
    if not isinstance(choices, list):
        choices = []
    labels = _choice_labels(config)

    expected_letter = _expected_choice_letter(expected, choices, labels)
    expected_index = _expected_choice_index(expected, choices)
    if (
        expected_letter is not None
        and choices
        and labels.index(expected_letter) >= len(choices)
    ):
        # A gold letter pointing past the configured options is a corpus-join
        # defect (wrong options list, or letter/index skew), never a model error.
        raise ScoringUnavailableError(
            f"multiple_choice gold is unusable: expected={expected!r} points past "
            f"the {len(choices)} configured choices."
        )
    if expected_letter is None and expected_index is None:
        # CJ-8. The GOLD is neither an A-H letter nor a member of `choices`, so
        # there is nothing to decide against. This is a corpus/gold defect, and
        # returning False recorded it as the MODEL being wrong — a systematic 0
        # on every row of a malformed slice, indistinguishable from a quality
        # gap.
        #
        # Every other unscoreable-gold class in this module already raises:
        # `math_verify` gold that will not parse (:1237/:1243), `f1_list` gold
        # that is not a JSON list (:1512), an unknown programmatic verifier. This
        # was the one left behind. `seeding_scoring.score_answer_or_error`
        # catches it and returns `(None, reason)`, and `eval_tower` EXCLUDES the
        # row from the quality denominator — it is not converted into a pass.
        raise ScoringUnavailableError(
            f"multiple_choice gold is unusable: expected={expected!r} is neither "
            f"a choice letter in {labels[0]}-{labels[-1]} nor one of the "
            f"{len(choices)} configured choices, "
            f"so no verdict can be reached. Fix the corpus join or supply "
            f"scoring_config['choices']; refusing to score the model wrong "
            f"against a gold that cannot be resolved."
        )

    parsed_letter = _extract_multiple_choice_letter(answer, labels)
    if parsed_letter is not None and expected_letter is not None:
        return parsed_letter == expected_letter

    parsed_index = _extract_multiple_choice_text_index(answer, choices)
    if parsed_index is not None and expected_index is not None:
        return parsed_index == expected_index

    # TD-21.11: no option letter or matching choice text was found anywhere
    # in the model's answer — the model failed to produce a comparable
    # verdict, distinct from parsing a letter/text that simply mismatches
    # gold (both branches above already returned in that case).
    return _unparseable_answer(
        "multiple_choice",
        f"no option letter or matching choice text found in the model's "
        f"answer (labels={labels[0]}-{labels[-1]}, choices={len(choices)})",
        config=config,
    )


#: Historical letter range. Rows that do not declare ``choice_labels`` keep it,
#: so sealed captures scored before 2026-09-17 re-score identically.
_DEFAULT_CHOICE_LABELS = "ABCDEFGH"


def _choice_labels(config: dict[str, Any]) -> str:
    """Resolve ``scoring_config['choice_labels']`` to a contiguous ``A..X`` string.

    Accepts a string (``"ABCDEFGHIJ"``) or a list of single letters. Anything
    that is not a contiguous run starting at ``A`` is a malformed row and
    raises: silently falling back to A-H is exactly how MMLU-Pro's I/J gold
    became unscoreable (PRB-T4, 2026-09-16).
    """
    raw = config.get("choice_labels")
    if raw is None:
        return _DEFAULT_CHOICE_LABELS
    if isinstance(raw, (list, tuple)):
        raw = "".join(str(x) for x in raw)
    labels = str(raw).strip().upper()
    expected = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(labels)]
    if not labels or labels != expected:
        raise ScoringUnavailableError(
            f"multiple_choice choice_labels={raw!r} is not a contiguous A.. range"
        )
    return labels


def _expected_choice_letter(
    expected: str, choices: list[Any], labels: str = _DEFAULT_CHOICE_LABELS
) -> str | None:
    expected_match = re.fullmatch(
        rf"\s*[\(\[\{{]?\s*([{labels}])\s*[\)\]\}}]?\s*\.?\s*",
        expected,
        re.IGNORECASE,
    )
    if expected_match:
        return expected_match.group(1).upper()

    idx = _expected_choice_index(expected, choices)
    if idx is not None and idx < len(labels):
        return labels[idx]
    return None


def _expected_choice_index(expected: str, choices: list[Any]) -> int | None:
    if not choices:
        return None
    expected_norm = _normalize_choice_text(expected)
    for idx, choice in enumerate(choices):
        if _normalize_choice_text(str(choice)) == expected_norm:
            return idx
    return None


def _normalize_choice_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", str(text), flags=re.DOTALL)
    text = re.sub(r"[*_`~]+", "", text)
    text = text.strip().lower()
    text = text.strip("\"'“”‘’()[]{}")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_multiple_choice_letter(
    answer: str, labels: str = _DEFAULT_CHOICE_LABELS
) -> str | None:
    # ⚠ NOT interchangeable with epyc-inference-research
    # `scripts/benchmark/answer_scoring.py:extract_letter_answer`, despite the
    # resemblance. Proven per-consumer 2026-08-11 (`mainC`, A10) rather than assumed:
    # this function is on the authority/sealed-capture path, so a silent swap would
    # RE-SCORE sealed evidence. Deltas, with this side named first:
    #   1. RANGE: A-H here, A-J there. "I"/"J" return None here and parse there.
    #   2. LAST-RESORT RULE, the big one: Strategy 5 below returns the LAST
    #      standalone letter UNCONDITIONALLY, so a verbose reply mentioning several
    #      letters always yields a guess. The canonical function accepts a bare
    #      letter only when exactly ONE candidate exists, else returns "". This side
    #      is systematically more PERMISSIVE — it scores answers the canonical one
    #      declines to parse.
    #   3. No `\boxed{...}` handling here; canonical honours it at second priority.
    #   4. Canonical also accepts `ANSWER = X`; this accepts only `is`/`:`.
    # Unifying them is a SCORING CHANGE, not a de-duplication. It needs a re-score of
    # the affected sealed captures, not just a diff — out of scope for additive A10.
    #
    # Strategy 1: Explicit "Answer: X" — take LAST match (verbose models repeat)
    # Negative lookahead prevents "option is correct" matching as letter "C"
    cls = f"[{labels}]"
    explicit_pat = rf"(?:answer|choice|option)\s*(?:is|:)\s*\(?({cls})\)?(?![a-zA-Z])"
    explicit_matches = re.findall(explicit_pat, answer, re.IGNORECASE)
    if explicit_matches:
        return explicit_matches[-1].upper()

    # Strategy 2: Letter on its own line near the end of output
    last_line_pat = rf"^\s*\(?({cls})\)?\s*$"
    line_matches = re.findall(last_line_pat, answer, re.MULTILINE)
    if line_matches:
        return line_matches[-1].upper()

    # Strategy 3: Letter at very start of output (before any prose)
    match = re.match(rf"\s*\(?({cls})\)?\s*[.:\-\n]", answer)
    if match:
        return match.group(1).upper()

    # Strategy 4: Bold letter — take LAST match
    bold_matches = re.findall(rf"\*\*({cls})\*\*", answer)
    if bold_matches:
        return bold_matches[-1].upper()

    # Strategy 5: Last standalone letter in the text (not first!). The pronoun
    # "I" is never a standalone-letter vote: once a row widens the range past H
    # ("I think ...") it would otherwise parse as option I.
    loose = cls if "I" not in labels else f"[{labels.replace('I', '')}]"
    standalone = re.findall(rf"\b({loose})\b", answer)
    if standalone:
        return standalone[-1].upper()

    return None


def _extract_multiple_choice_text_index(answer: str, choices: list[Any]) -> int | None:
    if not choices:
        return None

    answer_norm = _normalize_choice_text(answer)
    if not answer_norm:
        return None

    matches: list[tuple[int, int, int]] = []
    for idx, choice in enumerate(choices):
        choice_norm = _normalize_choice_text(str(choice))
        if not choice_norm:
            continue
        pattern = rf"(?<!\w){re.escape(choice_norm)}(?!\w)"
        found = list(re.finditer(pattern, answer_norm))
        if found:
            match = found[-1]
            matches.append((match.end(), len(choice_norm), idx))

    if not matches:
        return None
    return max(matches)[2]


def _score_stdin_program(code: str, test_code: str, preamble: str, timeout: int) -> bool:
    """Run a stdin/stdout program against TEST_CASES.

    For competitive programming (USACO, etc.) where solutions read from stdin
    and write to stdout.  Each test case is (input_str, expected_output_str).
    The program passes if ALL test cases produce the expected output.

    Strategy: write solution to a temp file, then run it once per test case
    with stdin piped in.  Compare stdout to expected output.
    """
    # Parse TEST_CASES from the test_code string
    try:
        ns: dict = {}
        exec(test_code, ns)
        cases = ns.get("TEST_CASES", [])
    except Exception:
        return False

    if not cases:
        return False

    full_code = preamble + code

    try:
        # Benchmark test code can create relative files (for example BCB190's
        # ``test.db``). Each scorer invocation needs a private CWD so parallel
        # rows cannot observe or delete one another's files.
        with tempfile.TemporaryDirectory(
            prefix="debug-scorer-", dir=_SCORER_TMP_ROOT
        ) as workdir_name:
            workdir = Path(workdir_name)
            sol_file = workdir / "solution.py"
            sol_file.write_text(full_code, encoding="utf-8")

            for inp, expected_out in cases:
                try:
                    result = subprocess.run(
                        [sys.executable, str(sol_file)],
                        input=inp,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                        cwd=workdir,
                    )
                except subprocess.TimeoutExpired:
                    return False

                if result.returncode != 0:
                    return False

                got = result.stdout.strip()
                want = expected_out.strip()
                if got != want:
                    return False

            return True
    except OSError as exc:
        raise ScoringUnavailableError(
            "code_execution could not create or execute its temporary stdin harness"
        ) from exc


def _has_executable_assertion(test_code: str) -> bool:
    for line in test_code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("assert ") or stripped.startswith("assert("):
            expr = stripped[6:].strip() if stripped.startswith("assert ") else stripped[7:].strip()
            expr = expr.split(",", 1)[0].strip().strip("()")
            if expr == "True":
                continue
            return True
    return False


def _has_unittest_case(test_code: str) -> bool:
    return "unittest.TestCase" in test_code or "(TestCase)" in test_code


def _score_code_execution(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Extract code from model output, run against test cases.

    Used for: HumanEval, MBPP.

    Config:
        test_code: Test code to append after the model's function.
        language: Programming language (default: "python").
        timeout: Execution timeout in seconds (default: 10).
        entry_point: Function name to test (for HumanEval).
    """
    language = config.get("language", "python")
    timeout = config.get("timeout", 10)
    test_code = config.get("test_code", "")
    entry_point = config.get("entry_point", "")
    entry_point_cases = config.get("entry_point_cases")

    if language != "python":
        # Only Python execution supported currently
        return False

    # Extract code block from model output
    code = _extract_code_block(answer, language)
    if not code:
        return False

    # Prepend common imports so extracted code with type annotations
    # (e.g. List[int], Optional[str]) doesn't crash on NameError.
    _TYPING_PREAMBLE = (
        "from typing import List, Optional, Tuple, Dict, Set, Any\n"
        "from collections import defaultdict, deque, Counter\n"
        "import math, heapq, bisect, itertools, functools\n\n"
    )

    # Detect stdin-based competitive programming solutions (USACO etc.)
    # These use input() to read from stdin, so we must feed test cases via stdin.
    _uses_stdin = "input()" in code or "sys.stdin" in code
    _has_test_cases = test_code.strip().startswith("TEST_CASES")

    if _has_test_cases:
        if not _uses_stdin:
            return False
        return _score_stdin_program(code, test_code, _TYPING_PREAMBLE, timeout)

    has_test_oracle = _has_executable_assertion(test_code) or _has_unittest_case(test_code)
    has_entrypoint_oracle = bool(
        entry_point and isinstance(entry_point_cases, list) and entry_point_cases
    )
    if test_code and not has_test_oracle:
        return False
    if not test_code and not has_entrypoint_oracle:
        if entry_point and expected:
            raise ScoringUnavailableError(
                "code_execution entry_point oracle requires executable "
                "entry_point_cases or test_code; refusing to synthesize a "
                "zero-argument assertion from expected text"
            )
        return False

    # Build full test script
    full_code = _PARENT_DEATH_GUARD + "\n" + _TYPING_PREAMBLE + code
    if test_code:
        full_code += "\n\n" + test_code
        if _has_unittest_case(test_code) and "unittest.main" not in test_code:
            full_code += "\n\nif __name__ == '__main__':\n    unittest.main()\n"
    elif entry_point:
        if not _is_safe_entry_point(entry_point):
            raise ScoringUnavailableError(
                f"code_execution entry_point {entry_point!r} is not a safe Python identifier path"
            )
        cases_literal = repr(entry_point_cases)
        full_code += (
            "\n\n"
            f"_EPYC_ENTRY_POINT_CASES = {cases_literal}\n"
            "for _case in _EPYC_ENTRY_POINT_CASES:\n"
            "    if isinstance(_case, dict):\n"
            "        _args = _case.get('args', [])\n"
            "        _kwargs = _case.get('kwargs', {})\n"
            "        _expected = _case.get('expected')\n"
            "    else:\n"
            "        _args, _expected = _case\n"
            "        _kwargs = {}\n"
            f"    assert {entry_point}(*_args, **_kwargs) == _expected\n"
        )

    # Execute in a private sandboxed subprocess. Benchmark test code is allowed
    # to create relative files, so a shared CWD corrupts concurrent scorers.
    try:
        with tempfile.TemporaryDirectory(
            prefix="debug-scorer-", dir=_SCORER_TMP_ROOT
        ) as workdir_name:
            workdir = Path(workdir_name)
            solution_path = workdir / "solution.py"
            solution_path.write_text(full_code, encoding="utf-8")
            process = subprocess.Popen(
                [sys.executable, str(solution_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=workdir,
                start_new_session=True,
            )
            try:
                process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
                raise ScoringUnavailableError("code_execution exceeded its configured timeout")
            return process.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except OSError as exc:
        raise ScoringUnavailableError(
            "code_execution could not create or execute its temporary harness"
        ) from exc


#: Whitespace carries no meaning *inside* a line of code for the purpose of
#: "is this the same statement" — ``for(int i=0;i<n;i++)`` and
#: ``for (int i = 0; i < n; i++)`` are one statement written two ways, and a
#: bug-fix oracle that rejects the second measures formatting, not fixing.
#: Indentation is stripped along with the rest; we never compare across lines,
#: so this stays safe for Python.
_CODE_WS = re.compile(r"\s+")

#: Fenced blocks in a model answer. Each block is scored as an independent
#: candidate (see ``_score_code_patch``), so a model may narrate around its code.
_CODE_FENCE = re.compile(r"```[A-Za-z0-9_+.#-]*[ \t]*\r?\n(.*?)```", re.DOTALL)


def _code_key(text: str) -> str:
    """Whitespace-free form of a code line/blob, for identity comparison."""
    return _CODE_WS.sub("", text)


def _code_patch_candidates(answer: str) -> list[str]:
    """The code blobs an answer offers, best-effort.

    Fenced blocks when present, otherwise the whole answer. Each is scored
    independently and any one passing is a pass, so an answer that quotes the
    ORIGINAL buggy code in one block and gives the fix in another is not punished
    for quoting — while an answer that only ever reproduces the buggy code has no
    passing candidate at all.
    """
    blocks = [b for b in _CODE_FENCE.findall(answer) if b.strip()]
    return blocks or [answer]


def _score_code_patch(answer: str, config: dict[str, Any]) -> bool:
    """Did the answer make the changes the reference patch makes?

    WHY THIS EXISTS (2026-08-12, debugbench oracle rebuild — epyc-root
    ``artifacts/audit/debugbench-oracle-vacuity-20260812.md``). The suite's old
    oracle was a 100-character PREFIX of the reference solution scored with
    ``substring``. That prefix is class/constructor boilerplate already present in
    the buggy code the model is handed, so echoing the input scored a PASS on 4 of
    4 pool rows and on 76.1% of the upstream corpus. A longer prefix is not a fix:
    the defect is containment in the input, not length.

    This verifier asks the only question the shipped data can answer without
    executable tests — **what did the answer CHANGE?**

    * ``required_lines`` — lines the reference solution has and the buggy code
      does not. Matched as a substring of the candidate's whitespace-free text, so
      re-indenting or splitting a statement across lines still matches.
    * ``forbidden_lines`` — lines the buggy code has and the reference solution
      does not: the broken statements. Matched as a whole normalised LINE, never
      as a substring, because a buggy ``return idx;`` is a substring of a corrected
      ``return idx + 1;`` and substring matching there would fail correct answers.

    Both sides must hold in one candidate block. Anti-echo is structural, not
    incidental: reproducing the buggy code reproduces the forbidden lines and omits
    the required ones. The builder (``scripts/benchmark/debugbench_oracle.py``)
    additionally re-runs this function against the buggy code and against the
    reference solution at build time and refuses to emit a row unless it fails the
    first and passes the second, so neither a vacuous nor an unsatisfiable row can
    reach the pool.

    Fails closed on an empty oracle, exactly as ``_score_substring`` does with an
    empty needle: no oracle is not a free pass.
    """
    required = [k for k in (_code_key(str(x)) for x in config.get("required_lines") or []) if k]
    forbidden = [k for k in (_code_key(str(x)) for x in config.get("forbidden_lines") or []) if k]
    if not required and not forbidden:
        return False
    for candidate in _code_patch_candidates(answer):
        flat = _code_key(candidate)
        if not all(need in flat for need in required):
            continue
        candidate_lines = {_code_key(line) for line in candidate.splitlines()}
        if any(bad in candidate_lines for bad in forbidden):
            continue
        return True
    return False


def _score_programmatic(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Run IFEval-style programmatic verifiers.

    Used for: IFEval — checks format constraints.

    Config:
        verifier: Name of verifier to run. Options:

            YAML prompt verifiers:
            - word_count_min/max/range: word count checks (threshold, min_val, max_val)
            - contains_keyword / no_keyword: keyword presence (keyword)
            - starts_with / ends_with: text prefix/suffix (text)
            - json_valid / all_uppercase / all_lowercase: format checks
            - bullet_list / numbered_list: list format checks
            - paragraph_count / sentence_count_min: structure checks (threshold)
            - comma_separated / title_case: format checks

            IFEval adapter verifiers (from dataset_adapters.py):
            - no_comma: answer contains no commas
            - has_title: first line is short + title-cased
            - placeholder_count: count of [placeholder] patterns (count)
            - bullet_count: minimum bullet points (count)
            - contains_keywords: all keywords present (keywords list)
            - no_forbidden_words: none of forbidden words present (forbidden list)
            - language: language check (always passes — no langdetect)
            - non_empty: answer is non-empty
            - highlighted_sections: contains **bold** or ## headings
            - word_count: word count with relation (count, relation)
            - sentence_count: sentence count with relation (count, relation)

            Bug-fix verifier (debugbench, built by scripts/benchmark/debugbench_oracle.py):
            - code_patch: answer contains every `required_lines` entry and no
              `forbidden_lines` line — see _score_code_patch.

        threshold: Numeric threshold for count-based verifiers.
        count: Alias for threshold (used by IFEval adapter).
        relation: "at_least" | "at_most" | "exactly" (IFEval word/sentence count).
        keyword: Keyword for contains/no_keyword verifiers.
        keywords: Keyword list for contains_keywords verifier.
        forbidden: Forbidden word list for no_forbidden_words verifier.
        required_lines / forbidden_lines: Code lines for the code_patch verifier.
        text: Text for starts_with/ends_with verifiers.
        min_val / max_val: Range for range-based verifiers.
    """
    verifier = config.get("verifier", "")
    threshold = config.get("threshold", 0)
    keyword = config.get("keyword", "")
    keywords = config.get("keywords", [])
    forbidden = config.get("forbidden", [])
    text = config.get("text", "")
    min_val = config.get("min_val", 0)
    max_val = config.get("max_val", 0)
    # IFEval adapter uses "count" and "relation" instead of threshold/min_val/max_val
    count = config.get("count") or threshold or 0
    relation = config.get("relation", "at_least")

    answer_stripped = answer.strip()
    words = answer_stripped.split()
    wc = len(words)
    lines = answer_stripped.split("\n")

    def _word_count_by_relation() -> bool:
        """Handle word_count/sentence_count with 'relation' from IFEval adapter."""
        if relation == "at_least":
            return wc >= count
        elif relation == "at_most":
            return wc <= count
        elif relation == "exactly":
            return wc == count
        return wc >= count  # default: at_least

    def _sentence_count_by_relation() -> bool:
        sc = len(re.findall(r"[.!?]+", answer_stripped))
        if relation == "at_least":
            return sc >= count
        elif relation == "at_most":
            return sc <= count
        elif relation == "exactly":
            return sc == count
        return sc >= count

    verifiers = {
        # Original verifiers (YAML prompts use these names)
        "word_count_min": lambda: wc >= (count or threshold),
        "word_count_max": lambda: wc <= (count or threshold),
        "word_count_range": lambda: min_val <= wc <= max_val,
        "contains_keyword": lambda: keyword.lower() in answer_stripped.lower(),
        "no_keyword": lambda: keyword.lower() not in answer_stripped.lower(),
        "starts_with": lambda: answer_stripped.lower().startswith(text.lower()),
        "ends_with": lambda: answer_stripped.rstrip(".!?").lower().endswith(text.lower()),
        "json_valid": lambda: _is_valid_json(answer_stripped),
        "all_uppercase": lambda: answer_stripped == answer_stripped.upper(),
        "all_lowercase": lambda: answer_stripped == answer_stripped.lower(),
        "bullet_list": lambda: any(
            line.strip().startswith(("- ", "* ", "• ")) for line in lines if line.strip()
        ),
        "numbered_list": lambda: any(
            re.match(r"^\d+[\.\)]\s", line.strip()) for line in lines if line.strip()
        ),
        "paragraph_count": lambda: (
            len([p for p in re.split(r"\n\s*\n", answer_stripped) if p.strip()])
            == (count or threshold)
        ),
        "sentence_count_min": lambda: (
            len(re.findall(r"[.!?]+", answer_stripped)) >= (count or threshold)
        ),
        "comma_separated": lambda: "," in answer_stripped and "\n" not in answer_stripped.strip(),
        # IFEval adapter verifiers (dataset_adapters.py emits these names)
        "no_comma": lambda: "," not in answer_stripped,
        "has_title": lambda: (
            bool(
                lines[0].strip()
                and len(lines[0].strip().split()) <= 10
                and lines[0].strip().istitle()
            )
            if lines
            else False
        ),
        "placeholder_count": lambda: len(re.findall(r"\[.*?\]", answer_stripped)) >= (count or 1),
        "bullet_count": lambda: (
            sum(1 for line in lines if line.strip().startswith(("- ", "* ", "• "))) >= (count or 1)
        ),
        "contains_keywords": lambda: (
            all(kw.lower() in answer_stripped.lower() for kw in keywords) if keywords else True
        ),
        "no_forbidden_words": lambda: (
            not any(fw.lower() in answer_stripped.lower() for fw in forbidden)
            if forbidden
            else True
        ),
        "language": lambda: True,  # Cannot verify without langdetect; pass through
        "non_empty": lambda: len(answer_stripped) > 0,
        "highlighted_sections": lambda: bool(
            re.search(r"\*\*[^*]+\*\*", answer_stripped)
            or re.search(r"^##\s+", answer_stripped, re.MULTILINE)
        ),
        # IFEval relation-based verifiers (word_count with at_least/at_most/exactly)
        "word_count": _word_count_by_relation,
        "sentence_count": _sentence_count_by_relation,
        "title_case": lambda: (
            all(w[0].isupper() for w in words if w and w[0].isalpha()) if words else False
        ),
        # Bug-fix (debugbench) verifier — scores the CHANGE, not the reproduction.
        "code_patch": lambda: _score_code_patch(answer, config),
    }

    fn = verifiers.get(verifier)
    if fn is None:
        # Mirror score_answer's "Unknown scoring method" convention: an
        # unrecognized verifier is a config defect, not a silent substring
        # match that would score arbitrary answers as correct.
        raise ValueError(f"Unknown programmatic verifier: {verifier!r}")

    return fn()


def _score_substring(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Check if expected text appears in output.

    Used for: Needle-in-haystack, simple factoid QA.

    Config:
        case_sensitive: Whether comparison is case-sensitive (default: False).

    Digit-group separators (commas/underscores/spaces sitting *between two
    digits*) are stripped from both sides before matching, so a correctly
    computed numeric answer formatted as "479,001,600" still matches the
    expected substring "479001600". Non-numeric text is untouched because the
    separator must be flanked by digits on both sides (e.g. "Hello, world" is
    left as-is). 2026-06-02: fixes the agentic factorial sentinel, which began
    failing on 06-01 once the compute-first prompt made the model emit
    comma-grouped results.
    """
    if config.get("language"):
        # PRB-T4 (2026-09-17). A row that declares a programming `language` but is
        # scored by `substring` has no correctness oracle: every such row in the
        # live pool (2,349 livecodebench + 3 real_suite_v1) carries the needle
        # "def ", which any Python answer — right or wrong — contains. The
        # adapter was rebuilt to an executable oracle on 2026-08-12 (research
        # cb0761b5), but a pool built before that still ships these rows. Refuse
        # rather than pass vacuously; the caller EXCLUDES the row.
        raise ScoringUnavailableError(
            f"substring oracle on a {config.get('language')!r} code row "
            f"(needle={(expected.strip() or config.get('substring'))!r}) cannot decide "
            "correctness; rebuild the pool so the row carries an executable "
            "code_execution oracle"
        )
    case_sensitive = config.get("case_sensitive", False)

    def _strip_digit_separators(s: str) -> str:
        return re.sub(r"(?<=\d)[,_ ](?=\d)", "", s)

    answer = _strip_digit_separators(answer)
    expected = _strip_digit_separators(expected)

    needle = expected.strip()
    if not needle:
        # Part of the pool carries the needle in `scoring_config["substring"]` and
        # leaves `expected` empty — instruction_precision ("P.S.") and agentic
        # ("send_email") are built that way. This function only ever read
        # `expected`, so those rows had no reachable oracle: 25 questions that the
        # tower then dropped as unscoreable, which is the `expected=='' never
        # scored` defect recorded in instrument_eras.yaml:known_dead_instrument_items.
        #
        # Fail-closed either way: an empty needle still returns False rather than
        # matching everything. Reading config here makes the oracle reachable; it
        # does not weaken the check.
        needle = _strip_digit_separators(str(config.get("substring", "") or "")).strip()
    if not needle:
        return False
    return _contains_text_unit(answer, needle, case_sensitive=case_sensitive)


def _score_f1(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Token-level F1 scoring for QA tasks.

    Used for: HotpotQA, SQuAD-style reading comprehension.

    Computes precision/recall/F1 at the token level after normalization.
    A prediction is considered correct if F1 >= threshold.

    Config:
        extract_pattern: Regex to extract answer (default: <answer> tag).
        threshold: Minimum F1 to count as correct (default: 0.5).
        normalize: Whether to normalize text (default: True).
    """
    pattern = config.get("extract_pattern", r"<answer>(.*?)</answer>")
    threshold = config.get("threshold", 0.5)
    normalize = config.get("normalize", True)

    # Extract answer: find the LAST occurrence — models may emit
    # the tag multiple times before settling on a final answer.
    compiled_pattern = _compile_single_group_pattern(pattern)
    matches = compiled_pattern.findall(answer)
    if matches:
        extracted = matches[-1].strip()
    else:
        # Legacy fallback: try #### pattern for backward compatibility
        legacy_matches = re.findall(r"####[ \t]*\n?(.+)", answer, re.IGNORECASE)
        if legacy_matches:
            extracted = legacy_matches[-1].strip()
        else:
            extracted = _extract_answer(answer, pattern)
    if extracted is None:
        # Fallback: use last non-empty line
        lines = [ln.strip() for ln in answer.strip().split("\n") if ln.strip()]
        extracted = lines[-1] if lines else ""

    if normalize:
        extracted = _normalize_text(extracted)
        expected = _normalize_text(expected)

    # Tokenize
    pred_tokens = extracted.split()
    gold_tokens = expected.split()

    if not gold_tokens:
        return len(pred_tokens) == 0

    if not pred_tokens:
        return False

    # Compute multiset token overlap so repeated entities are counted honestly.
    from collections import Counter

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    common = sum((pred_counts & gold_counts).values())

    if not common:
        return False

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1 >= threshold


def _normalize_text(text: str) -> str:
    """Normalize text for F1 scoring (SQuAD-style)."""
    import string
    import unicodedata

    # Fold diacritics before punctuation stripping so answer variants like
    # "Dusan Lajovic" and "Dušan Lajović" score as the same tokens.
    text = "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text


def _score_llm_judge(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Score using a local LLM as semantic equivalence judge.

    Used for semantic-oracle rows, including symbolic physics/math answers and
    long-form summary/reference-answer tasks where deterministic matching is
    insufficient.

    Calls an OpenAI-compatible endpoint to judge whether the model's answer is
    semantically equivalent to the expected answer.

    Endpoint resolution (realized-first, mirrors commits 5aa29f35/e97d4ed9):
    a hardcoded llama-server port (the old default 8082 = a worker_general
    *quarter* port) is dead on a quarters-only / eval-batch stack, so the judge
    now defaults to the ORCHESTRATOR API, which resolves the role to a LIVE
    backend itself. No new hardcoded llama port is introduced. An explicit
    ``judge_host``+``judge_port`` in ``scoring_config`` still wins (targeted
    override); otherwise the endpoint comes from ``ORCHESTRATOR_API_URL``
    (default ``http://localhost:8000``). A down/malformed judge remains an
    honest ScoringUnavailableError — we never launch anything and never
    silently fall back to substring.

    Protocol (two shapes — the earlier bug was speaking the wrong one):
        - Orchestrator API (the realized-first default / env path): the
          orchestrator does NOT serve a bare llama ``/completion``; its native
          eval ingress is ``POST /chat`` with an orchestrator payload
          (``real_mode``/``mock_mode``/``force_role``/``workload_class``) and a
          response carrying ``answer`` — NOT llama's ``choices[].message``.
          Posting the raw OpenAI/llama payload instead auto-routes the judge
          prompt through the frontdoor's multi-turn REPL loop (the ``/v1``
          OpenAI-compat ingress), which overruns the judge's short timeout and
          surfaces as a "malformed response" ScoringUnavailableError. We now
          POST ``/chat`` with ``force_mode="direct"`` (one direct LLM call, no
          REPL) and ``force_role`` pinned to a text role, and parse ``answer``
          — mirroring ``call_orchestrator_forced`` / eval_tower's rubric judge.
        - Explicit ``judge_url`` / ``judge_host``+``judge_port`` override: a
          direct llama-server target, so we keep the raw OpenAI-compatible
          ``/v1/chat/completions`` protocol and parse ``choices[].message``.

    Config:
        judge_host + judge_port: Explicit judge server (both required to
            override; targets ``http://{host}:{port}/v1/chat/completions``
            via the raw llama-server protocol).
        judge_url: Explicit full base URL override (wins over host/port; raw
            llama-server protocol).
        judge_role: Text role to pin the orchestrator judge to (force_role).
            Defaults to ``LLM_JUDGE_ROLE`` env, else the disjoint GPU
            ``architect_critic`` lane (the :8083 27B; ``architect_general``
            before the 2026-09-27 ARCHITECT SWAP).
        timeout: HTTP timeout in seconds (default: 30).
        _eval_batch_id: Internal EvalTower batch correlation identifier.
    """
    if config.get("per_nugget"):
        # CME-1: a nugget-rubric item (BEAM) is judged ONCE PER NUGGET on a
        # three-valued 0 / 0.5 / 1 scale and folded later. A single boolean here
        # would binarise that scale, which is exactly the fold confusion that put
        # one BEAM run at 49.0 and 55.7. Refuse loudly instead of answering a
        # different question.
        raise ScoringUnavailableError(
            "llm_judge_per_nugget_item: scoring_config.per_nugget is set; a boolean "
            "verdict would binarise the 0/0.5/1 nugget scale. Judge each nugget with "
            "request_llm_judge_text() (epyc-inference-research "
            "scripts/benchmark/judge_beam_run.py); refusing scorer fallback"
        )

    # First try a boundary-aware substring fast path; contained words such as
    # "cat" in "concatenate" must still go to the judge.
    if _contains_text_unit(answer, expected.strip()):
        return True

    # Preserve the whole candidate. The former last-line fallback silently
    # reduced summaries/tables to one trailing row, making the judge answer a
    # different question than the benchmark asked. A boxed mathematical answer
    # remains a useful compact specialization when present.
    boxed = re.findall(r"\\boxed\{(.+?)\}", answer, re.DOTALL)
    candidate = boxed[-1].strip() if boxed else answer.strip()

    judge_prompt = (
        "You are a strict semantic answer judge. Decide whether the candidate "
        "conveys the same correct substantive answer as the reference.\n\n"
        "For mathematical or physics answers, accept equivalent notation, "
        "symbolic rearrangements, and equivalent units. For summaries or "
        "long-form answers, require the reference's central claims without "
        "material contradiction; ignore harmless wording and formatting "
        "differences. Do not reward an answer merely for sharing keywords.\n\n"
        f"REFERENCE ANSWER:\n{expected}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}\n\n"
        "Return only the JSON boolean true or false."
    )

    verdict_raw = request_llm_judge_text(
        judge_prompt, config, max_tokens=8, output_schema={"type": "boolean"}
    )
    return _parse_judge_boolean_verdict(verdict_raw, config)


def _parse_judge_boolean_verdict(verdict_raw: str, config: dict[str, Any]) -> bool:
    """Parse the boolean judge verdict from ``_score_llm_judge`` (TD-21.9/21.10).

    The wire already asks for a JSON boolean via ``output_schema={"type":
    "boolean"}`` (TD-21.9's orchestrator branch) / ``response_format``
    (TD-21.10's raw llama-server branch, gated the same as here). The STRICT
    check below (exact ``"true"``/``"false"`` after stripping) always runs and
    is always counted via ``_record_judge_parse_outcome`` — so the rate a
    strict parse would see is observable regardless of the flag, matching the
    always-on counters elsewhere in this module.

    ``CONSTRAIN_JUDGE_OUTPUT=False`` (default): unchanged prefix-match
    behaviour (``.lower().startswith("true")``), byte-identical to every
    prior release.

    ``CONSTRAIN_JUDGE_OUTPUT=True``: the prefix-match artifact is removed —
    only an exact ``"true"``/``"false"`` is accepted. Anything else is a
    SCORER-side failure (the judge, not the model under test, produced
    unusable output despite the boolean schema) and raises
    ``ScoringUnavailableError`` directly. This is deliberately NOT
    ``AnswerParseError``: that subclass is reserved for a MODEL's own answer
    failing to fit its own required shape (TD-21.11..21.14); a judge failure
    is scorer-infrastructure unavailability, the same class this module
    already raises ``ScoringUnavailableError`` for on an unreachable judge or
    malformed envelope, and is therefore unconditional on
    ``EXCLUDE_UNPARSEABLE_ANSWERS`` (a different, model-side flag).
    """
    arm_key = (config or {}).get("_eval_batch_id")
    verdict = verdict_raw.strip()
    lowered = verdict.lower()
    strictly_parseable = lowered in ("true", "false")
    _record_judge_parse_outcome(
        "llm_judge_boolean",
        "parsed" if strictly_parseable else "unparseable",
        arm_key,
    )
    if not CONSTRAIN_JUDGE_OUTPUT:
        return lowered.startswith("true")
    if strictly_parseable:
        return lowered == "true"
    raise ScoringUnavailableError(
        f"llm_judge_unparseable_verdict: judge replied {verdict_raw[:80]!r}, "
        "neither exactly 'true' nor 'false' despite a boolean output_schema"
    )


def request_llm_judge_text(
    judge_prompt: str,
    config: dict[str, Any],
    *,
    max_tokens: int,
    output_schema: dict[str, Any] | None,
) -> str:
    """Send one prompt to the resolved llm_judge endpoint and return its raw text verdict.

    This is the transport ``_score_llm_judge`` uses, exposed so that a caller
    with a non-boolean contract can reuse the same endpoint resolution, the same
    protocol choice and the same fail-closed error taxonomy. The per-nugget BEAM
    judge (CME-1, ``epyc-inference-research
    scripts/benchmark/judge_beam_run.py``) is one such caller. It never parses
    the verdict: the returned text is stripped but not lowercased, and the caller
    owns the parse.

    Any transport, HTTP, shape or empty-answer failure raises
    ``ScoringUnavailableError``. It is never returned as text.
    """
    timeout = config.get("timeout", 30)
    judge_url = _resolve_llm_judge_base_url(config)
    use_orchestrator = _llm_judge_uses_orchestrator(config)

    import httpx

    try:
        if use_orchestrator:
            # Native orchestrator eval path. force_mode="direct" runs ONE direct
            # LLM call (no REPL loop); force_role pins the judge to a text role
            # so the equivalence prompt isn't auto-routed to a vision/code
            # specialist. Mirrors seeding_orchestrator.call_orchestrator_forced.
            client_deadline_unix_s = time.time() + float(timeout)
            payload = {
                "prompt": judge_prompt,
                "real_mode": True,
                "mock_mode": False,
                "force_mode": "direct",
                "force_role": _llm_judge_force_role(config),
                "workload_class": "eval_batch",
                "request_priority": "background",
                # Queueing must not consume the entire end-to-end judge
                # deadline. Scoring is admitted only after a quiescence guard;
                # five seconds is therefore a fault signal, not useful work.
                "max_queue_wait_ms": min(
                    5_000,
                    max(1_000, int(float(timeout) * 250)),
                ),
                "max_tokens": int(max_tokens),
                "timeout_s": int(timeout),
                "client_deadline_unix_s": client_deadline_unix_s,
                "allow_delegation": False,
                "output_schema": output_schema,
            }
            eval_batch_id = str(config.get("_eval_batch_id") or "").strip()
            if eval_batch_id:
                payload["batch_id"] = eval_batch_id
            resp = httpx.post(
                f"{judge_url}/chat",
                json=payload,
                # The server receives the authoritative deadline above. A small
                # transport grace lets it return its structured timeout instead
                # of making the client abandon a still-running backend request.
                timeout=float(timeout) + 5.0,
            )
            resp.raise_for_status()
            data = resp.json()
            verdict = str(data.get("answer") or "").strip()
            if data.get("error"):
                # A structured backend error (200-with-error body) or an empty
                # answer is scorer-unavailability, NOT a "false" verdict. Route
                # it through the shared handler below so it becomes an honest
                # ERROR row instead of a silently-wrong score.
                raise _JudgeResponseError(
                    "backend_error",
                    f"{type(data.get('error')).__name__}: {str(data.get('error'))[:120]}",
                )
            if not verdict:
                raise _JudgeResponseError("empty_answer", "answer field was empty")
        else:
            # Explicit judge_url / judge_host+judge_port override: a direct
            # llama-server target — keep the raw OpenAI-compatible protocol.
            raw_body: dict[str, Any] = {
                "messages": [{"role": "user", "content": judge_prompt}],
                "max_tokens": int(max_tokens),
                "temperature": 0.0,
            }
            # TD-21.10: this branch sent NO schema at all before this flag —
            # the orchestrator branch above already constrained its reply,
            # so a raw-endpoint judge was the one unconstrained lane. Forward
            # the same `output_schema` as an OpenAI `response_format`
            # envelope, matching the protocol this endpoint already speaks
            # (mirrors `LLMPrimitives`'s own `/v1` json_schema forwarding).
            if CONSTRAIN_JUDGE_OUTPUT and output_schema is not None:
                raw_body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "llm_judge_verdict", "schema": output_schema},
                }
            resp = httpx.post(
                f"{judge_url}/v1/chat/completions",
                json=raw_body,
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            try:
                verdict = str(data["choices"][0]["message"]["content"]).strip()
            except (KeyError, IndexError, TypeError) as exc:
                raise _JudgeResponseError("unexpected_shape", type(exc).__name__) from exc
            if not verdict:
                raise _JudgeResponseError("empty_answer", "content field was empty")
    except httpx.TimeoutException as exc:
        cause = exc
        category = "transport_timeout"
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
    except httpx.HTTPStatusError as exc:
        cause = exc
        status = getattr(getattr(exc, "response", None), "status_code", "unknown")
        category = f"http_status_{status}"
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
    except httpx.RequestError as exc:
        cause = exc
        category = "transport_error"
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
    except _JudgeResponseError as exc:
        cause = exc
        category = exc.category
        detail = exc.detail
    except json.JSONDecodeError as exc:
        cause = exc
        category = "invalid_json"
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        cause = exc
        # Some response stand-ins and older HTTP clients raise plain ValueError
        # for invalid JSON. Preserve that distinction from transport failure.
        category = "invalid_json" if isinstance(exc, ValueError) else "unexpected_shape"
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
    else:
        return verdict

    # The category and original exception class/message are deliberately first:
    # compact per-question sidecars truncate long errors. Never include prompts
    # or response bodies here.
    message = (
        f"llm_judge_{category}: endpoint={judge_url}; cause={detail}; refusing scorer fallback"
    )
    log.warning("LLM judge unavailable: %s", message)
    raise ScoringUnavailableError(message) from cause


def _resolve_llm_judge_base_url(config: dict[str, Any]) -> str:
    """Resolve the llm_judge base URL (scheme://host:port), realized-first.

    Precedence: explicit ``judge_url`` > explicit ``judge_host``+``judge_port``
    > ``ORCHESTRATOR_API_URL`` env > ``http://localhost:8000``. Routing through
    the orchestrator API (rather than a hardcoded llama-server port) lets the
    orchestrator resolve the judge role to a LIVE backend on a quarters-only
    fleet — no new hardcoded ports, per commits 5aa29f35/e97d4ed9.
    """
    import os

    explicit_url = str(config.get("judge_url") or "").strip()
    if explicit_url:
        return explicit_url.rstrip("/")
    # Both host AND port must be supplied to take the legacy direct-port path;
    # a lone hardcoded port default is exactly the dead-endpoint trap we fix.
    host = config.get("judge_host")
    port = config.get("judge_port")
    if host and port:
        return f"http://{host}:{port}".rstrip("/")
    return os.environ.get("ORCHESTRATOR_API_URL", "http://localhost:8000").rstrip("/")


def _llm_judge_uses_orchestrator(config: dict[str, Any]) -> bool:
    """Whether the judge routes through the orchestrator API (native ``/chat``).

    Mirrors ``_resolve_llm_judge_base_url``'s precedence: an explicit
    ``judge_url`` or ``judge_host``+``judge_port`` override is a DIRECT
    llama-server target (raw OpenAI ``/v1/chat/completions`` protocol,
    ``choices[].message`` responses); everything else — the
    ``ORCHESTRATOR_API_URL`` env or the ``http://localhost:8000`` default —
    is the orchestrator, which speaks its own ``/chat`` schema (``answer``).
    Speaking the wrong one is exactly the "malformed response" bug this fixes.
    """
    if str(config.get("judge_url") or "").strip():
        return False
    if config.get("judge_host") and config.get("judge_port"):
        return False
    return True


def _llm_judge_force_role(config: dict[str, Any]) -> str:
    """Text role to pin the orchestrator judge to (``force_role``).

    An equivalence judge must land on a text model, not be auto-routed to a
    vision/code specialist. Precedence: ``scoring_config['judge_role']`` >
    ``LLM_JUDGE_ROLE`` env > ``architect_critic`` (the disjoint GPU judge — the
    :8083 27B, labelled ``architect_general`` before the 2026-09-27 ARCHITECT SWAP).
    """
    import os

    role = str(config.get("judge_role") or "").strip()
    if role:
        return role
    return os.environ.get("LLM_JUDGE_ROLE", "").strip() or DEFAULT_LLM_JUDGE_ROLE


def _score_math_verify(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Score using Math-Verify library for symbolic mathematical comparison.

    Used for: MATH-500 — where equivalent expressions should match
    (e.g. \\frac{mg}{2} ≡ mg/2, x^2+1 ≡ 1+x^2, {1,2,3} ≡ {3,1,2}).

    Requires: pip install math-verify (Apache-2.0, HuggingFace).

    Error semantics — NO silent scorer fallback:
        - math-verify not installed → ScoringUnavailableError. We refuse to
          quietly score math with exact_match; that silent swap is what
          mis-scored every threaded math eval.
        - The GOLD (``expected``) answer raising on parse, or parsing to an
          empty extraction, is a dataset/gold defect → ScoringUnavailableError.
        - The MODEL's answer failing to parse is a *task* failure → False.
        - ``verify`` itself raising is a scorer defect → ScoringUnavailableError.

    Thread safety:
        math-verify guards BOTH ``parse`` and ``verify`` with ``signal.alarm``,
        which raises ``ValueError("... signal only works in main thread ...")``
        on any non-main thread. The library's documented remedy is to disable
        those timeouts (``parse(..., parsing_timeout=None)`` and
        ``verify(..., timeout_seconds=None)``); we pass both whenever we are
        off the main thread — the eval-level watchdog still bounds pathological
        wall time — so threaded scoring runs the real math_verify path instead
        of silently degrading (or, if only parse were fixed, ERRORing on every
        threaded verify() call).

    Config:
        extraction_mode: "latex" (default), "expr", or "string"
    """
    try:
        from math_verify import parse, verify
    except ImportError as exc:
        raise ScoringUnavailableError(
            "math-verify not installed but scoring_method=math_verify "
            "requested; refusing to silently fall back to exact_match"
        ) from exc

    parse_kwargs: dict[str, Any] = {}
    verify_kwargs: dict[str, Any] = {}
    if threading.current_thread() is not threading.main_thread():
        # math-verify's own documented remedy for threaded use; the
        # eval-level watchdog bounds pathological wall time.
        parse_kwargs["parsing_timeout"] = None
        verify_kwargs["timeout_seconds"] = None

    try:
        gold = parse(expected, **parse_kwargs)
    except Exception as exc:
        raise ScoringUnavailableError(
            "math_verify could not parse the GOLD/expected answer "
            f"{expected!r} (dataset/gold defect)"
        ) from exc
    if not gold:
        raise ScoringUnavailableError(
            "math_verify extracted nothing from the GOLD/expected answer "
            f"{expected!r} (dataset/gold defect)"
        )

    try:
        pred = parse(answer.strip(), **parse_kwargs)
    except Exception:
        # The model's own answer failing to parse is a task failure, not a
        # scorer-unavailability condition — score it wrong, don't raise.
        return False

    try:
        # gold-first argument order — verify() is asymmetric.
        return bool(verify(gold, pred, **verify_kwargs))
    except Exception as exc:
        raise ScoringUnavailableError(
            "math_verify.verify() raised while comparing parsed answers (scorer defect)"
        ) from exc


# ── f1_list (tulving_episodic) + structural_exact_match (longcot_mini) ────
# ADDITIVE scorers (audit SCORE-25 / SCORE-26). Before these landed, every row
# of the tulving_episodic suite (scoring_method="f1_list", 456 rows) and the
# longcot_mini suite (scoring_method="structural_exact_match", 402 rows) raised
# "Unknown scoring method" and was honestly EXCLUDED (REL-1) from the
# denominator. These two scorers handle ONLY those previously-erroring methods;
# NO other scorer's verdict changes (verified against the B7 golden corpus,
# which contains no f1_list / structural_exact_match rows). They reuse the B7
# primitives (final-answer anchoring, the shared ``_normalize_text`` boundary
# normaliser) and do not fork or modify any existing scorer's semantics.


def _score_f1_list(answer: str, expected: str, config: dict[str, Any]) -> bool:
    """Item-level (set-level) F1 for episodic-memory *list* answers.

    Used for: tulving_episodic — questions whose gold is a JSON list of answer
    items (locations, entity/person names, dates, event-content phrases).
    Distinct from the token-multiset ``f1`` scorer: here BOTH the gold and the
    parsed model answer are LISTS, matched item-to-item.

    Faithful to the reference deterministic scorer (epyc-inference-research
    ``tulving_episodic_adapter.score_f1_list``): greedy GT→prediction matching by
    per-item token-F1, a lenient precision denominator (``min(nb_pred, nb_gt)``,
    per the benchmark paper), and the group-0 hallucination policy (empty gold +
    any prediction ⇒ F1 0; empty gold + empty prediction ⇒ F1 1). The one
    deliberate substitution — per this task's B7-reuse mandate — is that per-item
    token normalization uses the shared B7 ``_normalize_text`` (SQuAD-style: NFKD
    diacritic fold, lowercase, punctuation strip, article removal, whitespace
    collapse) rather than the adapter's private NFC normaliser. That is strictly
    more lenient (folds accents, drops articles) and never fabricates a match.

    Config:
        threshold: final F1 pass/fail cutoff AND the per-item greedy-match cutoff
            (default 0.5 — the value every pool row carries).

    Gold format: ``expected`` MUST be a JSON list. A non-list / unparseable gold
    is a dataset defect ⇒ ScoringUnavailableError (an EXCLUDED row, never False).
    """
    threshold = config.get("threshold", 0.5)
    gold_items = _parse_gold_list(expected)
    pred_items, used_fallback = _extract_list_items_with_fallback(answer)
    f1 = _f1_list_score(pred_items, gold_items, threshold=threshold)
    passed = f1 >= threshold
    if passed or not used_fallback:
        return passed
    # TD-21.12 (corrected 2026-09-24): the crude one-per-line catch-all fired
    # (no bullet/numbered/comma list structure was recognised). If it still
    # produced candidate item(s), the model DID answer with something
    # comparable — a low F1 against gold is a genuine wrong answer, never
    # excluded, exactly like a cleanly extracted list that is simply the
    # wrong items. Only when the fallback found NOTHING at all (an
    # empty/whitespace-only answer, so `pred_items` is itself empty) is
    # there truly no candidate to compare — that is the real model-side
    # parse failure.
    if pred_items:
        return False
    return _unparseable_answer(
        "f1_list",
        "no bullet/numbered/comma list structure recognized, and the raw "
        "line-split fallback produced no candidate items at all "
        f"(empty/whitespace-only answer; threshold={threshold})",
        config=config,
    )


def _score_structural_exact_match(answer: str, expected: str, config: dict[str, Any]) -> bool:
    r"""Structural (canonicalized) equality for longcot_mini answers.

    Used for: longcot_mini — every prompt instructs the model to end with
    ``solution = <value>``, where ``<value>`` is a JSON value; ``expected`` is
    the canonical JSON of the gold value.

    Interpretation (derived from the suite's 402 rows — golds are JSON
    str / int / list / dict): "structural" equality means parse-then-
    canonicalize-then-compare, NOT string equality. Faithful to
    ``longcot_mini_adapter.score_structural``:
      1. Final-answer extraction: take the text after the LAST ``solution =``
         marker (the suite's final-answer anchor — the B7 last-occurrence
         convention; models echo the format instruction earlier, so the real
         answer is last). No marker ⇒ the model did not follow the required
         output format ⇒ a model-side parse failure (TD-21.13): ``False``
         today (``EXCLUDE_UNPARSEABLE_ANSWERS`` default OFF, unchanged from
         the original "task failure, not scorer-unavailability" call), or
         routed through the ``AnswerParseError``/EXCLUDED path once that
         flag is ratified on. See ``_unparseable_answer``.
      2. Parse a leading JSON / Python-literal value (balanced-bracket scan for
         containers, quoted-string scan, scalar fallback — never raises).
      3. Recursively canonicalize BOTH sides: dict keys sorted; list order
         preserved; numeric scalars (incl. numeric strings — ``391365`` ==
         ``"391365"`` == ``391365.0``) collapsed to one form; non-numeric string
         case PRESERVED (SMILES / FEN are case-sensitive) with surrounding
         whitespace stripped and internal runs collapsed.
      4. Pure structural ``==``.

    The uniform ``scoring_config["extract_pattern"] = r"solution\s*=\s*(.+)"`` is
    a single-line hint; the structural parser here supersedes it because gold
    values span multiple lines (JSON arrays/objects) that a single-line ``.+``
    cannot capture.

    Fully deterministic: no sampling, no network, no model-in-the-loop —
    identical inputs always yield identical verdicts.
    """
    gold_canon = _coerce_structural_gold(expected)
    tail = _extract_solution_tail(answer)
    if tail is None:
        return _unparseable_answer(
            "structural_exact_match",
            "no 'solution = ' marker found in the model's answer",
            config=config,
        )
    predicted = _canonicalize_structural(_parse_leading_structural_value(tail))
    return predicted == gold_canon


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_answer(text: str, pattern: str) -> str | None:
    """Extract answer from text using regex pattern."""
    compiled = _compile_single_group_pattern(pattern)
    match = compiled.search(text)
    if match and match.group(1):
        return match.group(1).strip()
    return None


def _compile_single_group_pattern(pattern: str) -> re.Pattern[str]:
    compiled = re.compile(pattern, re.IGNORECASE | re.DOTALL)
    if compiled.groups != 1:
        raise ValueError(
            f"extract_pattern must contain exactly one capture group, got {compiled.groups}"
        )
    return compiled


def _extract_boxed_answer(text: str) -> str | None:
    """Extract the final LaTeX \\boxed{...} payload, including nested braces."""
    last_start = text.rfind(r"\boxed{")
    if last_start < 0:
        return None
    i = last_start + len(r"\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def _final_answer_region(text: str) -> str:
    """Return the final answer-bearing line/region, not earlier explanation."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    marker = re.compile(r"\b(final\s+answer|answer|result)\b", re.IGNORECASE)
    for line in reversed(lines):
        if marker.search(line):
            return line
    return lines[-1]


def _contains_text_unit(
    haystack: str,
    needle: str,
    *,
    case_sensitive: bool = False,
) -> bool:
    flags = 0 if case_sensitive else re.IGNORECASE
    needle = needle.strip()
    if not needle:
        return False
    left = r"(?<!\w)" if needle[0].isalnum() else ""
    right = r"(?!\w)" if needle[-1].isalnum() else ""
    return re.search(f"{left}{re.escape(needle)}{right}", haystack, flags) is not None


def _is_safe_entry_point(entry_point: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", str(entry_point)))


def _extract_code_block(text: str, language: str = "python") -> str | None:
    """Extract code from markdown code block or raw code."""
    # Try markdown code block first
    patterns = [
        rf"```{language}\s*\n(.*?)```",
        r"```\w*\s*\n(.*?)```",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try to find a def/class statement (Python-specific)
    if language == "python":
        match = re.search(r"((?:def|class)\s+\w+.*?)(?:\n\n|\Z)", text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Last resort: if text looks like executable Python code, return it
    # Covers USACO-style stdin solutions (n = int(input()), sys.stdin, etc.)
    stripped = text.strip()
    if stripped and any(
        stripped.startswith(prefix)
        for prefix in (
            "def ",
            "class ",
            "import ",
            "from ",
            "n ",
            "t ",
            "for ",
            "while ",
            "if ",
            "#",
        )
    ):
        return stripped

    # Also accept if it contains input() — likely a competitive programming solution
    if "input()" in stripped or "sys.stdin" in stripped:
        return stripped

    return None


def _is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON.

    TD-21.14: this is IFEval's ``json_valid`` programmatic verifier (used
    only via ``_score_programmatic``'s ``verifiers["json_valid"]``). NOT
    converted to the ``AnswerParseError``/exclusion path: unlike the other
    TD-21.11..21.14 sites, this IS the correctness check being measured — a
    prompt that instructs "respond in valid JSON" and gets prose back has
    genuinely failed the task, so ``False`` is a real, correctly-scored
    verdict, not scorer-unavailability (`td-json-consumer-audit-20260924.md`
    J-08's "real oracle" reasoning applies here too, even though J-08 itself
    only lists the sibling ``_score_programmatic`` verifiers explicitly).

    Deliberately NOT switched to the shared ``fish_json`` extractor
    (``src/structured_output/repair.py``): ``fish_json`` REPAIRS trailing
    commas and stray closers and prefers a fenced block over a bare one,
    so it would make this oracle accept text that is not actually valid
    JSON — an ungated live-score change to a real correctness check, with
    no era boundary covering it (a 2026-09-24 main-session review caught
    this before it landed; see the TD-21.11..21.14 commit history). Stays
    the original naive ``find("{")``/``rfind("}")`` slice, which can
    mis-extract on multiple/nested JSON blobs, but never accepts malformed
    JSON as valid — the property this verifier's callers depend on.
    """
    # Try the whole text
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find JSON in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start >= 0 and end > start:
            try:
                json.loads(text[start : end + 1])
                return True
            except (json.JSONDecodeError, ValueError):
                pass

    return False


# ── f1_list helpers (tulving_episodic) ───────────────────────────────────

_F1_LIST_ABSTENTION_RE = re.compile(
    r"(?is)(none|n/a|i don'?t know\.?|i'?m not sure\.?|"
    r"i cannot (answer|determine).*|no information( available)?\.?|"
    r"not mentioned\.?|not available\.?)"
)
_F1_LIST_BULLET_RE = re.compile(r"^[\s]*[-•*]\s*(.+)$", re.MULTILINE)
_F1_LIST_NUMBERED_RE = re.compile(r"^[\s]*\d+[.)]\s*(.+)$", re.MULTILINE)


def _parse_gold_list(expected: str) -> list[str]:
    """Parse the tulving gold (a JSON list) into a list of item strings.

    A non-list or unparseable gold is a dataset/gold defect — raise so the
    caller records an EXCLUDED row rather than scoring a wrong answer.
    """
    try:
        parsed = json.loads(expected)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ScoringUnavailableError(
            f"f1_list gold is not valid JSON: {expected!r} (dataset/gold defect)"
        ) from exc
    if not isinstance(parsed, list):
        raise ScoringUnavailableError(
            f"f1_list gold must be a JSON list, got {type(parsed).__name__}: "
            f"{expected!r} (dataset/gold defect)"
        )
    return [str(x) for x in parsed]


def _extract_list_items_with_fallback(response: str) -> tuple[list[str], bool]:
    """Like ``_extract_list_items``, plus whether the raw one-per-line
    catch-all fired (``True``) rather than a recognised list structure or a
    clean abstention (``False``).

    TD-21.12: that flag distinguishes a model that answered in an
    unstructured-but-real way (heuristically split, genuinely scoreable)
    from one whose reply carries no list structure at all — the case
    ``_score_f1_list`` treats as a model-side parse failure when the
    resulting F1 also misses threshold.
    """
    if _F1_LIST_ABSTENTION_RE.fullmatch(response.strip()):
        return [], False
    bullets = _F1_LIST_BULLET_RE.findall(response)
    if bullets:
        return [b.strip() for b in bullets if b.strip()], False
    numbered = _F1_LIST_NUMBERED_RE.findall(response)
    if numbered:
        return [n.strip() for n in numbered if n.strip()], False
    lines = [ln.strip() for ln in response.strip().split("\n") if ln.strip()]
    candidates: list[str] = []
    for line in lines:
        if "," in line and len(line) < 200:
            candidates.extend(p.strip() for p in line.split(",") if p.strip())
    if candidates:
        return candidates, False
    return lines, True


def _extract_list_items(response: str) -> list[str]:
    """Parse a model response into a list of answer items.

    Faithful port of ``tulving_episodic_adapter._extract_list_from_response``:
    an explicit abstention ⇒ empty list; else a bullet list, else a numbered
    list, else comma-separated (short lines), else one item per line.
    """
    return _extract_list_items_with_fallback(response)[0]


def _token_f1_score(prediction: str, ground_truth: str) -> float:
    """Token-multiset F1 over two strings, normalized with the B7
    ``_normalize_text``.

    Mirrors the multiset-overlap math already used by ``_score_f1`` (repeated
    tokens counted honestly), returning F1 as a float. Empty vs empty ⇒ 1.0;
    empty vs non-empty ⇒ 0.0.
    """
    from collections import Counter

    pred_tokens = _normalize_text(prediction).split()
    gold_tokens = _normalize_text(ground_truth).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not common:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _f1_list_score(
    pred_items: list[str], gold_items: list[str], *, threshold: float = 0.5
) -> float:
    """Set-level F1 for list answers via greedy GT→prediction matching.

    Faithful port of ``tulving_episodic_adapter.score_f1_list`` — same greedy
    loop, same lenient precision denominator, same empty-gold policy.
    """
    nb_gt = len(gold_items)
    nb_pred = len(pred_items)
    if nb_gt == 0 and nb_pred == 0:
        return 1.0
    if nb_gt == 0:  # hallucination: predicted items where the gold is empty
        return 0.0
    if nb_pred == 0:  # miss: no prediction against a non-empty gold
        return 0.0

    remaining = list(pred_items)
    gt_scores: list[float] = []
    for gt_item in gold_items:
        best_score = 0.0
        best_idx = -1
        for i, pred in enumerate(remaining):
            s = _token_f1_score(pred, gt_item)
            if s > best_score:
                best_score = s
                best_idx = i
        gt_scores.append(best_score)
        if best_score >= threshold and best_idx >= 0:
            remaining.pop(best_idx)

    sum_scores = sum(gt_scores)
    nb_pred_lenient = min(nb_pred, nb_gt)
    precision = sum_scores / nb_pred_lenient if nb_pred_lenient > 0 else 0.0
    recall = sum_scores / nb_gt if nb_gt > 0 else 0.0
    if precision + recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── structural_exact_match helpers (longcot_mini) ────────────────────────

_SOLUTION_MARKER_RE = re.compile(r"solution\s*=\s*", re.IGNORECASE)


def _extract_solution_tail(response: str) -> str | None:
    """Return the text after the LAST ``solution =`` marker, stripped.

    Returns None when no marker is present. Faithful port of
    ``longcot_mini_adapter._extract_solution_text``.
    """
    if not response:
        return None
    matches = list(_SOLUTION_MARKER_RE.finditer(response))
    if not matches:
        return None
    tail = response[matches[-1].end() :]
    return tail.strip() or None


def _parse_leading_structural_value(text: str) -> Any:
    """Parse a JSON / Python-literal value from the start of ``text``.

    Balanced-bracket scan for ``[...]`` / ``{...}`` containers and a quoted-
    string scan for ``"..."`` / ``'...'``; scalar fallback to the first line.
    Never raises. Faithful port of ``longcot_mini_adapter._parse_leading_value``.
    """
    if text is None:
        return None
    s = text.strip()
    if not s:
        return ""
    if s[0] in "[{\"'":
        opener = s[0]
        if opener in "[{":
            closer = "]" if opener == "[" else "}"
            depth = 0
            in_str = False
            esc = False
            end = None
            for i, ch in enumerate(s):
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = s[:end] if end is not None else s
        else:  # quoted string
            candidate = s.split("\n", 1)[0]
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
        # Unparseable container/quoted token → strip surrounding quotes.
        return candidate.strip().strip("\"'")

    # Scalar path: first line, try JSON/literal, else raw string.
    first_line = s.split("\n", 1)[0].strip()
    stripped = (
        first_line[:-1]
        if first_line.endswith(".") and not first_line[:-1].endswith(".")
        else first_line
    )
    for candidate in (first_line, stripped):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(candidate)
            except Exception:
                continue
    return first_line


def _norm_structural_scalar(v: Any) -> Any:
    """Canonicalize a scalar. Numbers and numeric strings collapse to one
    canonical numeric form (``391365`` == ``"391365"`` == ``391365.0``); non-
    numeric strings keep their CASE (SMILES/FEN) with whitespace collapsed.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() else v
    if isinstance(v, str):
        t = " ".join(v.strip().split())
        if re.fullmatch(r"[+-]?\d+", t):
            return int(t)
        try:
            f = float(t)
            if not math.isfinite(f):
                return t
            return int(f) if f.is_integer() else f
        except (ValueError, TypeError):
            return t
    return v


def _canonicalize_structural(v: Any) -> Any:
    """Recursively canonicalize a parsed value for structural equality."""
    if isinstance(v, dict):
        return {
            str(k): _canonicalize_structural(val)
            for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(v, (list, tuple)):
        return [_canonicalize_structural(x) for x in v]
    return _norm_structural_scalar(v)


def _coerce_structural_gold(gold: Any) -> Any:
    """Accept gold as a parsed value OR its JSON string form; canonicalize."""
    if isinstance(gold, str):
        try:
            gold = json.loads(gold)
        except (json.JSONDecodeError, ValueError):
            pass
    return _canonicalize_structural(gold)


def score_batch(
    questions: list[dict[str, Any]],
    answers: list[str],
) -> list[dict[str, Any]]:
    """Score a batch of answers against their questions.

    Args:
        questions: List of question dicts with id, expected, scoring_method,
            scoring_config.
        answers: List of model answers (same order as questions).

    Returns:
        List of result dicts with id, passed, expected, actual_answer.
    """
    results = []
    for q, ans in zip(questions, answers):
        passed = score_answer(
            answer=ans,
            expected=q.get("expected", ""),
            scoring_method=q.get("scoring_method", "exact_match"),
            scoring_config=q.get("scoring_config"),
        )
        results.append(
            {
                "id": q.get("id", "unknown"),
                "suite": q.get("suite", "unknown"),
                "passed": passed,
                "expected": q.get("expected", ""),
                "answer_preview": ans[:200] if ans else "",
            }
        )
    return results
