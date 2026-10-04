"""Tier 1(a): deterministic ground-truth answer checks with pluggable extractors.

Every grader returns an `answer_check` dict:
    {grader, version, expected, extracted, status: correct|wrong|unanswered, ...}
or None when the item is not gradable here (no truth, or a scorer this library does not implement:
code_execution, programmatic verifiers, llm_judge). None routes the item to tier 2.

Built-in graders (registry `GRADERS`, extend with `register_grader` or `evaluate(graders=...)`):
  gsm8k      numeric. Marker extraction only, last of: <answer>..</answer>, `#### X`, \\boxed{X},
             "(final) answer is/: X", or a text that is just a number. NO last-number fallback: a
             reply truncated mid-derivation is `unanswered`, never graded on a stray number.
  letter     multiple choice, labels from truth.labels / scoring_config.choice_labels (default A-J).
  mmlu, gpqa letter with labels A-D;  mmlu_pro  letter with labels A-J.
             Extraction (most explicit first): <answer>X</answer>, ANSWER: X, \\boxed{X}, the LAST
             "answer/option/choice is (**X**)", a terse whole reply ("C", "(C).", "**C**"), a final
             line that is only the letter, else the single standalone label letter if exactly one
             exists. Follows benchmark/answer_scoring.extract_letter_answer (the research-repo
             canonical) plus <answer> precedence and a configurable range; pinned here, not imported,
             so a scorer edit elsewhere cannot silently change a gate's grader.
  needle     expected = list of strings (or one string); all must appear (case-sensitive unless
             truth.case_sensitive is false). An abstention ("abstain", "prompt injection", ...) is
             `unanswered` with `abstained: true`; a partial hit is `wrong`.
  exact      extract via scoring_config.extract_patterns / extract_pattern (question_pool shape),
             else <answer>..</answer>, else the whole reply; normalised (case, whitespace, `$`, `,`,
             trailing `.`) equality. A configured pattern that does not match is `unanswered`.
  regex      truth.pattern (or expected) re.search against the reply: correct / wrong.
  substring  expected in reply (scoring_config.case_sensitive, default False).
  f1         <answer>..</answer> token F1 >= threshold (scoring_config.threshold, default 0.8).

`grader_for(truth)` picks one: an explicit truth.grader wins, otherwise question_pool's
scoring_method (+ id prefix) maps multiple_choice -> letter/mmlu_pro, exact_match -> gsm8k (gsm8k ids
with numeric truth) or exact, f1 -> f1. `substring`, code_execution, programmatic and llm_judge are not
auto-mapped (presence checks and executors are not answer checks) and go to tier 2.
"""
from __future__ import annotations

import collections
import re
from typing import Callable

GRADERS_VERSION = "answer-graders.v1"
RANK = {"correct": 0, "wrong": 1, "unanswered": 1}

Grader = Callable[[str, dict], dict]
GRADERS: dict[str, Grader] = {}

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
_ABSTAIN_RE = re.compile(r"\babstain|prompt injection|injected instruction|cannot (?:comply|provide)|"
                         r"won't provide|will not provide", re.I)


def register_grader(name: str, fn: Grader) -> None:
    GRADERS[name] = fn


def _grader(name: str):
    def deco(fn: Grader) -> Grader:
        register_grader(name, fn)
        return fn
    return deco


def answer_region(text: str) -> str:
    """The reply minus any <think>..</think> block (graders never read reasoning as the answer)."""
    return _THINK_RE.sub("", text).strip()


def _check(grader: str, expected, extracted, status: str, **kw) -> dict:
    return {"grader": grader, "version": GRADERS_VERSION, "expected": expected, "extracted": extracted,
            "status": status, **kw}


def _cfg(truth: dict) -> dict:
    c = truth.get("scoring_config")
    return c if isinstance(c, dict) else {}


# ------------------------------------------------------------------------------ numeric (gsm8k)
_NUM = r"-?\$?\s*\d[\d,]*(?:\.\d+)?"


def parse_number(s: str | None) -> float | None:
    if s is None:
        return None
    t = re.sub(r"[\s$,]|\\!|\\,", "", str(s)).rstrip(".")
    t = re.sub(r"\\(?:text|mathrm)\{[^}]*\}$", "", t)
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)(?:%|\\%)?", t)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"\\d?frac\{(-?\d+)\}\{(\d+)\}", t)
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    return None


def extract_number(text: str) -> tuple[str | None, str | None]:
    """(raw, method) of the stated numeric answer, or (None, None)."""
    for method, pat in (("answer_tag", _ANSWER_TAG), ("hashes", re.compile(rf"####\s*({_NUM})")),
                        ("boxed", re.compile(r"\\boxed\{([^{}]*)\}")),
                        ("answer_is", re.compile(rf"(?:final answer|answer)\s*(?:is|:|=)\s*\**\s*({_NUM})", re.I))):
        hits = pat.findall(text)
        if hits:
            return hits[-1].strip(), method
    if re.fullmatch(rf"\s*{_NUM}\s*\.?\s*", text):
        return text.strip(), "whole"
    return None, None


@_grader("gsm8k")
def grade_gsm8k(text: str, truth: dict) -> dict:
    exp = truth.get("expected")
    raw, method = extract_number(answer_region(text))
    if raw is None:
        return _check("gsm8k", exp, None, "unanswered")
    got, want = parse_number(raw), parse_number(str(exp))
    if want is None:
        raise ValueError(f"gsm8k truth {exp!r} is not numeric")
    if got is None:
        return _check("gsm8k", exp, raw, "unanswered", method=method)
    ok = abs(got - want) <= 1e-6 * max(1.0, abs(want))
    return _check("gsm8k", exp, raw, "correct" if ok else "wrong", method=method)


# ------------------------------------------------------------------------------ letters
def extract_letter(text: str, labels: str = "ABCDEFGHIJ") -> str | None:
    s = answer_region(text)
    cls = f"[{labels}{labels.lower()}]"
    for pat in (rf"<answer>\s*\**\s*\(?({cls})\)?(?![A-Za-z])", rf"(?i:ANSWER)\s*[:=]\s*\**\s*\(?({cls})\)?\b",
                rf"\\boxed\{{\s*\(?({cls})\)?\s*\}}"):
        hits = re.findall(pat, s)
        if hits:
            return hits[-1].upper()
    # marker words are case-insensitive, the letter is NOT: "the answer is a function" is no vote for A
    hits = re.findall(rf"\b(?i:answer|option|choice|letter)\s*(?i:is|:|=|-)?\s*\**\s*\(?([{labels}])\)?\**(?![A-Za-z])", s)
    if hits:
        return hits[-1].upper()
    m = re.fullmatch(rf"\**\(?({cls})\)?[.):]?\**", s)
    if m:
        return m.group(1).upper()
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        m = re.fullmatch(rf"\**\(?({cls})\)?[.):]?\**", lines[-1])
        if m:
            return m.group(1).upper()
    loose = [x for x in re.findall(rf"\b([{labels}])\b", s) if x != "I"]
    return loose[0] if len(loose) == 1 else None


def _letter(name: str, labels: str | None = None) -> Grader:
    def grade(text: str, truth: dict) -> dict:
        labs = labels or truth.get("labels") or _cfg(truth).get("choice_labels") or "ABCDEFGHIJ"
        exp = str(truth.get("expected", "")).strip().upper()
        got = extract_letter(text, labs)
        if got is None:
            return _check(name, exp, None, "unanswered")
        return _check(name, exp, got, "correct" if got == exp else "wrong")
    return grade


register_grader("letter", _letter("letter"))
register_grader("mmlu", _letter("mmlu", "ABCD"))
register_grader("gpqa", _letter("gpqa", "ABCD"))
register_grader("mmlu_pro", _letter("mmlu_pro", "ABCDEFGHIJ"))


# ------------------------------------------------------------------------------ needle
def abstained(text: str) -> bool:
    return bool(_ABSTAIN_RE.search(text))


@_grader("needle")
def grade_needle(text: str, truth: dict) -> dict:
    exp = truth.get("expected")
    want = [exp] if isinstance(exp, str) else list(exp or [])
    cs = truth.get("case_sensitive", _cfg(truth).get("case_sensitive", True))
    hay = answer_region(text)
    norm = (lambda x: x) if cs else str.lower
    found = [w for w in want if norm(w) in norm(hay)]
    if len(found) == len(want) and want:
        return _check("needle", want, found, "correct")
    if abstained(hay):
        return _check("needle", want, found, "unanswered", abstained=True)
    return _check("needle", want, found, "wrong" if hay else "unanswered")


# ------------------------------------------------------------------------------ exact / regex / substring / f1
def _norm(x: str) -> str:
    return re.sub(r"[\s$,]+", " ", x.strip().lower()).strip().rstrip(".").strip()


def _extract_configured(text: str, cfg: dict) -> tuple[str | None, bool]:
    """(extracted, pattern_configured)."""
    pats = cfg.get("extract_patterns") or ([cfg["extract_pattern"]] if cfg.get("extract_pattern") else [])
    for pat in pats:
        hits = re.findall(pat, text, re.I | re.M)
        if hits:
            h = hits[-1]
            if isinstance(h, tuple):
                h = next((p for p in h if p), "")
            return str(h).strip(), True
    return None, bool(pats)


@_grader("exact")
def grade_exact(text: str, truth: dict) -> dict:
    exp = str(truth.get("expected", ""))
    s = answer_region(text)
    got, configured = _extract_configured(s, _cfg(truth))
    if got is None:
        if configured:
            return _check("exact", exp, None, "unanswered")
        tag = _ANSWER_TAG.findall(s)
        got = tag[-1] if tag else s
    if not got.strip():
        return _check("exact", exp, got, "unanswered")
    return _check("exact", exp, got, "correct" if _norm(got) == _norm(exp) else "wrong")


@_grader("regex")
def grade_regex(text: str, truth: dict) -> dict:
    pat = truth.get("pattern") or str(truth.get("expected", ""))
    s = answer_region(text)
    m = re.search(pat, s, re.S)
    return _check("regex", pat, m.group(0) if m else None, "correct" if m else ("wrong" if s else "unanswered"))


@_grader("substring")
def grade_substring(text: str, truth: dict) -> dict:
    exp = str(truth.get("expected", ""))
    cs = _cfg(truth).get("case_sensitive", False)
    s = answer_region(text)
    hit = exp in s if cs else exp.lower() in s.lower()
    return _check("substring", exp, exp if hit else None, "correct" if hit else ("wrong" if s else "unanswered"))


@_grader("f1")
def grade_f1(text: str, truth: dict) -> dict:
    exp = str(truth.get("expected", ""))
    thr = float(_cfg(truth).get("threshold", 0.8))
    tag = _ANSWER_TAG.findall(answer_region(text))
    if not tag:
        return _check("f1", exp, None, "unanswered")
    g, x = _norm(tag[-1]).split(), _norm(exp).split()
    common = sum((collections.Counter(g) & collections.Counter(x)).values())
    f1 = 0.0 if not common else 2 * common / (len(g) + len(x))
    return _check("f1", exp, tag[-1].strip(), "correct" if f1 >= thr else "wrong", f1=round(f1, 3))


# ------------------------------------------------------------------------------ selection
def grader_for(truth: dict | None) -> str | None:
    """Grader name for a truth row, or None (not gradable here -> tier 2)."""
    if not truth or truth.get("expected") in (None, "", "__patch__"):
        return None
    if truth.get("grader"):
        return truth["grader"]
    m, pid = truth.get("scoring_method"), str(truth.get("id", ""))
    if m == "multiple_choice":
        return "mmlu_pro" if pid.startswith("mmlu_pro") else "letter"
    if m == "exact_match":
        if pid.startswith("gsm8k") and parse_number(str(truth["expected"])) is not None:
            return "gsm8k"
        return "exact"
    if m == "f1":
        return "f1"
    # question_pool `substring` is NOT auto-mapped: on code suites it is a presence check
    # ("def ", a function name), which proves nothing about the answer. Opt in with grader="substring".
    return None


def grade(text: str, truth: dict | None, graders: dict[str, Grader] | None = None) -> dict | None:
    name = grader_for(truth)
    if name is None:
        return None
    table = {**GRADERS, **(graders or {})}
    if name not in table:
        raise KeyError(f"truth names grader {name!r}, which is not registered")
    return table[name](text, truth)
