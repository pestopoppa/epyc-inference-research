"""Canonical answer-extraction + objective-scoring primitives (single source of truth).

Promoted verbatim from v7_quality_gate_runner on 2026-07-24 so every scorer in the
stack shares ONE validated implementation instead of ~10 independent copies (each a
latent copy of the bare-letter / verbose-penalty bug). See
handoffs/active/scoring-infra-standardization.md. Regression tests: test_answer_scoring.py.

Module-level deps: re only. Fraction and sympy are imported lazily inside functions,
so importing this module is cheap and never requires sympy.
"""
from __future__ import annotations

import re

def extract_letter_answer(response: str) -> str:
    """Extract a single letter (A-J) from the model's response."""
    stripped = response.strip()

    # An explicit final-answer tag wins outright. The delimiter is REQUIRED:
    # without it this pattern happily matches the "i" of "answer is A".
    tagged = re.findall(r'ANSWER\s*[:=]\s*\**\s*\(?([A-Ja-j])\)?\b', stripped,
                        re.IGNORECASE)
    if tagged:
        return tagged[-1].upper()

    boxed = re.findall(r'\\boxed\{\s*\(?([A-Ja-j])\)?\s*\}', stripped)
    if boxed:
        return boxed[-1].upper()

    # Prefer explicit answer markers over arbitrary standalone letters, and
    # take the LAST one: under chain-of-thought the model says "answer" several
    # times while working, and only the final statement is its answer.
    matches = re.findall(
        r'\b(?:answer|option|choice|letter)\s*(?:is|:|=|\.|-)?\s*\(?([A-Ja-j])\)?\b',
        stripped,
        re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()

    # Accept terse responses like "C" or "C.".
    match = re.fullmatch(r'\(?([A-Ja-j])\)?[.)]?', stripped)
    if match:
        return match.group(1).upper()

    # A model that reasons and then puts a bare letter on its own final line
    # HAS answered. Without this, verbose arms fail to parse while terse arms
    # score fine -- a bias against exactly the models that show their work.
    # Requires the whole last line to be the letter, so a reply truncated
    # mid-derivation still (correctly) fails to parse.
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if lines:
        match = re.fullmatch(r'\**\(?([A-Ja-j])\)?[.):]?\**', lines[-1])
        if match:
            return match.group(1).upper()

    # Fall back only when there is exactly one candidate letter in the response.
    matches = re.findall(r'\b([A-Ja-j])\b', stripped)
    if len(matches) == 1:
        return matches[0].upper()
    return ""


def _normalize_numeric(value: str) -> str:
    """Normalize numeric answer strings while preserving non-numeric fallbacks."""
    stripped = value.strip()
    if re.fullmatch(r"\d+", stripped):
        return str(int(stripped))
    return stripped


from fractions import Fraction  # noqa: E402


def parse_math_number(raw: str):
    """Parse a competition-math answer to a float, or None if not a clean number.

    Handles the forms that appear in OlympiadBench 'Numerical' gold answers:
    plain int/decimal, \\frac{a}{b}, \\sqrt{n}, a\\sqrt{b}, \\pi, percentages,
    with $/\\boxed/\\left/\\right/degree/unit wrappers and an optional 'VAR='
    prefix stripped. Returns None on anything it cannot reduce to a number, so
    a suite can be filtered to only cleanly-scorable items and a model answer
    that is not a clean number simply fails to parse (reported, not miscounted).
    """
    if raw is None:
        return None
    s = str(raw).strip()
    # strip common wrappers
    s = s.replace("\\boxed", "").replace("\\left", "").replace("\\right", "")
    s = s.replace("$", "").replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    s = s.replace("{", "(").replace("}", ")").replace(" ", "")
    s = re.sub(r"\\text\(([^)]*)\)", "", s)
    s = re.sub(r"^[A-Za-z]=", "", s)                      # M= prefix
    s = re.sub(r"(\\circ|\^\(\\circ\)|degrees?|°)$", "", s)
    percent = s.endswith("%")
    s = s.rstrip("%")
    if not s:
        return None
    # \frac(a)(b) and \dfrac -> (a)/(b)
    s = re.sub(r"\\d?frac\(([^()]*)\)\(([^()]*)\)", r"((\1)/(\2))", s)
    # \sqrt(n) -> (n)**0.5 ; \pi -> pi
    s = re.sub(r"\\sqrt\(([^()]*)\)", r"((\1)**0.5)", s)
    s = re.sub(r"\\sqrt(\d+)", r"((\1)**0.5)", s)
    s = s.replace("\\pi", "pi").replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\", "")
    # implicit multiplication: 2( -> 2*(, )2 -> )*2, )( -> )*(
    s = re.sub(r"(\d)\(", r"\1*(", s)
    s = re.sub(r"\)(\d)", r")*\1", s)
    s = re.sub(r"\)\(", r")*(", s)
    s = re.sub(r"(\d)pi", r"\1*pi", s)
    if not re.fullmatch(r"[0-9pi.+\-*/()]*", s):
        return None
    try:
        import math
        val = eval(s, {"__builtins__": {}}, {"pi": math.pi})  # restricted, digits/ops only
        val = float(val)
        return val / 100.0 if percent else val
    except Exception:
        try:
            return float(Fraction(s))
        except Exception:
            return None


# ── Symbolic scoring (OlympiadBench hard tier: Expression / Tuple / set answers) ──
# sympy-backed equivalence for answers a numeric compare cannot handle (free
# variables, tuples, sets). Lazily imported + guarded so the runner still works
# without sympy for the numeric/MC suites.

def _latex_to_sympy_str(s: str):
    """Best-effort LaTeX -> sympy-parseable string; None if empty."""
    if s is None:
        return None
    s = str(s).strip()
    for a in ("$", "\\left", "\\right", "\\,", "\\!", "\\displaystyle", "\\boxed", " "):
        s = s.replace(a, "")
    s = s.strip(". ")
    if s.count("=") == 1:  # strip a leading  f(x)= / VAR= / m_{\max}=  -> keep RHS
        m = re.match(r"^[A-Za-z](_\{?\w+\}?)?(\([^)]*\))?=(.+)$", s)
        if m:
            s = m.group(3)
    s = s.replace("{", "(").replace("}", ")")
    s = re.sub(r"\\d?frac\(([^()]*)\)\(([^()]*)\)", r"((\1)/(\2))", s)
    s = re.sub(r"\\lfloor(.*?)\\rfloor", r"floor(\1)", s)
    s = re.sub(r"\\lceil(.*?)\\rceil", r"ceiling(\1)", s)
    s = re.sub(r"\\sqrt\(([^()]*)\)", r"sqrt(\1)", s)
    s = re.sub(r"\\sqrt(\w+)", r"sqrt(\1)", s)
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("\\pi", "pi")
    s = re.sub(r"\^", "**", s)
    s = s.replace("\\", "")
    return s


def _sympy_expr(s: str):
    ss = _latex_to_sympy_str(s)
    if not ss or len(ss) > 400:  # bound: pred is model output
        return None
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr, standard_transformations, implicit_multiplication_application)
        trans = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(ss, transformations=trans)
    except Exception:
        return None


def _split_top(s: str) -> list:
    """Split on top-level commas (not inside brackets)."""
    s = str(s).replace("$", "").strip().strip(".")
    parts, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts if p.strip()]


def _canon_elem(e: str):
    """Canonicalize one answer element (ordered tuple, number, or expr)."""
    inner = e.strip().strip("$").strip()
    if inner.startswith("(") and inner.endswith(")") and "," in inner:
        return ("T",) + tuple(str(_sympy_expr(x)) for x in _split_top(inner[1:-1]))
    v = parse_math_number(inner)
    if v is not None:
        return ("N", round(v, 9))
    ex = _sympy_expr(inner)
    return ("E", str(ex)) if ex is not None else None


def _is_set_answer(gold: str) -> bool:
    g = str(gold).replace("$", "").strip()
    return len(_split_top(g)) > 1 or (g.startswith("(") and "," in g)


def score_math_symbolic(response: str, gold: str) -> bool:
    r"""Compare a model \boxed answer to gold via numeric → set → sympy equivalence."""
    pred = extract_boxed(response)
    if not pred:
        return False
    # 1) numeric-first (robust for numeric-valued answers, incl. \sqrt/\frac)
    pv, gv = parse_math_number(pred), parse_math_number(gold)
    if pv is not None and gv is not None:
        return abs(pv - gv) <= 1e-4 * max(1.0, abs(gv))
    # 2) set / tuple answers (order-independent across elements)
    if _is_set_answer(gold):
        gset = {_canon_elem(x) for x in _split_top(gold)}
        pset = {_canon_elem(x) for x in _split_top(pred)}
        return (None not in gset) and gset == pset
    # 3) single symbolic expression
    ge, pe = _sympy_expr(gold), _sympy_expr(pred)
    if ge is None or pe is None:
        return False
    try:
        from sympy import simplify
        if simplify(ge - pe) == 0:
            return True
    except Exception:
        pass
    try:
        return bool(ge.equals(pe))
    except Exception:
        return False


def gold_symbolically_parseable(gold: str) -> bool:
    """True iff score_math_symbolic can canonicalize this gold (suite filter)."""
    if parse_math_number(gold) is not None:
        return True
    if _is_set_answer(gold):
        return all(_canon_elem(x) is not None for x in _split_top(gold))
    return _sympy_expr(gold) is not None


def extract_boxed(text: str) -> str:
    r"""Return the content of the LAST *complete* \boxed{...}, brace-balanced.

    Iterates \boxed occurrences from last to first and returns the first one that
    brace-closes. This matters when a response is TRUNCATED mid-\boxed (or loops
    on \boxed and gets cut): the final \boxed{... is incomplete, but an earlier
    complete \boxed{answer} is the model's real answer. Taking the last complete
    one recovers it instead of returning the cut-off fragment.

    Falls back to an 'ANSWER:'/'final answer' tail, then the last line.
    """
    starts = [m.start() for m in re.finditer(r"\\boxed", text)]
    for idx in reversed(starts):
        i = text.find("{", idx)
        if i == -1:
            continue
        depth = 0
        for j in range(i, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[i + 1:j].strip()
        # this \boxed never closed (truncated) -> try the previous one
    m = re.findall(r"(?:ANSWER|final answer)\s*[:=]\s*(.+)", text, re.IGNORECASE)
    if m:
        return m[-1].strip().rstrip(".")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def score_math_numeric(response: str, expected: str, rel_tol: float = 1e-4) -> bool:
    """Compare a model response's \\boxed answer to gold numerically."""
    a = parse_math_number(extract_boxed(response))
    b = parse_math_number(expected)
    if a is None or b is None:
        return False
    return abs(a - b) <= rel_tol * max(1.0, abs(b))


def _first_pattern_match(text: str, patterns: list) -> str:
    """Return the last match of the first pattern in `patterns` that hits."""
    for pattern in patterns:
        if not pattern:
            continue
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            match = matches[-1]
            if isinstance(match, tuple):
                match = next((part for part in match if part), "")
            return str(match).strip()
    return ""


def extract_exact_answer(response: str, scoring_config: dict) -> str:
    """Extract an exact-match answer using an adapter-provided config.

    `extract_patterns` (list) is tried in order, most-explicit first, so a
    stated final answer always outranks a stray digit in the working-out.
    `extract_pattern` (single) is the original behaviour, kept as-is.
    """
    stripped = response.strip()
    patterns = scoring_config.get("extract_patterns")
    if patterns:
        got = _first_pattern_match(stripped, list(patterns))
        return got if got else stripped
    pattern = scoring_config.get("extract_pattern")
    if pattern:
        matches = re.findall(pattern, stripped)
        if matches:
            match = matches[-1]
            if isinstance(match, tuple):
                match = next((part for part in match if part), "")
            return str(match).strip()
    return stripped


def score_response(response: str, expected: str, q: dict) -> bool:
    """Score one adapter question response."""
    scoring_method = q.get("scoring_method", "multiple_choice")
    scoring_config = q.get("scoring_config", {}) or {}

    if scoring_method == "multiple_choice":
        return extract_letter_answer(response) == expected.upper().strip()

    if scoring_method == "exact_match":
        got = extract_exact_answer(response, scoring_config)
        want = expected.strip()
        if scoring_config.get("normalize_numeric"):
            got = _normalize_numeric(got)
            want = _normalize_numeric(want)
        return got == want

    if scoring_method == "math_numeric":
        # Extract \boxed{} (brace-balanced) then compare numerically.
        return score_math_numeric(response, expected)

    if scoring_method == "math_symbolic":
        # \boxed{} + numeric → set/tuple → sympy symbolic equivalence.
        return score_math_symbolic(response, expected)

    return response.strip() == expected.strip()
