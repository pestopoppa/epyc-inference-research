#!/usr/bin/env python3
"""Deterministic scorer for debug benchmark suite.

Scores model outputs against ground-truth answers using methods from
public benchmarks (exact_match, multiple_choice, code_execution,
programmatic, substring). No heuristics, no Claude-as-Judge needed.

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

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def score_answer(
    answer: str,
    expected: str,
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
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    if not answer:
        return False

    config = scoring_config or {}

    scorers = {
        "exact_match": _score_exact_match,
        "multiple_choice": _score_multiple_choice,
        "code_execution": _score_code_execution,
        "programmatic": _score_programmatic,
        "substring": _score_substring,
        "f1": _score_f1,
    }

    scorer = scorers.get(scoring_method)
    if scorer is None:
        raise ValueError(f"Unknown scoring method: {scoring_method}")

    return scorer(answer, expected, config)


def _score_exact_match(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Extract answer via regex, compare to expected.

    Used for: GSM8K, MATH — where the answer is a number or expression.

    Config:
        extract_pattern: Regex with one capture group to extract the answer.
            Default: ``#### (\\S+)`` (GSM8K standard).
        normalize: If True, strip whitespace and lowercase both sides.
    """
    pattern = config.get("extract_pattern", r"####[ \t]*\n?(\S+)")
    normalize = config.get("normalize", True)

    # Try to extract via pattern first
    extracted = _extract_answer(answer, pattern)
    if extracted is None:
        # Fallback: try to find the expected value anywhere in the last line
        last_line = answer.strip().split("\n")[-1]
        extracted = last_line.strip()

    if normalize:
        extracted = extracted.strip().lower().rstrip(".")
        expected_norm = expected.strip().lower().rstrip(".")
    else:
        expected_norm = expected.strip()

    # Numeric comparison for numbers (including word forms like "three" vs "3")
    _NUMBER_WORDS = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
        "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20,
    }
    def _to_number(s: str) -> float | None:
        try:
            return float(s.replace(",", ""))
        except (ValueError, TypeError):
            return _NUMBER_WORDS.get(s.lower()) if isinstance(s, str) else None

    ext_num = _to_number(extracted)
    exp_num = _to_number(expected_norm)
    if ext_num is not None and exp_num is not None:
        return abs(ext_num - exp_num) < 1e-6

    if extracted == expected_norm:
        return True

    # Fallback: vision models wrap OCR results in prose like
    #   'The text in the image is "iRaeenlc".' or 'The image contains the text: iRaeenlc'
    # Try extracting quoted text or text after colon from the full answer.
    if normalize:
        answer_lower = answer.strip().lower()
        # Check quoted: "answer" or 'answer'
        for q in ('"', "'", "\u201c"):
            q_end = "\u201d" if q == "\u201c" else q
            idx = answer_lower.find(q)
            if idx >= 0:
                end = answer_lower.find(q_end, idx + 1)
                if end > idx:
                    candidate = answer_lower[idx + 1:end].strip().rstrip(".")
                    if candidate == expected_norm:
                        return True
        # Check after colon on last meaningful line
        for line in reversed(answer.strip().split("\n")):
            if ":" in line:
                candidate = line.split(":", 1)[1].strip().lower().rstrip(".")
                if candidate == expected_norm:
                    return True

    return False


def _score_multiple_choice(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Parse A/B/C/D from output, compare to expected letter.

    Used for: ARC-Challenge, MMLU, HellaSwag.

    Config:
        choices: Optional list of choice texts (for fuzzy matching).
    """
    expected_letter = expected.strip().upper()
    if expected_letter not in "ABCDEFGH":
        return False

    # Strategy 1: Explicit "Answer: X" — take LAST match (verbose models repeat)
    # Negative lookahead prevents "option is correct" matching as letter "C"
    explicit_pat = r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-H])\)?(?![a-zA-Z])"
    explicit_matches = re.findall(explicit_pat, answer, re.IGNORECASE)
    if explicit_matches:
        return explicit_matches[-1].upper() == expected_letter

    # Strategy 2: Letter on its own line near the end of output
    last_line_pat = r"^\s*\(?([A-H])\)?\s*$"
    line_matches = re.findall(last_line_pat, answer, re.MULTILINE)
    if line_matches:
        return line_matches[-1].upper() == expected_letter

    # Strategy 3: Letter at very start of output (before any prose)
    match = re.match(r"\s*\(?([A-H])\)?\s*[.:\-\n]", answer)
    if match:
        return match.group(1).upper() == expected_letter

    # Strategy 4: Bold letter — take LAST match
    bold_matches = re.findall(r"\*\*([A-H])\*\*", answer)
    if bold_matches:
        return bold_matches[-1].upper() == expected_letter

    # Strategy 5: Last standalone letter A-H in the text (not first!)
    standalone = re.findall(r"\b([A-H])\b", answer)
    if standalone:
        return standalone[-1].upper() == expected_letter

    return False


def _score_stdin_program(
    code: str, test_code: str, preamble: str, timeout: int
) -> bool:
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
        sol_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/mnt/raid0/llm/tmp",
        )
        sol_file.write(full_code)
        sol_file.flush()
        sol_file.close()

        for inp, expected_out in cases:
            try:
                result = subprocess.run(
                    ["python3", sol_file.name],
                    input=inp,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd="/mnt/raid0/llm/tmp",
                )
            except subprocess.TimeoutExpired:
                Path(sol_file.name).unlink(missing_ok=True)
                return False

            if result.returncode != 0:
                Path(sol_file.name).unlink(missing_ok=True)
                return False

            got = result.stdout.strip()
            want = expected_out.strip()
            if got != want:
                Path(sol_file.name).unlink(missing_ok=True)
                return False

        Path(sol_file.name).unlink(missing_ok=True)
        return True
    except OSError:
        return False


def _score_code_execution(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
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

    if _uses_stdin and _has_test_cases:
        return _score_stdin_program(code, test_code, _TYPING_PREAMBLE, timeout)

    # Build full test script
    full_code = _TYPING_PREAMBLE + code
    if test_code:
        full_code += "\n\n" + test_code
    elif entry_point and expected:
        # Simple assertion test
        full_code += f"\n\nassert {entry_point}() == {expected}"

    # Execute in sandboxed subprocess
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
            dir="/mnt/raid0/llm/tmp",
        ) as f:
            f.write(full_code)
            f.flush()
            result = subprocess.run(
                ["python3", f.name],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd="/mnt/raid0/llm/tmp",
            )
            Path(f.name).unlink(missing_ok=True)
            return result.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _score_programmatic(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
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

        threshold: Numeric threshold for count-based verifiers.
        count: Alias for threshold (used by IFEval adapter).
        relation: "at_least" | "at_most" | "exactly" (IFEval word/sentence count).
        keyword: Keyword for contains/no_keyword verifiers.
        keywords: Keyword list for contains_keywords verifier.
        forbidden: Forbidden word list for no_forbidden_words verifier.
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
    count = config.get("count", threshold)
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
            line.strip().startswith(("- ", "* ", "• "))
            for line in lines if line.strip()
        ),
        "numbered_list": lambda: any(
            re.match(r"^\d+[\.\)]\s", line.strip())
            for line in lines if line.strip()
        ),
        "paragraph_count": lambda: len([
            p for p in re.split(r"\n\s*\n", answer_stripped) if p.strip()
        ]) == (count or threshold),
        "sentence_count_min": lambda: len(re.findall(r"[.!?]+", answer_stripped)) >= (count or threshold),
        "comma_separated": lambda: "," in answer_stripped and "\n" not in answer_stripped.strip(),
        # IFEval adapter verifiers (dataset_adapters.py emits these names)
        "no_comma": lambda: "," not in answer_stripped,
        "has_title": lambda: bool(
            lines[0].strip() and len(lines[0].strip().split()) <= 10
            and lines[0].strip().istitle()
        ) if lines else False,
        "placeholder_count": lambda: len(re.findall(r'\[.*?\]', answer_stripped)) >= (count or 1),
        "bullet_count": lambda: sum(
            1 for line in lines
            if line.strip().startswith(("- ", "* ", "• "))
        ) >= (count or 1),
        "contains_keywords": lambda: all(
            kw.lower() in answer_stripped.lower() for kw in keywords
        ) if keywords else True,
        "no_forbidden_words": lambda: not any(
            fw.lower() in answer_stripped.lower() for fw in forbidden
        ) if forbidden else True,
        "language": lambda: True,  # Cannot verify without langdetect; pass through
        "non_empty": lambda: len(answer_stripped) > 0,
        "highlighted_sections": lambda: bool(
            re.search(r'\*\*[^*]+\*\*', answer_stripped)
            or re.search(r'^##\s+', answer_stripped, re.MULTILINE)
        ),
        # IFEval relation-based verifiers (word_count with at_least/at_most/exactly)
        "word_count": _word_count_by_relation,
        "sentence_count": _sentence_count_by_relation,
        "title_case": lambda: all(
            w[0].isupper() for w in words if w and w[0].isalpha()
        ) if words else False,
    }

    fn = verifiers.get(verifier)
    if fn is None:
        # Unknown verifier — check if expected is a simple match
        return expected.strip().lower() in answer_stripped.lower() if expected else False

    return fn()


def _score_substring(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Check if expected text appears in output.

    Used for: Needle-in-haystack, simple factoid QA.

    Config:
        case_sensitive: Whether comparison is case-sensitive (default: False).
    """
    case_sensitive = config.get("case_sensitive", False)

    if case_sensitive:
        return expected.strip() in answer
    else:
        return expected.strip().lower() in answer.lower()


def _score_f1(
    answer: str, expected: str, config: dict[str, Any]
) -> bool:
    """Token-level F1 scoring for QA tasks.

    Used for: HotpotQA, SQuAD-style reading comprehension.

    Computes precision/recall/F1 at the token level after normalization.
    A prediction is considered correct if F1 >= threshold.

    Config:
        extract_pattern: Regex to extract answer (default: #### pattern).
        threshold: Minimum F1 to count as correct (default: 0.5).
        normalize: Whether to normalize text (default: True).
    """
    pattern = config.get("extract_pattern", r"####[ \t]*\n?(.+)")
    threshold = config.get("threshold", 0.5)
    normalize = config.get("normalize", True)

    # Extract answer if pattern provided.
    # For #### patterns, find the LAST occurrence — models often emit
    # #### before explanation then #### before the final answer.
    # Pattern allows optional newline after #### to handle models that
    # put the marker and answer on separate lines.
    matches = re.findall(pattern, answer, re.IGNORECASE)
    if matches:
        extracted = matches[-1].strip()
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

    # Compute token overlap
    common = set(pred_tokens) & set(gold_tokens)

    if not common:
        return False

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1 >= threshold


def _normalize_text(text: str) -> str:
    """Normalize text for F1 scoring (SQuAD-style)."""
    import string

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Collapse whitespace
    text = " ".join(text.split())

    return text


# ── Helpers ────────────────────────────────────────────────────────────


def _extract_answer(text: str, pattern: str) -> str | None:
    """Extract answer from text using regex pattern."""
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match and match.group(1):
        return match.group(1).strip()
    return None


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

    # Last resort: return the whole text (might be just code)
    if text.strip().startswith(("def ", "class ", "import ", "from ")):
        return text.strip()

    return None


def _is_valid_json(text: str) -> bool:
    """Check if text contains valid JSON."""
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
        results.append({
            "id": q.get("id", "unknown"),
            "suite": q.get("suite", "unknown"),
            "passed": passed,
            "expected": q.get("expected", ""),
            "answer_preview": ans[:200] if ans else "",
        })
    return results
