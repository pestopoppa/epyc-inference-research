#!/usr/bin/env python3
from __future__ import annotations

"""
Algorithmic Quality Scorer for Benchmark Responses

Provides pattern-based scoring for benchmark responses:
- Score 0: Wrong answer or no answer
- Score 1: Partial answer (some correct elements)
- Score 2: Mostly correct answer
- Score 3: Fully correct answer

This module is shared with the orchestrator project.
"""

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ScoreResult:
    """Result of scoring a response."""

    score: int  # 0-3
    reason: str
    matched_patterns: list[str]
    negative_patterns: list[str]


# Scoring patterns by suite and question
# Format: {question_id: {"positive": [...], "negative": [...], "required": [...]}}
SCORING_PATTERNS: dict[str, dict[str, dict[str, list[str]]]] = {
    "thinking": {
        "t1_q1_logic": {
            "positive": [
                r"\bno\b",
                r"cannot\s+conclude",
                r"fallacy",
                r"undistributed\s+middle",
                r"invalid",
            ],
            "negative": [
                r"\byes\b.*conclude",
                r"we\s+can\s+conclude",
            ],
            "required": [r"\bno\b|cannot"],  # Must have "no" or "cannot"
        },
        "t1_q2_sequence": {
            "positive": [
                r"\b42\b",
                r"n\s*\(\s*n\s*\+\s*1\s*\)",
                r"difference.*increase",
                r"pattern",
            ],
            "negative": [
                r"\b40\b",
                r"\b44\b",
                r"\b38\b",
            ],
            "required": [r"\b42\b"],
        },
        "t1_q3_deduction": {
            "positive": [
                r"alice.*bird",
                r"bird.*alice",
                r"allergic.*fur",
                r"bob.*bird",
                r"carol.*cat|carol.*dog",
            ],
            "negative": [
                r"alice.*cat",
                r"alice.*dog",
            ],
            "required": [],
        },
        "t2_q1_multistep": {
            "positive": [
                r"11:30",
                r"11:\s*30",
                r"2\.5\s*hours?",
                r"150\s*minutes?",
            ],
            "negative": [
                r"11:00",
                r"12:00",
                r"10:30",
            ],
            "required": [r"11:30|11:\s*30"],
        },
        "t2_q2_hypothesis": {
            "positive": [
                r"confound",
                r"socioeconomic|SES|income|wealth",
                r"sleep",
                r"health",
                r"correlation.*causation",
                r"alternative",
                r"third\s+variable",
            ],
            "negative": [],
            "required": [],  # Need at least 3 alternatives, checked differently
        },
        "t2_q3_planning": {
            "positive": [
                r"A.*B.*D.*C",  # A before B, B before D, D immediately before C
                r"constraint",
                r"valid",
                r"ordering",
            ],
            "negative": [
                r"impossible",
                r"no\s+valid",
            ],
            "required": [],
        },
        "t3_q1_paradox": {
            "positive": [
                r"identity",
                r"continuity",
                r"material",
                r"spatio-?temporal",
                r"thesis",
                r"philosophy",
            ],
            "negative": [],
            "required": [],
        },
        "t3_q2_counterfactual": {
            "positive": [
                r"dissemination",
                r"collaboration",
                r"peer\s+review",
                r"reproducib",
                r"journal",
                r"conference",
                r"slower",
            ],
            "negative": [],
            "required": [],
        },
        "t3_q3_formal": {
            "positive": [
                r"hypothetical\s+syllogism",
                r"transitive|transitivity",
                r"chain\s+rule",
                r"modus\s+ponens",
                r"if.*then",
                r"therefore",
                r"proof|prove",
            ],
            "negative": [
                r"disprove",
                r"false|invalid",
            ],
            "required": [],
        },
        "t3_q4_metacognition": {
            "positive": [
                r"fermi",
                r"estimate|estimation",
                r"assumption",
                r"population",
                r"piano",
                r"bound|range",
                r"uncertain",
            ],
            "negative": [],
            "required": [],
        },
    },
    "coder": {
        "t1_q1_fizzbuzz": {
            "positive": [
                r"def\s+fizzbuzz|function\s+fizzbuzz",
                r"%\s*3|mod\s*3",
                r"%\s*15|%\s*3.*%\s*5",
                r"fizzbuzz",
            ],
            "negative": [],
            "required": [r"fizz|buzz"],
        },
        "t2_q1_cache": {
            "positive": [
                r"race\s+condition",
                r"thread.?safe",
                r"double.?check",
                r"lock.*before.*check|check.*lock",
            ],
            "negative": [],
            "required": [r"race|thread|lock|synchron"],
        },
    },
    "long_context": {
        "t1_q1_retrieval": {
            "positive": [
                r"\b2847\b",
                r"2,?847",
            ],
            "negative": [
                r"\b2000\b",
                r"\b3000\b",
            ],
            "required": [r"2847"],  # Must find the exact number
        },
        "t1_q2_summary": {
            "positive": [
                r"decision",
                r"action\s+item",
                r"assigned\s+to|assignee",
                r"agreed|approved|proceed",
            ],
            "negative": [],
            "required": [],  # Qualitative - need human review
        },
        "t1_q3_cross_reference": {
            "positive": [
                r"section\s+3",
                r"section\s+5",
                r"error\s+handling",
                r"relat|connect|refer",
            ],
            "negative": [],
            "required": [r"section"],
        },
        "t2_q1_needle": {
            "positive": [
                r"sk-proj-7x9mK2nP4qR8sT1uV3wY5zA",
                r"api.?key",
                r"secret|credential",
            ],
            "negative": [],
            "required": [r"sk-proj|7x9mK2n"],  # Must find the API key
        },
        "t2_q2_multi_file": {
            "positive": [
                r"config\.py",
                r"main\.py",
                r"data\s+flow|flow.*data",
                r"bug|issue|error",
            ],
            "negative": [],
            "required": [],
        },
        "t2_q3_extraction": {
            "positive": [
                r"\$\d",  # Dollar amounts
                r"\d{4}",  # Dates with years
                r"january|february|march|april|may|june|july|august|september|october|november|december",
                r"inc\.|corp|llc|company",
            ],
            "negative": [],
            "required": [],
        },
        "t3_q1_deep_needle": {
            "positive": [
                r"node-7",
                r"03:47:22",
                r"memory\s+corrupt",
                r"critical",
            ],
            "negative": [
                r"node-1\b",
                r"node-2\b",
            ],
            "required": [r"node-7"],  # Must identify correct server
        },
        "t3_q2_synthesis": {
            "positive": [
                r"q1|q2|quarter",
                r"market",
                r"competitor",
                r"performance|revenue|growth",
                r"increase|decrease|change",
            ],
            "negative": [],
            "required": [],
        },
        "t3_q3_architecture": {
            "positive": [
                r"architecture",
                r"pattern|design",
                r"class|module|component",
                r"hierarchy|structure",
            ],
            "negative": [],
            "required": [],
        },
    },
}


def score_response(suite: str, question_id: str, response: str) -> ScoreResult:
    """Score a response based on pattern matching.

    Args:
        suite: The benchmark suite name (e.g., 'thinking').
        question_id: The question identifier (e.g., 't1_q1_logic').
        response: The model's response text.

    Returns:
        ScoreResult with score (0-3) and explanation.
    """
    # Normalize response
    response_lower = response.lower()

    # Get patterns for this question
    patterns = SCORING_PATTERNS.get(suite, {}).get(question_id, {})
    if not patterns:
        # No patterns defined - return neutral score
        return ScoreResult(
            score=1,
            reason="No scoring patterns defined for this question",
            matched_patterns=[],
            negative_patterns=[],
        )

    positive = patterns.get("positive", [])
    negative = patterns.get("negative", [])
    required = patterns.get("required", [])

    # Count matches
    matched_positive = []
    matched_negative = []

    for pattern in positive:
        if re.search(pattern, response_lower, re.IGNORECASE):
            matched_positive.append(pattern)

    for pattern in negative:
        if re.search(pattern, response_lower, re.IGNORECASE):
            matched_negative.append(pattern)

    # Check required patterns
    required_met = True
    if required:
        for pattern in required:
            if not re.search(pattern, response_lower, re.IGNORECASE):
                required_met = False
                break

    # Calculate score
    if matched_negative:
        # Wrong answer detected
        score = 0
        reason = f"Wrong answer pattern detected: {matched_negative[0]}"
    elif not required_met:
        # Missing required answer
        score = 0
        reason = "Required answer pattern not found"
    elif len(matched_positive) == 0:
        # No positive matches
        score = 1
        reason = "No correct answer patterns found"
    elif len(matched_positive) == 1:
        # One positive match
        score = 2
        reason = f"Partial answer: {matched_positive[0]}"
    elif len(matched_positive) >= 2:
        # Multiple positive matches
        score = 3
        reason = f"Full answer with {len(matched_positive)} matching patterns"
    else:
        score = 1
        reason = "Unclear response"

    return ScoreResult(
        score=score,
        reason=reason,
        matched_patterns=matched_positive,
        negative_patterns=matched_negative,
    )


def get_quality_score(suite: str, question_id: str, response: str) -> int:
    """Convenience function to get just the score."""
    return score_response(suite, question_id, response).score


def add_scoring_patterns(
    suite: str, question_id: str, patterns: dict[str, list[str]]
) -> None:
    """Add or update scoring patterns for a question.

    Args:
        suite: Suite name.
        question_id: Question identifier.
        patterns: Dict with 'positive', 'negative', 'required' lists.
    """
    if suite not in SCORING_PATTERNS:
        SCORING_PATTERNS[suite] = {}
    SCORING_PATTERNS[suite][question_id] = patterns


if __name__ == "__main__":
    # Test the scorer
    print("=== Scorer Test ===\n")

    # Test t1_q1_logic (should say "no")
    test_responses = [
        ("thinking", "t1_q1_logic", "No, we cannot conclude that. This is a logical fallacy."),
        ("thinking", "t1_q1_logic", "Yes, we can conclude that some roses fade quickly."),
        ("thinking", "t1_q2_sequence", "The answer is 42. The pattern is n(n+1)."),
        ("thinking", "t1_q2_sequence", "The answer is 40."),
        ("thinking", "t2_q1_multistep", "The trains meet at 11:30 AM."),
    ]

    for suite, qid, response in test_responses:
        result = score_response(suite, qid, response)
        print(f"Suite: {suite}, Question: {qid}")
        print(f"  Response: {response[:50]}...")
        print(f"  Score: {result.score}/3")
        print(f"  Reason: {result.reason}")
        print()
