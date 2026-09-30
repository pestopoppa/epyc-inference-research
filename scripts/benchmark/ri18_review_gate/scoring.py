"""The pre-registered scorers, identical for every arm and policy (design §2.1).

S1: ``answer_scoring.extract_letter_answer`` (A-J; ``Answer: X`` supported) against the expected
letter. S2: ``answer_scoring.score_response`` -> ``math_symbolic``. An empty text is wrong.
``answer_scoring`` is imported from THIS research checkout (its sha goes into score.json).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

_BENCH_DIR = Path(__file__).resolve().parents[1]
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
from answer_scoring import extract_letter_answer, score_response  # noqa: E402

SCORER_PATH = _BENCH_DIR / "answer_scoring.py"


def scorer_sha256() -> str:
    return hashlib.sha256(SCORER_PATH.read_bytes()).hexdigest()


def is_correct(item: dict[str, Any], text: str | None) -> bool:
    if not text:
        return False
    if item["scoring_method"] == "multiple_choice":
        return extract_letter_answer(text).upper() == str(item["expected"]).strip().upper()
    if item["scoring_method"] == "math_symbolic":
        try:
            return bool(score_response(text, item["expected"],
                                       {"scoring_method": "math_symbolic",
                                        "scoring_config": item.get("scoring_config") or {}}))
        except ImportError:
            raise  # a missing sympy must never score every S2 item wrong silently
        except Exception:
            return False
    raise ValueError(f"{item['id']}: unsupported scoring method {item['scoring_method']!r}")
