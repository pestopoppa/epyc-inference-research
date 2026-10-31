"""Passive RRR over an explicit, ordered reflection-memory sequence.

Definition: intake-989#03, arXiv:2605.29463v2 section 3. A post-first
reflection is redundant if SequenceMatcher similarity to any earlier reflection
is >=0.85. RRR is redundant count / (M-1). This diagnoses recorded text only:
it does not remove memories, infer correctness, label frozen memory, or grade.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import platform
from typing import Sequence

SIMILARITY_THRESHOLD = 0.85


def reflection_redundancy(reflections: Sequence[str]) -> dict:
    """Preserve raw text/order; compare current reflection against every predecessor.

    An empty recorded string remains a string; missing/null text refuses rather
    than silently changing M. For M<2 the ratio is unknown, never a zero rate.
    SequenceMatcher uses its standard autojunk=True behavior, stated in output.
    """
    if isinstance(reflections, (str, bytes)) or not isinstance(reflections, (list, tuple)):
        raise ValueError("an ordered list of recorded reflection strings is required")
    if any(not isinstance(text, str) for text in reflections):
        raise ValueError("missing/non-string reflection text; sequence cannot be scored")
    rows = []
    for i, text in enumerate(reflections[1:], 1):
        similarities = [
            SequenceMatcher(None, text, earlier, autojunk=True).ratio()
            for earlier in reflections[:i]
        ]
        matches = [
            j for j, similarity in enumerate(similarities) if similarity >= SIMILARITY_THRESHOLD
        ]
        rows.append(
            {
                "reflection_index": i,
                "redundant": bool(matches),
                "matching_earlier_indices": matches,
                "max_similarity": max(similarities),
            }
        )
    numerator = sum(row["redundant"] for row in rows)
    denominator = max(0, len(reflections) - 1)
    return {
        "instrument": "reflection_redundancy.v1",
        "scope": "passive retrospective text diagnostic; no measurement warrant or adoption grade",
        "reflection_count": len(reflections),
        "redundant_count": numerator,
        "eligible_count": denominator,
        "rrr": numerator / denominator if denominator else None,
        "unknown_reason": None if denominator else "fewer_than_two_recorded_reflections",
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "matcher": "difflib.SequenceMatcher(None, current, earlier, autojunk=True).ratio()",
        "python_version": platform.python_version(),
        "comparisons": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reflections-json",
        type=Path,
        required=True,
        help="JSON array containing one explicit ordered reflection-memory sequence",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        raw = args.reflections_json.read_bytes()
        result = reflection_redundancy(json.loads(raw))
    except (OSError, ValueError, TypeError) as exc:
        parser.error(str(exc))
    result["input"] = {
        "path": str(args.reflections_json.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    producer = Path(__file__).resolve()
    result["producer"] = {
        "path": str(producer),
        "sha256": hashlib.sha256(producer.read_bytes()).hexdigest(),
    }
    with args.output.open("x") as output:
        json.dump(result, output, sort_keys=True, indent=2, allow_nan=False)
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
