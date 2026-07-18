#!/usr/bin/env python3
"""Select deterministic C-CRAB accept controls for GLM patch-review probes."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CORPUS = Path("/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl")
DEFAULT_N = 24

TEST_PATH_RE = re.compile(r"(^diff --git a/.*(?:test|tests|testing|spec).*$|^\+\+\+ b/.*(?:test|tests|testing|spec).*$)", re.I | re.M)
ADDED_TEST_EVIDENCE_RE = re.compile(
    r"^\+.*(?:def test_|class Test|assert|self\.assert|pytest\.|with pytest\.raises)",
    re.I | re.M,
)


@dataclass(frozen=True)
class AcceptControlRow:
    row_id: str
    gold_confidence: str
    executable_oracle: Any
    candidate_chars: int
    instance_id: str | None

    @property
    def hard_accept_control(self) -> bool:
        return self.gold_confidence not in {"", "observation"} or self.executable_oracle is not None


def read_rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("candidate_id") or "")


def clean_control(row: dict[str, Any]) -> bool:
    provenance = row.get("provenance")
    return isinstance(provenance, dict) and provenance.get("clean_control") is True


def instance_id(row: dict[str, Any]) -> str | None:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("instance_id")
    return str(value) if value else None


def candidate_has_test_path(candidate: str) -> bool:
    return TEST_PATH_RE.search(candidate) is not None


def candidate_has_added_test_evidence(candidate: str) -> bool:
    return ADDED_TEST_EVIDENCE_RE.search(candidate) is not None


def is_candidate(row: dict[str, Any], *, max_chars: int) -> bool:
    candidate = str(row.get("candidate") or "")
    if not row_id(row):
        return False
    if str(row.get("source_benchmark") or "").lower() != "c-crab":
        return False
    if str(row.get("source_suite") or "").lower() != "python":
        return False
    if str(row.get("gold_label") or "").lower() != "accept":
        return False
    if str(row.get("gold_source") or "").lower() != "merged_pr_accepted":
        return False
    if not clean_control(row):
        return False
    if str(row.get("defect_origin") or "").lower() != "natural":
        return False
    if row.get("ambiguous_tail") is not False:
        return False
    if len(candidate) >= max_chars:
        return False
    if not candidate_has_test_path(candidate):
        return False
    if not candidate_has_added_test_evidence(candidate):
        return False
    return True


def matching_rows(rows: Iterable[dict[str, Any]], *, max_chars: int) -> list[AcceptControlRow]:
    matches: list[AcceptControlRow] = []
    for row in rows:
        if not is_candidate(row, max_chars=max_chars):
            continue
        matches.append(
            AcceptControlRow(
                row_id=row_id(row),
                gold_confidence=str(row.get("gold_confidence") or "").lower(),
                executable_oracle=row.get("executable_oracle"),
                candidate_chars=len(str(row.get("candidate") or "")),
                instance_id=instance_id(row),
            )
        )
    matches.sort(key=lambda row: row.row_id)
    return matches


def select_rows(rows: Iterable[dict[str, Any]], *, n: int, max_chars: int) -> list[AcceptControlRow]:
    return matching_rows(rows, max_chars=max_chars)[:n]


def build_report(
    selected: list[AcceptControlRow],
    *,
    pool: list[AcceptControlRow],
    corpus: Path,
    n: int,
    max_chars: int,
) -> dict[str, Any]:
    hard = [row for row in selected if row.hard_accept_control]
    hard_pool = [row for row in pool if row.hard_accept_control]
    return {
        "schema": "glm52_ccrab_accept_control_filter.v1",
        "corpus": str(corpus),
        "requested_n": n,
        "matching_pool_n": len(pool),
        "selected_n": len(selected),
        "max_candidate_chars": max_chars,
        "decision_grade": len(hard) == len(selected) and len(selected) == n,
        "hard_accept_control_pool_n": len(hard_pool),
        "observation_only_pool_n": len(pool) - len(hard_pool),
        "hard_accept_control_n": len(hard),
        "observation_only_n": len(selected) - len(hard),
        "filter": {
            "source_benchmark": "c-crab",
            "source_suite": "python",
            "gold_label": "accept",
            "gold_source": "merged_pr_accepted",
            "provenance.clean_control": True,
            "defect_origin": "natural",
            "ambiguous_tail": False,
            "requires_test_like_path": True,
            "requires_added_test_or_assertion_evidence": True,
        },
        "selected_row_ids": [row.row_id for row in selected],
        "selected_rows": [asdict(row) | {"hard_accept_control": row.hard_accept_control} for row in selected],
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--max-candidate-chars", type=int, default=15000)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--row-ids-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n <= 0:
        print("--n must be positive", file=sys.stderr)
        return 2
    pool = matching_rows(read_rows(args.corpus), max_chars=args.max_candidate_chars)
    selected = pool[:args.n]
    report = build_report(selected, pool=pool, corpus=args.corpus, n=args.n, max_chars=args.max_candidate_chars)
    if args.json_out:
        write_text(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.row_ids_out:
        write_text(args.row_ids_out, "\n".join(report["selected_row_ids"]) + "\n")
    if not args.json_out and not args.row_ids_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
