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
DEFAULT_AUDIT_ROW_MAX_CHARS = 20000

TEST_PATH_RE = re.compile(r"(^diff --git a/.*(?:test|tests|testing|spec).*$|^\+\+\+ b/.*(?:test|tests|testing|spec).*$)", re.I | re.M)
ADDED_TEST_EVIDENCE_RE = re.compile(
    r"^\+.*(?:def test_|class Test|assert|self\.assert|pytest\.|with pytest\.raises)",
    re.I | re.M,
)
LONG_DIGIT_RUN_RE = re.compile(r"\d{12,19}")


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
    matches.sort(key=lambda row: (not row.hard_accept_control, row.row_id))
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


def _truncate(value: str, *, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(value) <= max_chars:
        return value, False
    return value[:max_chars], True


def _redact_long_digit_runs(value: str) -> tuple[str, bool]:
    redacted, count = LONG_DIGIT_RUN_RE.subn("[redacted-long-digit-run]", value)
    return redacted, count > 0


def build_audit_packet(
    rows: Iterable[dict[str, Any]],
    selected: list[AcceptControlRow],
    *,
    corpus: Path,
    max_row_chars: int = DEFAULT_AUDIT_ROW_MAX_CHARS,
) -> dict[str, Any]:
    by_id = {row_id(row): row for row in rows if row_id(row)}
    packet_rows: list[dict[str, Any]] = []

    for selected_row in selected:
        row = by_id[selected_row.row_id]
        task, task_truncated = _truncate(str(row.get("task") or ""), max_chars=max_row_chars)
        candidate, candidate_truncated = _truncate(str(row.get("candidate") or ""), max_chars=max_row_chars)
        task, task_redacted = _redact_long_digit_runs(task)
        candidate, candidate_redacted = _redact_long_digit_runs(candidate)
        decontamination = row.get("decontamination") if isinstance(row.get("decontamination"), dict) else {}
        provenance = row.get("provenance") if isinstance(row.get("provenance"), dict) else {}
        packet_rows.append({
            "row_id": selected_row.row_id,
            "instance_id": selected_row.instance_id,
            "source_benchmark": row.get("source_benchmark"),
            "source_suite": row.get("source_suite"),
            "gold_label": row.get("gold_label"),
            "gold_source": row.get("gold_source"),
            "gold_confidence": selected_row.gold_confidence,
            "hard_accept_control": selected_row.hard_accept_control,
            "executable_oracle": selected_row.executable_oracle,
            "repo": decontamination.get("repo"),
            "pull_number": decontamination.get("pull_number"),
            "base_commit": decontamination.get("base_commit"),
            "candidate_is": provenance.get("candidate_is"),
            "candidate_chars": selected_row.candidate_chars,
            "task": task,
            "task_truncated": task_truncated,
            "task_redacted_long_digit_runs": task_redacted,
            "candidate": candidate,
            "candidate_truncated": candidate_truncated,
            "candidate_redacted_long_digit_runs": candidate_redacted,
            "signoff": {
                "status": "unreviewed",
                "reviewer": None,
                "reviewed_at": None,
                "decision": None,
                "notes": None,
            },
        })

    return {
        "schema": "glm52_ccrab_accept_control_audit_packet.v1",
        "corpus": str(corpus),
        "purpose": "Full-candidate label-audit packet for GC-shadow-repair4b.2b hard accept-control signoff.",
        "decision_grade": False,
        "decision_grade_reason": "Rows require executable oracle or explicit signoff before use as hard accept controls.",
        "audit_instructions": [
            "Review the task and full candidate patch, not only metadata.",
            "Mark signoff.decision as hard_accept only if the patch is a complete plausible fix for the reported task.",
            "Mark signoff.decision as reject_or_ambiguous if the patch is incomplete, unrelated, under-tested, or requires execution to know.",
            "Do not convert observation-grade merged PR rows to decision-grade evidence without a reviewer name and rationale.",
        ],
        "selected_n": len(selected),
        "rows": packet_rows,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--max-candidate-chars", type=int, default=15000)
    parser.add_argument("--audit-packet-out", type=Path)
    parser.add_argument("--audit-row-max-chars", type=int, default=DEFAULT_AUDIT_ROW_MAX_CHARS)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--row-ids-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n <= 0:
        print("--n must be positive", file=sys.stderr)
        return 2
    rows = list(read_rows(args.corpus))
    pool = matching_rows(rows, max_chars=args.max_candidate_chars)
    selected = pool[:args.n]
    report = build_report(selected, pool=pool, corpus=args.corpus, n=args.n, max_chars=args.max_candidate_chars)
    if args.json_out:
        write_text(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.row_ids_out:
        write_text(args.row_ids_out, "\n".join(report["selected_row_ids"]) + "\n")
    if args.audit_packet_out:
        audit_packet = build_audit_packet(
            rows,
            selected,
            corpus=args.corpus,
            max_row_chars=args.audit_row_max_chars,
        )
        write_text(args.audit_packet_out, json.dumps(audit_packet, indent=2, sort_keys=True) + "\n")
    if not args.json_out and not args.row_ids_out and not args.audit_packet_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
