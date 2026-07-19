#!/usr/bin/env python3
"""Select deterministic C-CRAB hard negatives for GLM patch-review probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple


DEFAULT_CORPUS = Path("/mnt/raid0/llm/datasets/nearmiss-corpus-v1/rows.jsonl")
DEFAULT_N = 24
DEFAULT_SEED = 42
DEFAULT_MAX_CANDIDATE_CHARS = 15000


@dataclass(frozen=True)
class HardNegativeRow:
    row_id: str
    gold_source: str
    candidate_chars: int
    instance_id: str | None


class CorpusRead(NamedTuple):
    rows: list[dict[str, Any]]
    invalid_json_lines: int


def read_rows(path: Path) -> CorpusRead:
    rows: list[dict[str, Any]] = []
    invalid_json_lines = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_lines += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return CorpusRead(rows=rows, invalid_json_lines=invalid_json_lines)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("candidate_id") or "")


def instance_id(row: dict[str, Any]) -> str | None:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return None
    value = provenance.get("instance_id")
    return str(value) if value else None


def provenance_scoring_method(row: dict[str, Any]) -> str:
    provenance = row.get("provenance")
    if not isinstance(provenance, dict):
        return ""
    return str(provenance.get("scoring_method") or "").strip().lower()


def candidate_payload_scope(row: dict[str, Any]) -> str:
    if provenance_scoring_method(row) in {"substring", "exact_match"}:
        return "answer_fragment"
    return "full_candidate"


def is_candidate(row: dict[str, Any], *, max_chars: int) -> bool:
    candidate = str(row.get("candidate") or "")
    if not row_id(row):
        return False
    if str(row.get("domain") or "").lower() != "code":
        return False
    if str(row.get("source_benchmark") or "").lower() != "c-crab":
        return False
    if str(row.get("source_suite") or "").lower() != "python":
        return False
    if str(row.get("gold_label") or "").lower() != "reject":
        return False
    if str(row.get("gold_confidence") or "").lower() != "multi_oracle":
        return False
    if candidate_payload_scope(row) != "full_candidate":
        return False
    if not candidate or len(candidate) >= max_chars:
        return False
    return True


def stable_row_hash(seed: int, row_id_value: str) -> str:
    return hashlib.sha1(f"{seed}:glm52-hard-negative\x00{row_id_value}".encode("utf-8")).hexdigest()


def matching_rows(rows: Iterable[dict[str, Any]], *, max_chars: int) -> list[HardNegativeRow]:
    matches: list[HardNegativeRow] = []
    for row in rows:
        if not is_candidate(row, max_chars=max_chars):
            continue
        matches.append(
            HardNegativeRow(
                row_id=row_id(row),
                gold_source=str(row.get("gold_source") or ""),
                candidate_chars=len(str(row.get("candidate") or "")),
                instance_id=instance_id(row),
            )
        )
    matches.sort(key=lambda row: row.row_id)
    return matches


def select_rows(rows: Iterable[dict[str, Any]], *, n: int, max_chars: int, seed: int) -> list[HardNegativeRow]:
    pool = matching_rows(rows, max_chars=max_chars)
    pool.sort(key=lambda row: stable_row_hash(seed, row.row_id))
    return pool[:n]


def read_row_ids_file(path: Path) -> list[str]:
    row_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        row_id_value = stripped.split("#", 1)[0].strip()
        if row_id_value:
            row_ids.append(row_id_value)
    return row_ids


def build_report(
    selected: list[HardNegativeRow],
    *,
    pool: list[HardNegativeRow],
    corpus: Path,
    n: int,
    max_chars: int,
    seed: int,
    invalid_json_lines: int,
) -> dict[str, Any]:
    return {
        "schema": "glm52_ccrab_hard_negative_filter.v1",
        "corpus": str(corpus),
        "purpose": (
            "Deterministic reject-side companion controls for GC-shadow-repair4b.2b. "
            "This is no-inference prep and does not close the hard accept-control signoff gate."
        ),
        "requested_n": n,
        "matching_pool_n": len(pool),
        "selected_n": len(selected),
        "seed": seed,
        "max_candidate_chars": max_chars,
        "invalid_json_lines": invalid_json_lines,
        "decision_grade": False,
        "decision_grade_reason": (
            "This artifact validates only the reject-side companion controls; the combined "
            "GC-shadow-repair4b.2b gate remains blocked until accept controls have signed "
            "hard-accept status."
        ),
        "reject_side_decision_grade": len(selected) == n,
        "filter": {
            "domain": "code",
            "source_benchmark": "c-crab",
            "source_suite": "python",
            "gold_label": "reject",
            "gold_confidence": "multi_oracle",
            "candidate_payload_scope": "full_candidate",
        },
        "selected_row_ids": [row.row_id for row in selected],
        "selected_rows": [asdict(row) for row in selected],
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_markdown(path: Path, report: dict[str, Any], *, combined_row_ids_path: Path | None) -> None:
    lines = [
        "# GLM-5.2 C-CRAB Hard-Negative Companion",
        "",
        f"- Schema: `{report['schema']}`",
        f"- Corpus: `{report['corpus']}`",
        f"- Selected: `{report['selected_n']}` / requested `{report['requested_n']}`",
        f"- Matching pool: `{report['matching_pool_n']}`",
        f"- Seed: `{report['seed']}`",
        f"- Max candidate chars: `{report['max_candidate_chars']}`",
        f"- Invalid JSON lines skipped: `{report['invalid_json_lines']}`",
        f"- Decision grade: `{str(report['decision_grade']).lower()}`",
        f"- Reject-side decision grade: `{str(report['reject_side_decision_grade']).lower()}`",
        "",
        (
            "This is no-inference companion prep for `GC-shadow-repair4b.2b`; it does not "
            "make the observation-only accept controls decision-grade."
        ),
    ]
    if combined_row_ids_path is not None:
        lines.extend(["", f"- Combined accept+reject row ids: `{combined_row_ids_path}`"])
    lines.extend(["", "## Selected Row IDs", ""])
    lines.extend(f"- `{row_id_value}`" for row_id_value in report["selected_row_ids"])
    write_text(path, "\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-candidate-chars", type=int, default=DEFAULT_MAX_CANDIDATE_CHARS)
    parser.add_argument("--accept-row-ids", type=Path)
    parser.add_argument(
        "--accept-row-ids-note",
        default=(
            "accept rows are not proven signed unless this file came from "
            "glm52_ccrab_accept_control_signoff.py with decision_grade=true"
        ),
    )
    parser.add_argument("--combined-row-ids-out", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--row-ids-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n <= 0:
        print("--n must be positive", file=sys.stderr)
        return 2
    if args.max_candidate_chars <= 0:
        print("--max-candidate-chars must be positive", file=sys.stderr)
        return 2
    if bool(args.accept_row_ids) != bool(args.combined_row_ids_out):
        print("--accept-row-ids and --combined-row-ids-out must be provided together", file=sys.stderr)
        return 2

    read = read_rows(args.corpus)
    pool = matching_rows(read.rows, max_chars=args.max_candidate_chars)
    selected = select_rows(read.rows, n=args.n, max_chars=args.max_candidate_chars, seed=args.seed)
    report = build_report(
        selected,
        pool=pool,
        corpus=args.corpus,
        n=args.n,
        max_chars=args.max_candidate_chars,
        seed=args.seed,
        invalid_json_lines=read.invalid_json_lines,
    )

    if args.json_out:
        write_text(args.json_out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.row_ids_out:
        write_text(args.row_ids_out, "\n".join(report["selected_row_ids"]) + "\n")
    if args.combined_row_ids_out and args.accept_row_ids:
        accept_row_ids = read_row_ids_file(args.accept_row_ids)
        combined = [
            "# GC-shadow-repair4b.2b combined accept+reject row ids.",
            f"# accept_row_ids_source: {args.accept_row_ids}",
            f"# accept_row_ids_note: {args.accept_row_ids_note}",
            (
                "# do_not_execute_live_until: "
                "glm52_ccrab_accept_control_signoff_status_*.json has decision_grade=true"
            ),
            "",
            *accept_row_ids,
            *report["selected_row_ids"],
        ]
        write_text(args.combined_row_ids_out, "\n".join(combined) + "\n")
    if args.md_out:
        write_markdown(args.md_out, report, combined_row_ids_path=args.combined_row_ids_out)
    if not args.json_out and not args.row_ids_out and not args.combined_row_ids_out and not args.md_out:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
