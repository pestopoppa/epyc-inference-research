#!/usr/bin/env python3
"""Build a targeted Tulving follow-up manifest from a scored run.

The first K-MEM Tulving run showed three different failure modes that should
not be mixed into one larger rerun: zero-answer hallucination, event-content
recall, and chronology/order failures. This helper turns the scored JSON into a
small JSONL slice for the next clean window without starting inference.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from score_tulving_run import build_prompt_index
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from score_tulving_run import build_prompt_index


EVENT_RETRIEVAL_TYPES = {"Event contents", "Full event details"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _prompt_index_or_empty() -> dict[str, dict[str, Any]]:
    try:
        return build_prompt_index()
    except Exception:
        return {}


def _record_base(
    row: dict[str, Any],
    *,
    focus: str,
    prompt: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = prompt.get("metadata", {}) if prompt else {}
    record = {
        "question_id": row["question_id"],
        "focus": focus,
        "retrieval_type": row.get("retrieval_type", ""),
        "get_style": row.get("get_style", ""),
        "f1": row.get("f1"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "nb_gt": row.get("nb_gt"),
        "nb_pred": row.get("nb_pred"),
        "ground_truth_items": row.get("ground_truth_items", []),
        "matched_gt_items": row.get("matched_gt_items", []),
    }
    if "kendall_tau" in row:
        record["kendall_tau"] = row["kendall_tau"]
    if prompt:
        record["prompt"] = prompt.get("prompt", "")
        record["metadata"] = metadata
    return record


def build_followup_records(
    scored: dict[str, Any],
    *,
    prompt_index: dict[str, dict[str, Any]] | None = None,
    max_per_focus: int = 40,
    event_f1_ceiling: float = 0.25,
) -> list[dict[str, Any]]:
    prompt_index = prompt_index or {}
    rows = list(scored.get("per_question", []))
    records: list[dict[str, Any]] = []

    zero_answer = [
        row
        for row in rows
        if row.get("nb_gt") == 0 and row.get("nb_pred", 0) > 0
    ]
    zero_answer.sort(key=lambda row: (-int(row.get("nb_pred", 0)), row["question_id"]))
    for row in zero_answer[:max_per_focus]:
        record = _record_base(
            row,
            focus="zero_answer_abstention",
            prompt=prompt_index.get(row["question_id"]),
        )
        record["recommended_contract"] = "Return exactly [] or None when no matching event exists."
        records.append(record)

    event_content = [
        row
        for row in rows
        if row.get("retrieval_type") in EVENT_RETRIEVAL_TYPES
        and float(row.get("f1") or 0.0) <= event_f1_ceiling
        and row.get("nb_gt", 0) > 0
    ]
    event_content.sort(
        key=lambda row: (
            float(row.get("f1") or 0.0),
            -int(row.get("nb_gt", 0)),
            row["question_id"],
        )
    )
    for row in event_content[:max_per_focus]:
        record = _record_base(
            row,
            focus="event_content_recall",
            prompt=prompt_index.get(row["question_id"]),
        )
        record["recommended_contract"] = "Return only event-content answer items; no explanatory prose."
        records.append(record)

    chronology = [
        row
        for row in rows
        if row.get("get_style") == "chronological"
        and float(row.get("kendall_tau", 0.0)) < 0.5
    ]
    chronology.sort(
        key=lambda row: (
            float(row.get("kendall_tau", 0.0)),
            float(row.get("f1") or 0.0),
            row["question_id"],
        )
    )
    for row in chronology[:max_per_focus]:
        record = _record_base(
            row,
            focus="chronology_order",
            prompt=prompt_index.get(row["question_id"]),
        )
        record["recommended_contract"] = "Return answer items in chronological order only."
        records.append(record)

    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_focus = Counter(record["focus"] for record in records)
    return {
        "total_records": len(records),
        "by_focus": dict(sorted(by_focus.items())),
    }


def render_markdown(
    scored: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    score_path: Path,
) -> str:
    summary = scored.get("summary", {})
    focus_counts = Counter(record["focus"] for record in records)
    prompt_text_included = any("prompt" in record for record in records)
    lines = [
        "# Tulving Follow-Up Manifest",
        "",
        f"- Source score: `{score_path}`",
        f"- Run ID: `{summary.get('run_id')}`",
        f"- Model role: `{summary.get('model_role')}`",
        f"- Selected records: {len(records)}",
        f"- Prompt text included: {'yes' if prompt_text_included else 'no'}",
        f"- Source avg F1: {float(summary.get('avg_f1') or 0.0):.4f}",
        f"- Source Simple Recall: {float(summary.get('simple_recall_score') or 0.0):.4f}",
        f"- Source Chronological Awareness: {float(summary.get('chronological_awareness_score') or 0.0):.4f}",
        "",
        "## Selected Focus Areas",
        "",
        "| Focus | Records | Purpose |",
        "|---|---:|---|",
    ]
    purposes = {
        "zero_answer_abstention": "Measure abstention/list-contract repair for empty-answer prompts.",
        "event_content_recall": "Measure whether event/detail prompts improve under a stricter answer contract.",
        "chronology_order": "Measure chronological ordering separately from lexical recall.",
    }
    for focus, count in sorted(focus_counts.items()):
        lines.append(f"| {focus} | {count} | {purposes.get(focus, '')} |")

    lines.extend(
        [
            "",
            "## Acceptance Use",
            "",
            "Use this as a targeted follow-up slice only. It is not a promotion gate by itself; "
            "a passing follow-up should trigger a larger clean-window Tulving rerun before any "
            "memory-routing or retrieval-policy change.",
        ]
    )
    return "\n".join(lines)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("score_json", type=Path)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--max-per-focus", type=int, default=40)
    parser.add_argument("--event-f1-ceiling", type=float, default=0.25)
    parser.add_argument(
        "--no-prompt-enrichment",
        action="store_true",
        help="Skip adapter prompt enrichment; useful when dataset dependencies are unavailable.",
    )
    args = parser.parse_args()

    scored = _load_json(args.score_json)
    prompt_index = {} if args.no_prompt_enrichment else _prompt_index_or_empty()
    records = build_followup_records(
        scored,
        prompt_index=prompt_index,
        max_per_focus=args.max_per_focus,
        event_f1_ceiling=args.event_f1_ceiling,
    )

    write_jsonl(args.out_jsonl, records)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(render_markdown(scored, records, score_path=args.score_json) + "\n")
    print(json.dumps(summarize_records(records), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
