#!/usr/bin/env python3
"""No-inference direct-runner scaffold for GLM external pairwise gates.

The adapter materializes external ground-truth rows. This companion prepares the
exact prompts/plan and can score saved response text into reviewer-style
artifacts. It deliberately does not launch a model server yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import glm52_external_ground_truth_adapter as adapter

SCHEMA = "glm52_external_ground_truth_direct_runner.v1"
RUN_MANIFEST_SCHEMA = "glm52_external_ground_truth_direct_run_manifest.v1"
DEFAULT_RUBRIC_VERSION = "glm52_external_pairwise_exact_match_v1"
DEFAULT_ERA = "external_ground_truth_no_inference"


def canonical_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield row


def validate_pairwise_row(row: dict[str, Any], *, source: str) -> None:
    required = ("row_id", "task", "candidate", "candidate_b", "gold_label", "source_benchmark", "source_suite")
    missing = [key for key in required if not row.get(key)]
    if missing:
        raise ValueError(f"{source}: missing required field(s): {', '.join(missing)}")
    if row.get("gold_label") not in adapter.PAIRWISE_DECISIONS:
        raise ValueError(f"{source}: gold_label must be A or B")


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    for row in rows:
        validate_pairwise_row(row, source=str(path))
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "gold_label_counts": dict(Counter(str(row.get("gold_label")) for row in rows)),
        "source_counts": dict(Counter(f"{row.get('source_benchmark')}|{row.get('source_suite')}" for row in rows)),
        "row_ids": [str(row.get("row_id")) for row in rows],
    }


def build_plan(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    prompt_refusals: list[str] = []
    prompt_rows: list[dict[str, Any]] = []
    for row in rows:
        try:
            prompt_info = adapter.fit_pairwise_prompt_to_budget(
                row,
                context_length=args.context_length,
                max_completion_tokens=args.max_tokens,
                prompt_context_guard_tokens=args.prompt_guard_tokens,
                max_field_chars=args.max_field_chars,
            )
        except ValueError as exc:
            prompt_refusals.append(f"{row['row_id']}: {exc}")
            continue
        prompt_rows.append(
            {
                "row_id": row["row_id"],
                "prompt_token_count": prompt_info["prompt_token_count"],
                "prompt_token_max": prompt_info["prompt_token_max"],
                "truncation": prompt_info["truncation"],
            }
        )
    return {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "score-responses" if args.score_responses_jsonl else "dry-run",
        "observation_only": True,
        "measurement_protocol": "external_ground_truth_no_inference",
        "rows": summarize_rows(rows),
        "request": {
            "endpoint": "chat",
            "context_length": args.context_length,
            "max_tokens": args.max_tokens,
            "prompt_guard_tokens": args.prompt_guard_tokens,
            "max_field_chars": args.max_field_chars,
            "temperature": args.temperature,
            "seed": args.seed,
            "rubric_version": args.rubric_version,
            "era": args.era,
            "response_schema": {"decision": list(adapter.PAIRWISE_DECISIONS), "confidence": "number|null"},
        },
        "prompt_rows": prompt_rows,
        "output_dir": str(args.output_dir),
        "decisions_path": str(args.output_dir / "decisions.jsonl"),
        "execution_allowed": bool(rows) and not prompt_refusals,
        "refusal_reasons": prompt_refusals + ([] if rows else ["no rows"]),
    }


def response_text_from_row(row: dict[str, Any]) -> str:
    for key in ("response_text", "text", "content"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    response = row.get("response")
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            if isinstance(choices[0].get("text"), str):
                return choices[0]["text"]
    return ""


def score_saved_responses(rows: list[dict[str, Any]], response_rows: Iterable[dict[str, Any]], *, plan: dict[str, Any]) -> dict[str, Any]:
    rows_by_id = {row["row_id"]: row for row in rows}
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for response_row in response_rows:
        row_id = str(response_row.get("row_id") or "")
        if row_id not in rows_by_id:
            continue
        seen.add(row_id)
        gold_row = rows_by_id[row_id]
        scored = adapter.score_pairwise_text(response_text_from_row(response_row), str(gold_row["gold_label"]))
        decisions.append(
            {
                "decision_id": f"glm52-ext-{row_id}",
                "reviewer_model_quant": response_row.get("reviewer_model_quant", "glm_52_ud_iq2m"),
                "rubric_version": plan["request"]["rubric_version"],
                "corpus_id": str(gold_row.get("gold_source") or gold_row.get("source_benchmark")),
                "candidate_id": row_id,
                "domain": "judge_quality",
                "decision": scored["decision"],
                "confidence": scored.get("confidence"),
                "gold_label": gold_row["gold_label"],
                "gold_source": gold_row.get("gold_source"),
                "gold_instrument_version": gold_row.get("gold_instrument_version"),
                "source_benchmark": gold_row.get("source_benchmark"),
                "source_suite": gold_row.get("source_suite"),
                "correct": scored["correct"],
                "parse_failure": scored["parse_failure"],
                "era": plan["request"]["era"],
            }
        )
    missing = [row_id for row_id in rows_by_id if row_id not in seen]
    correct = sum(1 for row in decisions if row["correct"])
    parse_failures = sum(1 for row in decisions if row["parse_failure"] is not None)
    return {
        "decisions": decisions,
        "summary": {
            "n": len(decisions),
            "n_expected": len(rows),
            "missing_response_row_ids": missing,
            "accuracy": (correct / len(decisions)) if decisions else None,
            "correct": correct,
            "parse_failures": parse_failures,
            "parse_failure_rate": (parse_failures / len(decisions)) if decisions else None,
            "decision_counts": dict(Counter(row["decision"] for row in decisions)),
        },
    }


def write_score_outputs(args: argparse.Namespace, plan: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    assert args.score_responses_jsonl is not None
    scored = score_saved_responses(rows, read_jsonl(args.score_responses_jsonl), plan=plan)
    decisions_path = args.output_dir / "decisions.jsonl"
    with decisions_path.open("w", encoding="utf-8") as fh:
        for row in scored["decisions"]:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    run_manifest = {
        "schema": RUN_MANIFEST_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "observation_only": True,
        "measurement_protocol": plan["measurement_protocol"],
        "rows_jsonl": str(args.rows_jsonl),
        "responses_jsonl": str(args.score_responses_jsonl),
        "decisions_path": str(decisions_path),
        "n_scored": scored["summary"]["n"],
    }
    write_json(args.output_dir / "run_manifest.json", run_manifest)
    plan["score"] = scored["summary"]
    plan["run_manifest"] = run_manifest
    write_json(args.output_dir / "summary.json", plan)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-responses-jsonl", type=Path)
    parser.add_argument("--context-length", type=int, default=adapter.DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--max-tokens", type=int, default=adapter.DEFAULT_COMPLETION_TOKENS)
    parser.add_argument("--prompt-guard-tokens", type=int, default=adapter.DEFAULT_PROMPT_GUARD_TOKENS)
    parser.add_argument("--max-field-chars", type=int, default=adapter.DEFAULT_MAX_FIELD_CHARS)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--rubric-version", default=DEFAULT_RUBRIC_VERSION)
    parser.add_argument("--era", default=DEFAULT_ERA)
    args = parser.parse_args(argv)
    args.rows_jsonl = args.rows_jsonl.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.score_responses_jsonl is not None:
        args.score_responses_jsonl = args.score_responses_jsonl.expanduser().resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_rows(args.rows_jsonl)
        plan = build_plan(args, rows)
    except (ValueError, FileNotFoundError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "plan.json", plan)
    if not plan["execution_allowed"]:
        print("execution refused: " + "; ".join(plan["refusal_reasons"]), file=sys.stderr)
        return 3
    if args.score_responses_jsonl is not None:
        write_score_outputs(args, plan, rows)
        print(f"scored responses; wrote {args.output_dir / 'summary.json'}")
    else:
        print(f"dry-run wrote {args.output_dir / 'plan.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
