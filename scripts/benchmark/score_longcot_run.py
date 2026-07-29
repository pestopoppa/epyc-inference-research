#!/usr/bin/env python3
"""Deterministic post-run aggregator for the LongCoT-Mini suite (K-LCM-1 / RE-4).

``run_benchmark.py --suite longcot_mini`` stores raw model responses per
question but no benchmark-specific accuracy.  This helper is the thin,
DETERMINISTIC companion that rehydrates the LongCoT-Mini ground truth from the
landed adapter and aggregates a run-output file into:

  - overall accuracy over the scorable rows (default suite: 402 rows —
    chemistry 100 + chess 100 + cs 100 + math 102),
  - per-domain accuracy (chemistry / chess / cs / math),
  - canary-leak count — reported SEPARATELY; a leak invalidates that row's
    accuracy reading, so a clean (leak-excluded) accuracy is also emitted,
  - unscorable "logic" row count — the 105 null-gold rows, EXCLUDED from the
    accuracy denominator (never counted as correct).

It calls ``LongCoTMiniAdapter.compute_score_for_result`` (a sibling-landed,
fully deterministic structural scorer — NO LLM-judge, no network, no
model-in-the-loop) exactly as the adapter tests do, mirroring the
``score_tulving_run`` precedent (rehydrate gold from the adapter, iterate the
run rows, aggregate).  Identical inputs always yield identical output.

MEASUREMENT NOTE
----------------
The numbers this emits are OBSERVATION-grade (a non-saturated research
benchmark), NOT a decision-gating measurement.  They are for hypothesis
formation only; do not use them to gate a keep/revert/deploy/promote decision.

Safe to run on a partial result file while a benchmark is still active.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

# Allow standalone import outside the benchmarks package (mirrors siblings).
_BENCHMARK_DIR = Path(__file__).resolve().parent
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

from longcot_mini_adapter import LongCoTMiniAdapter  # noqa: E402

SUITE_KEY = "longcot_mini"
SCORABLE_DOMAINS = ("chemistry", "chess", "cs", "math")

# The measurement grade of every number this tool emits.
GRADE = "observation"
GRADE_NOTE = (
    "OBSERVATION-grade: non-saturated research benchmark; for hypothesis "
    "formation only, not a keep/revert/deploy/promote decision gate."
)


# ── prompt-index rehydration (mirrors score_tulving_run.build_prompt_index) ────


def build_prompt_index(include_unscorable: bool = True) -> dict[str, dict[str, Any]]:
    """Rehydrate the LongCoT-Mini gold from the adapter, keyed by prompt ``id``.

    ``include_unscorable=True`` (default) also loads the 105 null-gold ``logic``
    rows so that, if a run executed them, they can be resolved and reported as
    unscorable rather than dropped as ``missing``.
    """
    adapter = LongCoTMiniAdapter(include_unscorable=include_unscorable)
    return {item["id"]: item for item in adapter.extract_all()}


# ── run-output loading ────────────────────────────────────────────────────────


def load_run_rows(path: Path, suite_key: str = SUITE_KEY) -> list[tuple[str, dict]]:
    """Load a run-output file into a list of ``(question_id, row_dict)`` pairs.

    Accepts either:
      * a ``run_benchmark.py`` payload JSON: ``{"results": {suite: {qid: row}}}``
        (row carries at least ``response``), or
      * a bare ``{qid: row}`` mapping, or a JSON array of row dicts, or
      * JSONL — one row dict per line (id from ``question_id`` / ``id``).

    Never invokes a model; pure file parse.
    """
    text = path.read_text()
    stripped = text.strip()

    payload: Any = None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None

    rows: list[tuple[str, dict]] = []

    # run_benchmark.py payload shape
    if isinstance(payload, dict) and isinstance(payload.get("results"), dict):
        suite_map = payload["results"]
        if suite_key in suite_map and isinstance(suite_map[suite_key], dict):
            suite_rows = suite_map[suite_key]
        elif len(suite_map) == 1:
            (only,) = suite_map.values()
            suite_rows = only if isinstance(only, dict) else {}
        else:  # merge every suite dict (best effort)
            suite_rows = {}
            for sub in suite_map.values():
                if isinstance(sub, dict):
                    suite_rows.update(sub)
        for qid, row in suite_rows.items():
            if isinstance(row, dict):
                rows.append((str(qid), row))
        return rows

    # bare {qid: row} mapping (no 'results' wrapper)
    if isinstance(payload, dict):
        for qid, row in payload.items():
            if isinstance(row, dict):
                rows.append((str(qid), row))
        return rows

    # JSON array of row dicts
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                qid = row.get("question_id") or row.get("id") or ""
                rows.append((str(qid), row))
        return rows

    # JSONL fallback (one row dict per line)
    for line in stripped.splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            qid = row.get("question_id") or row.get("id") or ""
            rows.append((str(qid), row))
    return rows


def _resolve_prompt(qid: str, row: dict, prompt_index: dict[str, dict]) -> Optional[dict]:
    """Find the prompt_dict (carrying gold/canary/domain) for a run row.

    Primary: the rehydrated prompt index (keyed by prompt id). Fallback: an
    inline ``metadata`` block on the run row itself, if the run stored one.
    """
    pd = prompt_index.get(qid)
    if pd is not None:
        return pd
    if isinstance(row.get("metadata"), dict):
        return row
    return None


# ── aggregation (mirrors score_tulving_run.score_result_payload) ───────────────


def score_run_payload(
    rows: list[tuple[str, dict]],
    prompt_index: dict[str, dict[str, Any]],
    *,
    run_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Aggregate scored run rows into a deterministic LongCoT-Mini report.

    Args:
        rows: ``(question_id, run_row)`` pairs; each ``run_row`` supplies the
            model text under ``response``.
        prompt_index: id -> prompt_dict (with gold/canary/domain in metadata).
        run_meta: optional run identifiers (run_id, model_role, config_name).

    Returns a dict with ``summary``, ``per_domain``, ``per_question`` and the
    list of ids missing from the prompt index.
    """
    run_meta = run_meta or {}
    per_question: list[dict[str, Any]] = []
    missing: list[str] = []

    scorable_total = 0
    scorable_correct = 0
    canary_leaks = 0            # across ALL rows (scorable + unscorable)
    canary_leaks_scorable = 0   # leaks on scorable rows (invalidate readings)
    unscorable = 0
    infra_error_rows = 0
    # leak-excluded ("clean") accuracy over scorable rows
    clean_total = 0
    clean_correct = 0

    # domain -> counters
    per_domain: dict[str, dict[str, int]] = {}

    for qid, row in rows:
        # The runner persists infra failures incrementally for resume/audit
        # purposes.  They are not model answers and therefore never belong in
        # the quality denominator (REL-1).
        if row.get("excluded_from_scoring") is True:
            infra_error_rows += 1
            per_question.append(
                {
                    "question_id": qid,
                    "excluded": True,
                    "exclusion_reason": row.get("exclusion_reason") or "infra_error",
                    "error": row.get("error"),
                }
            )
            continue
        prompt_dict = _resolve_prompt(qid, row, prompt_index)
        if prompt_dict is None:
            missing.append(qid)
            continue

        response = row.get("response") or ""
        score = LongCoTMiniAdapter.compute_score_for_result(response, prompt_dict)

        leak = bool(score.get("canary_leak"))
        is_scorable = bool(score.get("is_scorable"))
        correct = score.get("correct")  # bool | None
        domain = score.get("domain") or "unknown"

        if leak:
            canary_leaks += 1

        per_question.append(
            {
                "question_id": qid,
                "domain": domain,
                "is_scorable": is_scorable,
                "correct": correct,
                "canary_leak": leak,
                "reason": score.get("reason"),
            }
        )

        if not is_scorable:
            unscorable += 1
            continue

        # scorable row
        scorable_total += 1
        dc = per_domain.setdefault(
            domain, {"correct": 0, "total": 0, "canary_leaks": 0}
        )
        dc["total"] += 1
        if correct:
            scorable_correct += 1
            dc["correct"] += 1
        if leak:
            canary_leaks_scorable += 1
            dc["canary_leaks"] += 1
        else:
            clean_total += 1
            if correct:
                clean_correct += 1

    # finalize per-domain accuracy
    per_domain_out: dict[str, dict[str, Any]] = {}
    for domain, dc in sorted(per_domain.items()):
        per_domain_out[domain] = {
            "correct": dc["correct"],
            "total": dc["total"],
            "accuracy": (dc["correct"] / dc["total"]) if dc["total"] else 0.0,
            "canary_leaks": dc["canary_leaks"],
        }

    overall_accuracy = (scorable_correct / scorable_total) if scorable_total else 0.0
    clean_accuracy = (clean_correct / clean_total) if clean_total else 0.0

    summary = {
        "grade": GRADE,
        "grade_note": GRADE_NOTE,
        "run_id": run_meta.get("run_id"),
        "model_role": run_meta.get("model_role"),
        "config_name": run_meta.get("config_name"),
        "suite": SUITE_KEY,
        "rows_in_run": len(rows),
        "scorable_rows": scorable_total,
        "scorable_correct": scorable_correct,
        "overall_accuracy": overall_accuracy,
        # a canary leak invalidates that row's reading, so also report accuracy
        # over the leak-free scorable rows:
        "overall_accuracy_excluding_canary_leaks": clean_accuracy,
        "clean_scorable_rows": clean_total,
        "canary_leak_count": canary_leaks,
        "canary_leaks_on_scorable_rows": canary_leaks_scorable,
        "unscorable_logic_rows": unscorable,
        "infra_error_rows": infra_error_rows,
        "missing_from_prompt_index": len(missing),
    }

    return {
        "summary": summary,
        "per_domain": per_domain_out,
        "missing_prompt_ids": missing,
        "per_question": per_question,
    }


# ── markdown rendering ─────────────────────────────────────────────────────────


def render_markdown(scored: dict[str, Any], run_path: Path) -> str:
    s = scored["summary"]
    lines = [
        "# LongCoT-Mini Run Score",
        "",
        f"> {s['grade_note']}",
        "",
        f"- Result file: `{run_path}`",
        f"- Run ID: `{s.get('run_id')}`",
        f"- Model role: `{s.get('model_role')}`",
        f"- Config: `{s.get('config_name')}`",
        f"- Scorable rows: {s['scorable_rows']}",
        f"- Infra-error rows excluded from scoring: {s['infra_error_rows']}",
        f"- Overall accuracy: {s['overall_accuracy']:.4f} "
        f"({s['scorable_correct']}/{s['scorable_rows']})",
        f"- Overall accuracy (excluding canary-leaked rows): "
        f"{s['overall_accuracy_excluding_canary_leaks']:.4f} "
        f"({s['clean_scorable_rows']} clean rows)",
        f"- Canary leaks: {s['canary_leak_count']} "
        f"({s['canary_leaks_on_scorable_rows']} on scorable rows — readings invalidated)",
        f"- Unscorable logic rows (excluded from accuracy): {s['unscorable_logic_rows']}",
        f"- Missing from prompt index: {s['missing_from_prompt_index']}",
        "",
        "## Per-domain accuracy",
        "",
        "| Domain | Correct | Total | Accuracy | Canary leaks |",
        "|---|---:|---:|---:|---:|",
    ]
    for domain, d in sorted(scored["per_domain"].items()):
        lines.append(
            f"| {domain} | {d['correct']} | {d['total']} | "
            f"{d['accuracy']:.4f} | {d['canary_leaks']} |"
        )
    lines.append("")
    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────


def _extract_run_meta(path: Path) -> dict[str, Any]:
    """Best-effort pull of run identifiers from a payload-shaped run file."""
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "run_id": payload.get("run_id"),
        "model_role": payload.get("model_role"),
        "config_name": payload.get("config_name"),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run", type=Path, help="Path to a run_benchmark.py run-output JSON or JSONL"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Write the JSON report here (a sibling .md summary is written too)",
    )
    parser.add_argument(
        "--out-md", type=Path, default=None,
        help="Explicit path for the markdown summary (overrides the sibling default)",
    )
    parser.add_argument(
        "--suite", default=SUITE_KEY,
        help=f"Suite key inside results[...] (default: {SUITE_KEY})",
    )
    parser.add_argument(
        "--include-unscorable", action="store_true", default=True,
        help="Load logic rows into the prompt index so they resolve (default on)",
    )
    args = parser.parse_args(argv)

    rows = load_run_rows(args.run, suite_key=args.suite)
    prompt_index = build_prompt_index(include_unscorable=args.include_unscorable)
    scored = score_run_payload(
        rows, prompt_index, run_meta=_extract_run_meta(args.run)
    )
    md = render_markdown(scored, args.run)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(scored, indent=2) + "\n")
        md_path = args.out_md or args.output.with_suffix(".md")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md + "\n")
    elif args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md + "\n")

    # Always print the short markdown summary to stdout.
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
