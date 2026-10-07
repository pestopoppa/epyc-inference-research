#!/usr/bin/env python3
"""Replay the autopilot journal under legacy and task-rate objective policies."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.autopilot_core.journal_reconstruction import (  # noqa: E402
    fold_supersession_events,
    reconstruct_archive_from_journal_rows,
)
from src.autopilot_core.pareto_math import dominates  # noqa: E402
from src.autopilot_core.tier_specs import (  # noqa: E402
    DEFAULT_FRONTIER_TIER,
    LEGACY_OBJECTIVE_POLICY,
    TASK_RATE_OBJECTIVE_POLICY,
    goodput_qph_from_row,
    task_rate_objectives_from_row,
    task_rate_qph_from_row_if_available,
)
from scripts.autopilot.journal_shards import resolve_journal_paths  # noqa: E402

DEFAULT_JOURNAL = REPO / "orchestration" / "autopilot_journal.jsonl"
DEFAULT_STATE = REPO / "orchestration" / "autopilot_state.json"
DEFAULT_REPORT_DIR = REPO / "orchestration" / "reports"
BASELINE_PROMOTION_EVENT_TYPE = "baseline_promotion"
QUALITY_FLOOR = 1.0


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    malformed = 0
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(row, dict):
                rows.append(row)
    return rows, malformed


def _effective_journal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    folded_rows, _ = fold_supersession_events(rows)
    return [row for row in folded_rows if row.get("trial_id") is not None]


def _read_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _details(row: dict[str, Any]) -> dict[str, Any]:
    eval_details = row.get("eval_details") or {}
    details = eval_details.get("details") or {}
    return details if isinstance(details, dict) else {}


def _n_questions(row: dict[str, Any]) -> int | None:
    details = _details(row)
    direct = row.get("n_questions") or details.get("total") or details.get("n_questions")
    n_questions = _as_int(direct)
    if n_questions is not None:
        return n_questions
    counts = details.get("per_suite_counts")
    if isinstance(counts, dict):
        total = 0
        for value in counts.values():
            count = _as_int(value)
            if count and count > 0:
                total += count
        return total or None
    return None


def _eval_wall_s(row: dict[str, Any]) -> float | None:
    eval_details = row.get("eval_details") or {}
    return (
        _as_float(row.get("eval_wall_s"))
        or _as_float(eval_details.get("eval_wall_s"))
        or _as_float(_details(row).get("eval_wall_s"))
    )


def _tokens_per_solved_task(row: dict[str, Any]) -> float | None:
    details = _details(row)
    existing = _as_float(details.get("tokens_per_solved_task"))
    if existing is not None:
        return existing
    tokens = _as_float(details.get("tokens_generated"))
    solved = _as_float(details.get("correct"))
    if tokens is None or solved is None or solved <= 0:
        return None
    return tokens / solved


def _fmt(value: Any, digits: int = 2) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"


def _trial_id(entry: dict[str, Any]) -> int | None:
    return _as_int(entry.get("trial_id"))


def _frontier_for(archive: dict[str, Any]) -> list[dict[str, Any]]:
    frontiers = archive.get("frontiers_by_tier") or {}
    return (
        frontiers.get(str(DEFAULT_FRONTIER_TIER))
        or frontiers.get(DEFAULT_FRONTIER_TIER)
        or archive.get("frontier")
        or []
    )


def _archive(
    rows: list[dict[str, Any]],
    state: dict[str, Any],
    *,
    objective_policy: str,
    current_run_only: bool,
) -> dict[str, Any]:
    try:
        pareto_epoch_ts = float(state.get("pareto_epoch_ts") or 0.0) or None
    except (TypeError, ValueError):
        pareto_epoch_ts = None
    try:
        deinflate_factor = float(state.get("pareto_pre_epoch_speed_factor", 0.5))
    except (TypeError, ValueError):
        deinflate_factor = 0.5
    try:
        exclude_before_ts = float(state.get("pareto_exclude_before_ts") or 0.0) or None
    except (TypeError, ValueError):
        exclude_before_ts = None
    archive = reconstruct_archive_from_journal_rows(
        rows,
        None,
        current_run_only=current_run_only,
        deinflate_before_ts=pareto_epoch_ts,
        deinflate_factor=deinflate_factor,
        exclude_before_ts=exclude_before_ts,
        objective_policy=objective_policy,
    )
    if archive is None:
        raise SystemExit("no reconstructable journal rows")
    return archive


def _dominators(
    row: dict[str, Any],
    task_rate_frontier: list[dict[str, Any]],
) -> list[int]:
    objectives = task_rate_objectives_from_row(row)
    if objectives is None:
        return []
    dominated_by: list[int] = []
    for entry in task_rate_frontier:
        if dominates(entry.get("objectives") or [], objectives):
            tid = _trial_id(entry)
            if tid is not None:
                dominated_by.append(tid)
    return dominated_by


def _row_table(
    title: str,
    entries: list[dict[str, Any]],
    rows_by_tid: dict[int, dict[str, Any]],
    task_rate_frontier: list[dict[str, Any]],
    *,
    include_dominators: bool,
) -> list[str]:
    lines = [f"## {title}", ""]
    if not entries:
        lines.extend(["None.", ""])
        return lines
    header = (
        "| Trial | Quality | Speed t/s | Wall s | N | task_rate q/h | "
        "goodput q/h | Tokens/solved | Dominated by task-rate |"
    )
    lines.extend([
        header,
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for entry in entries:
        tid = _trial_id(entry)
        row = rows_by_tid.get(tid or -1, {})
        dom = _dominators(row, task_rate_frontier) if include_dominators else []
        lines.append(
            "| "
            + " | ".join([
                str(tid if tid is not None else "n/a"),
                _fmt(row.get("quality"), 3),
                _fmt(row.get("speed"), 2),
                _fmt(_eval_wall_s(row), 1),
                str(_n_questions(row) or "n/a"),
                _fmt(task_rate_qph_from_row_if_available(row), 2),
                _fmt(goodput_qph_from_row(row), 2),
                _fmt(_tokens_per_solved_task(row), 1),
                ", ".join(str(item) for item in dom) or "n/a",
            ])
            + " |"
        )
    lines.append("")
    return lines


def _baseline_promotion_events(
    rows: list[dict[str, Any]],
    rows_by_tid: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    scoped_trial_ids = set(rows_by_tid)
    events: list[dict[str, Any]] = []
    for row in rows:
        if row.get("type") != BASELINE_PROMOTION_EVENT_TYPE:
            continue
        source_trial_id = _as_int(row.get("source_trial_id"))
        if source_trial_id is None or source_trial_id not in scoped_trial_ids:
            continue
        events.append(row)
    return sorted(
        events,
        key=lambda row: (
            row.get("timestamp", ""),
            _as_int(row.get("source_trial_id")) or -1,
        ),
    )


def _promotion_event_table(
    events: list[dict[str, Any]],
    rows_by_tid: dict[int, dict[str, Any]],
) -> list[str]:
    lines = ["## Baseline Promotion Evidence", ""]
    if not events:
        lines.extend(["None.", ""])
        return lines

    lines.extend([
        (
            "| Source trial | In replay rows | Tier | Previous q | New q | Delta | "
            "Result q | Result speed | Pareto status | Matrix | Speed mode | Reason |"
        ),
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|",
    ])
    for event in events:
        source_trial_id = _as_int(event.get("source_trial_id"))
        previous_quality = _as_float(event.get("previous_quality"))
        new_quality = _as_float(event.get("new_quality"))
        quality_delta = (
            new_quality - previous_quality
            if new_quality is not None and previous_quality is not None
            else None
        )
        proof = event.get("proof") if isinstance(event.get("proof"), dict) else {}
        result_metrics = (
            event.get("result_metrics")
            if isinstance(event.get("result_metrics"), dict)
            else {}
        )
        lines.append(
            "| "
            + " | ".join([
                str(source_trial_id if source_trial_id is not None else "n/a"),
                "yes" if source_trial_id in rows_by_tid else "no",
                str(_as_int(event.get("tier")) or "n/a"),
                _fmt(previous_quality, 3),
                _fmt(new_quality, 3),
                _fmt(quality_delta, 3),
                _fmt(result_metrics.get("quality"), 3),
                _fmt(result_metrics.get("speed"), 2),
                str(result_metrics.get("pareto_status") or "n/a"),
                str(proof.get("matrix_status") or "n/a"),
                str(proof.get("speed_metric_mode") or "n/a"),
                str(event.get("reason") or "n/a").replace("|", "\\|"),
            ])
            + " |"
        )
    lines.append("")
    return lines


def _quality_for_entry(
    entry: dict[str, Any],
    rows_by_tid: dict[int, dict[str, Any]],
) -> float | None:
    tid = _trial_id(entry)
    if tid is None:
        return None
    return _as_float(rows_by_tid.get(tid, {}).get("quality"))


def _write_report(
    path: Path,
    *,
    journal: Path,
    rows: list[dict[str, Any]],
    malformed: int,
    legacy: dict[str, Any],
    task_rate: dict[str, Any],
    current_run_only: bool,
) -> None:
    rows_by_tid = {
        tid: row
        for row in _effective_journal_rows(rows)
        if (tid := _as_int(row.get("trial_id"))) is not None
    }
    legacy_frontier = _frontier_for(legacy)
    task_rate_frontier = _frontier_for(task_rate)
    legacy_ids = {_trial_id(entry) for entry in legacy_frontier}
    task_rate_ids = {_trial_id(entry) for entry in task_rate_frontier}
    dropped_ids = sorted(tid for tid in legacy_ids - task_rate_ids if tid is not None)
    added_ids = sorted(tid for tid in task_rate_ids - legacy_ids if tid is not None)
    dropped = [entry for entry in legacy_frontier if _trial_id(entry) in dropped_ids]
    added = [entry for entry in task_rate_frontier if _trial_id(entry) in added_ids]

    drop_threshold = max(2, math.ceil(len(legacy_frontier) * 2 / 5))
    raw_drop_criterion_met = len(dropped_ids) >= drop_threshold
    low_quality_additions = [
        entry
        for entry in added
        if (quality := _quality_for_entry(entry, rows_by_tid)) is not None
        and quality < QUALITY_FLOOR
    ]
    promotion_ready = raw_drop_criterion_met and not low_quality_additions
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    scope = "latest trial-id reset segment" if current_run_only else "full journal"
    lines = [
        "# Task-rate / Goodput Replay Report",
        "",
        f"Generated: {generated}",
        f"Journal: `{journal}`",
        f"Scope: {scope}",
        f"Rows parsed: {len(rows)} ({malformed} malformed skipped)",
        "",
        "## Verdict",
        "",
        (
            f"{len(dropped_ids)} of {len(legacy_frontier)} legacy canonical T"
            f"{DEFAULT_FRONTIER_TIER} frontier points fall off under "
            f"`{TASK_RATE_OBJECTIVE_POLICY}`."
        ),
        (
            "Raw Fable drop criterion "
            f"(`>={drop_threshold} of {len(legacy_frontier)}`) is "
            f"{'met' if raw_drop_criterion_met else 'not met'} on this replay."
        ),
        (
            "Task-rate promotion readiness is "
            f"{'ready' if promotion_ready else 'not ready'}"
            + (
                " because task-rate frontier additions include "
                f"{len(low_quality_additions)} quality-floor violation(s)."
                if low_quality_additions
                else "."
            )
        ),
        "",
        "## Frontier Summary",
        "",
        "| Policy | Frontier points | All admitted entries | Hypervolume final |",
        "|---|---:|---:|---:|",
        (
            f"| `{LEGACY_OBJECTIVE_POLICY}` | {len(legacy_frontier)} | "
            f"{len(legacy.get('all_entries') or [])} | "
            f"{_fmt((legacy.get('hypervolume_history') or [[None, 0]])[-1][1], 4)} |"
        ),
        (
            f"| `{TASK_RATE_OBJECTIVE_POLICY}` | {len(task_rate_frontier)} | "
            f"{len(task_rate.get('all_entries') or [])} | "
            f"{_fmt((task_rate.get('hypervolume_history') or [[None, 0]])[-1][1], 4)} |"
        ),
        "",
    ]
    lines.extend(
        _row_table(
            "Legacy Frontier Points Dropped Under Task-rate",
            dropped,
            rows_by_tid,
            task_rate_frontier,
            include_dominators=True,
        )
    )
    lines.extend(
        _row_table(
            "Task-rate Frontier Additions",
            added,
            rows_by_tid,
            task_rate_frontier,
            include_dominators=False,
        )
    )
    lines.extend(
        _promotion_event_table(
            _baseline_promotion_events(rows, rows_by_tid),
            rows_by_tid,
        )
    )
    lines.extend([
        "## Notes",
        "",
        "- Dominance uses `(quality, task_rate_qph, reliability)` for the task-rate policy.",
        "- `goodput_qph` is reported as a diagnostic: `(quality / 3) * task_rate_qph`.",
        "- Legacy speed de-inflation is preserved for `legacy_4d_v1` and ignored for task-rate replay.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--current-run-only",
        action="store_true",
        help="replay only the latest contiguous trial-id segment",
    )
    args = parser.parse_args()

    # JRN-5: default base path fans out to every rotated shard; an explicit
    # non-base --journal path is read as given.
    rows: list[dict[str, Any]] = []
    malformed = 0
    for shard in resolve_journal_paths(args.journal):
        shard_rows, shard_malformed = _read_jsonl(shard)
        rows.extend(shard_rows)
        malformed += shard_malformed
    state = _read_state(args.state)
    legacy = _archive(
        rows,
        state,
        objective_policy=LEGACY_OBJECTIVE_POLICY,
        current_run_only=args.current_run_only,
    )
    task_rate = _archive(
        rows,
        state,
        objective_policy=TASK_RATE_OBJECTIVE_POLICY,
        current_run_only=args.current_run_only,
    )

    output = args.output
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        output = DEFAULT_REPORT_DIR / f"task_rate_goodput_replay_{stamp}.md"
    _write_report(
        output,
        journal=args.journal,
        rows=rows,
        malformed=malformed,
        legacy=legacy,
        task_rate=task_rate,
        current_run_only=args.current_run_only,
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
