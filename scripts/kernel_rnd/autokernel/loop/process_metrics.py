#!/usr/bin/env python3
"""Read-only retrospective process metrics for AutoKernel attempt stores.

This module deliberately does not open :class:`ExperimentStore`: that class can
create a database and enable WAL when used incorrectly.  Retrospective analysis
opens SQLite inputs with ``mode=ro`` and accepts exported JSON/JSONL as a second,
equally read-only source.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.process_metrics.v1"
MEASURED = frozenset({"kept", "keep_candidate", "measured_null", "regression",
                      "runtime_observed"})
#: `patch_rounds_exhausted` / `scope_blocked` / `hypothesis_retired` replace the
#: `refused_at_formation` an accepted hypothesis's spent patch rounds used to end in.
VALID_DENOMINATOR = MEASURED | {"refused_at_formation", "planner_transient", "bench_failed",
                                "patch_rounds_exhausted", "scope_blocked",
                                "hypothesis_retired", "authoring_failed",
                                "authoring_harness_failure"}


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def _critic_cohort(row: Mapping[str, Any]) -> str | None:
    """Return a cohort only when pass-1 disposition is positively recorded.

    Absence is not acceptance: historical runtime fast paths and legacy writers
    legitimately emitted no critic record at all.
    """
    value = row.get("critic_pass_1_rejected")
    if value in (0, 1) and not isinstance(value, bool):
        value = bool(value)
    if isinstance(value, bool):
        return "accepted_after_rejection" if value else "accepted_first_pass"
    verdict = row.get("critic_pass_1")
    if isinstance(verdict, Mapping):
        rejected = verdict.get("rejected_before_acceptance")
        if isinstance(rejected, bool) and verdict.get("accepted") is True:
            return "accepted_after_rejection" if rejected else "accepted_first_pass"
    return None


def _critic_decision(row: Mapping[str, Any]) -> bool | None:
    """True for an explicit pass-1 rejection, False for an explicit acceptance."""
    if row.get("status") == "critic_reject":
        return True
    verdict = row.get("critic_pass_1")
    if isinstance(verdict, Mapping) and isinstance(verdict.get("accepted"), bool):
        return not verdict["accepted"]
    return None


def calculate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    attempt_steps = 0
    measured = 0
    denominator = 0
    first_improvement = None
    best = 0.0
    auc = 0.0
    numeric_effect_steps = 0
    critic_decisions = {"rejected": 0, "accepted": 0}
    cohorts = {
        "accepted_after_rejection": {"eventual_measured": 0, "measured_null": 0},
        "accepted_first_pass": {"eventual_measured": 0, "measured_null": 0},
    }

    for step, row in enumerate(rows, 1):
        attempt_steps = step
        status = str(row.get("status") or "unknown")
        decision = _critic_decision(row)
        if decision is not None:
            critic_decisions["rejected" if decision else "accepted"] += 1
        if status in VALID_DENOMINATOR:
            denominator += 1
        if status in MEASURED:
            measured += 1
            effect = _finite_number(row.get("effect_fraction"))
            if effect is not None:
                numeric_effect_steps += 1
                best = max(best, effect)
                if first_improvement is None and effect > 0:
                    first_improvement = step
            cohort = _critic_cohort(row)
            if cohort is not None:
                cohorts[cohort]["eventual_measured"] += 1
                cohorts[cohort]["measured_null"] += status == "measured_null"
        auc += best

    for values in cohorts.values():
        count = values["eventual_measured"]
        values["eventual_measured_null_rate"] = (
            values["measured_null"] / count if count else None)

    after = cohorts["accepted_after_rejection"]["eventual_measured"]
    first = cohorts["accepted_first_pass"]["eventual_measured"]
    decision_total = sum(critic_decisions.values())
    critic_available = bool(after and first and decision_total)
    return {
        "schema": SCHEMA,
        "attempt_steps": attempt_steps,
        "first_improvement_step": first_improvement,
        "auc_best_so_far_effect_over_steps": auc,
        "final_best_so_far_effect_fraction": best,
        "numeric_effect_steps": numeric_effect_steps,
        "valid_step_ratio": measured / denominator if denominator else None,
        "valid_step_counts": {"measured": measured, "denominator": denominator},
        "critic_pass_1_comparison": {
            "available": critic_available,
            "unavailable_reason": (None if critic_available else
                "the rejection rate requires explicit pass-1 decisions and both outcome cohorts "
                "require explicit pass-1 lineage on eventual measured outcomes; absence of a "
                "rejection is not evidence of first-pass acceptance"),
            "cohorts": cohorts,
            "pass_1_decisions": critic_decisions,
            "pass_1_rejection_rate_available": bool(decision_total),
            "pass_1_rejection_rate": (
                critic_decisions["rejected"] / decision_total if decision_total else None),
            "measured_null_rate_difference_after_rejection_minus_first_pass": (
                cohorts["accepted_after_rejection"]["eventual_measured_null_rate"]
                - cohorts["accepted_first_pass"]["eventual_measured_null_rate"]
                if critic_available else None),
        },
    }


def _sqlite_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Stream bounded scalar projections; never materialize the payload column.

    The live GLM database has a multi-GiB payload corpus.  SQLite can avoid its
    overflow pages when only indexed/table scalar columns are selected.  Optional
    critic columns are projected only if a future schema adds them explicitly.
    """
    uri = path.resolve().as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(experiments)")}
        required = {"recorded_at", "status", "effect_fraction"}
        if not required <= columns:
            raise ValueError(f"{path}: experiments table lacks {sorted(required - columns)}")
        names = ["recorded_at", "status", "effect_fraction"]
        expressions = list(names)
        if "payload" in columns:
            # SQL returns two bounded values, never the potentially multi-MiB payload.
            names.extend(("critic_pass_1_rejected", "critic_pass_1"))
            expressions.extend((
                "json_extract(payload,'$.critic_pass_1_rejected')",
                "json_extract(payload,'$.critic_pass_1')"))
        else:
            optional = [name for name in ("critic_pass_1_rejected", "critic_pass_1")
                        if name in columns]
            names.extend(optional)
            expressions.extend(optional)
        query = (f"SELECT {','.join(expressions)} FROM experiments "
                 "ORDER BY recorded_at,rowid")
        for values in connection.execute(query):
            row = dict(zip(names, values))
            if isinstance(row.get("critic_pass_1"), str):
                try:
                    row["critic_pass_1"] = json.loads(row["critic_pass_1"])
                except json.JSONDecodeError:
                    row["critic_pass_1"] = None
            yield row
    finally:
        connection.close()


def load(path: Path) -> Iterable[dict[str, Any]]:
    if path.name == "experiments.db" or path.suffix == ".db":
        return _sqlite_rows(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        body = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        body = json.loads(text)
        if isinstance(body, dict):
            body = body.get("iterations", body.get("attempts"))
    if not isinstance(body, list) or not all(isinstance(row, dict) for row in body):
        raise ValueError(f"{path}: expected a JSON list or an object with iterations/attempts")
    return body


def analyze(stores: Sequence[Path]) -> dict[str, Any]:
    """Return per-store metrics and one chronological aggregate."""
    all_rows: list[dict[str, Any]] = []
    per_store = []
    for path in stores:
        rows = list(load(path))  # scalar projections only for SQLite inputs
        rows.sort(key=lambda row: str(row.get("recorded_at") or
                                      row.get("turn_recorded_at") or ""))
        per_store.append({"path": str(path), "metrics": calculate(rows)})
        all_rows.extend(rows)
    all_rows.sort(key=lambda row: str(row.get("recorded_at") or
                                      row.get("turn_recorded_at") or ""))
    return {"schema": "epyc.autokernel.process_metrics_collection.v1",
            "stores": per_store, "aggregate": calculate(all_rows)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stores", nargs="+", type=Path,
                        help="experiments.db, JSON attempt export, or JSONL attempt export")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(analyze(args.stores), sort_keys=True,
                     indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
