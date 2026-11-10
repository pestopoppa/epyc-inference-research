#!/usr/bin/env python3
"""Per-rule C4 telemetry and estimate-vs-achieved error (INF48-LIVE-3: RVP-C4-9, -11).

RVP-C4-9 (method from intake-1222#record Table 9; its NVIDIA numbers do not port):
on every eval-backend run, log one record per deterministic C4 rule --
(rule_id, fired?, downstream speedup) -- then fold to tasks-covered / fires /
hit-rate at >1.5x / avg speedup, and mark rules at or below the BASE RATE for pruning.
The base rate is the >1.5x rate over ALL logged task outcomes, fired or not: a rule
whose firings win no more often than chance adds nothing.

RVP-C4-11: each fired prescription carries its ESTIMATED speedup
(profile_report.SourcePattern.estimated_speedup); the reducer reports the relative
error |est - achieved| / achieved per rule and overall, mean AND worst -- the mean is
never the accuracy of a single prediction (GPA's 4.1% mean hides 42% / 26% rows).

Records are JSONL, append-only, one file per run. This module costs no GPU time.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.c4_rule_telemetry.v1"
FOLD_SCHEMA = "epyc.autokernel.c4_rule_telemetry_fold.v1"
HIT_THRESHOLD = 1.5


class TelemetryError(ValueError):
    pass


@dataclass(frozen=True)
class RuleRecord:
    run_id: str
    task_id: str
    rule_id: str
    fired: bool
    downstream_speedup: float | None      # None = run produced no valid measurement
    estimated_speedup: float | None = None

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "rule_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise TelemetryError(f"{name} must be non-empty text")
        if not isinstance(self.fired, bool):
            raise TelemetryError("fired must be a bool")
        for name in ("downstream_speedup", "estimated_speedup"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value <= 0:
                raise TelemetryError(f"{name} must be a positive finite number or null")
        if self.estimated_speedup is not None and not self.fired:
            raise TelemetryError("an estimate belongs to a fired prescription only")

    def as_dict(self) -> dict[str, Any]:
        return {"schema": SCHEMA, "run_id": self.run_id, "task_id": self.task_id,
                "rule_id": self.rule_id, "fired": self.fired,
                "downstream_speedup": self.downstream_speedup,
                "estimated_speedup": self.estimated_speedup}

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> "RuleRecord":
        if row.get("schema") != SCHEMA:
            raise TelemetryError(f"record schema must be {SCHEMA}")
        return cls(row.get("run_id"), row.get("task_id"), row.get("rule_id"),
                   row.get("fired"), row.get("downstream_speedup"),
                   row.get("estimated_speedup"))


def records_for_run(run_id: str, task_id: str, rule_ids: Iterable[str],
                    fired: Mapping[str, float | None], downstream_speedup: float | None,
                    ) -> list[RuleRecord]:
    """One record per KNOWN rule, fired or not -- the denominator is every rule.

    `fired` maps rule_id -> estimated speedup (or None when the rule fired without
    one, which `require_estimates` should already have refused upstream).
    """
    known = sorted(set(rule_ids))
    unknown = sorted(set(fired) - set(known))
    if unknown:
        raise TelemetryError(f"fired rules not in the rule set: {unknown}")
    return [RuleRecord(run_id, task_id, rule, rule in fired, downstream_speedup,
                       fired.get(rule) if rule in fired else None)
            for rule in known]


def records_from_report(report: Any, *, run_id: str, task_id: str,
                        downstream_speedup: float | None) -> list[RuleRecord]:
    """Bind a profile_report.ProfileReport to telemetry: the rule set is every
    manifest pattern, and a rule FIRED iff the deterministic pass emitted its row."""
    rule_ids = [pattern.pattern_id for pattern in report.manifest.patterns]
    fired = {row["pattern_id"]: row.get("estimated_speedup")
             for row in (*report.overlap_rows, *report.fuse_rows)}
    return records_for_run(run_id, task_id, rule_ids, fired, downstream_speedup)


def append_records(path: Path, records: Sequence[RuleRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), sort_keys=True) + "\n")


def load_records(paths: Iterable[Path]) -> list[RuleRecord]:
    out = []
    for path in paths:
        for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append(RuleRecord.from_dict(json.loads(line)))
            except (ValueError, TelemetryError) as exc:
                raise TelemetryError(f"{path}:{line_no}: {exc}") from exc
    return out


def _error_stats(pairs: Sequence[tuple[float, float]]) -> dict[str, Any]:
    errors = [abs(est - ach) / ach for est, ach in pairs]
    if not errors:
        return {"n": 0, "mean_rel_error": None, "worst_rel_error": None}
    return {"n": len(errors), "mean_rel_error": sum(errors) / len(errors),
            "worst_rel_error": max(errors)}


def fold(records: Sequence[RuleRecord], *, hit_threshold: float = HIT_THRESHOLD) -> dict:
    """tasks-covered / fires / hit-rate / avg speedup per rule, base rate, prune list,
    and estimate-vs-achieved error. Measured outcomes only: a null speedup counts as a
    fire but never as a hit and never in an average or an error."""
    seen: set[tuple[str, str, str]] = set()
    for r in records:
        key = (r.run_id, r.task_id, r.rule_id)
        if key in seen:
            raise TelemetryError(f"duplicate record {key}")
        seen.add(key)
    outcomes: dict[tuple[str, str], float | None] = {}
    for r in records:
        key = (r.run_id, r.task_id)
        if key in outcomes and outcomes[key] != r.downstream_speedup:
            raise TelemetryError(f"run {key} logged two different outcomes")
        outcomes[key] = r.downstream_speedup
    measured = [v for v in outcomes.values() if v is not None]
    base_rate = (sum(v > hit_threshold for v in measured) / len(measured)) if measured else None
    rules: dict[str, dict[str, Any]] = {}
    all_pairs: list[tuple[float, float]] = []
    for rule in sorted({r.rule_id for r in records}):
        rows = [r for r in records if r.rule_id == rule]
        fired = [r for r in rows if r.fired]
        fired_measured = [r.downstream_speedup for r in fired if r.downstream_speedup is not None]
        hits = sum(v > hit_threshold for v in fired_measured)
        hit_rate = hits / len(fired_measured) if fired_measured else None
        pairs = [(r.estimated_speedup, r.downstream_speedup) for r in fired
                 if r.estimated_speedup is not None and r.downstream_speedup is not None]
        all_pairs.extend(pairs)
        prune = (hit_rate is not None and base_rate is not None and hit_rate <= base_rate)
        rules[rule] = {
            "tasks_covered": len({(r.run_id, r.task_id) for r in fired}),
            "fires": len(fired),
            "fires_measured": len(fired_measured),
            "hits": hits,
            "hit_rate": hit_rate,
            "avg_speedup": (sum(fired_measured) / len(fired_measured)) if fired_measured else None,
            "fired_without_estimate": sum(r.estimated_speedup is None for r in fired),
            "estimate_error": _error_stats(pairs),
            "prune": prune,
        }
    return {
        "schema": FOLD_SCHEMA,
        "hit_threshold": hit_threshold,
        "tasks": len(outcomes),
        "tasks_measured": len(measured),
        "base_rate": base_rate,
        "rules": rules,
        "prune": sorted(rule for rule, row in rules.items() if row["prune"]),
        "estimate_error": _error_stats(all_pairs),
        "note": "mean_rel_error is NOT the accuracy of any single prediction; read worst_rel_error",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fold C4 per-rule telemetry JSONL")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--hit-threshold", type=float, default=HIT_THRESHOLD)
    args = parser.parse_args(argv)
    json.dump(fold(load_records(args.paths), hit_threshold=args.hit_threshold),
              sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
