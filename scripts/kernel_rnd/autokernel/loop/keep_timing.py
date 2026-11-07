#!/usr/bin/env python3
"""Per-stage wall timing of the AutoKernel KEEP path (observability only).

WHY. ~22 h of DS41 wall was attributed to anchor rows by inter-row gaps
(ak-anchor-reuse-20261107/DECISION.md), but nothing recorded what the keep path
spends its time on: clean anchor build vs guard build vs digest vs A/A vs headline
vs reprofile. The suspected duplicate (a clean AND an incremental build of the same
commit per keep) could not be quantified either.

WHAT. `KeepTimer` collects one record per stage -- monotonic start/end, wall-clock
ISO start/end, duration, plus the build identity (commit, build dir) for build
stages, or `skipped` + `reason`. `run.py` attaches `KeepTimer.to_dict()` to the
`kept` row's payload as the additive field `keep_path_timing` (schema below). It
never changes control flow: every method swallows its own faults, and a stage that
raises is recorded with `error` and the exception re-raised unchanged.

`summarize` / `main` is the read-only report: per-stage duration totals across a
store's `kept` rows, and a flag for every keep that built the same commit twice.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import statistics
import sys
import time
from typing import Any, Iterable, Iterator, Mapping

SCHEMA = "epyc.autokernel.keep_path_timing.v1"
#: Payload key on the `kept` experiment row.
FIELD = "keep_path_timing"
#: Stages that BUILD a commit -- the duplicate-build check compares their identities.
BUILD_STAGES = ("anchor_clean_build", "anchor_guard_build", "anchor_guard_heal_build")


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class KeepTimer:
    """Collects stage records for ONE keep. Never raises from its own bookkeeping."""

    def __init__(self, *, head: str | None = None,
                 clock=time.monotonic, wall=_iso) -> None:
        self.head = head
        self._clock = clock
        self._wall = wall
        self.stages: list[dict[str, Any]] = []
        try:
            self._t0 = clock()
            self._t0_iso = wall()
        except Exception:  # pragma: no cover - observability must not fault
            self._t0, self._t0_iso = None, None

    @contextmanager
    def stage(self, name: str, **identity: Any) -> Iterator[dict[str, Any]]:
        """Time one stage. The yielded dict may be updated with extra facts."""
        rec: dict[str, Any] = {"stage": name}
        try:
            rec.update({k: v for k, v in identity.items() if v is not None})
            rec["start_monotonic"] = self._clock()
            rec["started_at"] = self._wall()
        except Exception:  # pragma: no cover
            pass
        try:
            yield rec
        except BaseException as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"[:512]
            raise
        finally:
            try:
                rec["end_monotonic"] = self._clock()
                rec["ended_at"] = self._wall()
                rec["duration_s"] = round(rec["end_monotonic"] - rec["start_monotonic"], 3)
                self.stages.append(rec)
            except Exception:  # pragma: no cover
                pass

    def timed(self, name: str, fn, **identity: Any):
        """Wrap a callable so every call is a stage (same args, same return/raise)."""
        def wrapper(*args: Any, **kwargs: Any):
            with self.stage(name, **identity):
                return fn(*args, **kwargs)
        return wrapper

    def skip(self, name: str, reason: str, **facts: Any) -> None:
        try:
            now, iso = self._clock(), self._wall()
            self.stages.append({"stage": name, "skipped": True, "reason": str(reason),
                                "start_monotonic": now, "end_monotonic": now,
                                "started_at": iso, "ended_at": iso, "duration_s": 0.0,
                                **{k: v for k, v in facts.items() if v is not None}})
        except Exception:  # pragma: no cover
            pass

    def to_dict(self) -> dict[str, Any]:
        try:
            end, end_iso = self._clock(), self._wall()
            total = None if self._t0 is None else round(end - self._t0, 3)
        except Exception:  # pragma: no cover
            end, end_iso, total = None, None, None
        return {"schema": SCHEMA, "authority": "diagnostic_only", "head": self.head,
                "started_at": self._t0_iso, "ended_at": end_iso, "total_s": total,
                "stages": [dict(s) for s in self.stages],
                "duplicate_builds": duplicate_builds(self.stages)}


def duplicate_builds(stages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Commits built more than once within one keep (any build stage, any dir)."""
    seen: dict[str, list[Mapping[str, Any]]] = {}
    for s in stages:
        if s.get("stage") in BUILD_STAGES and s.get("commit") and not s.get("skipped"):
            seen.setdefault(str(s["commit"]), []).append(s)
    return [{"commit": commit, "builds": len(items),
             "stages": [i.get("stage") for i in items],
             "build_dirs": [i.get("build_dir") for i in items],
             "build_s": round(sum(float(i.get("duration_s") or 0.0) for i in items), 3)}
            for commit, items in seen.items() if len(items) > 1]


def summarize(payloads: Iterable[Mapping[str, Any] | None]) -> dict[str, Any]:
    """Per-stage durations across `kept` payloads; rows without timing are counted."""
    per: dict[str, list[float]] = {}
    skipped: dict[str, int] = {}
    dups: list[dict[str, Any]] = []
    timed = untimed = 0
    totals: list[float] = []
    for payload in payloads:
        timing = (payload or {}).get(FIELD) if isinstance(payload, Mapping) else None
        if not isinstance(timing, Mapping) or not isinstance(timing.get("stages"), list):
            untimed += 1
            continue
        timed += 1
        if isinstance(timing.get("total_s"), (int, float)):
            totals.append(float(timing["total_s"]))
        for s in timing["stages"]:
            if not isinstance(s, Mapping):
                continue
            name = str(s.get("stage"))
            if s.get("skipped"):
                skipped[name] = skipped.get(name, 0) + 1
                continue
            if isinstance(s.get("duration_s"), (int, float)):
                per.setdefault(name, []).append(float(s["duration_s"]))
        for d in duplicate_builds(s for s in timing["stages"] if isinstance(s, Mapping)):
            dups.append({"head": timing.get("head"), **d})
    stages = {name: {"n": len(v), "total_s": round(sum(v), 3),
                     "median_s": round(statistics.median(v), 3),
                     "max_s": round(max(v), 3), "skipped": skipped.get(name, 0)}
              for name, v in per.items()}
    for name, n in skipped.items():
        stages.setdefault(name, {"n": 0, "total_s": 0.0, "median_s": None,
                                 "max_s": None, "skipped": n})
    return {"schema": SCHEMA + ".report", "keeps_timed": timed, "keeps_untimed": untimed,
            "keep_total_s": round(sum(totals), 3), "stages": stages,
            "duplicate_builds": dups}


def read_store(store_root: Path) -> list[Mapping[str, Any] | None]:
    """Read-only: every `kept` row's decoded payload from `experiments.db`."""
    db = Path(store_root) / "experiments.db"
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT payload FROM experiments WHERE status = 'kept' "
                            "ORDER BY recorded_at, rowid").fetchall()
    finally:
        conn.close()
    out: list[Mapping[str, Any] | None] = []
    for (raw,) in rows:
        try:
            out.append(json.loads(raw))
        except (TypeError, ValueError):
            out.append(None)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize keep-path stage timing (read-only).")
    parser.add_argument("store", type=Path, help="AutoKernel store root (holds experiments.db)")
    parser.add_argument("--json", action="store_true", help="print the JSON report")
    args = parser.parse_args(argv)
    report = summarize(read_store(args.store))
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"keeps timed {report['keeps_timed']}, untimed {report['keeps_untimed']}, "
          f"keep wall {report['keep_total_s'] / 3600:.2f} h")
    for name, s in sorted(report["stages"].items(), key=lambda kv: -kv[1]["total_s"]):
        print(f"  {name:28s} n={s['n']:<4d} total={s['total_s'] / 60:8.1f} min "
              f"median={s['median_s'] if s['median_s'] is not None else '-'} s "
              f"skipped={s['skipped']}")
    for d in report["duplicate_builds"]:
        print(f"  DUPLICATE BUILD keep {str(d['head'])[:12]}: commit {d['commit'][:12]} "
              f"built {d['builds']}x ({', '.join(map(str, d['stages']))}), "
              f"{d['build_s'] / 60:.1f} min")
    return 0


__all__ = ["BUILD_STAGES", "FIELD", "SCHEMA", "KeepTimer", "duplicate_builds",
           "main", "read_store", "summarize"]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
