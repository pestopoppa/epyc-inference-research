#!/usr/bin/env python3
"""RI-16: per-stage routing-decision latency from the progress JSONL.

Reads the ``stage_ms`` telemetry that live ``/chat`` requests write
(``src/runtime/routing_stage_timing.py``) and prints p50 / p95 / max per stage with n.

One row per request: the ``task_completed`` / ``task_failed`` record is preferred
(final: it includes ``mode``, ``review_gate`` and the final ``total``); a request
with only a ``routing_decision`` record (still running, or failed before its
completion was logged) contributes its pre-execution snapshot. ``n`` per stage
counts only requests where that stage RAN (``None`` = did not run, not 0 ms).

Pure offline read — no server, no inference.

Usage:
    python3 scripts/analysis/routing_stage_latency.py                  # all logs
    python3 scripts/analysis/routing_stage_latency.py --since 24h      # last 24 h
    python3 scripts/analysis/routing_stage_latency.py \\
        --since 2026-09-30T00:00 --until 2026-09-30T12:00 --path chat --json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

SCRIPT_DIR = Path(__file__).resolve().parent
ORCH_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOG_DIR = ORCH_ROOT / "logs" / "progress"

#: Mirrors ``src.runtime.routing_stage_timing.STAGE_KEYS`` (kept literal so this
#: reader has no src import); unknown stages found in the data are appended.
STAGE_ORDER: tuple[str, ...] = (
    "memrl_init",
    "priors",
    "route",
    "xmas",
    "factual_risk",
    "failure_veto",
    "difficulty",
    "trinity",
    "route_total",
    "mode",
    "routing_context",
    "review_gate",
    "review_verdict",
    "total",
)
PATHS = ("chat", "unified_stream", "legacy_stream")
_COMPLETION_EVENTS = frozenset({"task_completed", "task_failed"})
_DECISION_EVENT = "routing_decision"
_RELATIVE_RE = re.compile(r"^(\d+(?:\.\d+)?)([smhd])$")
_UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_time(value: str | None, *, now: datetime | None = None) -> datetime | None:
    """ISO-8601 (naive = UTC) or a relative span like ``90m`` / ``24h`` / ``7d``."""
    if value is None:
        return None
    text = value.strip()
    match = _RELATIVE_RE.match(text)
    if match:
        now = now or datetime.now(timezone.utc)
        return now - timedelta(seconds=float(match.group(1)) * _UNIT_SECONDS[match.group(2)])
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _log_files(log_dir: Path) -> list[Path]:
    return sorted(
        [*log_dir.glob("*.jsonl"), *log_dir.glob("*.jsonl.gz")], key=lambda p: p.name
    )


def _file_may_overlap(path: Path, since: datetime | None, until: datetime | None) -> bool:
    """Skip day files (``YYYY-MM-DD.jsonl``) wholly outside the window."""
    match = re.match(r"^(\d{4}-\d{2}-\d{2})", path.name)
    if not match:
        return True
    # one day of slack either side: the file date is the writer's calendar day
    day_start = datetime.fromisoformat(match.group(1)).replace(tzinfo=timezone.utc) - timedelta(days=1)
    day_end = day_start + timedelta(days=3)
    if since is not None and day_end < since:
        return False
    if until is not None and day_start > until:
        return False
    return True


def iter_events(log_dir: Path, since: datetime | None = None, until: datetime | None = None) -> Iterator[dict]:
    for path in _log_files(log_dir):
        if not _file_may_overlap(path, since, until):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if '"stage_ms"' not in line:  # cheap prefilter
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def collect_requests(
    events: Iterable[dict],
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """One record per task_id: ``{"path", "stage_ms", "final", "timestamp"}``."""
    requests: dict[str, dict[str, Any]] = {}
    for event in events:
        event_type = event.get("event_type")
        if event_type not in _COMPLETION_EVENTS and event_type != _DECISION_EVENT:
            continue
        data = event.get("data") or {}
        stage_ms = data.get("stage_ms")
        task_id = event.get("task_id")
        if not isinstance(stage_ms, dict) or not task_id:
            continue
        try:
            ts = parse_time(str(event.get("timestamp")))
        except ValueError:
            continue
        if since is not None and ts < since:
            continue
        if until is not None and ts > until:
            continue
        routing_path = data.get("routing_path") or "unknown"
        if path is not None and routing_path != path:
            continue
        is_final = event_type in _COMPLETION_EVENTS
        known = requests.get(task_id)
        if known is not None and known["final"] and not is_final:
            continue  # never replace a completion with an earlier snapshot
        requests[task_id] = {
            "path": routing_path,
            "stage_ms": stage_ms,
            "final": is_final,
            "timestamp": ts,
        }
    return requests


def percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list (q in [0, 100])."""
    if not sorted_values:
        raise ValueError("empty")
    rank = max(1, math.ceil(q / 100.0 * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def summarize(requests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    for record in requests.values():
        for stage, value in record["stage_ms"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.setdefault(stage, []).append(float(value))
    order = [s for s in STAGE_ORDER if s in values] + sorted(
        s for s in values if s not in STAGE_ORDER
    )
    stages = {}
    for stage in order:
        ordered = sorted(values[stage])
        stages[stage] = {
            "n": len(ordered),
            "p50": percentile(ordered, 50),
            "p95": percentile(ordered, 95),
            "max": ordered[-1],
        }
    by_path: dict[str, int] = {}
    for record in requests.values():
        by_path[record["path"]] = by_path.get(record["path"], 0) + 1
    return {
        "requests": len(requests),
        "final_records": sum(1 for r in requests.values() if r["final"]),
        "by_path": dict(sorted(by_path.items())),
        "stages": stages,
    }


def format_table(summary: dict[str, Any], header: str) -> str:
    lines = [
        header,
        f"requests={summary['requests']} (final={summary['final_records']}) "
        f"by_path={summary['by_path']}",
        f"{'stage':<16}{'n':>7}{'p50_ms':>12}{'p95_ms':>12}{'max_ms':>12}",
    ]
    for stage, row in summary["stages"].items():
        lines.append(
            f"{stage:<16}{row['n']:>7}{row['p50']:>12.3f}{row['p95']:>12.3f}{row['max']:>12.3f}"
        )
    if not summary["stages"]:
        lines.append("(no stage_ms records in window)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="progress JSONL dir")
    parser.add_argument("--since", help="window start: ISO-8601 (UTC if naive) or relative (24h, 90m, 7d)")
    parser.add_argument("--until", help="window end: ISO-8601 or relative")
    parser.add_argument("--path", choices=[*PATHS, "all"], default="all", help="routing path filter")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    now = datetime.now(timezone.utc)
    since = parse_time(args.since, now=now)
    until = parse_time(args.until, now=now)
    path = None if args.path == "all" else args.path
    if not args.log_dir.is_dir():
        print(f"log dir not found: {args.log_dir}", file=sys.stderr)
        return 2
    requests = collect_requests(
        iter_events(args.log_dir, since, until), since=since, until=until, path=path
    )
    summary = summarize(requests)
    summary["window"] = {
        "since": since.isoformat() if since else None,
        "until": until.isoformat() if until else None,
        "path": args.path,
        "log_dir": str(args.log_dir),
    }
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        header = (
            f"routing stage latency  window=[{summary['window']['since'] or '-'} .. "
            f"{summary['window']['until'] or '-'}]  path={args.path}"
        )
        print(format_table(summary, header))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
