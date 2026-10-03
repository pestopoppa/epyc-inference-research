#!/usr/bin/env python3
"""Prefix-cache effectiveness per llama-server, from the serving-call records (UFH14-B4).

Reads ``logs/serving_calls/serving_calls.jsonl`` (and its rotated shards ``.1`` …)
written by ``src/backends/serving_calls.py`` and reports, per port and per
(port, role):

  hit_tok          sum(cache_n) / sum(cache_n + prompt_n)            HIGHER is better
  prefill_s        sum(prompt_ms) / 1000                              LOWER is better
  cold_large       calls with prompt_n >= 8192 and cache_n < 1024     LOWER is better
                   (the definition used by docs/reviews/prefill-share-20261003.md)
  missed_reuse     calls whose prompt shared >= D leading characters with an EARLIER
                   call to the same port inside --window-min (``request.prefix_fp``),
                   yet reused fewer than half of the ~D/3 tokens that prefix implies
  missed_prefill_s prefill seconds attributable to missed reuse:     LOWER is better
                   prompt_ms * min(1, (D/3 - cache_n) / prompt_n)

``missed_prefill_share = missed_prefill_s / prefill_s`` is the B4 headline: the part
of prefill the server could have skipped had the prefix still been cached and
reachable. It is a LOWER BOUND on avoidable prefill (fingerprints only see exact
character-prefix agreement at fixed depths).

Records written before UFH14-B4 carry no ``prefix_fp``; for them only hit_tok,
prefill_s and cold_large are computed and ``fp_coverage`` says how many calls the
missed-reuse figures rest on.

Before/after: ``--split-at <ISO-UTC>`` reports both sides and the deltas.

Usage:
  python3 scripts/analysis/prefix_cache_report.py [--log PATH] [--since ISO] [--until ISO]
      [--split-at ISO] [--port 8083] [--window-min 60] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG = REPO / "logs" / "serving_calls" / "serving_calls.jsonl"
CHARS_PER_TOKEN = 3.0
COLD_NEW_TOKENS = 8192
COLD_CACHED_MAX = 1024
MISSED_REUSE_FRACTION = 0.5


def parse_ts(value: str | None) -> float | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def shard_paths(log: Path) -> list[Path]:
    """Oldest first: ``f.N`` … ``f.1``, then ``f``."""
    rotated = sorted(
        (p for p in log.parent.glob(log.name + ".*") if p.suffix[1:].isdigit()),
        key=lambda p: -int(p.suffix[1:]),
    )
    return rotated + ([log] if log.exists() else [])


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    for path in paths:
        try:
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict):
                        yield rec
        except OSError:
            continue


@dataclass
class Call:
    ts: float
    ts_end: float
    port: int
    role: str
    cache_n: int
    prompt_n: int
    prompt_ms: float
    fp: dict[str, Any] | None


def to_call(rec: dict[str, Any]) -> Call | None:
    """A dispatched call with server timings, or None."""
    if not rec.get("dispatched", True):
        return None
    timings = rec.get("timings") or {}
    if not isinstance(timings, dict) or "prompt_n" not in timings:
        return None
    ts = parse_ts(rec.get("ts_start"))
    port = (rec.get("server") or {}).get("port")
    if ts is None or not isinstance(port, int):
        return None
    try:
        cache_n = int(timings.get("cache_n") or 0)
        prompt_n = int(timings.get("prompt_n") or 0)
        prompt_ms = float(timings.get("prompt_ms") or 0.0)
    except (TypeError, ValueError):
        return None
    fp = (rec.get("request") or {}).get("prefix_fp")
    return Call(
        ts=ts,
        ts_end=parse_ts(rec.get("ts_end")) or ts,
        port=port,
        role=str(rec.get("role") or "?"),
        cache_n=cache_n,
        prompt_n=prompt_n,
        prompt_ms=prompt_ms,
        fp=fp if isinstance(fp, dict) else None,
    )


def _depths(fp: dict[str, Any]) -> list[tuple[int, str]]:
    out = []
    for key, value in fp.items():
        if key.startswith("c") and key[1:].isdigit() and isinstance(value, str):
            out.append((int(key[1:]), value))
    return sorted(out, reverse=True)  # deepest first


@dataclass
class Agg:
    calls: int = 0
    cache_n: int = 0
    prompt_n: int = 0
    prefill_ms: float = 0.0
    cold_large: int = 0
    cold_large_ms: float = 0.0
    fp_calls: int = 0
    missed_reuse: int = 0
    missed_prefill_ms: float = 0.0
    missed_tokens: float = 0.0
    concurrent_dupes: int = 0

    def add(self, other: "Agg") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def summary(self) -> dict[str, Any]:
        total = self.cache_n + self.prompt_n
        prefill_s = self.prefill_ms / 1000.0
        return {
            "calls": self.calls,
            "prompt_tokens": total,
            "hit_tok": round(self.cache_n / total, 4) if total else None,
            "prefill_s": round(prefill_s, 3),
            "cold_large": self.cold_large,
            "cold_large_s": round(self.cold_large_ms / 1000.0, 3),
            "cold_large_share": round(self.cold_large_ms / self.prefill_ms, 4) if self.prefill_ms else None,
            "fp_coverage": round(self.fp_calls / self.calls, 4) if self.calls else None,
            "missed_reuse": self.missed_reuse,
            "missed_tokens": int(self.missed_tokens),
            "missed_prefill_s": round(self.missed_prefill_ms / 1000.0, 3),
            "missed_prefill_share": (
                round(self.missed_prefill_ms / self.prefill_ms, 4) if self.prefill_ms else None
            ),
            "concurrent_dupes": self.concurrent_dupes,
        }


def analyse(calls: list[Call], window_s: float) -> dict[tuple[int, str], Agg]:
    """Aggregate per (port, role). Calls must be sorted by start time."""
    aggs: dict[tuple[int, str], Agg] = defaultdict(Agg)
    # per port: fingerprint value -> (last start ts, last end ts) of a call carrying it
    seen: dict[int, dict[str, tuple[float, float]]] = defaultdict(dict)
    for call in calls:
        agg = aggs[(call.port, call.role)]
        agg.calls += 1
        agg.cache_n += call.cache_n
        agg.prompt_n += call.prompt_n
        agg.prefill_ms += call.prompt_ms
        if call.prompt_n >= COLD_NEW_TOKENS and call.cache_n < COLD_CACHED_MAX:
            agg.cold_large += 1
            agg.cold_large_ms += call.prompt_ms
        if not call.fp:
            continue
        agg.fp_calls += 1
        port_seen = seen[call.port]
        for depth, value in _depths(call.fp):
            prior = port_seen.get(f"{depth}:{value}")
            if prior is None or call.ts - prior[0] > window_s:
                continue
            implied = depth / CHARS_PER_TOKEN
            if prior[1] > call.ts:
                agg.concurrent_dupes += 1  # same prefix still in flight elsewhere: a D5 case
            if call.cache_n < MISSED_REUSE_FRACTION * implied:
                missed = implied - call.cache_n
                agg.missed_reuse += 1
                agg.missed_tokens += missed
                if call.prompt_n > 0:
                    agg.missed_prefill_ms += call.prompt_ms * min(1.0, missed / call.prompt_n)
            break  # only the deepest shared depth counts
        for depth, value in _depths(call.fp):
            port_seen[f"{depth}:{value}"] = (call.ts, call.ts_end)
    return aggs


def load_calls(paths: list[Path], since: float | None, until: float | None, port: int | None) -> list[Call]:
    calls = []
    for rec in iter_records(paths):
        call = to_call(rec)
        if call is None:
            continue
        if since is not None and call.ts < since:
            continue
        if until is not None and call.ts >= until:
            continue
        if port is not None and call.port != port:
            continue
        calls.append(call)
    calls.sort(key=lambda c: c.ts)
    return calls


def report(calls: list[Call], window_s: float) -> dict[str, Any]:
    aggs = analyse(calls, window_s)
    by_port: dict[int, Agg] = defaultdict(Agg)
    for (port, _role), agg in aggs.items():
        by_port[port].add(agg)
    return {
        "ports": {str(p): a.summary() for p, a in sorted(by_port.items())},
        "roles": {f"{p}:{r}": a.summary() for (p, r), a in sorted(aggs.items())},
    }


def _delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for port in sorted(set(before["ports"]) | set(after["ports"])):
        b, a = before["ports"].get(port, {}), after["ports"].get(port, {})
        row = {}
        for key in ("hit_tok", "cold_large_share", "missed_prefill_share"):
            if b.get(key) is not None and a.get(key) is not None:
                row[key] = round(a[key] - b[key], 4)
        out[port] = row
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--since")
    ap.add_argument("--until")
    ap.add_argument("--split-at", help="ISO-UTC boundary: report before / after and deltas")
    ap.add_argument("--port", type=int)
    ap.add_argument("--window-min", type=float, default=60.0)
    ap.add_argument("--json", type=Path, help="write the full report here")
    args = ap.parse_args(argv)

    paths = shard_paths(args.log)
    if not paths:
        print(f"no serving-call log at {args.log}", file=sys.stderr)
        return 2
    window_s = args.window_min * 60.0
    calls = load_calls(paths, parse_ts(args.since), parse_ts(args.until), args.port)
    out: dict[str, Any] = {
        "schema": "epyc.orchestrator.prefix_cache_report.v1",
        "log": [str(p) for p in paths],
        "window_min": args.window_min,
        "directions": {
            "hit_tok": "higher_better",
            "prefill_s": "lower_better",
            "cold_large_share": "lower_better",
            "missed_prefill_share": "lower_better",
        },
    }
    if args.split_at:
        split = parse_ts(args.split_at)
        if split is None:
            print("--split-at is not ISO-8601", file=sys.stderr)
            return 2
        out["before"] = report([c for c in calls if c.ts < split], window_s)
        out["after"] = report([c for c in calls if c.ts >= split], window_s)
        out["delta_after_minus_before"] = _delta(out["before"], out["after"])
    else:
        out.update(report(calls, window_s))

    text = json.dumps(out, indent=2, sort_keys=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
