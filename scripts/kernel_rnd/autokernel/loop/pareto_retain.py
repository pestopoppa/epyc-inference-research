"""AKX-PARETO-RETAIN: keep the per-workload Pareto front of candidates that missed the bar.

Intake-2669 s3.5: pick the Pareto front over (quality-delta, speed), then apply the quality
bar. A candidate that missed the keep bar but is non-dominated on its workload is still
worth trying on OTHER workloads before fresh generation. Both axes are higher-is-better
(``quality_delta`` = candidate minus anchor quality, ``speed`` = speedup/throughput ratio).
Pure over records; the JSONL store helpers are the only I/O.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA = "epyc.autokernel.pareto_retained.v1"
STORE_NAME = "pareto-retain.jsonl"


def _num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    ge = a["quality_delta"] >= b["quality_delta"] and a["speed"] >= b["speed"]
    return ge and (a["quality_delta"] > b["quality_delta"] or a["speed"] > b["speed"])


def non_dominated(rows: Iterable[Mapping[str, Any]]) -> list[dict]:
    rows = [dict(r) for r in rows]
    return [r for r in rows if not any(_dominates(o, r) for o in rows if o is not r)]


def retain(existing: Sequence[Mapping[str, Any]], candidates: Iterable[Mapping[str, Any]], *,
           campaign_id: str, campaign_seq: int) -> list[dict]:
    """Return the new retained set after offering ``candidates`` (this campaign).

    A candidate is retained only if it MISSED the keep bar (``kept`` falsy; kept ones are
    already in the champion path), has finite axes, and is non-dominated on its workload
    among retained rows of the same campaign. Other campaigns' rows pass through."""
    fresh = []
    for c in candidates:
        if c.get("kept") or not _num(c.get("quality_delta")) or not _num(c.get("speed")) \
                or not c.get("workload_id") or not c.get("candidate_id"):
            continue
        fresh.append({"schema": SCHEMA, "workload_id": c["workload_id"],
                      "candidate_id": c["candidate_id"], "patch_id": c.get("patch_id"),
                      "campaign_id": campaign_id, "campaign_seq": campaign_seq,
                      "quality_delta": c["quality_delta"], "speed": c["speed"]})
    keep = [dict(r) for r in existing if r.get("campaign_seq") != campaign_seq]
    same = [dict(r) for r in existing if r.get("campaign_seq") == campaign_seq] + fresh
    seen, merged = set(), []
    for wid in sorted({r["workload_id"] for r in same}):
        for r in non_dominated(x for x in same if x["workload_id"] == wid):
            if (wid, r["candidate_id"]) not in seen:
                seen.add((wid, r["candidate_id"]))
                merged.append(r)
    return keep + merged


def front(records: Sequence[Mapping[str, Any]], *, campaigns: int = 2,
          workload_id: str | None = None) -> dict[str, list[dict]]:
    """Census query: the per-workload front over the last ``campaigns`` campaigns."""
    last = sorted({r["campaign_seq"] for r in records}, reverse=True)[:campaigns]
    out: dict[str, list[dict]] = {}
    for wid in sorted({r["workload_id"] for r in records}):
        if workload_id is not None and wid != workload_id:
            continue
        rows = non_dominated(r for r in records
                             if r["workload_id"] == wid and r["campaign_seq"] in last)
        if rows:
            out[wid] = sorted(rows, key=lambda r: (-r["speed"], -r["quality_delta"],
                                                   r["candidate_id"]))
    return out


def retained_for_new_workload(records: Sequence[Mapping[str, Any]], workload_id: str, *,
                              campaigns: int = 2) -> list[dict]:
    """Candidates retained on OTHER workloads, best first, one per candidate id."""
    seen, out = set(), []
    for wid, rows in front(records, campaigns=campaigns).items():
        if wid == workload_id:
            continue
        for r in rows:
            if r["candidate_id"] not in seen:
                seen.add(r["candidate_id"])
                out.append(r)
    return sorted(out, key=lambda r: (-r["speed"], -r["quality_delta"], r["candidate_id"]))


def screen_order(candidates: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]],
                 workload_id: str, *, campaigns: int = 2) -> list[dict]:
    """Order a cross-workload screen: retained-front candidates first, then the rest
    (fresh generation) in their given order. Candidates carry ``candidate_id``."""
    rank = {r["candidate_id"]: i for i, r in
            enumerate(retained_for_new_workload(records, workload_id, campaigns=campaigns))}
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda p: (0, rank[p[1].get("candidate_id")])
                 if p[1].get("candidate_id") in rank else (1, p[0]))
    return [{**dict(c), "pareto_retained": c.get("candidate_id") in rank} for _, c in indexed]


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("schema") == SCHEMA:
            rows.append(row)
    return rows


def save(path: Path, records: Iterable[Mapping[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records),
                   encoding="utf-8")
    tmp.replace(path)
    return path


__all__ = ["SCHEMA", "STORE_NAME", "front", "load", "non_dominated", "retain",
           "retained_for_new_workload", "save", "screen_order"]
