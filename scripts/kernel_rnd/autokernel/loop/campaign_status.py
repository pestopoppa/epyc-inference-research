#!/usr/bin/env python3
"""Campaign status publisher (AK-RUNEND-REPORT, narrowed): decides nothing.

Publishes per-stage state + ETA, idle bins, named escalations and ONE decision package
as an atomically written JSON at <flags>/campaign-status/<row-id>.json -- the keeper's
existing status surface (queue_status.py) reads it beside the grant. No second report.

Input is a plain snapshot mapping (the shape the scheduler/heartbeat status already
publishes, flattened to JSON):
  now                       epoch seconds
  rows[]   {id, state: RUNNING|BACKLOG|..., started_at, eta_s | (progress, elapsed_s), heartbeat_at}
  bins[]   {name, idle_since, idle_until?, reason?}   idle spans of a compute bin
  escalations[] {id, text, decision?}  decision = the ONE operator-only decision, one sentence
  judgment_calls[] {topic, options[{name, tradeoff, criteria{crit: PASS|FAIL}}], recommendation}

Outputs: per-running-row heartbeat/ETA lines; idle-bin summary (idle > 10 min while a
row was BACKLOG); escalations only when they name a decision (others are downgraded
to `dropped`); judgment calls as ONE decision package, recommendation first.

CLI (read-only render):  python -m scripts.kernel_rnd.autokernel.loop.campaign_status SNAPSHOT.json [--json]
Publish: publish(flags_dir, row_id, snapshot).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

IDLE_MIN_S = 600.0
STALE_HEARTBEAT_S = 900.0


def _hms(s: float) -> str:
    s = max(0, int(s))
    return f"{s // 3600}h{s % 3600 // 60:02d}m"


def row_lines(snapshot: Mapping[str, Any]) -> list[str]:
    now = float(snapshot.get("now", 0))
    out = []
    for r in snapshot.get("rows", []):
        if r.get("state") != "RUNNING":
            continue
        eta = r.get("eta_s")
        if eta is None and r.get("progress") and r.get("elapsed_s"):
            p = float(r["progress"])
            eta = float(r["elapsed_s"]) * (1 - p) / p if 0 < p < 1 else None
        hb = r.get("heartbeat_at")
        age = now - float(hb) if hb is not None else None
        flag = " STALE-HEARTBEAT" if age is not None and age > STALE_HEARTBEAT_S else ""
        out.append(f"{r['id']}: running {_hms(now - float(r['started_at'])) if r.get('started_at') is not None else '?'}, "
                   f"ETA {_hms(eta) if eta is not None else 'unknown'}, "
                   f"heartbeat {_hms(age) + ' ago' if age is not None else 'none'}{flag}")
    return out


def idle_bins(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    now = float(snapshot.get("now", 0))
    backlog = [r["id"] for r in snapshot.get("rows", []) if r.get("state") == "BACKLOG"]
    if not backlog:
        return []  # idle with nothing queued is not unused compute
    out = []
    for b in snapshot.get("bins", []):
        end = float(b["idle_until"]) if b.get("idle_until") is not None else now
        dur = end - float(b["idle_since"])
        if dur > IDLE_MIN_S:
            out.append({"bin": b["name"], "idle_s": dur, "reason": b.get("reason") or "unstated",
                        "backlog_rows": backlog})
    return out


def escalations(snapshot: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Admission test: an escalation must name the operator-only decision."""
    kept, dropped = [], []
    for e in snapshot.get("escalations", []):
        decision = (e.get("decision") or "").strip()
        if decision:
            kept.append({"id": e.get("id"), "decision": decision, "text": e.get("text", "")})
        else:
            dropped.append(str(e.get("id")))
    return kept, dropped


def decision_package(snapshot: Mapping[str, Any]) -> dict[str, Any] | None:
    calls = snapshot.get("judgment_calls", [])
    if not calls:
        return None
    items = []
    for c in calls:
        crit = sorted({k for o in c.get("options", []) for k in o.get("criteria", {})})
        items.append({
            "topic": c["topic"],
            "recommendation": c.get("recommendation") or "none stated",
            "options": [{"name": o["name"], "tradeoff": o.get("tradeoff", ""),
                         "criteria": {k: o.get("criteria", {}).get(k, "n/a") for k in crit}}
                        for o in c.get("options", [])],
        })
    return {"decided_by": "operator", "items": items}


def build_report(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    esc, dropped = escalations(snapshot)
    return {"rows": row_lines(snapshot), "idle_bins": idle_bins(snapshot),
            "escalations": esc, "escalations_dropped_no_decision": dropped,
            "decision_package": decision_package(snapshot)}


SCHEMA = "epyc.autokernel.campaign_status.v1"


def publish(flags_dir: str | Path, row_id: str, snapshot: Mapping[str, Any]) -> Path:
    """Atomically write the payload next to the grants; the only write in this module."""
    if not row_id or "/" in row_id or row_id.startswith("."):
        raise ValueError(f"bad row id {row_id!r}")
    d = Path(flags_dir) / "campaign-status"
    d.mkdir(parents=True, exist_ok=True)
    payload = {"schema": SCHEMA, "row_id": row_id, "as_of": snapshot.get("now"), **build_report(snapshot)}
    tmp = d / f".{row_id}.json.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, d / f"{row_id}.json")
    return d / f"{row_id}.json"


def render(report: Mapping[str, Any]) -> str:
    L = []
    pkg = report["decision_package"]
    if pkg:
        L.append("DECISION PACKAGE (operator decides; items independent of it continue)")
        for it in pkg["items"]:
            L.append(f"  {it['topic']}  -- RECOMMEND: {it['recommendation']}")
            for o in it["options"]:
                c = ", ".join(f"{k}={v}" for k, v in o["criteria"].items())
                L.append(f"    - {o['name']}: {o['tradeoff']} [{c}]")
    for e in report["escalations"]:
        L.append(f"ESCALATION {e['id']}: operator decision needed: {e['decision']}")
    L.append("ROWS")
    L += [f"  {x}" for x in report["rows"]] or ["  (none running)"]
    L.append("IDLE BINS (> 10 min with rows in BACKLOG)")
    L += [f"  {b['bin']}: idle {_hms(b['idle_s'])}, reason: {b['reason']}, backlog {len(b['backlog_rows'])}"
          for b in report["idle_bins"]] or ["  (none)"]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AK campaign status render (read-only).")
    p.add_argument("snapshot", type=Path)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)
    rep = build_report(json.loads(a.snapshot.read_text()))
    print(json.dumps(rep, indent=2, sort_keys=True) if a.json else render(rep))
    return 0


__all__ = ["SCHEMA", "build_report", "publish", "decision_package", "escalations", "idle_bins", "main", "render", "row_lines"]

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
