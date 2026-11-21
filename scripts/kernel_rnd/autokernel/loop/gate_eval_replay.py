#!/usr/bin/env python3
"""Replay gate_eval over filed AutoKernel verdicts and measure agreement (AK parallel loop P1).

Read-only over the compute-flags root: filing-done/<row>.verdict.json (owner, filed_at, row_verdict,
end_line, result_path), the row's out/RESULT.json, ready/*.json (successor `supersedes` links).

Lifecycle facts come from rowflow state, not from the human verdict (ready/.withdrawn/ is NOT a
signal: filed manifests of every kind are archived there). A row that produced no completed RESULT
(no RESULT, or its run did not end rc 0 complete) is SUPERSEDED when a successor
exists (a manifest names it in `supersedes`, the -rN naming convention, or an ended reason
`superseded…`), else WITHDRAWN when it ended rc 75 with an owner-written reason.

  python3 -m autokernel.loop.gate_eval_replay [--since 2026-11-06] [--owner autokernel] [--json OUT]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from . import gate_eval

FLAGS = Path("/mnt/raid0/llm/tmp/compute-flags")
TMP = Path("/mnt/raid0/llm/tmp")
_NORM = {"bit_exact_pass": "pass", "partial": "inconclusive", "abstain": "inconclusive"}
#: Ended reasons the launcher/keeper writes itself (not an owner's lifecycle decision).
_MACHINE_REASONS = re.compile(r"^(complete|stopped|killed|timeout|partial|crash|grant|window|smoke|"
                              r"released-early|stale|keeper)", re.I)


def norm(v: Any) -> str:
    s = str(v or "").strip()
    return s.upper() if s.upper() in ("SUPERSEDED", "WITHDRAWN") else _NORM.get(s.lower(), s.lower())


def successors(flags: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in sorted((flags / "ready").glob("*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sup = m.get("supersedes") if isinstance(m, dict) else None
        for s in ([sup] if isinstance(sup, str) else sup or []):
            if isinstance(s, str) and s:
                out.setdefault(s, str(m.get("id") or p.stem))
    return out


_REV = re.compile(r"^(?P<base>.+?)(?:-r(?P<n>\d+)(?P<tail>q\d+)?)?$")


def known_ids(flags: Path) -> set[str]:
    ids = {p.stem for p in (flags / "ready").glob("*.json")}
    ids |= {p.name[: -len(".ended")] for p in (flags / "grants").glob("*.ended")}
    return ids


def successor_by_name(row: str, ids: set[str]) -> str | None:
    """`x` -> `x-r2`, `x-r3` -> `x-r4`… (the rowflow -rN supersede convention), when it exists."""
    m = _REV.match(row)
    base, n = m.group("base"), int(m.group("n") or 1)
    for k in range(n + 1, n + 10):
        for cand in (f"{base}-r{k}", f"{base}-r{k}{m.group('tail') or ''}"):
            if cand in ids:
                return cand
    return None


def result_for(rec: dict[str, Any]) -> tuple[Path | None, Any]:
    cands = [TMP / rec["id"] / "out" / "RESULT.json"]
    rp = rec.get("result_path") or ""
    if rp:
        cands.insert(0, Path(rp).with_name("RESULT.json"))
        if rp.endswith(".json"):
            cands.append(Path(rp))
    for c in cands:
        if c.is_file():
            try:
                return c, json.loads(c.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return c, None
    return None, None


def lifecycle(rec: dict[str, Any], ended: dict[str, str], succ: dict[str, str], has_result: bool,
              ids: set[str] | None = None) -> dict[str, Any]:
    """Rowflow facts for gate_eval's no-result rule (operator ruling 2026-11-21): the successor row id,
    if any. gate_eval decides SUPERSEDED (successor) / WITHDRAWN (none) for a row with no finished RESULT."""
    reason = ended.get("rc") and ended.get("reason", "") or ""
    if has_result and ended.get("rc", "0") == "0":     # rc 0 with a RESULT = finished (any end reason)
        return {}
    nxt = succ.get(rec["id"]) or successor_by_name(rec["id"], ids or set())
    if nxt or re.match(r"superseded", reason, re.I):
        return {"successor": nxt or reason}
    return {}


def lifecycle_legacy(rec: dict[str, Any], ended: dict[str, str], succ: dict[str, str], has_result: bool,
                     ids: set[str] | None = None) -> dict[str, Any]:
    """Only a row that produced no verdict-bearing run (no RESULT, or it did not end rc 0 complete)
    is SUPERSEDED / WITHDRAWN; a completed row keeps its own measured verdict even if re-run."""
    reason = ended.get("reason", "")
    completed = has_result and ended.get("rc", "0") == "0" and reason in ("", *gate_eval.COMPLETE_REASONS)
    if completed:
        return {}
    nxt = succ.get(rec["id"]) or successor_by_name(rec["id"], ids or set())
    if nxt or re.match(r"superseded", reason, re.I):
        return {"superseded_by": nxt or reason}
    if ended.get("rc") == "75" and reason and not _MACHINE_REASONS.match(reason):
        return {"withdrawn": reason[:120]}
    return {}


def load_records(flags: Path, since: str, owner: str) -> list[dict[str, Any]]:
    out = []
    for p in sorted((flags / "filing-done").glob("*.verdict.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if d.get("owner") == owner and str(d.get("filed_at", "")) >= since:
            out.append(d)
    return out


_LIFE = ("SUPERSEDED", "WITHDRAWN")


def ruled_label(human: str, v: dict[str, Any]) -> str:
    """The human label re-read under the 2026-11-21 filing rule: a no-result row's correct label is the
    rule's (SUPERSEDED with a successor, else WITHDRAWN); a lifecycle label on a row that RAN to a finished
    RESULT means "ran but could not decide" = inconclusive; a pass on a declared DROP is inconclusive under
    FILER-DROP-AS-PASS-GUARD (epyc-root bba1c010a, file_result.coerce_declared_nonpass). Everything else
    keeps the owner's label."""
    if v.get("gate_id") == gate_eval.NO_CLAIM_GATE:
        return v["filed_verdict"]
    if human == "pass" and re.match(r"^\s*DROP\s*\(\s*diagnostic", str(v.get("native_verdict") or ""), re.I):
        return "inconclusive"
    if human in _LIFE and v.get("verdict") not in _LIFE:
        return "inconclusive"
    return human


def load_categories(md: Path | None) -> dict[str, str]:
    """row -> disagreement category from the P1 replay report's 'Every disagreement' table."""
    out: dict[str, str] = {}
    if md and md.is_file():
        for line in md.read_text(encoding="utf-8").splitlines():
            m = re.match(r"^\| `([^`]+)` \|.*\| (A|B2|B|C|D|E|F|G|H) \|$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def _agree_by(rows: list[dict[str, Any]], key: str, field: str) -> dict[str, str]:
    groups: dict[str, list[bool]] = collections.defaultdict(list)
    for r in rows:
        groups[r[key]].append(r[field])
    return {g: f"{sum(v)}/{len(v)}" for g, v in sorted(groups.items())}


def replay(flags: Path = FLAGS, since: str = "2026-11-06", owner: str = "autokernel",
           categories: dict[str, str] | None = None, legacy: bool = False) -> dict[str, Any]:
    succ = successors(flags)
    ids = known_ids(flags)
    policy = gate_eval.load_policy()
    consts = gate_eval._constants()
    rows = []
    for rec in load_records(flags, since, owner):
        ended = gate_eval.parse_ended(rec.get("end_line") or "")
        path, result = result_for(rec)
        life = (lifecycle_legacy if legacy else lifecycle)(rec, ended, succ, isinstance(result, dict), ids)
        v = gate_eval.evaluate(result if isinstance(result, dict) else None, ended=ended, row=rec["id"],
                               policy=policy, lifecycle=life, consts=consts)
        human = norm(rec.get("row_verdict"))
        ruled = ruled_label(human, v)
        rows.append({"row": rec["id"], "human": human, "machine": v["filed_verdict"], "ruled": ruled,
                     "agree_ruled": ruled == v["filed_verdict"], "policy_decided": v.get("policy_decided"),
                     "group": (categories or {}).get(rec["id"], "agreed-in-P1"),
                     "verdict": v["verdict"], "gate_id": v["gate_id"], "layer": v["layer"],
                     "native": (v.get("native_verdict") or "")[:100], "result": str(path) if path else None,
                     "agree": human == v["filed_verdict"], "filed_at": rec.get("filed_at"),
                     "checks": v["checks"], "human_reason": str(rec.get("reason") or "")[:240]})
    n = len(rows)
    agree = sum(r["agree"] for r in rows)
    confusion = collections.Counter((r["human"], r["machine"]) for r in rows)
    agree_r = sum(r["agree_ruled"] for r in rows)
    pd = [r for r in rows if r["policy_decided"]]
    return {"schema": gate_eval.SCHEMA + ".replay", "since": since, "owner": owner, "n": n,
            "agree": agree, "agreement": round(agree / n, 4) if n else None,
            "agree_ruled": agree_r, "agreement_ruled": round(agree_r / n, 4) if n else None,
            "by_group_strict": _agree_by(rows, "group", "agree"),
            "by_group_ruled": _agree_by(rows, "group", "agree_ruled"),
            "by_human_ruled": _agree_by(rows, "ruled", "agree_ruled"),
            "policy_decided": f"{sum(r['agree_ruled'] for r in pd)}/{len(pd)}",
            "disagreements_ruled": [r for r in rows if not r["agree_ruled"]],
            "by_layer": dict(collections.Counter(r["layer"] for r in rows)),
            "policy_gate_rows": sum(1 for r in rows if r["checks"]),
            "confusion": {f"{h}->{m}": c for (h, m), c in sorted(confusion.items())},
            "disagreements": [r for r in rows if not r["agree"]], "rows": rows}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flags", type=Path, default=FLAGS)
    ap.add_argument("--since", default="2026-11-06")
    ap.add_argument("--owner", default="autokernel")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--categories", type=Path, help="P1 replay report (.md) whose disagreement table names groups")
    ap.add_argument("--legacy", action="store_true", help="pre-ruling lifecycle inference (the P1 baseline)")
    a = ap.parse_args(argv)
    r = replay(a.flags, a.since, a.owner, load_categories(a.categories), a.legacy)
    if a.json:
        a.json.write_text(json.dumps(r, indent=1, sort_keys=True), encoding="utf-8")
    print(f"replayed {r['n']} filed verdicts (owner {a.owner}, since {a.since}): agree {r['agree']} "
          f"= {100 * (r['agreement'] or 0):.1f}%; policy-gated rows {r['policy_gate_rows']}; layers {r['by_layer']}")
    print(f"ruled agreement (2026-11-21 filing rule): {r['agree_ruled']}/{r['n']} = "
          f"{100 * (r['agreement_ruled'] or 0):.1f}%; policy-decided rows agreeing {r['policy_decided']}")
    print(f"  by group strict {r['by_group_strict']}")
    print(f"  by group ruled  {r['by_group_ruled']}")
    for k, c in r["confusion"].items():
        print(f"  {k:32s} {c}")
    for d in r["disagreements"]:
        print(f"  DISAGREE {d['row']}: human {d['human']} machine {d['machine']} ({d['gate_id']}) native={d['native'][:60]!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
