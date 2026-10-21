#!/usr/bin/env python3
"""CJ-18 — judge-family affinity (discrete CAPA, plain error consistency) on the frozen
CJ-15 fixture. Zero inference: reads frozen JSON, builds no model client, no dependency.

Items are the per-criterion verdicts of the rows that carry per-judge typed verdicts
(`observed_outputs.<judge>.verdicts` or `.values`) against `expected_criterion_verdicts`.
Denominator rule: every item of every such row counts; an invalid/missing verdict is the
symbol INVALID, is wrong against gold, and never agrees with anything (two INVALIDs do not
share an answer). Judge-versus-generating-role pairs read `generator_outputs.<role>` from a
row when present; the frozen fixture carries none, so that table is reported empty.

  plain error consistency  EC = P(both right or both wrong)
  discrete CAPA            (c_obs - c_exp) / (1 - c_exp), c_obs = P(same valid output),
                           c_exp = p1*p2 + (1-p1)(1-p2)/(K-1), K = 2 (true/false verdicts)
  CAPA > 0 means the pair agrees beyond what their accuracies alone imply.

Bootstrap: percentile 95% interval over ROWS (the criteria of one row are not independent),
seeded, 2000 resamples.
"""

from __future__ import annotations

import itertools
import json
import random
from pathlib import Path
from typing import Any, Sequence

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "data" / "typed_decisions" / "cj15_judge_cascade_v1"
INVALID = "INVALID"
K = 2


def _sym(v: Any) -> Any:
    return v if isinstance(v, bool) else INVALID


def load_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Per-row {row_id, gold, outputs{name: [symbols]}} for rows with >=1 per-judge verdict list."""
    path = path or FIXTURE_DIR / "source_records.jsonl"
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        gold = r["expected_criterion_verdicts"]
        outs: dict[str, list[Any]] = {}
        for name, o in (r.get("observed_outputs") or {}).items():
            vec = o.get("verdicts") if o.get("verdicts") is not None else o.get("values")
            outs[name] = [_sym(x) for x in vec] if vec is not None else [INVALID] * len(gold)
        for name, vec in (r.get("generator_outputs") or {}).items():
            outs["role:" + name] = [_sym(x) for x in vec]
        if outs:
            rows.append({"row_id": r["row_id"], "gold": list(gold), "outputs": outs})
    return rows


def _stats(rows: Sequence[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    n = both = same = ra = rb = 0
    for r in rows:
        if a not in r["outputs"] or b not in r["outputs"]:
            continue
        for g, x, y in zip(r["gold"], r["outputs"][a], r["outputs"][b]):
            n += 1
            ca, cb = x == g, y == g
            ra += ca
            rb += cb
            both += ca == cb
            same += x == y and x != INVALID
    if n == 0:
        return {"n": 0, "ec": None, "capa": None, "acc_a": None, "acc_b": None}
    pa, pb = ra / n, rb / n
    c_obs = same / n
    c_exp = pa * pb + (1 - pa) * (1 - pb) / (K - 1)
    capa = None if c_exp >= 1 else (c_obs - c_exp) / (1 - c_exp)
    return {"n": n, "ec": both / n, "capa": capa, "acc_a": pa, "acc_b": pb, "c_obs": c_obs, "c_exp": c_exp}


def _ci(vals: list[float]) -> list[float] | None:
    if not vals:
        return None
    vals = sorted(vals)
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(round(0.975 * (len(vals) - 1)))]]


def affinity_table(rows: Sequence[dict[str, Any]], *, resamples: int = 2000, seed: int = 18) -> dict[str, Any]:
    names = sorted({k for r in rows for k in r["outputs"]})
    judges = [n for n in names if not n.startswith("role:")]
    roles = [n for n in names if n.startswith("role:")]
    pairs = list(itertools.combinations(judges, 2)) + [(j, r) for j in judges for r in roles]
    rng = random.Random(seed)
    table = []
    for a, b in pairs:
        point = _stats(rows, a, b)
        ec, capa = [], []
        for _ in range(resamples):
            s = _stats([rows[rng.randrange(len(rows))] for _ in rows], a, b)
            if s["ec"] is not None:
                ec.append(s["ec"])
            if s["capa"] is not None:
                capa.append(s["capa"])
        table.append({"a": a, "b": b, **point, "ec_ci95": _ci(ec), "capa_ci95": _ci(capa),
                      "capa_defined_resamples": len(capa), "resamples": resamples})
    per_judge = {n: sum(x == g for r in rows if n in r["outputs"] for g, x in zip(r["gold"], r["outputs"][n]))
                 for n in names}
    items = {n: sum(len(r["gold"]) for r in rows if n in r["outputs"]) for n in names}
    invalid = {n: sum(x == INVALID for r in rows if n in r["outputs"] for x in r["outputs"][n]) for n in names}
    return {"rows": len(rows), "judges": judges, "generating_roles": roles,
            "items_per_actor": items, "correct_per_actor": per_judge, "invalid_per_actor": invalid,
            "pairs": table}


def main() -> int:
    print(json.dumps(affinity_table(load_rows()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
