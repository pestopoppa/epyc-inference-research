#!/usr/bin/env python3
"""AGG-1.16b (agg113/PREDECLARED_AGG116B.md): v3 vs v2 on the long_term_debt golds of the AGG-1.13 set. OFFLINE, no model.

Reads the two stores written by ``agg113v3.py --phase replay`` (store.v2check.sqlite re-canonicalised under v2,
store.v3.sqlite = v2 + harvest delta under canonical.v3) read-only, and scores every filled gold cell with the frozen
``agg1314_zero_cell_diag.diagnose_cell`` HIT rule (raw) and the canonical row for (doc, field, period == fy).

    python3 agg116b.py --replay-dir <agg113v3 replay out> --out agg116b_result.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import agg1314_zero_cell_diag as Z  # noqa: E402

GOLDS = HERE / "agg113" / "ltd_golds_agg116b.csv"
PREDECLARED = HERE / "agg113" / "PREDECLARED_AGG116B.md"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def canon_row(conn: Any, doc: str, field: str, fy: str) -> dict[str, Any] | None:
    r = conn.execute("SELECT value, unit, period, rule FROM facts_canonical WHERE doc_id = ? AND field = ? AND period = ?",
                     [doc, field, fy]).fetchone()
    return dict(r) if r else None


def score(store: Path, cells: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    conn = Z._ro(store)
    try:
        out = {}
        for c in cells:
            d = Z.diagnose_cell(c, Z.facts_for(conn, c["doc"], c["field"]))
            cr = canon_row(conn, c["doc"], c["field"], c["fy"])
            ok = cr is not None and cr["value"] is not None and Z._close(float(cr["value"]), c["gold"], Z.TOL)
            out[f"{c['doc']}|{c['field']}|{c['fy']}"] = {"raw_hit": d["verdict"] == "HIT", "raw_verdict": d["verdict"],
                                                       "raw_candidate": d["candidate"], "canonical_ok": ok, "canonical_row": cr}
        return out
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = list(csv.DictReader(GOLDS.open(newline="")))
    cells = [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"]), "fy": r["fiscal_year"]}
             for r in rows if r["status"] == "filled"]
    excluded = [{"cell": f"{r['doc_name']}|{r['metric']}|{r['fiscal_year']}", "reason": r["note"]} for r in rows if r["status"] != "filled"]
    s2 = score(a.replay_dir / "store.v2check.sqlite", cells)
    s3 = score(a.replay_dir / "store.v3.sqlite", cells)
    per = {k: {"gold": c["gold"], "v2": s2[k], "v3": s3[k]} for k, c in zip(s2, cells)}
    tot: dict[str, Any] = {}
    for m, key in (("raw", "raw_hit"), ("canonical", "canonical_ok")):
        for fy in ("2022", "2021", "all"):
            ks = [k for k in per if fy == "all" or k.endswith("|" + fy)]
            tot[f"{m}_{fy}"] = {"n": len(ks), "v2": sum(per[k]["v2"][key] for k in ks), "v3": sum(per[k]["v3"][key] for k in ks),
                                "fixes": sorted(k for k in ks if per[k]["v3"][key] and not per[k]["v2"][key]),
                                "regressions": sorted(k for k in ks if per[k]["v2"][key] and not per[k]["v3"][key])}
    rep = json.loads((a.replay_dir / "result.replay.json").read_text())
    m1 = rep.get("m1", {})
    reg76 = {"raw": {"v2": m1["raw"]["v2"], "v3": m1["raw"]["v3"], "regressions": m1["raw"]["regressions"]},
             "canonical": {"v2": m1["canonical"]["v2"], "v3": m1["canonical"]["v3"], "regressions": m1["canonical"]["regressions"]},
             "replay_valid": rep.get("replay_valid"), "verdict": rep.get("verdict")}
    no_reg = not any(t["regressions"] for t in tot.values()) and not reg76["raw"]["regressions"] and not reg76["canonical"]["regressions"]
    ge = tot["raw_all"]["v3"] >= tot["raw_all"]["v2"] and tot["canonical_all"]["v3"] >= tot["canonical_all"]["v2"]
    res = {"task": "AGG-1.16b long_term_debt out-of-sample v3 vs v2 (offline, no model)", "model_calls": 0, "label": "post-data",
           "predeclared_sha256": sha(PREDECLARED), "golds_sha256": sha(GOLDS),
           "stores_sha256": {p: sha(a.replay_dir / p) for p in ("store.v2check.sqlite", "store.v3.sqlite")},
           "replay_dir": str(a.replay_dir), "n_filled": len(cells), "excluded": excluded, "totals": tot, "regression_76": reg76,
           "per_cell": per,
           "verdict": ("v3_ge_v2_no_regressions" if ge and no_reg else "v3_regresses" if not no_reg else "v3_below_v2")}
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("n_filled", "totals", "regression_76", "verdict")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
