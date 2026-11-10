#!/usr/bin/env python3
"""AGG-0.2 L3-only M1 report (informational; stdlib only, no model, no parsing).

Reads the corpus store a finished ``l3_load.py`` run left behind and compares the
``facts`` table (L3: parsed tables + regex spans) with the gold evidence cells of
``facts_sheet.csv`` (rows with ``status == filled``). Same match rule as the
predeclared M1: same doc, field, period; value within 0.5% relative after
conversion to USD millions.

This is NOT the Phase-0 M1 kill-gate reading: PREDECLARED.md computes M1 after
L4d, and the gate trips below 0.70 only then. The L3-only number is the floor L4
has to lift. When no sheet row is ``filled`` (today's state: AGG-0.1 hand-fill is
open) M1 is reported as ``null`` with ``gold_cells == 0``, never as 0.0, and the
per-cell candidate coverage (does ANY L3 fact exist for the doc/metric/period)
is still reported so the run is not empty.

Values are compared by absolute value: cash-flow outflows print as ``(1,234)``
in tables while the sheet stores capex/dividends positive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent

SCHEMA = "agg_l3_m1_report.v1"
REL_TOL = 0.005
M1_KILL_MIN = 0.70  # mirrored from PREDECLARED.md / agg_phase0.KILL["m1_min"]; applies after L4d only

# sheet metric -> L3 canonical field names (facts_det.FIELD_ALIASES targets)
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "revenue": ("revenue",),
    "net_income": ("net_income", "net_income_attributable"),
    "capex": ("capex",),
    "total_assets": ("total_assets",),
    "operating_cash_flow": ("operating_cash_flow",),
    "total_current_assets": ("total_current_assets",),
    "total_current_liabilities": ("total_current_liabilities",),
    "dividends_paid": ("dividends_paid",),
}
# unit label (facts.unit) -> multiplier to USD millions; "unknown" is assumed millions and counted
TO_MILLIONS = {"USD_millions": 1.0, "USD_thousands": 1e-3, "USD_billions": 1e3, "USD": 1e-6,
               "unknown": 1.0}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_gold(sheet: Path) -> list[dict[str, Any]]:
    out = []
    with sheet.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"].strip().lower() != "filled" or not r["value_usd_m"].strip():
                continue
            out.append({"doc_id": r["doc_name"], "metric": r["metric"],
                        "period": str(r["fiscal_year"]).strip(), "value": float(r["value_usd_m"])})
    return out


def sheet_cells(sheet: Path) -> list[tuple[str, str, str]]:
    with sheet.open(newline="") as fh:
        return [(r["doc_name"], r["metric"], str(r["fiscal_year"]).strip()) for r in csv.DictReader(fh)]


def load_facts(store: Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT doc_id, field, period, value, unit FROM facts WHERE value IS NOT NULL").fetchall()
    finally:
        conn.close()
    return [dict(zip(("doc_id", "field", "period", "value", "unit"), r)) for r in rows]


def report(store: Path, sheet: Path) -> dict[str, Any]:
    facts = load_facts(store)
    idx: dict[tuple[str, str, str], list[float]] = {}
    unit_counts: dict[str, int] = {}
    for f in facts:
        unit_counts[f["unit"] or "none"] = unit_counts.get(f["unit"] or "none", 0) + 1
        mult = TO_MILLIONS.get(f["unit"])
        if mult is None or f["period"] is None:  # percent / unit-less: cannot be a USD cell
            continue
        idx.setdefault((f["doc_id"], f["field"], str(f["period"])), []).append(abs(f["value"]) * mult)

    def candidates(doc: str, metric: str, period: str) -> list[float]:
        return [v for fld in METRIC_FIELDS[metric] for v in idx.get((doc, fld, period), [])]

    gold = load_gold(sheet)
    per_metric: dict[str, dict[str, int]] = {}
    per_doc: dict[str, dict[str, int]] = {}
    found = 0
    for g in gold:
        gv = abs(g["value"])
        hit = any(abs(v - gv) <= max(gv * REL_TOL, 1e-9) for v in candidates(g["doc_id"], g["metric"], g["period"]))
        found += hit
        for bucket, key in ((per_metric, g["metric"]), (per_doc, g["doc_id"])):
            b = bucket.setdefault(key, {"gold": 0, "found": 0})
            b["gold"] += 1
            b["found"] += int(hit)
    cells = sheet_cells(sheet)
    with_cand = sum(1 for c in cells if candidates(*c))
    m1 = (found / len(gold)) if gold else None
    return {
        "schema": SCHEMA,
        "kind_note": "L3-only, informational; the Phase-0 M1 kill gate is read after L4d",
        "store": str(store),
        "facts_sheet_sha256": sha256_file(sheet),
        "n_facts": len(facts),
        "unit_counts": dict(sorted(unit_counts.items())),
        "unit_unknown_assumed_millions": unit_counts.get("unknown", 0),
        "gold_cells": len(gold),
        "found": found,
        "m1_l3_only": m1,
        "m1_status": "no_gold_cells_filled" if not gold else "computed",
        "m1_kill_min_after_l4d": M1_KILL_MIN,
        "l3_below_kill_min": (m1 is not None and m1 < M1_KILL_MIN),
        "sheet_cells": len(cells),
        "cells_with_any_l3_candidate": with_cand,
        "candidate_coverage": (with_cand / len(cells)) if cells else None,
        "per_metric": per_metric,
        "per_doc": per_doc,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--sheet", type=Path, default=HERE / "facts_sheet.csv")
    ap.add_argument("--out", type=Path, default=None, help="also write the report JSON here")
    a = ap.parse_args(argv)
    if not a.store.exists():
        print(f"store missing: {a.store}", file=sys.stderr)
        return 2
    rep = report(a.store, a.sheet)
    text = json.dumps(rep, indent=1, sort_keys=True)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
