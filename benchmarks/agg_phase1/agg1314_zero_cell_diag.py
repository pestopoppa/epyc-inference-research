#!/usr/bin/env python3
"""AGG-1.3 r2 / AGG-1.4 r2 zero-cell diagnosis. No model call; every store is opened read-only (``mode=ro``).

Why this exists (2026-11-12). Two AGG runs scored exactly zero on the cells their NEW extraction should have produced:
  * AGG-1.3 r2: 0/24 on the three replaced FY2021 slots (``agg13.m1_hits_fy``: period must contain "2021");
  * AGG-1.4 r2: 0/9 on the backfilled documents (``phase1_arms.m1_hits``: period contains "2022", value within 0.5 %).
The cause is NOT in the scorers. It is two different configuration defects upstream of them, both visible in the saved
stores:
  * AGG-1.3: ``agg_phase0.METRICS`` bakes the fiscal year into every L4 field description ("..., FY2022 (USD millions)");
    the three replaced documents are FY2021 filings with no FY2022 column, so the model returns ``[]`` for most units
    and labels the four facts it does return ``FY2022`` -> a strict ``2021`` period filter can never match them.
  * AGG-1.4: for ``long_term_debt`` the model emitted ``unit: "USD"`` on every fact (13 of 13) while its ``value`` is the
    number printed in the filing in USD millions; ``facts_llm.reconcile`` trusts the label and scales by 1e-6, so
    2,467 is stored as 0.002467 USD_millions. The scorer's "no unit leniency" rule then (correctly, by its text) rejects it.

This module never changes a scorer, a gold, a gate or a pinned runner file. It provides:
  * ``diagnose_cell`` / ``diagnose`` -- per-cell classification of why a gold cell did not match, on a saved store;
  * ``fy_contract_problems`` -- a PREFLIGHT guard a re-run must pass: a field description must not name a fiscal year
    other than the one the gold cells are for;
  * ``scale_signature`` -- counts L4 facts carrying the 1e-6 mis-scale signature;
  * ``rescore`` -- strict (the frozen rule, bit-for-bit) next to DIAGNOSTIC variants. The diagnostic variants are NOT
    gate numbers: using them as gate numbers is a change of predeclared scorer semantics and needs a dated amendment.

CLI:  agg1314_zero_cell_diag.py 13|14 <out_dir> [--json PATH]
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
TOL = 0.005                      # the frozen rel_tol (phase0_arms.Cfg.rel_tol)
NEAR_TOL = 0.02                  # "right line, rounded source" band, diagnostic only
M1_GATE = 0.75
REPLACED_13 = ("MICROSOFT_2022_10K", "JOHNSON_JOHNSON_2022_10K", "3M_2022_10K")
UNIT_SCALE = {"USD_thousands": 1e-3, "USD_millions": 1.0, "USD_billions": 1e3, "USD": 1e-6}   # frozen M1R_UNIT_SCALE
CLASSES = ("HIT", "PERIOD_LABEL", "UNIT_SCALE", "NEAR_ROUNDED", "WRONG_VALUE", "NO_FACT")


def _ro(path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _close(a: float, b: float, tol: float) -> bool:
    return abs(abs(a) - abs(b)) <= tol * abs(b) + 1e-9


def load_cells13(cells_csv: Path) -> list[dict[str, Any]]:
    """Filled AGG-1.3 gold cells of the replaced slots (FY2021)."""
    return [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"].replace(",", "")),
             "fy": str(r["fiscal_year"])}
            for r in csv.DictReader(cells_csv.open(newline="")) if r["status"] == "filled" and r["value_usd_m"].strip()]


def load_cells14(cells_csv: Path) -> list[dict[str, Any]]:
    """Filled AGG-1.4 long_term_debt gold cells (FY2022)."""
    return [{"doc": r["doc_name"], "field": r["metric"], "gold": float(r["value_usd_m"].replace(",", "")),
             "fy": str(r["fiscal_year"])}
            for r in csv.DictReader(cells_csv.open(newline="")) if r["status"] == "filled" and r["value_usd_m"].strip()]


def facts_for(conn: sqlite3.Connection, doc: str, field: str) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(
        "SELECT period, value, unit, extractor, page, quote FROM facts WHERE doc_id = ? AND field = ?", [doc, field])]


def stored_value(f: dict[str, Any]) -> float:
    """The value the frozen scorer sees: the stored value times the scale of its stored unit."""
    return float(f["value"]) * UNIT_SCALE.get(f.get("unit"), 1.0)


def diagnose_cell(cell: dict[str, Any], facts: list[dict[str, Any]], *, tol: float = TOL) -> dict[str, Any]:
    """Why a gold cell did (not) match. Order of the verdicts is most- to least-recoverable:
    HIT (frozen rule), PERIOD_LABEL (value right, period lacks the cell's fiscal year), UNIT_SCALE (value right after a
    x1e6 / x1e3 / x1e-3 / x1e-6 correction), NEAR_ROUNDED (within NEAR_TOL), WRONG_VALUE, NO_FACT."""
    g, fy = cell["gold"], cell["fy"]
    rows = [f for f in facts if f.get("value") is not None]
    best = None
    verdict = "NO_FACT" if not rows else "WRONG_VALUE"
    for f in rows:
        v, has_fy = stored_value(f), fy in str(f.get("period") or "")
        rank = None
        if _close(v, g, tol):
            rank = ("HIT", 0) if has_fy else ("PERIOD_LABEL", 1)
        elif any(_close(v * k, g, tol) for k in (1e6, 1e3, 1e-3, 1e-6)):
            rank = ("UNIT_SCALE", 2)
        elif _close(v, g, NEAR_TOL):
            rank = ("NEAR_ROUNDED", 3)
        if rank and (best is None or rank[1] < best[0][1]):
            best = (rank, f, v)
    if best:
        verdict = best[0][0]
    pick = best[1] if best else (rows[0] if rows else None)
    return {"doc": cell["doc"], "field": cell["field"], "gold": g, "fy": fy, "verdict": verdict,
            "n_facts": len(rows),
            "candidate": None if pick is None else {"period": pick.get("period"), "stored_value": pick["value"],
                                                    "stored_unit": pick.get("unit"), "scored_value": stored_value(pick),
                                                    "extractor": pick.get("extractor"), "page": pick.get("page"),
                                                    "quote": (pick.get("quote") or "")[:90]}}


def diagnose(conn: sqlite3.Connection, cells: Iterable[dict[str, Any]], *, tol: float = TOL) -> list[dict[str, Any]]:
    return [diagnose_cell(c, facts_for(conn, c["doc"], c["field"]), tol=tol) for c in cells]


def scale_signature(conn: sqlite3.Connection, field: str | None = None) -> dict[str, Any]:
    """L4 facts stored with |value| < 1 USD_millions: the signature of a model-emitted ``unit: "USD"`` on a value
    already in millions (``reconcile`` multiplies by 1e-6). Per field."""
    q = "SELECT field, COUNT(*) n, SUM(ABS(value) < 1) lt1 FROM facts WHERE extractor LIKE 'l4%'"
    args: list[Any] = []
    if field:
        q += " AND field = ?"
        args.append(field)
    q += " GROUP BY field ORDER BY field"
    return {r["field"]: {"n": r["n"], "abs_lt_1": int(r["lt1"] or 0)} for r in conn.execute(q, args)}


_FY_RE = re.compile(r"\bFY\s?(\d{4})\b")


def fy_contract_problems(descriptions: dict[str, str], cells: Iterable[dict[str, Any]]) -> list[str]:
    """Preflight for any run that points an L4 field description at gold cells of a given fiscal year: the description
    must not name a different fiscal year than the cells' ``fy`` for that field. ``METRICS`` says "FY2022" in every
    description; the AGG-1.3 replaced slots are FY2021 filings, so this returns one problem per such field."""
    want: dict[str, set[str]] = {}
    for c in cells:
        want.setdefault(c["field"], set()).add(str(c["fy"]))
    out = []
    for field, fys in sorted(want.items()):
        named = set(_FY_RE.findall(descriptions.get(field, "")))
        bad = sorted(named - fys)
        if bad:
            out.append(f"{field}: description names FY{','.join(bad)} but the gold cells are FY{','.join(sorted(fys))}")
    return out


def _summ(found: int, gold: int) -> dict[str, Any]:
    return {"gold": gold, "found": found, "m1": (found / gold) if gold else None}


def rescore(diags: list[dict[str, Any]], *, ignore: Iterable[str] = ()) -> dict[str, Any]:
    """Strict = verdict HIT (the frozen rule, exactly). Diagnostic variants add recoverable classes. ``ignore`` unused
    classes are never added. Diagnostic numbers are NOT gate numbers."""
    n = len(diags)
    cnt = {k: sum(d["verdict"] == k for d in diags) for k in CLASSES}
    strict = cnt["HIT"]
    return {"classes": cnt, "strict": _summ(strict, n),
            "diag_period_agnostic": _summ(strict + cnt["PERIOD_LABEL"], n),
            "diag_unit_normalised": _summ(strict + cnt["UNIT_SCALE"], n),
            "diag_both": _summ(strict + cnt["PERIOD_LABEL"] + cnt["UNIT_SCALE"], n)}


def run13(out: Path, cells_csv: Path = HERE / "agg13_cells.csv") -> dict[str, Any]:
    conn = _ro(out / "store.sqlite")
    try:
        d = diagnose(conn, load_cells13(cells_csv))
        res = json.loads((out / "result.cpu.json").read_text())
        g4 = res["g4"]
        changed = rescore(d)
        ret = g4["retained"]
        tot = lambda c: {"gold": g4["overall"]["gold"], "found": ret["found"] + c["found"],  # noqa: E731
                         "m1": (ret["found"] + c["found"]) / g4["overall"]["gold"]}
        return {"run": "agg13", "cells": d, "changed": changed, "retained_recorded": ret,
                "g4_overall": {"strict_as_recorded": g4["overall"], "diag_period_agnostic": tot(changed["diag_period_agnostic"]),
                               "gate": M1_GATE},
                "scale_signature": scale_signature(conn)}
    finally:
        conn.close()


def run14(out: Path, cells_csv: Path = HERE / "agg14_cells.csv") -> dict[str, Any]:
    conn = _ro(out / "store.sqlite")
    try:
        res = json.loads((out / "result.cpu.json").read_text())
        order = set(res["order"])
        cells = load_cells14(cells_csv)
        d = diagnose(conn, cells)
        l3_facts = {c["doc"]: [f for f in facts_for(conn, c["doc"], c["field"]) if str(f["extractor"]).startswith("l3")]
                    for c in cells}
        l3d = [diagnose_cell(c, l3_facts[c["doc"]]) for c in cells]
        bf = [x for x in d if x["doc"] in order]
        return {"run": "agg14", "cells": d, "backfilled_docs": rescore(bf), "all_cells": rescore(d),
                "l3_only_strict": _summ(sum(x["verdict"] == "HIT" for x in l3d), len(cells)),
                "g2_recorded": res["g2"] if "g2" in res else None, "gate": M1_GATE,
                "scale_signature": scale_signature(conn, "long_term_debt")}
    finally:
        conn.close()


def main(argv: list[str]) -> int:
    if len(argv) < 3 or argv[1] not in ("13", "14"):
        print(__doc__)
        return 2
    out = Path(argv[2])
    rep = run13(out) if argv[1] == "13" else run14(out)
    text = json.dumps(rep, indent=1, default=str)
    if "--json" in argv:
        Path(argv[argv.index("--json") + 1]).write_text(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
