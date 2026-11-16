#!/usr/bin/env python3
"""AGG-1.18 pilot miss diagnosis, OFFLINE (no model, no server). OUT-OF-SAMPLE pilot of the held-out set (NETFLIX_2019, SALESFORCE_2023,
BOEING_2019; 48 cells), run row agg118-ingest-cpu-20261116, stopped at the predeclared futility gate ``v4_equals_v3_on_every_pilot_cell``.

Reads a COPY of the v3 pilot store (``--store`` is copied into ``--scratch`` first; the original is never opened) and the saved typed window
replies (results.jsonl, read only). Every cell not raw-HIT under agg116b.score is classified by the FIRST matching step, in the
agg111s2 / agg117 order, with one L3 sub-step split out:

  RETRIEVAL   the gold value (in the filing's own scale) is in no stored chunk text
  L3_PERIOD   an L3 fact carries the gold value under the right SCHEMA field but another period
  L3_LABEL    an L3 fact carries the gold value in the right period (or unlabelled) but under a non-schema label slug
  L4_REPLY    no saved typed reply carries the gold value for the field and period
  VALUE_IN_QUOTE / HARVEST / CANONICAL / GOLD   (later steps; reached only if all the above pass)

The gold of every miss is also checked against the stored text (the golds were written by a single extractor, so a gold error is a finding).
Two counterfactuals are reported for the misses only (not a regression check, which belongs to the fix task): (a) the Salesforce statement
header ``Fiscal Year Ended January 31, / 2023 2022 2021`` shifted by the fy_end rule (one month-day broadcast over the year columns);
(b) the non-schema label slugs mapped to their schema field.

    python3 agg118_pilot_diag.py --orch-root <orch @ 88bba4fd> --store <out>/store.pilot.v3.sqlite --results <out>/results.jsonl \
        --scratch <NEW dir> --out agg118_pilot_diag.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
for _p in (str(HERE.parent / "agg_phase0"), str(HERE), str(HERE / "agg113")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PILOT = ("NETFLIX_2019_10K", "SALESFORCE_2023_10K", "BOEING_2019_10K")
SCALE = {"NETFLIX_2019_10K": 1000.0}          # statements in thousands; every other pilot filing in millions
FY_SHIFT_DOCS = {"SALESFORCE_2023_10K"}        # fiscal year ends January 31: AGG-1.13 fy_end rule labels it fiscal 2022
# label slugs seen on the misses -> schema field (counterfactual (b)); the normalised phrase is what a synonym entry would need
SLUG_FIELD = {"net_loss_earnings": "net_income", "net_cash_used_in_operating_activities": "operating_cash_flow",
              "net_cash_used_provided_by_operating_activities": "operating_cash_flow",
              "property_plant_and_equipment_additions": "capex"}
UNIT_M = {"USD_millions": 1.0, "USD_thousands": 1e-3, "USD_billions": 1e3, "USD": 1e-6}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a: float, b: float) -> bool:
    return abs(abs(a) - abs(b)) <= max(0.5e-3, 0.005 * abs(b))


def to_m(v: float | None, unit: str | None) -> float | None:
    return None if v is None or unit not in UNIT_M else float(v) * UNIT_M[unit]


def fmt_num(x: float) -> str:
    return f"{abs(x):,.0f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--orch-root", required=True, type=Path)
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--scratch", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    sys.path.insert(0, str(a.orch_root.resolve()))
    os.environ.setdefault("ORCHESTRATOR_FEATURE_AGG_LAYER", "1")
    import agg116b as B  # noqa: PLC0415
    import run_agg118 as R  # noqa: PLC0415
    from src.aggregation import facts_det as fd  # noqa: PLC0415

    a.scratch.mkdir(parents=True, exist_ok=False)
    st = a.scratch / "v3.sqlite"
    shutil.copyfile(a.store, st)
    inp = R.load_inputs()
    cells = [c for c in inp["cells"] if c["doc"] in PILOT]
    sc = B.score(st, cells)
    tw = [json.loads(x) for x in a.results.open()]
    tw = [r for r in tw if r.get("kind") == "t_window" and r.get("final") and r.get("doc") in PILOT]
    gold_rows = {f"{r['doc_name']}|{r['metric']}|{r['fiscal_year']}": r for r in inp["rows"]}

    conn = sqlite3.connect(f"file:{st}?mode=ro", uri=True)
    hashes = dict(conn.execute("SELECT doc_id, doc_hash FROM documents").fetchall()) if "doc_id" in [
        r[1] for r in conn.execute("PRAGMA table_info(documents)")] else {}

    def text_has(doc: str, needle: str) -> list[int]:
        h = hashes.get(doc)
        q = "SELECT page_start FROM chunks WHERE instr(text, ?) > 0" + (" AND doc_hash = ?" if h else "")
        return sorted({r[0] for r in conn.execute(q, (needle, h) if h else (needle,))})

    per: dict[str, Any] = {}
    for c in cells:
        k = R.ckey(c)
        s = sc[k]
        if s["raw_hit"] and s["canonical_ok"]:
            continue
        doc, field, fy, gold = c["doc"], c["field"], c["fy"], c["gold"]
        printed = fmt_num(gold * SCALE.get(doc, 1.0))
        pages_txt = text_has(doc, printed)
        l3 = [dict(zip(("field", "period", "value", "unit", "extractor", "page", "quote"), r)) for r in conn.execute(
            "SELECT field, period, value, unit, extractor, page, substr(quote, 1, 100) FROM facts WHERE doc_id = ? AND extractor LIKE 'l3%'",
            (doc,))]
        l3v = [f for f in l3 if (m := to_m(f["value"], f["unit"])) is not None and close(m, gold)]
        if not l3v and doc in SCALE:   # textrow facts of a thousands filing: compare in its own scale
            l3v = [f for f in l3 if f["value"] is not None and close(float(f["value"]) / SCALE[doc], gold)]
        schema_wrong_period = [f for f in l3v if f["field"] == field and str(f["period"]) != fy]
        slug_right = [f for f in l3v if f["field"] != field and str(f["period"]) in (fy, "None") and SLUG_FIELD.get(f["field"]) == field]
        l4 = [{"chunk_id": r["chunk_id"], **{kk: f.get(kk) for kk in ("period", "value", "unit", "quote")}}
              for r in tw if r["doc"] == doc for f in r["facts"] if f.get("field") == field]
        l4_gold = [f for f in l4 if f["value"] is not None and close(float(f["value"]), gold) and str(f["period"]).endswith(fy)]
        l4_periods = sorted({str(f["period"]) for f in l4})
        if not s["raw_hit"]:
            if not pages_txt:
                step = "RETRIEVAL"
            elif schema_wrong_period:
                step = "L3_PERIOD"
            elif slug_right:
                step = "L3_LABEL"
            elif not l4_gold:
                step = "L4_REPLY"
            else:
                step = "VALUE_IN_QUOTE_OR_HARVEST"
        else:
            step = "CANONICAL"
        g = gold_rows[k]
        gold_check = {"gold": gold, "sheet_quote": g["quote"], "sheet_page": g["page"], "printed": printed,
                      "printed_on_stored_pages": pages_txt[:8], "gold_value_in_sheet_quote": printed in g["quote"].replace("$ ", "$"),
                      "verdict": "gold_ok" if pages_txt and printed in g["quote"] else "CHECK"}
        cf = {}
        if step == "L3_PERIOD" and doc in FY_SHIFT_DOCS:
            cf["fy_end_broadcast_shift"] = any(str(int(f["period"]) - 1) == fy for f in schema_wrong_period if str(f["period"]).isdigit())
        if step == "L3_LABEL":
            cf["slug_to_schema"] = {f["field"]: SLUG_FIELD[f["field"]] for f in slug_right}
            cf["normalised_label"] = sorted({fd.normalise_label(f["field"].replace("_", " ")) for f in slug_right})
            cf["in_textrow_synonyms"] = {n: n in fd.TEXTROW_SYNONYMS for n in cf["normalised_label"]}
        per[k] = {"raw_hit": s["raw_hit"], "canonical_ok": s["canonical_ok"], "raw_verdict": s["raw_verdict"],
                  "raw_candidate": s["raw_candidate"], "canonical_value": (s["canonical_row"] or {}).get("value"),
                  "first_step": step, "l3_gold_value_facts": l3v[:6], "l4_periods_replied": l4_periods,
                  "l4_reply_has_gold": bool(l4_gold), "gold_check": gold_check, "counterfactual": cf}
    conn.close()

    raw_miss = {k: v for k, v in per.items() if not v["raw_hit"]}
    by_step: dict[str, list[str]] = {}
    for k, v in raw_miss.items():
        by_step.setdefault(v["first_step"], []).append(k)
    fixable = sum(1 for v in raw_miss.values() if v["counterfactual"].get("fy_end_broadcast_shift") or v["counterfactual"].get("slug_to_schema"))
    res = {"task": "AGG-1.18 pilot miss diagnosis (offline, no model)", "model_calls": 0,
           "label": "OUT-OF-SAMPLE pilot (3 held-out docs, 48 cells); diagnosis is post-hoc on those cells. The 7 unrun held-out docs stay unseen.",
           "inputs_sha256": {str(a.store): sha(a.store), str(a.results): sha(a.results),
                             "facts_sheet.csv": sha(R.D118 / "facts_sheet.csv"),
                             "orch facts_det.py": sha(a.orch_root / "src/aggregation/facts_det.py")},
           "n_cells": len(cells), "raw_hits": sum(v["raw_hit"] for v in sc.values()),
           "canonical_ok": sum(v["canonical_ok"] for v in sc.values()),
           "raw_misses_by_first_step": {k: sorted(v) for k, v in by_step.items()},
           "canonical_only_misses": sorted(k for k, v in per.items() if v["raw_hit"]),
           "gold_errors": sorted(k for k, v in per.items() if v["gold_check"]["verdict"] != "gold_ok"),
           "raw_misses_fixed_by_counterfactuals": fixable,
           "raw_if_fixed": f"{sum(v['raw_hit'] for v in sc.values()) + fixable}/{len(cells)}",
           "findings": {
               "L3_PERIOD": "Salesforce (FY ends Jan 31): the statement header is ONE month-day over three years ('Fiscal Year Ended January 31,' / "
                            "'2023 2022 2021'); fy_end amendment 2 pairs month-days with years only when counts match (the Walmart guard), so "
                            "the columns keep the filer's labels 2023/2022/2021 and every FY/FY-1 cell is a year off. The manifest test "
                            "(first year == manifest FY + 1) already separates Salesforce (2023 vs 2022) from Walmart (2022 vs 2022), so broadcasting "
                            "a lone month-day is gated by the same rule. L4 also labels the filer's fiscal 2022 as FY2022 (value 26,492).",
               "L3_LABEL": "Netflix/Boeing: L3 text-row extracts the FY-1 value with the right period but the caption is not in TEXTROW_SYNONYMS "
                           "('net cash used in operating activities', 'net cash used provided by operating activities', 'net loss earnings', "
                           "'property plant and equipment additions'), so it is stored under a slug and never scores.",
               "L4_REPLY": "Secondary on the same 4 cells: every saved typed reply for these fields carries the current year only (FY2019); "
                           "the L4 stage never supplies FY-1, so FY-1 depends entirely on L3.",
               "CANONICAL": "BOEING net_income 2019: raw HIT (-636), canonical picks -677 from the equity-statement row "
                            "'Net earnings/(loss) (636) (41) (677)' (column total, L4 reply on page 57).",
               "unseen_vs_seen": "raw 36/48 (0.75) here vs 71/76 (0.93) on AGG-1.13 docs; 8 of the 12 misses are one filing's period labels, "
                                 "4 are caption-vocabulary gaps on FY-1; with both fixed the pilot would read 48/48 raw (counterfactual, misses only)."},
           "per_cell": per}
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("raw_hits", "canonical_ok", "raw_misses_by_first_step", "canonical_only_misses", "gold_errors",
                                          "raw_misses_fixed_by_counterfactuals", "raw_if_fixed")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
