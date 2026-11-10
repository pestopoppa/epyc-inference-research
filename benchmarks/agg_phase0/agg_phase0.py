#!/usr/bin/env python3
"""AGG-0.1 question set + gold derivation + M1-M5 scorer + kill gate (stdlib only).

Plan of record: aggregation-plan-fable.md rev 2, section (c) Phase 0.

Subcommands:
    build   write corpus.json, single_fact.jsonl, facts_sheet.csv (if absent),
            aggregation_questions.json from the FinanceBench jsonl
    golds   derive golds.jsonl + GOLD_REVIEW.md from facts_sheet.csv
            (every gold is ``verified: false`` until a named reviewer signs it)
    score   compute M1-M5 and the predeclared kill gate from a records jsonl

No model, network or PDF parsing happens here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
FB_DIR = Path("/mnt/raid0/llm/datasets/financebench-20261109")
FB_JSONL_SHA256 = "a5a2aa673e573e55675fc3c0f9aa38c1cf59d2abc91edb077534f71f10a71877"
FB_PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf"

FB5 = ["AMERICANWATERWORKS_2022_10K", "CORNING_2022_10K", "JPMORGAN_2022_10K",
       "PAYPAL_2022_10K", "VERIZON_2022_10K"]
# +5: the FY2022 10-Ks that carry metrics-generated FinanceBench questions, so the
# whole corpus is one fiscal year and cross-company aggregation is well-posed.
PLUS5 = ["AES_2022_10K", "COCACOLA_2022_10K", "GENERALMILLS_2022_10K",
         "LOCKHEEDMARTIN_2022_10K", "PEPSICO_2022_10K"]
DOCS = FB5 + PLUS5

SINGLE_FACT_IDS = [
    "financebench_id_00070", "financebench_id_00005", "financebench_id_01346",
    "financebench_id_00080", "financebench_id_00216", "financebench_id_00566",
    "financebench_id_10420", "financebench_id_06272", "financebench_id_03718",
    "financebench_id_04481",
]

METRICS = {
    "revenue": "Total revenues / net sales / operating revenues, FY2022 (USD millions)",
    "net_income": "Net income attributable to the company's shareholders, FY2022 (USD millions)",
    "capex": "Capital expenditures (purchases of PP&E), FY2022 cash flow statement (USD millions, positive)",
    "total_assets": "Total assets at FY2022 year end (USD millions)",
    "operating_cash_flow": "Net cash provided by operating activities, FY2022 (USD millions)",
    "total_current_assets": "Total current assets at FY2022 year end (USD millions); NA if unclassified balance sheet",
    "total_current_liabilities": "Total current liabilities at FY2022 year end (USD millions); NA if unclassified",
    "dividends_paid": "Cash dividends paid to common shareholders, FY2022 (USD millions, positive)",
}

# FinanceBench gold answers that pin a value or sign the sheet must agree with.
FB_ANCHORS = [
    {"doc": "CORNING_2022_10K", "expr": "total_current_assets-total_current_liabilities",
     "check": "sign", "expect": 1, "source": "financebench_id_00005 (+$831M, operating items only)"},
    {"doc": "AMERICANWATERWORKS_2022_10K", "expr": "total_current_assets-total_current_liabilities",
     "check": "approx", "expect": -1561, "source": "financebench_id_00070 (-$1561M)"},
    {"doc": "PAYPAL_2022_10K", "expr": "total_current_assets-total_current_liabilities",
     "check": "sign", "expect": 1, "source": "financebench_id_00080 (+$1.6Bn)"},
    {"doc": "COCACOLA_2022_10K", "expr": "dividends_paid/net_income", "check": "approx",
     "expect": 0.80, "source": "financebench_id_06272 (0.80)"},
    {"doc": "AES_2022_10K", "expr": "net_income", "check": "sign", "expect": -1,
     "source": "financebench_id_10420 (ROA -0.02)"},
]

AGG_QUESTIONS: list[dict[str, Any]] = [
    {"id": "agg01", "expr": "revenue", "op": "argmax", "q": "Which company in the corpus reported the highest FY2022 total revenue?"},
    {"id": "agg02", "expr": "revenue", "op": "argmin", "q": "Which company in the corpus reported the lowest FY2022 total revenue?"},
    {"id": "agg03", "expr": "revenue", "op": "sum", "q": "What is the combined FY2022 total revenue of all companies in the corpus, in USD millions?"},
    {"id": "agg04", "expr": "revenue", "op": "count_gt", "t": 50000, "q": "How many companies in the corpus reported FY2022 total revenue above $50 billion?"},
    {"id": "agg05", "expr": "revenue", "op": "top3", "q": "List the three companies with the highest FY2022 total revenue, highest first."},
    {"id": "agg06", "expr": "net_income", "op": "argmax", "q": "Which company reported the highest FY2022 net income attributable to shareholders?"},
    {"id": "agg07", "expr": "net_income", "op": "count_lt", "t": 0, "q": "How many companies in the corpus reported a net loss attributable to shareholders for FY2022?"},
    {"id": "agg08", "expr": "net_income", "op": "sum", "q": "What is the combined FY2022 net income attributable to shareholders across the corpus, in USD millions?"},
    {"id": "agg09", "expr": "capex", "op": "argmax", "q": "Which company had the largest FY2022 capital expenditures?"},
    {"id": "agg10", "expr": "capex", "op": "sum", "q": "What were total FY2022 capital expenditures across all companies in the corpus, in USD millions?"},
    {"id": "agg11", "expr": "capex", "op": "count_gt", "t": 5000, "q": "How many companies spent more than $5 billion on capital expenditures in FY2022?"},
    {"id": "agg12", "expr": "total_assets", "op": "argmax", "q": "Which company had the largest total assets at FY2022 year end?"},
    {"id": "agg13", "expr": "total_assets", "op": "top3", "q": "List the three companies with the largest total assets at FY2022 year end, largest first."},
    {"id": "agg14", "expr": "operating_cash_flow", "op": "argmax", "q": "Which company generated the most cash from operating activities in FY2022?"},
    {"id": "agg15", "expr": "operating_cash_flow-net_income", "op": "count_gt", "t": 0, "q": "For how many companies did FY2022 operating cash flow exceed net income attributable to shareholders?"},
    {"id": "agg16", "expr": "total_current_assets-total_current_liabilities", "op": "count_gt", "t": 0, "q": "Among companies with a classified balance sheet, how many had positive working capital (total current assets minus total current liabilities) at FY2022 year end?"},
    {"id": "agg17", "expr": "total_current_assets-total_current_liabilities", "op": "argmin", "q": "Among companies with a classified balance sheet, which had the most negative working capital at FY2022 year end?"},
    {"id": "agg18", "expr": "dividends_paid", "op": "sum", "q": "What were total FY2022 cash dividends paid to common shareholders across the corpus, in USD millions?"},
    {"id": "agg19", "expr": "capex/revenue", "op": "argmax", "q": "Which company had the highest FY2022 capital expenditures as a share of total revenue?"},
    {"id": "agg20", "expr": "net_income/revenue", "op": "count_gt", "t": 0.10, "q": "How many companies had an FY2022 net margin (net income attributable / total revenue) above 10%?"},
]

SHEET_COLUMNS = ["doc_name", "company", "metric", "fiscal_year", "value_usd_m", "status",
                 "page", "quote", "reviewer", "note"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_fb(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------------- build
def build(fb_dir: Path, out: Path) -> dict[str, Any]:
    jsonl = fb_dir / "financebench_open_source.jsonl"
    jsha = sha256_file(jsonl)
    if jsha != FB_JSONL_SHA256:
        raise SystemExit(f"FinanceBench jsonl sha256 mismatch: {jsha}")
    rows = load_fb(jsonl)
    company = {r["doc_name"]: r["company"] for r in rows}
    docs, missing = [], []
    for d in DOCS:
        pdf = fb_dir / "pdfs" / f"{d}.pdf"
        rec = {"doc_name": d, "company": company[d], "pdf": str(pdf),
               "sha256": sha256_file(pdf) if pdf.exists() else None,
               "bytes": pdf.stat().st_size if pdf.exists() else None,
               "source_url": FB_PDF_URL.format(doc=d),
               "fb_questions": [r["financebench_id"] for r in rows if r["doc_name"] == d]}
        docs.append(rec)
        if rec["sha256"] is None:
            missing.append(d)
    corpus = {"schema": "agg_phase0_corpus.v1", "fb_jsonl_sha256": jsha, "docs": docs,
              "missing_pdfs": missing,
              "selection": "FB5 pins + the five FY2022 10-Ks carrying metrics-generated questions"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus.json").write_text(json.dumps(corpus, indent=1) + "\n")
    by_id = {r["financebench_id"]: r for r in rows}
    with (out / "single_fact.jsonl").open("w") as f:
        for qid in SINGLE_FACT_IDS:
            r = by_id[qid]
            assert r["doc_name"] in DOCS, qid
            f.write(json.dumps({
                "id": qid, "doc_name": r["doc_name"], "question": r["question"],
                "gold": r["answer"], "question_type": r["question_type"],
                "evidence_pages": [e.get("evidence_page_num") for e in r["evidence"]],
                "arms": ["P-large", "P-small", "B-today"], "gold_source": "financebench",
                "verified": False}) + "\n")
    (out / "aggregation_questions.json").write_text(json.dumps(
        {"schema": "agg_phase0_questions.v1", "fiscal_year": 2022, "questions": AGG_QUESTIONS,
         "metrics": METRICS, "anchors": FB_ANCHORS}, indent=1) + "\n")
    sheet = out / "facts_sheet.csv"
    if not sheet.exists():  # never overwrite reviewer work
        with sheet.open("w", newline="") as f:
            w = csv.DictWriter(f, SHEET_COLUMNS)
            w.writeheader()
            for d in DOCS:
                for m in METRICS:
                    w.writerow({"doc_name": d, "company": company[d], "metric": m,
                                "fiscal_year": 2022, "value_usd_m": "", "status": "todo"})
    return corpus


# ------------------------------------------------------------------------- golds
def read_sheet(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as f:
        return {(r["doc_name"], r["metric"]): r for r in csv.DictReader(f)}


def _cell(sheet, doc, metric) -> float | None | str:
    r = sheet.get((doc, metric))
    if r is None or r["status"] == "todo" or r["value_usd_m"].strip() == "" and r["status"] != "na":
        return "missing"
    if r["status"] == "na":
        return None
    return float(r["value_usd_m"].replace(",", ""))


def eval_expr(expr: str, sheet, doc: str) -> float | None | str:
    """``a``, ``a-b`` or ``a/b``. Returns 'missing' if a cell is unfilled, None if NA."""
    for opch in ("-", "/"):
        if opch in expr:
            a, b = expr.split(opch)
            va, vb = _cell(sheet, doc, a), _cell(sheet, doc, b)
            if "missing" in (va, vb):
                return "missing"
            if va is None or vb is None or (opch == "/" and vb == 0):
                return None
            return va - vb if opch == "-" else va / vb
    return _cell(sheet, doc, expr)


def compute_gold(q: dict[str, Any], values: dict[str, float]) -> Any:
    items = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    op = q["op"]
    if op == "argmax":
        return items[0][0]
    if op == "argmin":
        return items[-1][0]
    if op == "sum":
        return round(sum(values.values()), 2)
    if op == "count_gt":
        return sum(1 for v in values.values() if v > q["t"])
    if op == "count_lt":
        return sum(1 for v in values.values() if v < q["t"])
    if op == "top3":
        return [k for k, _ in items[:3]]
    raise ValueError(op)


def derive_golds(out: Path) -> list[dict[str, Any]]:
    sheet = read_sheet(out / "facts_sheet.csv")
    company = {d: r["company"] for (d, _), r in sheet.items()}
    golds = []
    for q in AGG_QUESTIONS:
        values, missing, na = {}, [], []
        for d in DOCS:
            v = eval_expr(q["expr"], sheet, d)
            if v == "missing":
                missing.append(d)
            elif v is None:
                na.append(d)
            else:
                values[company[d]] = v
        gold = compute_gold(q, values) if not missing and values else None
        golds.append({"id": q["id"], "question": q["q"], "expr": q["expr"], "op": q["op"],
                      "threshold": q.get("t"), "gold": gold, "per_company": values,
                      "excluded_na": na, "missing_cells": missing,
                      "gold_source": "derived:facts_sheet.csv", "verified": False})
    anchors = []
    for a in FB_ANCHORS:
        v = eval_expr(a["expr"], sheet, a["doc"])
        if v in ("missing", None):
            ok = None
        elif a["check"] == "sign":
            ok = math.copysign(1, v) == a["expect"]
        else:
            ok = abs(v - a["expect"]) <= abs(a["expect"]) * 0.02
        anchors.append(dict(a, value=v if v != "missing" else None, agrees=ok))
    with (out / "golds.jsonl").open("w") as f:
        for g in golds:
            f.write(json.dumps(g) + "\n")
    (out / "anchor_checks.json").write_text(json.dumps(anchors, indent=1) + "\n")
    write_review_sheet(out, golds, anchors)
    return golds


def write_review_sheet(out: Path, golds, anchors) -> None:
    sf = [json.loads(l) for l in (out / "single_fact.jsonl").read_text().splitlines()]
    corpus = json.loads((out / "corpus.json").read_text())
    L = ["# AGG Phase 0 — gold review sheet (operator hand-verification)", "",
         "Every gold below is `verified: false`. **No Phase-0 run may start until each question "
         "used is verified against the filing by a named reviewer; an unverified question is "
         "excluded, never assumed** (plan §(c), AGG-0.1).", "",
         "## How to verify", "",
         "1. Fill `facts_sheet.csv`: one row per (doc, metric). Set `value_usd_m`, `page` (PDF page), "
         "a verbatim `quote`, `reviewer`, and `status` = `filled` (or `na` with a `note` when the "
         "filing has no such line item, e.g. an unclassified balance sheet).",
         "2. Run `python3 agg_phase0.py golds` — it recomputes every gold and the FinanceBench anchor "
         "checks below. Any `agrees: false` anchor means a sheet error; fix before signing.",
         "3. For each question, tick the `verified` cell here AND set `verified: true`, `verified_by`, "
         "`verified_at` in `verification_record.json` (the runner reads that file, not this one).", "",
         f"Corpus: {len(corpus['docs'])} FY2022 10-Ks; missing PDFs: "
         f"{', '.join(corpus['missing_pdfs']) or 'none'}.", "",
         "## FinanceBench anchor checks", "", "| doc | expr | expect | sheet value | agrees | source |",
         "|---|---|---|---|---|---|"]
    for a in anchors:
        L.append(f"| {a['doc']} | `{a['expr']}` | {a['check']} {a['expect']} | {a['value']} | "
                 f"{a['agrees']} | {a['source']} |")
    L += ["", "## 20 cross-document aggregation questions", "",
          "| id | question | rule | derived gold | NA-excluded | missing cells | verified | reviewer |",
          "|---|---|---|---|---|---|---|---|"]
    for g in golds:
        rule = f"{g['op']}({g['expr']}{'' if g['threshold'] is None else ', ' + str(g['threshold'])})"
        L.append(f"| {g['id']} | {g['question']} | `{rule}` | {json.dumps(g['gold'])} | "
                 f"{len(g['excluded_na'])} | {len(g['missing_cells'])} | [ ] | |")
    L += ["", "## 10 single-fact questions (FinanceBench golds; P-large, P-small, B-today)", "",
          "| id | doc | question | FinanceBench gold | evidence page(s) | verified | reviewer |",
          "|---|---|---|---|---|---|---|"]
    for s in sf:
        qq = s["question"].replace("|", "/").replace("\n", " ")[:160]
        gg = s["gold"].replace("|", "/").replace("\n", " ")[:160]
        L.append(f"| {s['id']} | {s['doc_name']} | {qq} | {gg} | {s['evidence_pages']} | [ ] | |")
    (out / "GOLD_REVIEW.md").write_text("\n".join(L) + "\n")
    rec = out / "verification_record.json"
    if not rec.exists():
        rec.write_text(json.dumps({"schema": "agg_phase0_verification.v1", "questions": {
            q: {"verified": False, "verified_by": None, "verified_at": None}
            for q in [g["id"] for g in golds] + [s["id"] for s in sf]}}, indent=1) + "\n")


# ------------------------------------------------------------------------- score
KILL = {"m1_min": 0.70, "m3_prefill_ratio_max": 2.0, "m3_calls_per_100_chunks_max": 150,
        "m2_agg_margin_pp_min": 25.0, "median_calls_max": 8,
        "m5_acc_tol_questions": 1, "m5_calls_tol": 1, "m5_prefill_tol": 2000}


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return float("nan") if n == 0 else (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2)


def score(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Records (one per line in the run's results.jsonl), kinds:
    ``ingest``  {prefill_tokens, corpus_tokens, llm_calls, n_chunks}
    ``m1``      {gold_facts, found}
    ``answer``  {arm, qid, qclass: agg|single, correct: true|false|null, calls, prefill_tokens}
    ``manifest``{arm: P-small, qid, built_layers: [...]}
    """
    recs = list(records)
    ing = [r for r in recs if r["kind"] == "ingest"]
    m1r = [r for r in recs if r["kind"] == "m1"]
    ans = [r for r in recs if r["kind"] == "answer"]
    man = [r for r in recs if r["kind"] == "manifest"]

    def acc(arm, qclass):
        xs = [r for r in ans if r["arm"] == arm and r["qclass"] == qclass]
        return (sum(1 for r in xs if r["correct"] is True) / len(xs)) if xs else None, len(xs)

    m1 = (sum(r["found"] for r in m1r) / sum(r["gold_facts"] for r in m1r)) if m1r else None
    pre = sum(r["prefill_tokens"] for r in ing)
    ctok = sum(r["corpus_tokens"] for r in ing)
    calls = sum(r["llm_calls"] for r in ing)
    chunks = sum(r["n_chunks"] for r in ing)
    m3 = {"prefill_ratio": pre / ctok if ctok else None,
          "calls_per_100_chunks": 100 * calls / chunks if chunks else None}
    pl = [r for r in ans if r["arm"] == "P-large"]
    m4 = {"median_calls": _median([r["calls"] for r in pl]),
          "median_prefill": _median([r["prefill_tokens"] for r in pl])}
    p_agg, n_agg = acc("P-large", "agg")
    b1_agg, _ = acc("B1", "agg")
    m2 = {"P-large_agg": p_agg, "B1_agg": b1_agg, "P-large_single": acc("P-large", "single")[0],
          "n_agg": n_agg}
    small = {r["qid"]: r for r in ans if r["arm"] == "P-small"}
    today = {r["qid"]: r for r in ans if r["arm"] == "B-today"}
    common = sorted(set(small) & set(today))
    m5 = {"n": len(common),
          "acc_diff_questions": abs(sum(small[q]["correct"] is True for q in common)
                                    - sum(today[q]["correct"] is True for q in common)),
          "max_extra_calls": max((small[q]["calls"] - today[q]["calls"] for q in common), default=None),
          "max_extra_prefill": max((small[q]["prefill_tokens"] - today[q]["prefill_tokens"]
                                    for q in common), default=None),
          "indexes_built": sorted({l for r in man if r.get("arm") == "P-small"
                                   for l in r["built_layers"] if l in ("L2", "L3", "L4")})}
    trips = []
    if m1 is None or m1 < KILL["m1_min"]:
        trips.append("M1")
    if (m3["prefill_ratio"] is None or m3["prefill_ratio"] > KILL["m3_prefill_ratio_max"]
            or m3["calls_per_100_chunks"] > KILL["m3_calls_per_100_chunks_max"]):
        trips.append("M3")
    if p_agg is None or b1_agg is None or (p_agg - b1_agg) * 100 < KILL["m2_agg_margin_pp_min"]:
        trips.append("M2")
    if not (m4["median_calls"] <= KILL["median_calls_max"]):
        trips.append("M4")
    if (m5["n"] == 0 or m5["acc_diff_questions"] > KILL["m5_acc_tol_questions"]
            or m5["max_extra_calls"] > KILL["m5_calls_tol"]
            or m5["max_extra_prefill"] > KILL["m5_prefill_tol"] or m5["indexes_built"]):
        trips.append("M5")
    return {"M1": m1, "M2": m2, "M3": m3, "M4": m4, "M5": m5, "kill_gate_tripped": trips,
            "verdict": "superseded_v2_not_viable" if trips else "go_phase_1"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "golds", "score"))
    ap.add_argument("--fb-dir", type=Path, default=FB_DIR)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--records", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "build":
        c = build(a.fb_dir, a.out)
        print(json.dumps({"missing_pdfs": c["missing_pdfs"]}))
    elif a.cmd == "golds":
        g = derive_golds(a.out)
        print(json.dumps({"golds": len(g), "complete": sum(x["gold"] is not None for x in g)}))
    else:
        recs = [json.loads(l) for l in a.records.read_text().splitlines() if l.strip()]
        print(json.dumps(score(recs), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
