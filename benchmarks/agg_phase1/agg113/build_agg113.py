#!/usr/bin/env python3
"""AGG-1.13 step A: out-of-sample corpus selection, corpus manifest, question set, golds, margin audit and verification
record (stdlib only). Mirrors ``benchmarks/agg_phase1/build_phase1.py`` (AGG-1.1 step A, research 3296c19f4).

No model, network or PDF parsing happens here (cell values live in ``facts_sheet.csv``; they were read from the PDF text and
cross-checked by ``../verify_phase1_cells.py``, the only code that opens a PDF).

Subcommands:
    select   write selection.json: the eligible-document rule, its audit table, the design-use scan, the salted pilot order
    corpus   write corpus.json (the selected docs, sha256 pinned) with its manifest digest from a page-count table
    golds    derive aggregation_questions.json, golds.jsonl, verification_record.json and GOLD_REVIEW.md from facts_sheet.csv
    check    recompute everything in memory and fail non-zero if a committed file differs (CI/self-test)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
P1 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(P1.parent / "agg_phase0"))
import agg113_questions as Q  # noqa: E402
import agg_phase0 as A0  # noqa: E402

FB_DIR = Path("/mnt/raid0/llm/datasets/financebench-20261113-agg113")
FB_PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf"
FB_DOCINFO_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_document_information.jsonl"
DOCINFO = HERE / "financebench_document_information.jsonl"
SALT = "agg113-20261113"
METRICS = list(A0.METRICS)
MARGIN_MIN = Q.MARGIN_MIN

# Per-filing audit of the "fiscal 2022 as the filing itself labels it" rule (AGG-1.1's selection text). Evidence = the cover
# page "fiscal year ended" date and the label the filing prints for that year, read from the downloaded PDF text.
# ``dl`` = the PDF was downloaded and read for this audit; "no" = decided from the cover-date convention only (see ``why``).
AUDIT = {
    "AMCOR_2022_10K": {"fye": "2022-06-30", "label": "fiscal 2022", "fy2022": True, "dl": "pdfs"},
    "BESTBUY_2022_10K": {"fye": "2022-01-29", "label": "fiscal 2022", "fy2022": True, "dl": "pdfs"},
    "BLOCK_2022_10K": {"fye": "2022-12-31", "label": "2022", "fy2022": True, "dl": "pdfs"},
    "FOOTLOCKER_2022_10K": {"fye": "2022-01-29", "label": "2021 Form 10-K (fiscal 2021)", "fy2022": False, "dl": "audit_pdfs",
                            "why": "the filing prints '2021 Form 10-K' on every page: its fiscal 2021"},
    "INTEL_2022_10K": {"fye": "2022-12-31", "label": "2022", "fy2022": True, "dl": "pdfs"},
    "NIKE_2022_10K": {"fye": "2022-05-31", "label": "fiscal 2022", "fy2022": True, "dl": "pdfs"},
    "ORACLE_2022_10K": {"fye": "2022-05-31", "label": "fiscal 2022", "fy2022": True, "dl": "pdfs"},
    "PG_E_2022_10K": {"fye": "2022-12-31", "label": "2022", "fy2022": True, "dl": "pdfs"},
    "WALMART_2022_10K": {"fye": "2022-01-31", "label": "fiscal 2022", "fy2022": True, "dl": "pdfs"},
    "FOOTLOCKER_2023_10K": {"fye": "2023-01-28", "label": "2022 Form 10-K (fiscal 2022)", "fy2022": True, "dl": "pdfs",
                            "why": "FinanceBench names this file 2023; the filing prints '2022 Form 10-K' and 'Fiscal year 2022 ... "
                                   "52 weeks ended January 28, 2023'"},
    "ULTABEAUTY_2023_10K": {"fye": "2023-01-28", "label": "fiscal 2022", "fy2022": True, "dl": "pdfs",
                            "why": "the filing labels the year ended January 28, 2023 'fiscal 2022'"},
    "BESTBUY_2023_10K": {"fye": "2023-01-28", "label": "fiscal 2023", "fy2022": False, "dl": "audit_pdfs"},
    "WALMART_2023_10K": {"fye": "2023-01-31", "label": "fiscal 2023", "fy2022": False, "dl": "audit_pdfs"},
    "SALESFORCE_2023_10K": {"fye": "2023-01-31", "label": "fiscal 2023", "fy2022": False, "dl": "audit_pdfs"},
    # mid-2023 fiscal year ends: a filing that names its year by the year it ends in cannot call a year ended in mid-2023 fiscal
    # 2022; none was downloaded, the January-end retailers (the only filers whose label can lag the calendar) were.
    "AMCOR_2023_10K": {"fye": "2023-06-30", "label": "fiscal 2023", "fy2022": False, "dl": "no"},
    "FEDEX_2023_10K": {"fye": "2023-05-31", "label": "fiscal 2023", "fy2022": False, "dl": "no"},
    "GENERALMILLS_2023_10K": {"fye": "2023-05-28", "label": "fiscal 2023", "fy2022": False, "dl": "no"},
    "MICROSOFT_2023_10K": {"fye": "2023-06-30", "label": "fiscal 2023", "fy2022": False, "dl": "no"},
    "NIKE_2023_10K": {"fye": "2023-05-31", "label": "fiscal 2023", "fy2022": False, "dl": "no"},
    "ORACLE_2023_10K": {"fye": "2023-05-31", "label": "fiscal 2023", "fy2022": False, "dl": "no"},
}
FY_END = {d: a["fye"] for d, a in AUDIT.items()}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest_docs(docs: list[dict[str, Any]]) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace) of the docs array."""
    return hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def salted(doc: str) -> str:
    return hashlib.sha256(f"{SALT}|{doc}".encode()).hexdigest()


def design_use() -> set[str]:
    """Every ``X_YYYY_10K`` document name mentioned in any AGG design/runner/test file outside this directory."""
    names: set[str] = set()
    root = P1.parent.parent
    files = [p for p in P1.rglob("*") if p.is_file() and HERE not in p.parents and p.suffix in (".py", ".md", ".json", ".csv", ".jsonl")]
    files += list((root / "benchmarks" / "agg_phase0").rglob("*"))
    files += [p for p in (root / "tests").glob("test_agg*") if p.name != "test_agg113.py"]  # this step's own test names the corpus
    for p in files:
        if p.is_file() and p.suffix in (".py", ".md", ".json", ".csv", ".jsonl"):
            names |= set(re.findall(r"\b[A-Za-z_]+_20\d\d_10K\b", p.read_text(errors="replace")))
    return names


def used_28() -> list[str]:
    return [d["doc_name"] for d in json.loads((P1 / "corpus.json").read_text())["docs"]]


def build_selection() -> dict[str, Any]:
    info = [json.loads(l) for l in DOCINFO.read_text().splitlines() if l.strip()]
    k10 = [r for r in info if r["doc_type"] == "10k"]
    seen28 = set(used_28())
    seen13 = {d["doc_name"] for d in json.loads((P1 / "corpus_agg13.json").read_text())["docs"]}
    used = design_use()
    fixture = sorted(n for n in used if n not in seen28 and n not in {r["doc_name"] for r in k10})
    cand = [r for r in k10 if r["doc_period"] in (2022, 2023) and r["doc_name"] not in seen28 and r["doc_name"] not in seen13]
    rows, eligible = [], []
    for r in sorted(cand, key=lambda r: r["doc_name"]):
        a = AUDIT[r["doc_name"]]  # KeyError = an unaudited candidate: the rule cannot be applied silently
        in_design = r["doc_name"] in used
        ok = a["fy2022"] and not in_design
        pdf = FB_DIR / a["dl"] / f"{r['doc_name']}.pdf"
        rows.append({"doc_name": r["doc_name"], "company": r["company"], "gics_sector": r["gics_sector"],
                     "pdf_sha256": sha256_file(pdf) if a["dl"] != "no" and pdf.exists() else None,
                     "financebench_doc_period": r["doc_period"], "fiscal_year_end": a["fye"], "filing_label": a["label"],
                     "own_label_fiscal_2022": a["fy2022"], "pdf_read_for_audit": a["dl"] != "no",
                     "named_in_agg1x_design_files": in_design, "eligible": ok, **({"why": a["why"]} if "why" in a else {})})
        if ok:
            eligible.append(r["doc_name"])
    chosen = sorted(eligible)  # fewer than 15 eligible: the rule takes all of them
    pilot_order = sorted(chosen, key=salted)
    return {
        "schema": "agg113_selection.v1",
        "rule": [
            "R1 universe: FinanceBench document-information rows with doc_type '10k' (sha256 of the file below).",
            "R2 candidates: doc_period 2022 or 2023 (a January-end filer's fiscal 2022 can sit in either FinanceBench period); "
            "every candidate not in the AGG-1.1 28-document corpus.json and not in corpus_agg13.json must have a row in AUDIT.",
            "R3 fiscal 2022 as the filing labels it (the AGG-1.1 selection text): the cover 'fiscal year ended' date and the "
            "label the filing prints decide, not the FinanceBench period.",
            "R4 never used: a document name that appears in any benchmarks/agg_phase0, benchmarks/agg_phase1 (outside "
            "this directory) or tests/test_agg* file is excluded (scanned by this script).",
            "R5 take ALL eligible documents when fewer than 15 are eligible (no sampling is possible); with 15 or more, the "
            "15 smallest sha256('agg113-20261113|doc_name') would be taken. The salted order also fixes the PILOT: the first 3.",
        ],
        "salt": SALT, "n_eligible": len(eligible), "target_n_docs": 15,
        "shortfall_reason": ("FinanceBench's whole public pdfs/ directory holds 37 FY2022 10-Ks by doc_period; 28 are in the "
                             "AGG-1.1 corpus, leaving 9, of which FOOTLOCKER_2022_10K is fiscal 2021 by its own label and two "
                             "January-end filings (FOOTLOCKER_2023_10K, ULTABEAUTY_2023_10K) are fiscal 2022 by theirs: 10 docs."),
        "financebench_document_information": {"url": FB_DOCINFO_URL, "file": DOCINFO.name, "sha256": sha256_file(DOCINFO),
                                              "rows": len(info), "tenk_rows": len(k10)},
        "candidates": rows, "selected": chosen, "pilot_order_by_salted_hash": pilot_order, "pilot_docs": pilot_order[:3],
        "salted_hash": {d: salted(d) for d in chosen},
        "design_use_scan": {"n_names_found": len(used), "of_which_in_28": len(used & seen28),
                            "test_fixture_placeholders_not_real_docs": fixture},
        "excluded_candidates": [r["doc_name"] for r in rows if not r["eligible"]],
    }


# ------------------------------------------------------------------------------------------------ corpus
def build_corpus(pages: dict[str, int]) -> dict[str, Any]:
    sel = json.loads((HERE / "selection.json").read_text())
    info = {json.loads(l)["doc_name"]: json.loads(l) for l in DOCINFO.read_text().splitlines() if l.strip()}
    docs = []
    for d in sel["selected"]:
        pdf = FB_DIR / "pdfs" / f"{d}.pdf"
        docs.append({"doc_name": d, "company": info[d]["company"], "gics_sector": info[d]["gics_sector"],
                     "pdf": str(pdf), "sha256": sha256_file(pdf), "bytes": pdf.stat().st_size, "pages": pages[d],
                     "source_url": FB_PDF_URL.format(doc=d), "fiscal_year_end": FY_END[d]})
    return {"schema": "agg113_corpus.v1", "fiscal_year": 2022, "n_docs": len(docs),
            "selection": "selection.json (rule R1-R5; all eligible FinanceBench fiscal-2022 10-Ks not used in AGG-1.x design)",
            "selection_sha256": sha256_file(HERE / "selection.json"), "pilot_docs": sel["pilot_docs"],
            "manifest_digest_sha256": digest_docs(docs), "docs": docs}


# ------------------------------------------------------------------------------------------------ golds
def load_sheet(path: Path):
    sheet = A0.read_sheet(path)
    company = {d: r["company"] for (d, _), r in sheet.items()}
    docs = sorted({d for d, _ in sheet})
    return sheet, company, docs


def _cells(expr: str) -> list[str]:
    return expr.replace("/", "-").split("-")


def _where_ok(sheet, doc: str, w: dict[str, Any]) -> bool:
    v = A0.eval_expr(w["expr"], sheet, doc)
    if v in ("missing", None):
        return False
    return v > w["t"] if w["cmp"] == ">" else v < w["t"]


def compute(q: dict[str, Any], sheet, company: dict[str, str], docs: list[str]) -> dict[str, Any]:
    values: dict[str, float] = {}
    missing, na, filtered, evidence = [], [], [], []
    for d in docs:
        if d in q.get("exclude_docs", []):
            continue
        if q.get("where") and not _where_ok(sheet, d, q["where"]):
            filtered.append(d)
            continue
        v = A0.eval_expr(q["expr"], sheet, d)
        if v == "missing":
            missing.append(d)
        elif v is None:
            na.append(d)
        else:
            values[company[d]] = v
            for m in _cells(q["expr"]) + (_cells(q["where"]["expr"]) if q.get("where") else []):
                r = sheet[(d, m)]
                evidence.append({"doc": d, "metric": m, "value": float(r["value_usd_m"]), "page": int(r["page"])})
    gold = A0.compute_gold(q, values) if not missing and values else None
    return {"gold": gold, "per_company": values, "excluded_na": na, "missing_cells": missing,
            "filtered_out_by_where": filtered, "evidence": evidence}


def margin(q: dict[str, Any], values: dict[str, float], sheet, docs: list[str], company: dict[str, str]) -> dict[str, Any]:
    """Smallest relative gap that decides the answer, INCLUDING the gaps of a ``where`` filter (stricter than AGG-1.1)."""
    items = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    op, gaps = q["op"], []
    if op in ("argmax", "top3") and len(items) > 1:
        n = 1 if op == "argmax" else 3
        gaps = [Q.rel(items[i][1], items[i + 1][1]) for i in range(min(n, len(items) - 1))]
    elif op == "argmin" and len(items) > 1:
        gaps = [Q.rel(items[-1][1], items[-2][1])]
    elif op in ("count_gt", "count_lt"):
        t = q["t"]
        gaps = [Q.rel(v, t) for v in values.values() if v != t]
        if t == 0:
            gaps = [1.0 if v != 0 else 0.0 for v in values.values()]
        if any(v == t for v in values.values()):
            gaps.append(0.0)
    if q.get("where"):
        w = q["where"]
        for d in docs:
            if d in q.get("exclude_docs", []):
                continue
            wv = A0.eval_expr(w["expr"], sheet, d)
            if wv in ("missing", None):
                continue
            gaps.append(1.0 if w["t"] == 0 and wv != 0 else Q.rel(wv, w["t"]) if w["t"] else (0.0 if wv == 0 else 1.0))
    g = min(gaps) if gaps else None
    return {"min_rel_gap": None if g is None else round(g, 4), "near_tie": g is not None and g < MARGIN_MIN}


VERIFICATION_STANDARD = (
    "One reader (the AGG-1.13 step A agent, reading the extracted page text) plus a deterministic cross-extractor check: the "
    "quote must be found, whitespace/$-normalised, on the stated PDF page in BOTH pypdfium2 and pypdf text "
    "(../verify_phase1_cells.py -> facts_sheet_xcheck.json), and the stated value must be a number printed in the quote. A cell "
    "that fails is NA/unverified and excluded, never repaired by assertion. No LLM call produced or scored any value. This is "
    "the AGG-1.1 step A standard for its 144 non-Phase-0 cells.")


def select_questions(sheet, company, docs) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gen = Q.Gen(sheet, docs, company)
    kept, dropped, taken = [], [], {f: 0 for f in Q.QUOTA}
    for q in Q.templates(gen):
        f = q["family"]
        if taken[f] >= Q.QUOTA[f]:
            continue
        c = compute(q, sheet, company, docs)
        mg = margin(q, c["per_company"], sheet, docs, company)
        bad = [f"{e['doc']}:{e['metric']}" for e in c["evidence"]
               if sheet[(e["doc"], e["metric"])]["verified"] != "true" or sheet[(e["doc"], e["metric"])]["status"] != "filled"]
        if c["gold"] is None or mg["near_tie"] or bad or c["missing_cells"] or len(c["per_company"]) < 3:
            dropped.append({"id": q["id"], "family": f, "expr": q["expr"], "op": q["op"], "t": q.get("t"),
                            "reason": ("near-tie" if mg["near_tie"] else "unverified/missing cell" if bad or c["missing_cells"]
                                       else "fewer than 3 companies" if c["gold"] is not None else "no gold"), "margin": mg})
            continue
        taken[f] += 1
        kept.append(q)
    short = {f: Q.QUOTA[f] - n for f, n in taken.items() if n < Q.QUOTA[f]}
    assert not short, f"quota not met after the margin audit: {short}"
    for i, q in enumerate(kept, 1):  # contiguous final ids; the template id is kept for traceability
        q["template_id"] = q["id"]
        q["id"] = f"n13_{i:02d}"
    return kept, dropped


def derive(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    sheet, company, docs = load_sheet(out / "facts_sheet.csv")
    assert len(sheet) == len(docs) * len(METRICS), (len(sheet), len(docs))
    questions, dropped = select_questions(sheet, company, docs)
    golds, rec = [], {}
    for q in questions:
        c = compute(q, sheet, company, docs)
        bad = [f"{e['doc']}:{e['metric']}" for e in c["evidence"]
               if sheet[(e["doc"], e["metric"])]["verified"] != "true" or sheet[(e["doc"], e["metric"])]["status"] != "filled"]
        na_unverified = [f"{d}:{m}" for d in docs for m in _cells(q["expr"]) if (d, m) in sheet
                         and sheet[(d, m)]["status"] == "na" and sheet[(d, m)]["verified"] != "true"
                         and d not in q.get("exclude_docs", [])]
        mg = margin(q, c["per_company"], sheet, docs, company)
        ok = c["gold"] is not None and not bad and not na_unverified and not c["missing_cells"]
        g = {"id": q["id"], "template_id": q["template_id"], "family": q["family"], "question": q["q"], "expr": q["expr"],
             "op": q["op"], "threshold": q.get("t"), "where": q.get("where"), "gold": c["gold"],
             "per_company": c["per_company"], "excluded_na": c["excluded_na"], "missing_cells": c["missing_cells"],
             "filtered_out_by_where": c["filtered_out_by_where"], "excluded_by_rule": q.get("exclude_docs", []),
             "exclude_reason": q.get("exclude_reason"), "margin": mg, "n_cells": len(c["evidence"]),
             "evidence": c["evidence"], "gold_source": "derived:facts_sheet.csv", "verified": ok}
        golds.append(g)
        rec[q["id"]] = {"verified": ok, "verified_by": VERIFICATION_STANDARD if ok else None,
                        "verified_at": "2026-11-13" if ok else None, "unverified_cells": bad + na_unverified,
                        "n_cells": len(c["evidence"]), "near_tie": mg["near_tie"], "min_rel_gap": mg["min_rel_gap"]}
    return golds, {"schema": "agg113_verification.v1", "standard": VERIFICATION_STANDARD, "questions": rec,
                   "dropped_templates": dropped}, questions, dropped


def review_sheet(golds: list[dict[str, Any]], n_docs: int) -> str:
    L = ["# AGG-1.13 - gold review sheet", "",
         f"Generated by `build_agg113.py golds` from `facts_sheet.csv` ({n_docs} docs x 8 metrics). Every gold lists the cells it "
         "depends on (`evidence` in golds.jsonl: doc, metric, value, PDF page); the verbatim span for each cell is the `quote` "
         "column of `facts_sheet.csv`. A question is `verified` only if every contributing cell is a verified filled cell; an "
         "unverified question is excluded from scoring, never assumed.", "", f"Standard: {VERIFICATION_STANDARD}", "",
         "| id | family | rule | gold | n companies | excluded NA | min rel gap | near-tie | verified |",
         "|---|---|---|---|---|---|---|---|---|"]
    for g in golds:
        w = g["where"]
        rule = f"{g['op']}({g['expr']}{'' if g['threshold'] is None else ', ' + str(g['threshold'])})"
        if w:
            rule += f" where {w['expr']} {w['cmp']} {w['t']}"
        L.append(f"| {g['id']} | {g['family']} | `{rule}` | {json.dumps(g['gold'])} | {len(g['per_company'])} | "
                 f"{len(g['excluded_na'])} | {g['margin']['min_rel_gap']} | {'YES' if g['margin']['near_tie'] else 'no'} | "
                 f"{'[x]' if g['verified'] else '[ ]'} |")
    return "\n".join(L) + "\n"


def questions_doc(questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {"schema": "agg113_questions.v1", "fiscal_year": 2022, "n_questions": len(questions),
            "generator": "agg113_questions.py (thresholds by pick_threshold, margin audit, per-family quota)",
            "questions": questions, "metrics": A0.METRICS}


def write_all(out: Path) -> dict[str, Any]:
    golds, rec, questions, dropped = derive(out)
    with (out / "golds.jsonl").open("w") as f:
        for g in golds:
            f.write(json.dumps(g) + "\n")
    (out / "verification_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "aggregation_questions.json").write_text(json.dumps(questions_doc(questions), indent=1) + "\n")
    sheet, _, docs = load_sheet(out / "facts_sheet.csv")
    (out / "GOLD_REVIEW.md").write_text(review_sheet(golds, len(docs)))
    return {"n_questions": len(golds), "verified": sum(g["verified"] for g in golds),
            "unverified": [g["id"] for g in golds if not g["verified"]], "dropped_templates": [d["id"] for d in dropped],
            "near_tie": [g["id"] for g in golds if g["margin"]["near_tie"]]}


def _same(a: Any, b: Any) -> bool:
    """Structural equality; floats compare by isclose (sum() is compensated on py3.12, plain on py3.11)."""
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool)
                and math.isclose(a, b, rel_tol=1e-9, abs_tol=0.0101))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def check(out: Path) -> int:
    golds, rec, questions, _ = derive(out)
    ok = True
    on_disk = [json.loads(l) for l in (out / "golds.jsonl").read_text().splitlines()]
    if not _same(on_disk, json.loads(json.dumps(golds))):
        print("golds.jsonl differs from a fresh derivation")
        ok = False
    if not _same(json.loads((out / "verification_record.json").read_text()), json.loads(json.dumps(rec))):
        print("verification_record.json differs")
        ok = False
    if not _same(json.loads((out / "aggregation_questions.json").read_text()), json.loads(json.dumps(questions_doc(questions)))):
        print("aggregation_questions.json differs")
        ok = False
    corpus = json.loads((out / "corpus.json").read_text())
    if corpus["manifest_digest_sha256"] != digest_docs(corpus["docs"]):
        print("corpus manifest digest mismatch")
        ok = False
    sel = json.loads((out / "selection.json").read_text())
    if sorted(d["doc_name"] for d in corpus["docs"]) != sel["selected"]:
        print("corpus docs differ from selection.json")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("select", "corpus", "golds", "check"))
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--pages", type=Path, help="JSON {doc_name: page_count} (corpus only)")
    a = ap.parse_args(argv)
    if a.cmd == "select":
        s = build_selection()
        (a.out / "selection.json").write_text(json.dumps(s, indent=1) + "\n")
        print(json.dumps({"n_eligible": s["n_eligible"], "selected": s["selected"], "pilot": s["pilot_docs"]}))
    elif a.cmd == "corpus":
        c = build_corpus(json.loads(a.pages.read_text()))
        (a.out / "corpus.json").write_text(json.dumps(c, indent=1) + "\n")
        print(json.dumps({"n_docs": c["n_docs"], "manifest_digest_sha256": c["manifest_digest_sha256"]}))
    elif a.cmd == "golds":
        print(json.dumps(write_all(a.out)))
    else:
        return check(a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
