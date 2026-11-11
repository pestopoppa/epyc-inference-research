#!/usr/bin/env python3
"""AGG-1.1 step A: Phase 1 corpus manifest, gold derivation, margin audit and verification record (stdlib only).

No model, network or PDF parsing happens here (cell values live in ``facts_sheet.csv``; they were read from the
PDF text and cross-checked by ``verify_phase1_cells.py``, which is the only file here that opens a PDF).

Subcommands:
    corpus   write corpus.json (28 docs, sha256 pinned) with its manifest digest from a page-count table
    golds    derive golds.jsonl, verification_record.json and GOLD_REVIEW.md from facts_sheet.csv
    check    recompute everything in memory and fail non-zero if a committed file differs (CI/self-test)

Question semantics, scoring and kill gates are those of ``benchmarks/agg_phase0`` (imported, not copied).
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
sys.path.insert(0, str(HERE.parent / "agg_phase0"))
import agg_phase0 as A0  # noqa: E402
import phase1_questions as Q  # noqa: E402

FB_DIR0 = Path("/mnt/raid0/llm/datasets/financebench-20261109")
FB_DIR1 = Path("/mnt/raid0/llm/datasets/financebench-20261111-agg1")
FB_PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf"

NEW_DOCS = [
    "3M_2022_10K", "ACTIVISIONBLIZZARD_2022_10K", "ADOBE_2022_10K", "AMAZON_2022_10K", "AMD_2022_10K",
    "AMERICANEXPRESS_2022_10K", "APPLE_2022_10K", "BOEING_2022_10K", "COSTCO_2022_10K", "CVSHEALTH_2022_10K",
    "EBAY_2022_10K", "JOHNSON_JOHNSON_2022_10K", "KRAFTHEINZ_2022_10K", "MCDONALDS_2022_10K",
    "MGMRESORTS_2022_10K", "MICROSOFT_2022_10K", "NETFLIX_2022_10K", "PFIZER_2022_10K",
]
COMPANY = {  # company display names used in prompts and golds (verbatim corpus names)
    "3M_2022_10K": "3M", "ACTIVISIONBLIZZARD_2022_10K": "Activision Blizzard", "ADOBE_2022_10K": "Adobe",
    "AMAZON_2022_10K": "Amazon", "AMD_2022_10K": "AMD", "AMERICANEXPRESS_2022_10K": "American Express",
    "APPLE_2022_10K": "Apple", "BOEING_2022_10K": "Boeing", "COSTCO_2022_10K": "Costco",
    "CVSHEALTH_2022_10K": "CVS Health", "EBAY_2022_10K": "eBay", "JOHNSON_JOHNSON_2022_10K": "Johnson & Johnson",
    "KRAFTHEINZ_2022_10K": "Kraft Heinz", "MCDONALDS_2022_10K": "McDonald's", "MGMRESORTS_2022_10K": "MGM Resorts",
    "MICROSOFT_2022_10K": "Microsoft", "NETFLIX_2022_10K": "Netflix", "PFIZER_2022_10K": "Pfizer",
}
# fiscal-year end of the filing's own "fiscal 2022" where it is not 31 Dec 2022
FY_END = {"ADOBE_2022_10K": "2022-12-02", "APPLE_2022_10K": "2022-09-24", "COSTCO_2022_10K": "2022-08-28",
          "MICROSOFT_2022_10K": "2022-06-30", "JOHNSON_JOHNSON_2022_10K": "2023-01-01 (fiscal 2022)",
          "KRAFTHEINZ_2022_10K": "2022-12-31", "AMD_2022_10K": "2022-12-31"}
DOCS = list(A0.DOCS) + NEW_DOCS
assert len(DOCS) == 28 and len(set(DOCS)) == 28

MARGIN_MIN = 0.015  # relative gap below which a ranking / threshold call is "near-tie"
VERIFICATION_STANDARD = (
    "Phase 0 cells (80): two independent readers (blind second sheet), per agg_phase0/verification_record.json. "
    "Phase 1 new cells (144): one reader (the AGG-1.1 step A agent, reading the extracted page text) plus a "
    "deterministic cross-extractor check: the quote must be found, whitespace/$-normalised, on the stated PDF page "
    "in BOTH pypdfium2 and pypdf text (verify_phase1_cells.py). No LLM call produced or scored any value.")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest_docs(docs: list[dict[str, Any]]) -> str:
    """sha256 of the canonical JSON (sorted keys, no whitespace) of the docs array."""
    return hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------ corpus
def build_corpus(pages: dict[str, int]) -> dict[str, Any]:
    fb_rows = [json.loads(l) for l in (FB_DIR0 / "financebench_open_source.jsonl").read_text().splitlines() if l.strip()]
    fbq: dict[str, list[str]] = {}
    for r in fb_rows:
        fbq.setdefault(r["doc_name"], []).append(r["financebench_id"])
    p0 = {d["doc_name"]: d for d in json.loads((HERE.parent / "agg_phase0" / "corpus.json").read_text())["docs"]}
    docs = []
    for d in DOCS:
        in0 = d in p0
        pdf = (FB_DIR0 if in0 else FB_DIR1) / "pdfs" / f"{d}.pdf"
        sha = sha256_file(pdf)
        if in0 and sha != p0[d]["sha256"]:
            raise SystemExit(f"{d}: sha256 differs from the Phase 0 pin")
        docs.append({
            "doc_name": d, "company": p0[d]["company"] if in0 else COMPANY[d],
            "phase0": in0, "pdf": str(pdf), "sha256": sha, "bytes": pdf.stat().st_size, "pages": pages[d],
            "source_url": FB_PDF_URL.format(doc=d), "fiscal_year_end": FY_END.get(d, "2022-12-31"),
            "fb_questions": fbq.get(d, [])})
    return {"schema": "agg_phase1_corpus.v1", "fiscal_year": 2022,
            "fb_jsonl_sha256": A0.FB_JSONL_SHA256, "n_docs": len(docs),
            "n_phase0_docs": sum(d["phase0"] for d in docs),
            "selection": ("the 10 Phase 0 FinanceBench FY2022 10-Ks (unchanged, same pins) + 18 more FinanceBench "
                          "FY2022 10-Ks chosen for sector spread, unit traps (USD thousands: MGM Resorts, Netflix), "
                          "reversed column order (Amazon), net losses (Amazon, Boeing, eBay), no-dividend filers and "
                          "an unclassified balance sheet (American Express); fiscal 2022 as each filing labels it"),
            "manifest_digest_sha256": digest_docs(docs), "docs": docs}


# ------------------------------------------------------------------------------------------------ golds
def _where_ok(sheet, doc: str, w: dict[str, Any]) -> bool:
    v = A0.eval_expr(w["expr"], sheet, doc)
    if v in ("missing", None):
        return False
    return v > w["t"] if w["cmp"] == ">" else v < w["t"]


def _cells(expr: str) -> list[str]:
    out = []
    for tok in expr.replace("/", "-").split("-"):
        out.append(tok)
    return out


def compute(q: dict[str, Any], sheet, company: dict[str, str]) -> dict[str, Any]:
    values: dict[str, float] = {}
    missing, na, filtered, evidence = [], [], [], []
    for d in DOCS:
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


def margin(q: dict[str, Any], values: dict[str, float]) -> dict[str, Any]:
    """Smallest relative gap that decides the answer; a gap < MARGIN_MIN is a near-tie (flagged, not hidden)."""
    items = sorted(values.items(), key=lambda kv: (-kv[1], kv[0]))
    op, gaps = q["op"], []

    def rel(a: float, b: float) -> float:
        den = max(abs(a), abs(b))
        return abs(a - b) / den if den else 0.0

    if op in ("argmax", "top3") and len(items) > 1:
        n = 1 if op == "argmax" else 3
        gaps = [rel(items[i][1], items[i + 1][1]) for i in range(min(n, len(items) - 1))]
    elif op == "argmin" and len(items) > 1:
        gaps = [rel(items[-1][1], items[-2][1])]
    elif op in ("count_gt", "count_lt"):
        t = q["t"]
        gaps = [rel(v, t) for v in values.values() if v != t]
        if t == 0:
            gaps = [1.0 if v != 0 else 0.0 for v in values.values()]
    g = min(gaps) if gaps else None
    return {"min_rel_gap": None if g is None else round(g, 4), "near_tie": g is not None and g < MARGIN_MIN}


def load_sheet(path: Path):
    sheet = A0.read_sheet(path)
    company = {d: r["company"] for (d, _), r in sheet.items()}
    return sheet, company


def derive(out: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sheet, company = load_sheet(out / "facts_sheet.csv")
    n_expect = len(DOCS) * len(A0.METRICS)
    assert len(sheet) == n_expect, (len(sheet), n_expect)
    golds, rec = [], {}
    for q in Q.ALL:
        c = compute(q, sheet, company)
        # a question is verified only if every contributing cell is a verified cell (filled+true) and the answer exists
        bad = []
        for e in c["evidence"]:
            r = sheet[(e["doc"], e["metric"])]
            if r["verified"] != "true" or r["status"] != "filled":
                bad.append(f"{e['doc']}:{e['metric']}")
        na_unverified = [f"{d}:{m}" for d in DOCS for m in _cells(q["expr"]) if (d, m) in sheet
                         and sheet[(d, m)]["status"] == "na" and sheet[(d, m)]["verified"] != "true"
                         and d not in q.get("exclude_docs", [])]
        mg = margin(q, c["per_company"])
        ok = c["gold"] is not None and not bad and not na_unverified and not c["missing_cells"]
        g = {"id": q["id"], "family": q["family"], "continuity": q["family"] == "continuity",
             "phase0_id": q.get("phase0_id"), "question": q["q"], "expr": q["expr"], "op": q["op"],
             "threshold": q.get("t"), "where": q.get("where"), "gold": c["gold"], "per_company": c["per_company"],
             "excluded_na": c["excluded_na"], "missing_cells": c["missing_cells"],
             "filtered_out_by_where": c["filtered_out_by_where"],
             "excluded_by_rule": q.get("exclude_docs", []), "exclude_reason": q.get("exclude_reason"),
             "text_changed_from_phase0": q.get("text_changed_from_phase0", False),
             "margin": mg, "n_cells": len(c["evidence"]), "evidence": c["evidence"],
             "gold_source": "derived:facts_sheet.csv", "verified": ok}
        golds.append(g)
        rec[q["id"]] = {"verified": ok, "verified_by": VERIFICATION_STANDARD if ok else None,
                        "verified_at": "2026-11-11" if ok else None,
                        "unverified_cells": bad + na_unverified, "n_cells": len(c["evidence"]),
                        "near_tie": mg["near_tie"], "min_rel_gap": mg["min_rel_gap"]}
    return golds, {"schema": "agg_phase1_verification.v1", "standard": VERIFICATION_STANDARD, "questions": rec}


def review_sheet(golds: list[dict[str, Any]], sheet) -> str:
    L = ["# AGG Phase 1 - gold review sheet", "",
         "Generated by `build_phase1.py golds` from `facts_sheet.csv` (28 docs x 8 metrics). Every gold lists the "
         "cells it depends on (`evidence` in golds.jsonl: doc, metric, value, PDF page); the verbatim span for each "
         "cell is the `quote` column of `facts_sheet.csv`. A question is `verified` only if every contributing cell "
         "is a verified filled cell; an unverified question is excluded from scoring, never assumed.", "",
         f"Standard: {VERIFICATION_STANDARD}", "",
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


def write_all(out: Path) -> dict[str, Any]:
    golds, rec = derive(out)
    sheet, _ = load_sheet(out / "facts_sheet.csv")
    with (out / "golds.jsonl").open("w") as f:
        for g in golds:
            f.write(json.dumps(g) + "\n")
    (out / "verification_record.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "aggregation_questions.json").write_text(json.dumps(
        {"schema": "agg_phase1_questions.v1", "fiscal_year": 2022, "n_continuity": len(Q.CONTINUITY),
         "n_new": len(Q.NEW), "questions": Q.ALL, "metrics": A0.METRICS}, indent=1) + "\n")
    (out / "GOLD_REVIEW.md").write_text(review_sheet(golds, sheet))
    return {"n_questions": len(golds), "verified": sum(g["verified"] for g in golds),
            "unverified": [g["id"] for g in golds if not g["verified"]],
            "near_tie": [g["id"] for g in golds if g["margin"]["near_tie"]]}


def check(out: Path) -> int:
    golds, rec = derive(out)
    ok = True
    on_disk = [json.loads(l) for l in (out / "golds.jsonl").read_text().splitlines()]
    if on_disk != json.loads(json.dumps(golds)):
        print("golds.jsonl differs from a fresh derivation")
        ok = False
    if json.loads((out / "verification_record.json").read_text()) != rec:
        print("verification_record.json differs")
        ok = False
    corpus = json.loads((out / "corpus.json").read_text())
    if corpus["manifest_digest_sha256"] != digest_docs(corpus["docs"]):
        print("corpus manifest digest mismatch")
        ok = False
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("corpus", "golds", "check"))
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--pages", type=Path, help="JSON {doc_name: page_count} (corpus only)")
    a = ap.parse_args(argv)
    if a.cmd == "corpus":
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
