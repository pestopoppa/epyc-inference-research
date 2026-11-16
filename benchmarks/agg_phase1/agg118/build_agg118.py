#!/usr/bin/env python3
"""AGG-1.18 step 1: the held-out corpus manifest and the gold facts sheet (no model, no pipeline output).

    python3 -I build_agg118.py write  --pdfium-site <site-packages with pypdfium2>   writes corpus.json, facts_sheet.csv, gold_check.json
    python3 -I build_agg118.py check  --pdfium-site <...>                            recompute in memory; non-zero if a committed file differs

Every gold quote is located in the pypdfium2 page text (whitespace collapsed per line): page and 1-based line are recorded, the quote must be
a substring of that line, and every printed number of the gold value must appear in the quote. pypdfium2 is the ONLY extractor (no
second-extractor cross-check in this step; AGG-1.13 used pypdf as a second reader).
PDFs are untrusted data: run with ``python3 -I``; this script lives outside the PDF directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gold_spec as G  # noqa: E402

CORPUS_ID = "agg118_heldout10"
FB_DIR = Path("/mnt/raid0/llm/datasets/financebench-agg118-heldout")
FB_PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf"   # the URL form of AGG-1.13 build_agg113.py, reused exactly
DOCINFO = HERE.parent / "agg113" / "financebench_document_information.jsonl"
METRICS = ["revenue", "net_income", "capex", "total_assets", "operating_cash_flow", "total_current_assets",
           "total_current_liabilities", "dividends_paid"]                                # agg_phase0 METRICS (the 8 SCHEMA_FIELDS)
FIELDS = METRICS + ["long_term_debt"]
PILOT_DOCS = ["NETFLIX_2019_10K", "SALESFORCE_2023_10K", "BOEING_2019_10K"]
FY_END = {"FEDEX_2023_10K": "2023-05-31", "SALESFORCE_2023_10K": "2023-01-31", "PEPSICO_2019_10K": "2019-12-28"}
REVIEWER = "agg118-reader1 pypdfium2 2026-11-16"
COLS = ["doc_name", "company", "metric", "fiscal_year", "value_usd_m", "status", "page", "line", "quote", "reviewer", "note", "unit",
        "confidence", "verified"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def digest_docs(docs: list[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def page_lines(pdfium: Any, pdf: Path) -> list[list[str]]:
    d = pdfium.PdfDocument(str(pdf))
    out = []
    for i in range(len(d)):
        t = d[i].get_textpage().get_text_range().replace("\r\n", "\n").replace("\r", "\n")
        out.append([re.sub(r"\s+", " ", l).strip() for l in t.split("\n")])
    return out


def fmt(v: float | None, div: int) -> str:
    if v is None:
        return ""
    x = v / div
    return str(int(x)) if x == int(x) else f"{x:.3f}".rstrip("0").rstrip(".")


def build(pdfium: Any) -> tuple[dict[str, Any], str, dict[str, Any]]:
    info = {json.loads(l)["doc_name"]: json.loads(l) for l in DOCINFO.read_text().splitlines() if l.strip()}
    docs, rows, checks, problems = [], [], [], []
    for d, (company, fy, fy1, div, fynote) in G.DOCS.items():
        pdf = FB_DIR / "pdfs" / f"{d}.pdf"
        pages = page_lines(pdfium, pdf)
        docs.append({"doc_name": d, "company": info[d]["company"], "gics_sector": info[d]["gics_sector"], "pdf": str(pdf),
                     "sha256": sha256_file(pdf), "bytes": pdf.stat().st_size, "pages": len(pages),
                     "source_url": FB_PDF_URL.format(doc=d), "fiscal_year_end": FY_END.get(d, f"{fy}-12-31"),
                     "fiscal_year": fy, "fiscal_year_prior": fy1})
        for m in FIELDS:
            page, quote, v0, v1, status, note, conf = G.SPEC[d][m]
            hits = [i + 1 for i, l in enumerate(pages[page - 1]) if quote in l]
            if len(hits) != 1:
                problems.append(f"{d} {m}: quote found on {len(hits)} lines of page {page}")
            line = hits[0] if hits else None
            for year, v in ((fy, v0), (fy1, v1)):
                ok_num = v is None or f"{abs(v):,}" in quote
                if not ok_num:
                    problems.append(f"{d} {m} {year}: printed number {abs(v):,} not in the quote")
                checks.append({"doc": d, "metric": m, "fiscal_year": year, "page": page, "line": line, "quote_found": bool(hits),
                               "number_in_quote": ok_num})
                n = "; ".join(x for x in (note, fynote, f"columns {fy} {fy1}") if x)
                rows.append({"doc_name": d, "company": company, "metric": m, "fiscal_year": year, "value_usd_m": fmt(v, div),
                             "status": status, "page": page, "line": line, "quote": quote, "reviewer": REVIEWER, "note": n,
                             "unit": "USD millions", "confidence": conf, "verified": "true" if hits and ok_num else "false"})
    corpus = {"schema": "agg118_corpus.v1", "corpus_id": CORPUS_ID, "n_docs": len(docs),
              "selection": "AGG-1.18 box: FEDEX_2023 + SALESFORCE_2023 (never-seen issuers) + 8 FY2019 10-Ks of AGG-1.1-only issuers "
                           "(not AGG-1.13, not the replacement docs); MGMRESORTS_2019 and NETFLIX_2019 report in USD thousands",
              "source": "patronus-ai/financebench main, " + FB_PDF_URL, "extractor": "pypdfium2 (single extractor)",
              "pilot_docs": PILOT_DOCS, "manifest_digest_sha256": digest_docs(docs), "docs": docs}
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    st = [r["status"] for r in rows]
    check = {"schema": "agg118_gold_check.v1", "extractor": "pypdfium2", "n_cells": len(rows), "filled": st.count("filled"),
             "na": st.count("na"), "excluded": st.count("EXCLUDED"), "verified": sum(r["verified"] == "true" for r in rows),
             "problems": problems, "cells": checks}
    return corpus, buf.getvalue(), check


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("write", "check"))
    ap.add_argument("--pdfium-site", default="/mnt/raid0/llm/venv/lib/python3.12/site-packages")
    a = ap.parse_args(argv)
    sys.path.append(a.pdfium_site)
    import pypdfium2  # noqa: PLC0415
    corpus, sheet, check = build(pypdfium2)
    check["pypdfium2_version"] = str(getattr(pypdfium2, "PYPDFIUM_INFO", "5.3.0"))
    out = {"corpus.json": json.dumps(corpus, indent=1) + "\n", "facts_sheet.csv": sheet, "gold_check.json": json.dumps(check, indent=1) + "\n"}
    if a.cmd == "write":
        for k, v in out.items():
            (HERE / k).write_text(v)
        print(json.dumps({k: check[k] for k in ("n_cells", "filled", "na", "excluded", "verified", "problems")}))
        return 1 if check["problems"] else 0
    bad = [k for k, v in out.items() if not (HERE / k).exists() or (HERE / k).read_text() != v]
    print("check ok" if not bad else f"differs: {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
