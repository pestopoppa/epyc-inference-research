#!/usr/bin/env python3
"""AGG-1.4 field pick: the lowest L3 recall on the 28-document store among fields no L4 unit has extracted.

No model, no network. Reads (read-only, ``immutable=1``) the AGG-1.1 CPU store and the 28 corpus PDFs
(``corpus.json`` paths, pypdfium2 text). Deterministic: same store + same PDFs -> same JSON.

Rule (written into ``PREDECLARED_AGG14.md`` before any AGG-1.4 data):

1. CANDIDATES = the canonical L3 field names that ``facts_det.FIELD_ALIASES`` can emit, MINUS every field that has an
   L4 unit in the store (the 8 AGG-1.1 metrics) and MINUS aliases of those 8 (``net_income_attributable`` is
   ``net_income``; ``revenue``, ``capex``, ... are the metrics themselves). Nothing else is excluded here.
2. APPLICABLE docs of a candidate = documents whose PDF text has a statement-style line: a line that starts with one of
   the field's label aliases and carries a number of >= 3 digits within 40 characters. This is the deterministic,
   gold-free denominator for "recall" (a label printed with a figure means the filing states the field).
3. L3 RECALL PROXY = (applicable docs with >= 1 L3 fact for the field whose period contains the fiscal year) / applicable.
   Counted per document, not per value: it is an UPPER bound of true recall (a wrong value counts as a hit).
4. ELIGIBLE = applicable >= 20 (a field with fewer than 20 documents cannot support M1 >= 0.75 against a gate with
   a binomial SE below ~0.1; an ineligible field is reported, never silently dropped).
5. PICK = lowest proxy among the eligible; ties broken by field name.

The units to backfill are the documents with NO canonical row for (field, fiscal year) after L3 + the existing L4
(``select_canonical``, in memory): that is the lazy backfill's trigger set.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# label aliases per candidate (from facts_det.FIELD_ALIASES prefixes, plus the verbatim variants filings print)
LABELS: dict[str, list[str]] = {
    "operating_income": ["operating income", "income from operations"],
    "total_liabilities": ["total liabilities"],
    "depreciation_amortization": ["depreciation and amortization"],
    "cash_and_equivalents": ["cash and cash equivalents"],
    "inventories": ["inventories", "total inventories"],
    "accounts_payable": ["accounts payable"],
    "long_term_debt": ["long-term debt"],
    "total_equity": ["total equity", "total shareholders' equity", "total stockholders' equity",
                     "total shareholders’ equity", "total stockholders’ equity"],
    "cogs": ["cost of sales", "cost of goods sold", "cost of revenue", "cost of revenues", "cost of products sold"],
}
L4_FIELDS = ("revenue", "net_income", "capex", "total_assets", "operating_cash_flow", "total_current_assets",
             "total_current_liabilities", "dividends_paid")
ALIAS_OF_L4 = ("net_income_attributable",)
NUM = re.compile(r"^[^\d\n]{0,40}\(?\$?\s*\(?\d[\d,]{2,}")
MIN_APPLICABLE = 20


def candidates() -> list[str]:
    """Canonical field names facts_det can emit, minus L4 fields and their aliases (the LABELS keys, asserted)."""
    return sorted(LABELS)


def page_texts(pdf: str) -> list[str]:
    import pypdfium2 as pdfium  # noqa: PLC0415
    d = pdfium.PdfDocument(pdf)
    return [d[i].get_textpage().get_text_range() for i in range(len(d))]


def applicable_docs(texts: dict[str, list[str]], labels: list[str]) -> set[str]:
    out: set[str] = set()
    for doc, pages in texts.items():
        hit = False
        for pg in pages:
            for line in pg.splitlines():
                low = line.strip().lower()
                if any(low.startswith(a) and NUM.match(low[len(a):]) for a in labels):
                    hit = True
                    break
            if hit:
                break
        if hit:
            out.add(doc)
    return out


def pick(store_path: Path, corpus_path: Path, fiscal_year: str = "2022",
         texts: dict[str, list[str]] | None = None) -> dict:
    corpus = json.loads(corpus_path.read_text())
    docs = [d["doc_name"] for d in corpus["docs"]]
    if texts is None:
        texts = {d["doc_name"]: page_texts(d["pdf"]) for d in corpus["docs"]}
    c = sqlite3.connect(f"file:{store_path}?immutable=1", uri=True)
    l4_in_store = {r[0] for r in c.execute("SELECT DISTINCT field FROM facts WHERE extractor = 'l4_llm.v1'")}
    assert set(L4_FIELDS) == l4_in_store, (sorted(l4_in_store), L4_FIELDS)
    assert not (set(LABELS) & (set(L4_FIELDS) | set(ALIAS_OF_L4)))
    rows = []
    for f in candidates():
        app = applicable_docs(texts, LABELS[f])
        cov = {r[0] for r in c.execute(
            "SELECT DISTINCT doc_id FROM facts WHERE field = ? AND extractor LIKE 'l3%' AND period LIKE ?",
            (f, f"%{fiscal_year}%"))}
        hit = cov & app
        rows.append({"field": f, "applicable_docs": len(app), "l3_hit_among_applicable": len(hit),
                     "l3_docs_total": len(cov), "l3_recall_proxy": round(len(hit) / len(app), 4) if app else None,
                     "eligible": len(app) >= MIN_APPLICABLE, "applicable": sorted(app), "l3_docs": sorted(cov)})
    elig = [r for r in rows if r["eligible"] and r["l3_recall_proxy"] is not None]
    best = min(elig, key=lambda r: (r["l3_recall_proxy"], r["field"]))
    return {"schema": "agg14_field_pick.v1", "fiscal_year": fiscal_year, "n_docs": len(docs),
            "rule": "agg14_field_pick.py module docstring (steps 1-5)",
            "l4_fields_in_store": sorted(l4_in_store), "candidates": rows,
            "picked": best["field"], "picked_proxy": best["l3_recall_proxy"],
            "picked_applicable": best["applicable_docs"], "picked_l3_docs": best["l3_docs"]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libpath", action="append", default=[])
    ap.add_argument("--store", type=Path, default=Path("/mnt/raid0/llm/tmp/agg1-ingest-cpu-20261111/out/store.sqlite"))
    ap.add_argument("--corpus", type=Path, default=HERE / "corpus.json")
    ap.add_argument("--out", type=Path, default=HERE / "agg14_field_pick.json")
    a = ap.parse_args(argv)
    sys.path[:0] = a.libpath
    res = pick(a.store, a.corpus)
    a.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for r in res["candidates"]:
        print(f"{r['field']:28s} applicable {r['applicable_docs']:2d}  L3 hit {r['l3_hit_among_applicable']:2d}  "
              f"proxy {r['l3_recall_proxy']}  eligible {r['eligible']}")
    print("PICK", res["picked"], res["picked_proxy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
