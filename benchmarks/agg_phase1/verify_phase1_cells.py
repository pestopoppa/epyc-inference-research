#!/usr/bin/env python3
"""AGG-1.1 step A cell cross-check: every cell's quote must be found on its stated PDF page in TWO independent
text extractors (pypdfium2 and pypdf), and the stated value must equal a number printed in that quote.

No model is called. Needs the 28 PDFs (corpus.json paths), ``pypdfium2`` and ``pypdf`` (+ ``cryptography`` for
AES-encrypted filings). Pass ``--libpath`` for site-packages dirs that hold them. Exit 1 if any cell fails.

Output ``facts_sheet_xcheck.json``: per cell {pdfium_found, pypdf_found, value_in_quote, pages_with_value}
(``pages_with_value`` = how many distinct PDF pages print the same figure, a corroboration count, informational).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sq(s: str) -> str:
    return re.sub(r"[\s$]+", "", s)


def number_strings(value: float) -> list[str]:
    """Printed forms of a USD-millions value: as millions (with the filing's decimals) and as thousands."""
    out = []
    for v, dec in ((value, None), (value * 1000, 0)):
        v = round(v, 3)
        if dec == 0:
            if abs(v - round(v)) > 1e-6:
                continue
            out.append(f"{int(round(abs(v))):,}")
        else:
            s = f"{abs(v):,.3f}".rstrip("0").rstrip(".")
            out.append(s)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--libpath", action="append", default=[])
    ap.add_argument("--sheet", type=Path, default=HERE / "facts_sheet.csv")
    ap.add_argument("--corpus", type=Path, default=HERE / "corpus.json")
    ap.add_argument("--out", type=Path, default=HERE / "facts_sheet_xcheck.json")
    a = ap.parse_args(argv)
    sys.path[:0] = a.libpath
    import pypdf  # noqa: PLC0415
    import pypdfium2 as pdfium  # noqa: PLC0415

    pdfs = {d["doc_name"]: d["pdf"] for d in json.loads(a.corpus.read_text())["docs"]}
    rows = list(csv.DictReader(a.sheet.open(newline="")))
    cache_pdfium: dict[str, list[str]] = {}
    cache_pypdf: dict[str, pypdf.PdfReader] = {}
    res, fails = [], 0
    for r in rows:
        doc, page = r["doc_name"], int(r["page"] or 0)
        if not r["quote"]:
            res.append({"doc": doc, "metric": r["metric"], "skipped": "no quote (NA with no page evidence)"})
            continue
        if doc not in cache_pdfium:
            pd = pdfium.PdfDocument(pdfs[doc])
            cache_pdfium[doc] = [pd[i].get_textpage().get_text_range() for i in range(len(pd))]
            cache_pypdf[doc] = pypdf.PdfReader(pdfs[doc])
        q = sq(r["quote"])
        pdfium_found = q in sq(cache_pdfium[doc][page - 1])
        pypdf_found = q in sq(cache_pypdf[doc].pages[page - 1].extract_text())
        value_ok, nstr, corro = None, [], None
        if r["status"] == "filled":
            nstr = number_strings(float(r["value_usd_m"]))
            value_ok = any(n in r["quote"] for n in nstr)
            corro = sum(1 for t in cache_pdfium[doc] if any(n in t for n in nstr))
        ok = pdfium_found and pypdf_found and value_ok is not False
        fails += not ok
        res.append({"doc": doc, "metric": r["metric"], "page": page, "pdfium_found": pdfium_found,
                    "pypdf_found": pypdf_found, "value_in_quote": value_ok, "pages_with_value": corro,
                    "ok": ok})
        if not ok:
            print("FAIL", doc, r["metric"], page, r["quote"][:90])
    summary = {"cells": len(rows), "checked": sum(1 for x in res if "ok" in x), "fails": fails,
               "single_page_cells": sum(1 for x in res if x.get("pages_with_value") == 1)}
    a.out.write_text(json.dumps({"schema": "agg_phase1_xcheck.v1", "summary": summary, "cells": res}, indent=1) + "\n")
    print(json.dumps(summary))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
