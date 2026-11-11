#!/usr/bin/env python3
"""AGG-1.3 step A: the changed-corpus manifest (``corpus_agg13.json``). No model, no network.

The AGG-1.1 corpus with 3 slots swapped to their FY2021 filings (``PREDECLARED_AGG13.md``, "The 3 replaced documents").
A replaced slot keeps its ``doc_name`` (the store ``doc_id``); the new file name, sha256, bytes, pages and the sha256 of
the file it replaces are recorded beside it. The 25 other entries are copied byte-for-byte from ``corpus.json``.

    build_agg13.py corpus [--libpath DIR]   write corpus_agg13.json (needs pypdfium2 for page counts)
    build_agg13.py check                    recompute in memory and fail non-zero if the committed file differs
                                             (stdlib only: page counts are read from the committed file)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FB_DIR1 = Path("/mnt/raid0/llm/datasets/financebench-20261111-agg1/pdfs")
FB_PDF_URL = "https://github.com/patronus-ai/financebench/raw/main/pdfs/{doc}.pdf"
FISCAL_YEAR = 2021
# slot (store doc_id) -> replacement FinanceBench file stem; fiscal-year end of the replacement filing
REPLACEMENTS = {
    "MICROSOFT_2022_10K": ("MICROSOFT_2021_10K", "2021-06-30"),
    "JOHNSON_JOHNSON_2022_10K": ("JOHNSON_JOHNSON_2021_10K", "2022-01-02 (fiscal 2021)"),
    "3M_2022_10K": ("3M_2021_10K", "2021-12-31"),
}
SELECTION = ("the AGG-1.1 corpus (28 FinanceBench FY2022 10-Ks) with the 3 AGG-1.1 pilot documents replaced by the same "
             "companies' FY2021 10-Ks (the nearest other fiscal year for which FinanceBench has a 10-K for all three); "
             "a replaced slot keeps its doc_name as the store doc_id")


def digest_docs(docs: list[dict[str, Any]]) -> str:
    """Same canonical digest as ``build_phase1.digest_docs`` (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(docs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(base: dict[str, Any], pages: dict[str, int], fb_dir: Path = FB_DIR1) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    for d in base["docs"]:
        if d["doc_name"] not in REPLACEMENTS:
            docs.append(dict(d))
            continue
        stem, fye = REPLACEMENTS[d["doc_name"]]
        pdf = fb_dir / f"{stem}.pdf"
        e = dict(d)
        e.update(pdf=str(pdf), sha256=sha256_file(pdf), bytes=pdf.stat().st_size, pages=pages[stem],
                 source_url=FB_PDF_URL.format(doc=stem), fiscal_year_end=fye, fiscal_year=FISCAL_YEAR,
                 file_name=stem, replaces={"file_name": d["doc_name"], "sha256": d["sha256"], "bytes": d["bytes"],
                                           "pages": d["pages"]}, fb_questions=[])
        docs.append(e)
    return {"schema": "agg13_corpus.v1", "base": "corpus.json (AGG-1.1)", "fiscal_year_replaced": FISCAL_YEAR,
            "n_docs": len(docs), "n_replaced": len(REPLACEMENTS), "selection": SELECTION,
            "manifest_digest_sha256": digest_docs(docs), "docs": docs}


def page_counts(libpath: list[str], fb_dir: Path = FB_DIR1) -> dict[str, int]:
    sys.path[:0] = libpath
    import pypdfium2 as pdfium  # noqa: PLC0415
    return {stem: len(pdfium.PdfDocument(str(fb_dir / f"{stem}.pdf"))) for stem, _ in REPLACEMENTS.values()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("corpus", "check"))
    ap.add_argument("--libpath", action="append", default=[])
    ap.add_argument("--out", type=Path, default=HERE)
    a = ap.parse_args(argv)
    base = json.loads((HERE / "corpus.json").read_text())
    target = a.out / "corpus_agg13.json"
    if a.cmd == "corpus":
        new = build(base, page_counts(a.libpath))
        target.write_text(json.dumps(new, indent=1) + "\n")
        print(f"wrote {target}: {new['n_docs']} docs, {new['n_replaced']} replaced, digest {new['manifest_digest_sha256']}")
        return 0
    cur = json.loads(target.read_text())
    pages = {d["file_name"]: d["pages"] for d in cur["docs"] if "file_name" in d}
    new = build(base, pages)
    ok = new == cur
    print("corpus_agg13.json " + ("matches" if ok else "DIFFERS from a fresh build"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
