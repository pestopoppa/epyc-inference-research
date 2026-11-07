#!/usr/bin/env python3
"""One-document probe of the orchestrator's ORIGINAL PDF fast path (PIP-04 / ODL-013 baseline arm).

Runs under the ORCHESTRATOR interpreter (not the research venv). Mirrors `PDFRouter.extract()` with
PDF_EXTRACTOR=pdftotext: `pdftotext -layout`, then the router's own entropy/garbage/word-length quality gate.
Prints one JSON line {text, latency_ms, method, needs_ocr, quality_score, chars}. The LightOnOCR leg is not run;
`needs_ocr` records where the router WOULD have left the fast path.
"""
import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orch-root", required=True)
    ap.add_argument("--pdf", required=True)
    a = ap.parse_args()
    os.environ["PDF_EXTRACTOR"] = "pdftotext"
    sys.path.insert(0, a.orch_root)
    from src.services.pdf_router import PDFRouter  # noqa: E402

    router = PDFRouter(temp_dir=tempfile.mkdtemp(prefix="legacy_router_"))
    t0 = time.perf_counter()
    text, _ = router._extract_with_pdftotext(Path(a.pdf))
    quality, needs_ocr = router._assess_text_quality(text)
    latency = (time.perf_counter() - t0) * 1000.0
    print(json.dumps({"text": text, "latency_ms": latency, "method": "pdftotext",
                      "needs_ocr": bool(needs_ocr or not text), "quality_score": quality, "chars": len(text)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
