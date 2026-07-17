"""Model-gated engines emitted as Wave-3 batch-manifest entry stubs.

These engines require model inference (GPU/OCR/sidecar) and are EXCLUDED from the
deterministic Wave-2 comparison. Instead each is returned as a structured
``ManifestEntryStub`` describing exactly what the Wave-3 inference-batch loop must
run, its preconditions, and the artifacts it should produce. They deliberately
REUSE the deterministic wiring: each still emits ``<stem>.md`` predictions scored
by the same OmniDocBench config, so the only new thing at Wave-3 is the (gated)
extraction step.

Sources (orchestrator, read-only):
  * LightOnOCR        -> PDFRouter._extract_with_lightonocr (OCR server, port from config)
  * ODL hybrid        -> PDFRouter._extract_with_opendataloader_hybrid
                         (opendataloader-pdf[hybrid] + sidecar :5002; docling-fast default)
  * PaddleOCR-VL-1.6  -> VL document parser (GGUF+mmproj, llama-mtmd CPU path);
                         OmniDocBench v1.6 SOTA reference engine (intake follow-up)
"""

from __future__ import annotations

from .schemas import ManifestEntryStub

_PRED_DIR = "$RUN_DIR/predictions/<engine>"
_EXPECTED = [
    f"{_PRED_DIR}/<gt_stem>.md  (one markdown prediction per GT page)",
    "$RUN_DIR/config/<engine>.yaml  (OmniDocBench config; reuses DETERMINISTIC_METRIC_CONFIG)",
    "opendataloader-bench/result/<save_name>_metric_result.json  (structural/table/reading_order scores)",
    "$RUN_DIR/<engine>_run_manifest.json  (speed rows: per-page extraction latency_ms)",
]


def model_gated_stubs() -> list[ManifestEntryStub]:
    """The full model-gated engine set as Wave-3 manifest entries."""
    return [
        ManifestEntryStub(
            entry_id="ODLB-W3-01-lightonocr",
            engine="lightonocr",
            description=(
                "Scanned/image PDF OCR path. Run LightOnOCR over the same GT-page "
                "PDFs, write <stem>.md predictions, score with the deterministic "
                "OmniDocBench config for a like-for-like structural/table/order "
                "comparison against pdftotext/ODL-local/LiteParse."
            ),
            preconditions=[
                "LightOnOCR server reachable (config.server_urls.ocr_server; /ocr/pdf)",
                "Source born-digital+scanned PDFs mapped to each GT page (see input-modality note in README)",
                "Quiet window: no competing llama-server / bench / autopilot compute (B7 rider)",
            ],
            command=(
                "python -m scripts.benchmark.odl_bench.adapter run "
                "--engine lightonocr --pdf-manifest <pdf_manifest.json> "
                "--gt <OmniDocBench GT json> --run-dir $RUN_DIR --score"
            ),
            env={"PDF_ROUTER_FORCE_OCR": "1"},
            expected_artifacts=list(_EXPECTED),
            notes=(
                "Backend symbol: PDFRouter._extract_with_lightonocr (async). The adapter's "
                "run() wraps it via extract(force_ocr=True). INFERENCE — Wave-3 only."
            ),
        ),
        ManifestEntryStub(
            entry_id="ODLB-W3-02-odl-hybrid",
            engine="opendataloader_hybrid",
            description=(
                "ODL hybrid table extraction (docling-fast sidecar) for table-fidelity "
                "uplift vs ODL-local. Same prediction+scoring path; the delta of interest "
                "is table TEDS."
            ),
            preconditions=[
                "opendataloader-pdf[hybrid] installed; sidecar 'opendataloader-pdf-hybrid --port 5002' live",
                "GET http://127.0.0.1:5002/health == {'status':'ok'}",
                "ORCHESTRATOR_ODL_TABLE_BACKEND=hybrid set for the run",
                "Structural/table-heavy source corpus with GT (demo_data lacks source PDFs)",
            ],
            command=(
                "python -m scripts.benchmark.odl_bench.adapter run "
                "--engine opendataloader_hybrid --pdf-manifest <pdf_manifest.json> "
                "--gt <GT json> --run-dir $RUN_DIR --score"
            ),
            env={
                "ORCHESTRATOR_ODL_TABLE_BACKEND": "hybrid",
                "ORCHESTRATOR_ODL_HYBRID_URL": "http://127.0.0.1:5002",
            },
            expected_artifacts=list(_EXPECTED),
            notes=(
                "Backend symbol: PDFRouter._extract_with_opendataloader_hybrid. Deterministic "
                "extraction algorithm but MODEL-GATED on the docling-fast sidecar => Wave-3. "
                "Sidecar already proven live 2026-07-06 (see opendataloader handoff)."
            ),
        ),
        ManifestEntryStub(
            entry_id="ODLB-W3-03-paddleocr-vl",
            engine="paddleocr_vl_1_6",
            description=(
                "VL document parser reference arm (PaddleOCR-VL-1.6, OmniDocBench v1.6 "
                "SOTA 96.33). Establishes the model-parser ceiling for structural/table/"
                "reading-order vs the deterministic fast paths."
            ),
            preconditions=[
                "PaddleOCR-VL-1.6 GGUF + mmproj present; llama-mtmd CPU (or MI210) server up",
                "Per-page image OR PDF inputs mapped to GT pages",
                "Operator approval for inference run + quiet window (B7 rider)",
                "NOT wired into pdf_router today — needs a Wave-3 engine adapter (documented gap)",
            ],
            command=(
                "python -m scripts.benchmark.odl_bench.adapter run-model "
                "--engine paddleocr_vl_1_6 --gt <GT json> --image-root <GT image dir> "
                "--run-dir $RUN_DIR --allow-inference --score"
            ),
            env={},
            expected_artifacts=list(_EXPECTED),
            reuses_deterministic_wiring=True,
            notes=(
                "Wave-3 producer consumes GT page images directly and emits <stem>.md. "
                "INFERENCE; keep it explicit and quiet-window gated."
            ),
        ),
    ]


def model_gated_manifest() -> dict:
    """JSON-ready manifest of all model-gated stubs (for the Wave-3 compiler)."""
    return {
        "schema": "odl_bench.model_gated_manifest.v1",
        "wave": 3,
        "reuses": "scripts/benchmark/odl_bench (deterministic wiring)",
        "entries": [s.to_dict() for s in model_gated_stubs()],
    }
