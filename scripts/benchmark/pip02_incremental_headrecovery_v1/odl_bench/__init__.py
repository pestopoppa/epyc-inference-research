"""odl_bench — EPYC PDF/VL producers wired into opendataloader-bench.

Wave-2 B3. Deterministic structural comparison (pdftotext vs ODL-local vs
LiteParse) over the OmniDocBench harness; Wave-3 model-gated producers are
explicit opt-in commands that reuse the same prediction/scoring artifacts.

Public surface:
    from odl_bench.adapter import OdlBenchAdapter
    from odl_bench.backends import resolve_backend, availability_report, register_backend
    from odl_bench.manifest_stubs import model_gated_stubs, model_gated_manifest
    from odl_bench.paddleocr_vl import PaddleOcrVlProducer
    from odl_bench.comparison import build_existing_comparison
    from odl_bench.intrinsic import score_chunks, score_prediction_dir, DefaultChunker
"""

from __future__ import annotations

__all__ = [
    "adapter",
    "backends",
    "schemas",
    "run_configs",
    "manifest_stubs",
    "bootstrap",
    "paddleocr_vl",
    "comparison",
    "intrinsic",
]
