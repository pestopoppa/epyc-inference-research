"""odl_bench — EPYC deterministic PDF backends wired into opendataloader-bench.

Wave-2 B3. Deterministic structural comparison (pdftotext vs ODL-local vs
LiteParse) over the OmniDocBench harness; model-gated engines are emitted as
Wave-3 manifest-entry stubs instead of being run here. No inference.

Public surface:
    from odl_bench.adapter import OdlBenchAdapter
    from odl_bench.backends import resolve_backend, availability_report, register_backend
    from odl_bench.manifest_stubs import model_gated_stubs, model_gated_manifest
"""

from __future__ import annotations

__all__ = ["adapter", "backends", "schemas", "run_configs", "manifest_stubs", "bootstrap"]
