"""AKU-10f4: evidence-owner coverage advances through the real native capture append."""
from __future__ import annotations

from .test_native_capture_control import _produce
from .test_candidate_manifest import _base, _state
from .test_native_retention_catalog import _complete_controller


def test_installed_catalog_evidence_coverage_advances_through_real_append(
        tmp_path, monkeypatch):
    _resolved, _anchors, seed, controller, manager = _complete_controller(
        tmp_path, monkeypatch)
    try:
        records = controller._retention_owner_records
        before = records["evidence"]
        assert before.generation[0] == seed.seed_digest and not before.gaps
        (tmp_path / "capture").mkdir()
        result, payloads, _context, _store = _produce(controller, tmp_path / "capture")
        assert result.execution_complete and len(payloads) == 2
        after = controller._retention_owner_records["evidence"]
        assert after.record_digest != before.record_digest  # hook ran on the append path
        assert after.generation == before.generation and not after.gaps
        assert set(controller._native_records) == set(payloads)
        # The re-derived record matches what the hook wrote: coverage is current.
        assert controller._derive_retention_owner_locked("evidence") == after
        manager.initialize(request_id="init", state=_state(_base()), manifest=_base())
        assert manager.retention_view().uncertain_scopes == ()
    finally:
        controller.close()
