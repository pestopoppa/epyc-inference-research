"""AKU-10f3: owner coverage records replay from their additive journal kind."""
from __future__ import annotations

import pytest

from .. import journal as journal_module
from . import retention_owner_journal as owner_journal
from .test_candidate_manifest import _base, _state
from .test_native_retention_catalog import _complete_controller, _controller

KIND = journal_module.KIND_RETENTION_OWNER_RECORDED


def _kinds(controller):
    return [entry.kind for entry in controller._journal.read_all()]


def test_records_journal_on_change_and_restore_on_replay_without_rederiving(
        tmp_path, monkeypatch):
    resolved, _anchors, _seed, controller, manager = _complete_controller(tmp_path, monkeypatch)
    try:
        assert _kinds(controller).count(KIND) == 5  # one per class at catalog install
        manager.initialize(request_id="init", state=_state(_base()), manifest=_base())
        written = _kinds(controller).count(KIND)
        assert written == 7  # candidate_manifest + build_dependency changed
        controller._register_retention_owners_locked("evidence")  # unchanged -> no append
        assert _kinds(controller).count(KIND) == written
        records = dict(controller._retention_owner_records)
    finally:
        controller.close()
    reopened = _controller(resolved, tmp_path / "service")
    derived = []
    real = reopened._derive_retention_owner_locked
    monkeypatch.setattr(reopened, "_derive_retention_owner_locked",
                        lambda owner_class: (derived.append(owner_class), real(owner_class))[1])
    with reopened:
        assert derived == []  # every class restored from the journal
        assert {k: v.record_digest for k, v in reopened._retention_owner_records.items()} \
            == {k: v.record_digest for k, v in records.items()}
        assert _kinds(reopened).count(KIND) == written  # startup appends nothing


def test_no_restorable_records_derives_in_memory_and_appends_nothing(tmp_path, monkeypatch):
    resolved, _anchors, _seed, controller, _manager = _complete_controller(tmp_path, monkeypatch)
    controller.close()
    payloads = []
    monkeypatch.setattr(owner_journal, "replayed_records",
                        lambda items, generation: payloads.extend(items) or {})
    with _controller(resolved, tmp_path / "service") as reopened:
        assert set(reopened._retention_owner_records) == set(owner_journal.catalog.OWNER_CLASSES)


def test_tampered_payload_is_refused():
    with pytest.raises(owner_journal.OwnerJournalRefused):
        owner_journal.validate({"schema": owner_journal.SCHEMA, "owner_class": "evidence",
                                "record": {}, "record_digest": "0" * 64})
