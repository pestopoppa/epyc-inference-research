"""Enrollment replay must retain request identities and frozen artifact pins."""
import copy

import pytest

from . import campaign, campaign_control
from .test_campaign import _manifest, _registry, _target


@pytest.mark.parametrize("explicit_null", [False, True])
def test_missing_artifact_resolution_and_restart(tmp_path, explicit_null):
    raw = _manifest(production=[_target("prod")])
    manifest = campaign.CampaignManifest.from_dict(raw)
    full = _registry()
    missing = copy.deepcopy(full)
    if explicit_null:
        missing["model"]["model-a"] = None
    else:
        del missing["model"]["model-a"]
    initial = campaign.resolve_manifest(manifest, registry_snapshot=missing)
    store = tmp_path / "service"
    with campaign_control.CampaignController(initial, store) as owner:
        first = owner.enroll(manifest, missing)
        assert first.targets[0].status == "missing_artifact"
        assert owner.enroll(manifest, full) == first
        moved = copy.deepcopy(full)
        moved["build"]["production-v9"]["sha256"] = "f" * 64
        with pytest.raises(campaign_control.ControlRefused, match="identity moved"):
            owner.enroll(manifest, moved, resolve_missing=True)
        resolved = owner.enroll(manifest, full, resolve_missing=True)
        assert resolved.targets[0].status == "ready"
        assert owner.enroll(manifest, full, resolve_missing=True) == resolved
        assert resolved.targets[0].revision == first.targets[0].revision + 1
        assert resolved.targets[0].baseline == first.targets[0].baseline
    with campaign_control.CampaignController(initial, store) as owner:
        assert owner.enroll(manifest, missing) == resolved
        assert owner.enrollment_snapshot()[manifest.request_id] == resolved.to_dict()
        changed = copy.deepcopy(raw)
        changed["objective_ref"] = "different"
        with pytest.raises(campaign_control.ControlRefused, match="different manifest"):
            owner.enroll(campaign.CampaignManifest.from_dict(changed), full)
        assert owner.resolved == initial


def test_replay_refuses_modified_projection_and_conflicting_request(tmp_path):
    raw = _manifest(production=[_target("prod")])
    manifest = campaign.CampaignManifest.from_dict(raw)
    registry = _registry()
    initial = campaign.resolve_manifest(manifest, registry_snapshot=registry)
    from . import enrollment_journal
    event = enrollment_journal.make_event("ENROLLED", manifest, registry, [], initial)
    damaged = copy.deepcopy(event)
    damaged["resolved"]["targets"][0]["revision"] += 1
    with pytest.raises(enrollment_journal.EnrollmentRefused, match="projection differs"):
        enrollment_journal.project([damaged], manifest.campaign_id, initial)
    different = campaign.CampaignManifest.from_dict({**raw, "objective_ref": "other"})
    with campaign_control.CampaignController(initial, tmp_path / "service") as owner:
        with pytest.raises(campaign_control.ControlRefused, match="identity differs"):
            owner.enroll(different, registry)


def test_later_request_preserves_moved_baseline_across_restart(tmp_path):
    raw = _manifest(production=[_target("prod")])
    manifest = campaign.CampaignManifest.from_dict(raw)
    initial = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    later_raw = _manifest(request_id="request-2", production=[
        _target("prod", request_id="request-2")])
    later = campaign.CampaignManifest.from_dict(later_raw)
    store = tmp_path / "service"
    with campaign_control.CampaignController(initial, store) as owner:
        owner.enroll(manifest, _registry())
        changed = owner.enroll(later, _registry(baseline_digit="f"))
        assert changed.targets[0].status == "mismatched_ref"
        assert changed.targets[0].baseline == initial.targets[0].baseline
    with campaign_control.CampaignController(initial, store) as owner:
        assert owner.enroll(later, _registry()) == changed


def test_journal_rejects_malformed_native_enrollment(tmp_path):
    from .. import journal
    from . import enrollment_journal
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    initial = campaign.resolve_manifest(manifest, registry_snapshot=_registry())
    event = enrollment_journal.make_event("ENROLLED", manifest, _registry(), [], initial)
    book = journal.Journal(str(tmp_path / "journal"), campaign_id=manifest.campaign_id)
    book.initialize()
    malformed = copy.deepcopy(event)
    malformed["resolved"]["manifest_digest"] = "0" * 64
    with pytest.raises(ValueError, match="identity differs"):
        book.append(journal.KIND_ENROLLMENT, malformed)
    book.append(journal.KIND_ENROLLMENT, event)
    assert book.read_all()[0].payload == event


def test_missing_artifact_can_resolve_while_capability_remains_unsupported(tmp_path):
    manifest = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    registry = _registry()
    registry["recipe"]["recipe-a"] = {"status": "unsupported_capability", "reason": "fixture"}
    missing = copy.deepcopy(registry)
    del missing["model"]["model-a"]
    initial = campaign.resolve_manifest(manifest, registry_snapshot=missing)
    assert initial.targets[0].status == "unsupported_capability"
    with campaign_control.CampaignController(initial, tmp_path / "service") as owner:
        owner.enroll(manifest, missing)
        result = owner.enroll(manifest, registry, resolve_missing=True)
        assert result.targets[0].status == "unsupported_capability"
        assert result.targets[0].execution.model is not None
        assert all(not reason.endswith(":missing") for reason in result.targets[0].missing)


def test_older_request_can_resolve_after_a_later_enrollment(tmp_path):
    first = campaign.CampaignManifest.from_dict(_manifest(production=[_target("prod")]))
    full = _registry()
    missing = copy.deepcopy(full)
    del missing["model"]["model-a"]
    initial = campaign.resolve_manifest(first, registry_snapshot=missing)
    later = campaign.CampaignManifest.from_dict(_manifest(
        request_id="request-2", production=[_target("prod", request_id="request-2")]))
    store = tmp_path / "service"
    with campaign_control.CampaignController(initial, store) as owner:
        owner.enroll(first, missing)
        owner.enroll(later, missing)
        resolved = owner.enroll(first, full, resolve_missing=True)
        assert resolved.targets[0].status == "ready"
    with campaign_control.CampaignController(initial, store) as owner:
        assert owner.enroll(first, full) == resolved
        assert owner.enrollment_snapshot()[later.request_id]["targets"][0]["status"] == "missing_artifact"
