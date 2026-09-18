"""Real controller-worker target-profile execution with immutable native result."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from .. import journal
from . import actor_preparation_state as state
from . import campaign_control, target_profile_execution as execution
from .test_actor_lifecycle import _publish_test_owner_accessors
from .test_actor_preparation import _resolved
from .test_campaign_control import _command
from .test_worker_lifecycle import MockProvider


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_absent_profile_mechanism_is_visible_unsupported(tmp_path):
    controller = object.__new__(campaign_control.CampaignController)
    with pytest.raises(execution.ProfileExecutionRefused, match="unsupported/absent"):
        execution.TargetProfileExecution(controller=controller, mechanism=None)


@pytest.mark.parametrize("return_code", [0, 7])
def test_selected_profile_requires_success_after_binding_held_cost(tmp_path, return_code):
    target = "a" * 64
    loaded = {
        "target_revision_digest": target, "model_digest": "1" * 64,
        "quantization": "Q4_K_M", "recipe_digest": "2" * 64,
        "executable_digest": "3" * 64, "dso_digest": "4" * 64,
    }
    output = {
        "schema": execution.PROFILE_OUTPUT_SCHEMA,
        "profile_content": {"hotspots": [{"symbol": "kernel_x", "samples": 8}]},
        "loaded_identity": loaded,
        "artifact_identity": {"kind": "profile-json", "sha256": "5" * 64},
        "measurement_carrier": {
            "schema": "epyc.autokernel.profile_measurement_carrier.v1",
            "profile_source_id": execution.PROFILE_SOURCE_ID,
            "validation_source_id": execution.VALIDATION_SOURCE_ID,
            "run_id": "profile-run-1",
            "profile_claim_tuple": {"claim_id": "profile-claim-1"},
            "validation_claim_tuple": {"claim_id": "validation-claim-1"},
        },
    }
    binary = tmp_path / "fake-profiler"
    binary.write_text(
        "#!/usr/bin/env python3\nimport json\nprint(" + repr(json.dumps(output))
        + ")\nraise SystemExit(" + str(return_code) + ")\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    resolved = _resolved()
    (tmp_path / "containers").mkdir()
    provider = MockProvider(tmp_path / "containers")
    store = tmp_path / "store"
    controller = campaign_control.CampaignController(
        resolved, store, snapshot_version=2, lifecycle_provider=provider,
        readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    _publish_test_owner_accessors(controller)
    request = {
        "schema": execution.PROFILE_REQUEST_SCHEMA, "target_revision_digest": target,
        "stage_proposal": {"target_revision": target, "stage_class": "prerequisite"},
        "profile_contract": {"mechanism_id": "fake-profile-v1"},
    }
    stage_plan = state.digest(request)
    selected_stage = "6" * 64
    issued = {
        "catalog_id": "7" * 64, "transition_id": "8" * 64,
        "selection": {"status": "selected", "proposal_digest": selected_stage},
        "catalog": {"work_by_stage_digest": {selected_stage: {
            "kind": "profile_preparation", "payload": request,
            "stage_plan_digest": stage_plan,
            "stage_plan_binding": "preparation_contract"}}},
    }
    controller._driver_issued = {issued["catalog_id"]: issued}
    mechanism = execution.ProfileMechanism(
        "fake-profile-v1", binary, _sha(binary), tmp_path,
        {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, loaded,
        1.0, 0.5, 4096)
    producer = execution.TargetProfileExecution(
        controller=controller, mechanism=mechanism)
    if return_code:
        with pytest.raises(execution.ProfileExecutionRefused,
                           match="terminal was not successful"):
            producer.prepare(
                profile_request=request, catalog_id=issued["catalog_id"],
                transition_id=issued["transition_id"], stage_plan_digest=stage_plan,
                clock_domain="test-monotonic", verified_at=10.0, valid_until=20.0)
        assert controller._test_held_receipt_calls == 1
        assert not controller._actor_profile_execution_reservations
        settlement = next(iter(controller._actor_profile_execution_settlements.values()))
        assert settlement[0].ended_at - settlement[0].started_at == pytest.approx(0.25)
        entries = journal.Journal(
            str(store / "journal"), campaign_id=resolved.campaign_id).read_all()
        assert not any(entry.kind == journal.KIND_ACTOR_PREPARATION
                       and entry.payload["event"] == "PROFILE_VERIFIED"
                       for entry in entries)
        controller.close()
        return
    receipt = producer.prepare(
        profile_request=request, catalog_id=issued["catalog_id"],
        transition_id=issued["transition_id"], stage_plan_digest=stage_plan,
        clock_domain="test-monotonic", verified_at=10.0, valid_until=20.0)
    assert controller._test_held_receipt_calls == 1
    assert not controller._actor_profile_execution_reservations
    controller.register_target_profile_producer(producer)
    assert controller.verified_target_profile(
        request={"unused": "actor request"}, request_digest="9" * 64,
        stage_plan_digest="0" * 64, campaign_digest=controller.config_digest,
        target_revision_digest=target, now=15.0,
        clock_domain="test-monotonic") == receipt
    assert receipt.target_revision_digest == target
    assert receipt.target_profile_digest == state.digest({
        "profile_content": output["profile_content"], "loaded_identity": loaded,
        "artifact_identity": output["artifact_identity"]})
    entries = journal.Journal(
        str(store / "journal"), campaign_id=resolved.campaign_id).read_all()
    assert any(entry.kind == journal.KIND_ACTOR_PREPARATION
               and entry.payload["event"] == "PROFILE_VERIFIED" for entry in entries)
    controller.close()
    with campaign_control.CampaignController(
            resolved, store, snapshot_version=2,
            readiness_check=lambda: (True, None)) as recovered:
        replayed = recovered.current_actor_profile(target)
        assert replayed is not None
        assert replayed.target_profile_digest == receipt.target_profile_digest


def test_selected_work_is_reserved_before_profiler_bytes_are_read(tmp_path):
    target = "a" * 64
    missing = tmp_path / "does-not-exist"
    resolved = _resolved()
    controller = campaign_control.CampaignController(
        resolved, tmp_path / "store", snapshot_version=2,
        readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(resolved, "resume", "resume", 0))
    mechanism = execution.ProfileMechanism(
        "missing", missing, "1" * 64, tmp_path, {}, {
            "target_revision_digest": target, "model_digest": "2" * 64,
            "quantization": "Q4_K_M", "recipe_digest": "3" * 64,
            "executable_digest": "4" * 64, "dso_digest": "5" * 64,
        }, 1.0, 0.5, 1024)
    request = {
        "schema": execution.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target,
        "stage_proposal": {"target_revision": target},
        "profile_contract": {"mechanism_id": "missing"},
    }
    with pytest.raises(campaign_control.ControlRefused,
                       match="exact issued transition"):
        execution.TargetProfileExecution(
            controller=controller, mechanism=mechanism).prepare(
                profile_request=request, catalog_id="6" * 64,
                transition_id="7" * 64,
                stage_plan_digest=state.digest(request), clock_domain="test",
                verified_at=1.0, valid_until=2.0)
    controller.close()


def test_publication_rejects_output_not_matching_second_owner_read(tmp_path, monkeypatch):
    original = campaign_control.CampaignController.record_verified_target_profile
    seen = {"called": False}

    def reject_mismatch(self, value, **kwargs):
        prior = self.read_worker_stdout

        def mismatched(**read_request):
            raw = prior(**read_request)
            body = json.loads(raw.decode())
            body["profile_content"] = {"forged": True}
            return json.dumps(body, sort_keys=True).encode()

        self.read_worker_stdout = mismatched
        try:
            return original(self, value, **kwargs)
        finally:
            self.read_worker_stdout = prior
            seen["called"] = True

    monkeypatch.setattr(campaign_control.CampaignController,
                        "record_verified_target_profile", reject_mismatch)
    with pytest.raises(campaign_control.ControlRefused,
                       match="controller-read worker stdout"):
        test_selected_profile_requires_success_after_binding_held_cost(tmp_path, 0)
    assert seen["called"]
