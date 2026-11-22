"""Real controller-worker target-profile execution through the public driver path.

AKU-06d: every acceptance test here reaches the profiler through the actual
UnifiedCampaignDriver-issued sequence (tick -> materialize_profile -> registered
producer -> contained fixture child -> provider-authored held cost -> controller-read
stdout). No private issuance projection, trusted-receipt cache or worker accessor is
seeded; the profiler, provider and loaded identities are labelled fixtures.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import time
from types import SimpleNamespace

import pytest

from .. import journal
from . import actor_preparation_state as state
from . import campaign_control, scheduling, target_profile_execution as execution
from . import unified_driver, worker_lifecycle
from .test_actor_preparation import _resolved
from .test_campaign_control import _command
from .test_unified_driver import runtime_driver
from .test_unified_planner import scheduler as make_scheduler
from .test_worker_lifecycle import ReceiptProvider


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_output(loaded, run_id="profile-run-1"):
    return {
        "schema": execution.PROFILE_OUTPUT_SCHEMA,
        "profile_content": {"hotspots": [{"symbol": "kernel_x", "samples": 8}]},
        "loaded_identity": loaded,
        "artifact_identity": {"kind": "profile-json", "sha256": "5" * 64},
        "measurement_carrier": {
            "schema": "epyc.autokernel.profile_measurement_carrier.v1",
            "profile_source_id": execution.PROFILE_SOURCE_ID,
            "validation_source_id": execution.VALIDATION_SOURCE_ID,
            "run_id": run_id,
            "profile_claim_tuple": {"claim_id": "profile-claim-1"},
            "validation_claim_tuple": {"claim_id": "validation-claim-1"},
        },
    }


def selected_profile_owner(tmp_path: Path, *, return_code: int = 0, stdout: str | None = None,
                           register: bool = True):
    """One driver-issued selected profile transition on a real v3 controller."""
    driver, _fixture_engine, enrolled, target, target_digest = runtime_driver()
    base_config, _ = make_scheduler()
    config = scheduling.SchedulerConfig.from_dict(
        base_config.to_dict() | {"config_id": enrolled.campaign_id})
    initial = scheduling.initial_state(config, enrolled.campaign_id)
    engine = scheduling.SchedulerEngine(config, initial)
    driver.scheduler = engine
    cost = scheduling.ResourceVector(1.0, (), 0)
    stage = scheduling.StageProposal(
        proposal_id="profile:actual", submitted_at=1, backend="cpu",
        target_revision=target_digest, alias_identity=target.workload_signature,
        frontier_id=target_digest, production_frontier=True, seed_id=None,
        stage_class="prerequisite", estimated_duration_seconds=10,
        estimated_claims=cost, eligible=True, eligibility_ref="2" * 64,
        reservation_kind=None, full_region=True,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    driver.profiles = {}
    driver.profile_requests = {target_digest: unified_driver.ProfilePreparationRequest.from_dict({
        "schema": unified_driver.PROFILE_REQUEST_SCHEMA,
        "target_revision_digest": target_digest, "stage_proposal": stage.to_dict(),
        "profile_contract": {"schema": unified_driver.PROFILE_CONTRACT_SCHEMA,
                             "adapter_id": "fixture-profile", "adapter_digest": "9" * 64}})}
    containers = tmp_path / "containers"
    containers.mkdir()
    provider = ReceiptProvider(containers)
    store = tmp_path / "store"
    controller = campaign_control.CampaignController(
        enrolled, store, snapshot_version=3, scheduler_engine=engine,
        lifecycle_provider=provider, readiness_check=lambda: (True, None))
    controller.__enter__()
    controller.apply_command(_command(enrolled, "resume", "resume", 0))
    driver.controller = controller
    issued = driver.tick(now=1)
    selected = driver.materialize_profile(issued)
    loaded = {
        "target_revision_digest": target_digest, "model_digest": "1" * 64,
        "quantization": "fixture", "recipe_digest": "2" * 64,
        "executable_digest": "3" * 64, "dso_digest": "4" * 64}
    output = profile_output(loaded)
    body = json.dumps(output) if stdout is None else stdout
    profiler = tmp_path / "fake-profiler"
    profiler.write_text(
        "#!/usr/bin/env python3\nprint(" + repr(body) + ")\nraise SystemExit("
        + str(return_code) + ")\n")
    profiler.chmod(profiler.stat().st_mode | stat.S_IXUSR)
    mechanism = execution.ProfileMechanism(
        "fixture-profile", profiler, _sha(profiler), tmp_path,
        {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}, loaded, 1.0, 0.5, 4096)
    producer = execution.TargetProfileExecution(controller=controller, mechanism=mechanism)
    if register:
        controller.register_target_profile_producer(producer)
    return SimpleNamespace(
        driver=driver, controller=controller, issued=issued, selected=selected,
        producer=producer, mechanism=mechanism, loaded=loaded, output=output,
        target=target_digest, store=store, provider=provider, enrolled=enrolled,
        config=config, initial=initial, engine=engine)


def prepare_selected(owner, producer=None, *, selected_work="selected"):
    now = time.monotonic()
    work = owner.selected
    return (producer or owner.producer).prepare(
        profile_request=work.profile_request.to_dict(), catalog_id=work.catalog_id,
        transition_id=work.transition_id, stage_plan_digest=work.stage_plan_digest,
        clock_domain=worker_lifecycle.monotonic_clock_domain(),
        verified_at=now, valid_until=now + 30.0,
        selected_work=work if selected_work == "selected" else selected_work)


def _journal_rows(owner):
    return journal.Journal(str(owner.store / "journal"),
                           campaign_id=owner.enrolled.campaign_id).read_all()


def _held_events(owner):
    return [row.payload for row in _journal_rows(owner)
            if row.kind == journal.KIND_WORKER_LIFECYCLE
            and row.payload["event"] == "OWNED_HELD_RECEIPT_RETAINED"]


def _assert_spent_without_profile(owner):
    controller = owner.controller
    assert not controller._actor_profile_execution_reservations
    held, terminal = next(iter(controller._actor_profile_execution_settlements.values()))
    [retained] = _held_events(owner)
    assert retained["request_id"] == terminal.request_id
    assert held.to_dict() == retained["data"]["receipt"]
    assert held.ended_at - held.started_at > 0
    assert held.proposal_id == owner.selected.selection.proposal.proposal_id
    assert not any(row.kind == journal.KIND_ACTOR_PREPARATION
                   and row.payload["event"] == "PROFILE_VERIFIED" for row in _journal_rows(owner))


def test_absent_profile_mechanism_is_visible_unsupported(tmp_path):
    controller = object.__new__(campaign_control.CampaignController)
    with pytest.raises(execution.ProfileExecutionRefused, match="unsupported/absent"):
        execution.TargetProfileExecution(controller=controller, mechanism=None)


@pytest.mark.parametrize("return_code", [0, 7])
def test_selected_profile_requires_success_after_binding_held_cost(tmp_path, return_code):
    owner = selected_profile_owner(tmp_path, return_code=return_code)
    controller = owner.controller
    if return_code:
        with pytest.raises(execution.ProfileExecutionRefused,
                           match="terminal was not successful"):
            prepare_selected(owner)
        _assert_spent_without_profile(owner)
        controller.close()
        return
    receipt = prepare_selected(owner)
    assert not controller._actor_profile_execution_reservations
    assert len(_held_events(owner)) == 1
    assert controller.verified_target_profile(
        request={"unused": "actor request"}, request_digest="9" * 64,
        stage_plan_digest="0" * 64, campaign_digest=controller.config_digest,
        target_revision_digest=owner.target, now=time.monotonic(),
        clock_domain=worker_lifecycle.monotonic_clock_domain()) == receipt
    assert receipt.target_profile_digest == state.digest({
        "profile_content": owner.output["profile_content"], "loaded_identity": owner.loaded,
        "artifact_identity": owner.output["artifact_identity"]})
    [published] = [row.payload for row in _journal_rows(owner)
                   if row.kind == journal.KIND_ACTOR_PREPARATION]
    assert published["event"] == "PROFILE_VERIFIED"
    assert published["catalog_id"] == owner.selected.catalog_id
    assert published["transition_id"] == owner.selected.transition_id
    controller.close()
    with campaign_control.CampaignController(
            owner.enrolled, owner.store, snapshot_version=3,
            scheduler_engine=scheduling.SchedulerEngine(owner.config, owner.initial),
            readiness_check=lambda: (True, None)) as recovered:
        replayed = recovered.current_actor_profile(owner.target)
        assert replayed is not None
        assert replayed.target_profile_digest == receipt.target_profile_digest


def test_malformed_profile_output_settles_spent_cost_without_profile(tmp_path):
    owner = selected_profile_owner(tmp_path, stdout="not one json object {")
    try:
        with pytest.raises(execution.ProfileExecutionRefused, match="UTF-8 JSON"):
            prepare_selected(owner)
        _assert_spent_without_profile(owner)
    finally:
        owner.controller.close()


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


@pytest.mark.parametrize("case", ["no_selected_work", "unregistered_producer",
                                  "no_registered_producer"])
def test_profiler_requires_selected_admission_by_registered_producer(tmp_path, case):
    owner = selected_profile_owner(tmp_path, register=case != "no_registered_producer")
    counter = tmp_path / "profiler-ran"
    owner.mechanism.binary.write_text(
        f"#!/usr/bin/env python3\nopen({str(counter)!r}, 'w').write('ran')\n")
    try:
        if case == "no_selected_work":
            with pytest.raises(campaign_control.ControlRefused, match="selected profile advice"):
                prepare_selected(owner, selected_work=None)
        else:
            other = execution.TargetProfileExecution(
                controller=owner.controller, mechanism=owner.mechanism)
            with pytest.raises(campaign_control.ControlRefused, match="registered producer"):
                prepare_selected(owner, other)
        assert owner.provider.authorization is None
        assert not counter.exists()
        assert not owner.controller._actor_profile_execution_reservations
        assert not _held_events(owner)
    finally:
        owner.controller.close()


def test_schema_valid_caller_events_are_not_verified_profiles(tmp_path, monkeypatch):
    """A self-consistent PROFILE_VERIFIED event is refused without the producer admission,
    or when it names a loaded identity other than the registered producer's mechanism."""
    original = campaign_control.CampaignController.record_verified_target_profile
    captured = {}

    def foreign_identity(self, value, **kwargs):
        forged = dict(value)
        forged["loaded_identity"] = dict(value["loaded_identity"], model_digest="e" * 64)
        forged["target_profile_digest"] = state.digest({
            name: forged[name] for name in
            ("profile_content", "loaded_identity", "artifact_identity")})
        state.validate_event(forged)  # schema-valid and self-consistent
        captured.update(event=forged, **kwargs)
        return original(self, forged, **kwargs)

    monkeypatch.setattr(campaign_control.CampaignController,
                        "record_verified_target_profile", foreign_identity)
    owner = selected_profile_owner(tmp_path)
    try:
        with pytest.raises(campaign_control.ControlRefused, match="registered producer"):
            prepare_selected(owner)
        _assert_spent_without_profile(owner)
        monkeypatch.setattr(campaign_control.CampaignController,
                            "record_verified_target_profile", original)
        reservation = captured["reservation"]
        caller_reservation = execution.TargetProfileExecutionReservation(
            *(getattr(reservation, name) for name in (
                "reservation_id", "catalog_id", "transition_id", "profile_request_digest",
                "stage_plan_digest", "request_id", "stage_id", "supervisor_incarnation",
                "control_revision", "max_output_bytes")), object(), owner.producer)
        caller_event = dict(captured["event"], loaded_identity=owner.loaded)
        caller_event["target_profile_digest"] = state.digest({
            name: caller_event[name] for name in
            ("profile_content", "loaded_identity", "artifact_identity")})
        with pytest.raises(campaign_control.ControlRefused, match="current owner authority"):
            owner.controller.record_verified_target_profile(
                caller_event, reservation=caller_reservation,
                terminal=captured["terminal"],
                provider_cost_receipt=captured["provider_cost_receipt"])
        assert owner.controller.current_actor_profile(owner.target) is None
    finally:
        owner.controller.close()


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
