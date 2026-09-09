"""Hermetic planned-worker bridge tests; fake placement is not containment proof."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

import pytest

from .. import schemas
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import observation_binding as ob
from . import planned_serving as ps
from . import scheduling
from . import unified_planner as up
from . import unified_worker as uw
from . import worker_lifecycle as wl
from .test_worker_lifecycle import Harness
from .test_lifecycle_observation import _budgets, _fixture, _write_process
from .test_experiment_plan import plan_dict, raw_dict
from . import test_unified_planner as fixtures


def _prompt(template):
    body = {"prompt": "one frozen prompt", "n_predict": template.n_predict,
            "temperature": template.temperature, "top_p": template.top_p,
            "top_k": template.top_k, "cache_prompt": False}
    request = {"prompt_id": "p1", **body,
               "request_digest": hashlib.sha256(json.dumps(
                   body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    manifest = {"schema": ps.PROMPT_SCHEMA, "version": "worker-test-v1",
                "prompts": [request]}
    return ps.FrozenPromptManifest.from_dict(
        {**manifest, "digest": schemas.content_hash(manifest)})


def _prepared(tmp_path, *, backend="cpu", continuation_allowed=False):
    anchor = fixtures.canonical_recipe(backend=backend)
    campaign = fixtures.campaign_for_recipe(anchor, campaign_id="camp-1")
    target = campaign.targets[0]
    target_digest = up._target_digest(target)
    anchors = fixtures.prepared(
        campaign, {target_digest: fixtures.runtime_anchor(target, anchor)})
    pair = up.enumerate_runtime_dimensions(
        anchors.recipes[target_digest], [fixtures.dimension()])[0]
    raw = plan_dict(n=1, instrument="serving", unit="process")
    raw.update(campaign_id="camp-1", target_revision=target_digest,
               metric="aggregate_tok_s", changed_factors=["threads"],
               required_witnesses=["native-capture-v1"],
               continuation_allowed=continuation_allowed)
    for unit in raw["expected_units"]:
        unit["expected_prompt_ids"] = ["p1"]
    raw["anchor_identity"] = ps.arm_identity(pair.anchor.template, pair.anchor)
    raw["candidate_identity"] = ps.arm_identity(pair.candidate.template, pair.candidate)
    plan = ep.ExperimentPlan.from_dict(raw)

    # Construct one already-selected advisory dispatch without invoking mutable planner state.
    claim = fixtures.claim(backend, pair=pair, target=target)
    claim_row = claim.to_dict()
    claim_row["control_identity"] = uw._plain(up.serving_arm_identity(pair.anchor))
    claim_row["intervention_identity"] = uw._plain(
        up.serving_arm_identity(pair.candidate))
    claim = fixtures.evidence.ClaimKey.from_dict(claim_row)
    proposal = up.UnifiedProposal.from_dict({
        "schema": up.PROPOSAL_SCHEMA, "proposal_id": "worker-runtime",
        "target_revision_digest": target_digest, "backend": backend,
        "parent_identity": {"target_ids": list(target.target_ids)},
        "control_identity": uw._plain(up.serving_arm_identity(pair.anchor)),
        "intervention_identity": uw._plain(up.serving_arm_identity(pair.candidate)),
        "kind": "runtime_recipe", "mechanism_id": "threads", "estimand": "level",
        "metric": "aggregate_tok_s", "metric_direction": "higher",
        "effect_question": uw._plain(claim.effect_question), "changed_factors": ["threads"],
        "instrument": "serving", "unit": "process",
        "required_witnesses": ["native-capture-v1"], "stage_class": "search",
        "estimated_duration_seconds": 10.0, "experiment_plan_digest": plan.digest,
        "native_artifact_sink_ref": "native:capture", "runtime_pair": pair.to_dict(),
        "claim_key": claim.to_dict(), "evidence_snapshot": {"epoch": "worker-test"}})
    vector = scheduling.ResourceVector(
        1.0, ("ROCm0",) if backend == "gpu" else (), 0)
    stage = scheduling.StageProposal(
        proposal.proposal_id, 1.0, backend, target_digest, "worker-alias", None, False,
        None, "search", 10.0, vector, True, "worker-test", None, True, (), False)
    selection = scheduling.Selection(
        "selected", (), stage, "coverage", 0, 1, "worker-scheduler", "1" * 64,
        1, vector.digest, stage.digest, 0.0, 0.0)
    intent = {"schema": up.INTENT_SCHEMA, "status": "ready_comparison",
        "proposal_digest": proposal.digest, "stage_proposal_digest": stage.digest,
        "experiment_plan_digest": plan.digest, "claim_key": claim.to_dict(),
        "effect_question": uw._plain(claim.effect_question),
        "arm_scalars_are_gain_evidence": False}
    dispatch = up.DispatchRequest(
        selection.to_dict(), proposal.to_dict(), intent).to_dict()

    base = {"campaign_id": "camp-1", "config_digest": "c" * 64,
        "supervisor_id": "supervisor-1", "supervisor_incarnation": 2,
        "config_generation": 3, "instrument_id": "planned-serving/v1",
        "protocol_id": plan.protocol_ref, "protocol_status": plan.protocol_status,
        "source_identities": {
            "anchor": {"source_revision": "a" * 40,
                       "model_sha256": pair.anchor.model.sha256,
                       "build_sha256": pair.anchor.executable.sha256,
                       "recipe_hash": pair.anchor.template.recipe_hash},
            "candidate": {"source_revision": "b" * 40,
                          "model_sha256": pair.candidate.model.sha256,
                          "build_sha256": pair.candidate.executable.sha256,
                          "recipe_hash": pair.candidate.template.recipe_hash}}}
    body = {"schema": uw.PREPARED_SCHEMA, "dispatch": dispatch,
            "plan": plan.to_dict(), "prompt_manifest": _prompt(pair.anchor.template).to_dict(),
            "runtime_pair": proposal.to_dict()["runtime_pair"], "capture_context_base": base,
            "artifact_root": str(tmp_path / "artifacts"), "previous": None,
            "max_stage_seconds": 30.0, "teardown_seconds": 2.0}
    return uw.PreparedPlannedServingStage.from_dict(
        {**body, "prepared_digest": uw._digest(body)})


def _start(prepared, *, lineage="lineage-1"):
    process = wl.process_identity(os.getpid())
    body = {"schema": uw.START_SCHEMA, "nonce": "nonce-0123456789abcdef",
        "request_id": "request-1",
        "prepared_digest": prepared.prepared_digest, "plan_digest": prepared.plan.digest,
        "lineage_id": lineage, "stage_id": "stage-1",
        "campaign_id": "camp-1", "config_digest": "c" * 64,
        "config_generation": 3, "supervisor_id": "supervisor-1",
        "supervisor_incarnation": 2, "worker_id": "worker-1", "worker_generation": 7,
        "grant_id": "grant-1", "grant_generation": 4,
        "container_id": "epyc-autokernel-test", "child_process": process.to_dict(),
        "cgroup_identity": {"path": "/mock/cgroup", "dev": 1, "ino": 2,
                            "uid": os.getuid(), "nlink": 1, "mode": 0o40700},
        "clock_domain": "test-monotonic", "provider_deadline": 20.0, "sequence": 0}
    return uw.WorkerStart.from_dict({**body, "start_digest": uw._digest(body)})


class Authority:
    def __init__(self, *, stop_at=None):
        self.stop_at = stop_at
        self.units = []

    def admit(self, *, sequence, plan_digest, unit, prior_completion_digest):
        self.units.append((sequence, unit.unit_id, prior_completion_digest))
        allowed = sequence != self.stop_at
        return uw.UnitPermit(sequence, unit.unit_id, unit.process_id,
            f"fence-{sequence}", 10.0, "grant-1", 4,
            "epyc-autokernel-test", allowed, "paused by parent" if not allowed else "held")

    def complete(self, *, sequence, fence, observation, native_observation=None):
        if native_observation is not None:
            assert set(native_observation) == {"locator", "sha256", "verified"}
        return ps.StageCompletion(fence.fence_id, True, {
            name: ep.Witness("pass", f"{name}:{sequence}") for name in
            ("native-capture-v1", "identity", "teardown", "contention", "placement",
             "residency")}, "clean", None)

    def verify_continuation(self, raw, plan, prompts, previous_lineage_id):
        return True


def _measure(template, build_dir, port, **kwargs):
    prompt_id, body = kwargs["frozen_requests"][0]
    kwargs["observation"].append({
        "schema": "epyc.autokernel.serving_observation.v1", "process_pid": os.getpid(),
        "requests": [{"phase": "measurement", "slot_index": 0,
            "prompt_id": prompt_id, "request_sha256": hashlib.sha256(body).hexdigest(),
            "predicted_n": template.n_predict, "predicted_per_second": 10.0,
            "terminal": True, "error": None}],
        "residency": {"status": "mocked"}, "teardown": "terminated", "failure": None})
    return 10.0


def _observed_measure(template, build_dir, port, **kwargs):
    session = kwargs["observation_session"]
    session.start()
    session.phase("load")
    session.attach_target(101)
    for phase in ("placement", "health", "warmup", "measurement"):
        session.phase(phase)
    session.checkpoint("measurement_end")
    session.phase("teardown")
    session.finish()
    return _measure(template, build_dir, port, **kwargs)


def _v2_prepared(tmp_path):
    prepared = _prepared(tmp_path)
    store = mc.ArtifactStore(prepared.artifact_root)
    instrument = ob.seal_loaded_instrument(
        store=store, measurement_callable=_observed_measure,
        fence_clock=time.monotonic, serving_timer=time.time)
    store.close()
    plan_row = prepared.plan.to_dict()
    plan_row |= {"schema": ep.PLAN_SCHEMA_V2, "loaded_instrument": instrument.to_dict(),
        "anchor_identity": ps.arm_identity(
            prepared.runtime_pair.anchor.template, prepared.runtime_pair.anchor,
            loaded_instrument=instrument.to_dict()),
        "candidate_identity": ps.arm_identity(
            prepared.runtime_pair.candidate.template, prepared.runtime_pair.candidate,
            loaded_instrument=instrument.to_dict())}
    plan = ep.ExperimentPlan.from_dict(plan_row)
    dispatch = uw._plain(prepared.dispatch)
    proposal_row = dispatch["proposal"] | {"experiment_plan_digest": plan.digest}
    proposal = up.UnifiedProposal.from_dict(proposal_row)
    dispatch["proposal"] = proposal.to_dict()
    dispatch["experiment_intent"]["proposal_digest"] = proposal.digest
    dispatch["experiment_intent"]["experiment_plan_digest"] = plan.digest
    dispatch = up.DispatchRequest(**dispatch).to_dict()
    body = prepared.body() | {"schema": uw.PREPARED_SCHEMA_V2,
        "dispatch": dispatch, "plan": plan.to_dict(),
        "capture_context_base": uw._plain(prepared.capture_context_base)
            | {"instrument_id": instrument.identity_sha256}}
    return uw.PreparedPlannedServingStage.from_dict(
        {**body, "prepared_digest": uw._digest(body)}), instrument


def _terminal_and_fence(prepared, start, reference, *, current=True):
    terminal = wl.TerminalWorker(
        start.worker_id, start.worker_generation, "request-1", prepared.plan.digest,
        start.lineage_id, start.stage_id, start.grant_id, start.grant_generation,
        start.container_id, 0, uw._digest(reference.to_dict()), True, None)
    fence = nc.TrustedWorkerResultFence(
        start.campaign_id, start.config_digest, start.config_generation,
        start.supervisor_id, start.supervisor_incarnation, start.worker_id,
        start.worker_generation, start.grant_id, start.container_id,
        start.lineage_id, current, current)
    return terminal, fence


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_actual_planned_consumer_seals_then_parent_fence_ingests(tmp_path, backend):
    prepared = _prepared(tmp_path, backend=backend)
    start = _start(prepared)
    authority = Authority()
    reference = uw.run_prepared_stage(
        prepared, start, _test_authority=authority, _test_membership_probe=lambda _: None,
        _test_measure=_measure, clock=lambda: 1.0,
        wall_clock=iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                         "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z")).__next__)
    entries = {}
    terminal, fence = _terminal_and_fence(prepared, start, reference)
    receipts = uw.ingest_deferred_result(
        reference, prepared=prepared, start=start, terminal=terminal, fence=fence,
        capture_transaction=lambda key, value: entries.setdefault(
            key, json.loads(json.dumps(value))))
    assert len(receipts) == len(entries) == 2
    assert [row[1] for row in authority.units] == [
        unit.unit_id for unit in prepared.plan.expected_units]
    assert all(item["carrier"]["status"] == "measurement" for item in entries.values())


def test_missing_real_membership_refuses_and_pause_stops_fixed_successor(tmp_path):
    prepared = _prepared(tmp_path)
    start = _start(prepared)
    with pytest.raises(ps.TrustedStageProviderRequired, match="not connected"):
        uw.run_prepared_stage(prepared, start, _test_measure=_measure, clock=lambda: 1.0)
    with pytest.raises(ps.UnsupportedContainment):
        uw.run_prepared_stage(prepared, start, _test_authority=Authority(),
                              _test_measure=_measure,
                              clock=lambda: 1.0)
    authority = Authority(stop_at=2)
    reference = uw.run_prepared_stage(
        prepared, start, _test_authority=authority, _test_membership_probe=lambda _: None,
        _test_measure=_measure, clock=lambda: 1.0,
        wall_clock=iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z")).__next__)
    store = uw.mc.ArtifactStore(prepared.artifact_root)
    result = uw.PlannedWorkerResult.from_dict(
        store.read(reference.result_locator, reference.result_sha256))
    assert result.body["completed_unit_ids"] == ("anchor-0",)
    assert result.body["run"]["execution_complete"] is False


def test_inherited_socket_authority_is_bounded_and_returns_typed_permit(tmp_path):
    prepared, start = _prepared(tmp_path), None
    start = _start(prepared)
    child, parent = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)

    def watcher():
        request = uw.read_bounded_message(parent)
        unit = request["unit"]
        body = {"schema": uw.UNIT_PERMIT_SCHEMA, "nonce": start.nonce,
                "sequence": request["sequence"], "unit_id": unit["unit_id"],
                "process_generation_id": unit["process_id"], "fence_id": "fence-1",
                "valid_until": 10.0, "grant_id": start.grant_id,
                "grant_generation": start.grant_generation,
                "container_id": start.container_id, "allowed": True, "reason": "held"}
        uw.write_bounded_message(parent, {**body, "permit_digest": uw._digest(body)})
        parent.close()

    thread = threading.Thread(target=watcher)
    thread.start()
    authority = uw.InheritedUnitAuthority(child, start=start, clock=lambda: 1.0)
    unit = prepared.plan.expected_units[0]
    permit = authority.admit(sequence=1, plan_digest=prepared.plan.digest,
                             unit=unit, prior_completion_digest=None)
    authority.close()
    thread.join(timeout=1.0)
    assert not thread.is_alive() and permit.unit_id == unit.unit_id and permit.allowed


def test_bounded_v2_observation_messages_bind_unit_worker_and_actual_pid(tmp_path):
    prepared = _prepared(tmp_path)
    start = _start(prepared)
    unit = prepared.plan.expected_units[0]
    fence = ps.StageFence(
        "fence-observation", unit.unit_id, unit.process_id, start.lineage_id,
        start.grant_id, start.container_id, start.clock_domain, 10.0,
        start.supervisor_id, start.supervisor_incarnation, start.config_generation,
        start.worker_id, start.worker_generation)
    instrument = ob.LoadedInstrumentReference(
        "a" * 64, True, mc.StoredArtifact("instrument.json", "b" * 64, True))
    worker_binding = {"worker_id": start.worker_id,
        "worker_incarnation": start.worker_generation, "grant_id": start.grant_id,
        "grant_generation": start.grant_generation,
        "container_identity": uw._plain(start.cgroup_identity)}
    budgets = {"max_samples": 16, "max_pending_markers": 8, "max_processes": 8,
        "max_read_bytes": 65536, "max_proc_entries": 32,
        "max_retained_bytes": 1024 * 1024, "max_map_entries": 16,
        "max_fd_entries": 16, "max_cpu_ids": 16, "max_numa_rows": 8,
        "max_dso_entries": 4, "phase_ack_timeout_s": 0.1,
        "join_timeout_s": 0.1, "max_probe_duration_s": 0.1}
    binding = ob.ObservationUnitBinding.from_dict(ob.ObservationUnitBinding(
        "obs-1", unit.unit_id, unit.process_id, fence.fence_id,
        start.clock_domain, start.child_process.boot_id, worker_binding,
        start.container_id, "claim:1", {"logical_cpus": [0], "gpu_devices": []},
        {"logical_cpus": [0], "numa_nodes": [0], "thp_mode": "madvise"},
        (), (), 1.0, 2.0, budgets, instrument).to_dict())
    child, parent = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    errors = []

    def watcher():
        try:
            request = uw.read_bounded_message(parent)
            assert request["schema"] == uw.OBSERVATION_BINDING_REQUEST_SCHEMA
            body = {"schema": uw.OBSERVATION_BINDING_SCHEMA, "nonce": start.nonce,
                    "sequence": 1, "binding": binding.to_dict()}
            uw.write_bounded_message(
                parent, {**body, "response_digest": uw._digest(body)})
            request = uw.read_bounded_message(parent)
            assert request["schema"] == uw.OBSERVATION_TARGET_REQUEST_SCHEMA
            response = {"schema": uw.OBSERVATION_TARGET_RECEIPT_SCHEMA,
                "nonce": start.nonce, "sequence": 1, "unit_id": unit.unit_id,
                "process_generation_id": unit.process_id, "fence_id": fence.fence_id,
                "pid": request["pid"], "start_ticks": 999,
                "boot_id": start.child_process.boot_id,
                "worker_binding": worker_binding, "binding_ref": "event:descendant"}
            uw.write_bounded_message(
                parent, {**response, "response_digest": uw._digest(response)})
        except BaseException as exc:
            errors.append(exc)
        finally:
            parent.close()

    thread = threading.Thread(target=watcher)
    thread.start()
    authority = uw.InheritedUnitAuthority(child, start=start, clock=lambda: 1.0)
    assert authority.observation_binding(
        sequence=1, unit=unit, fence=fence,
        recipe_identity_digest=prepared.runtime_pair.anchor.execution_digest) == binding
    target = authority.observation_target(
        sequence=1, unit=unit, fence=fence, pid=123)
    authority.close()
    thread.join(timeout=1.0)
    assert not thread.is_alive() and errors == []
    assert target["pid"] == 123 and target["binding_ref"] == "event:descendant"


def test_observation_requests_share_parent_retained_key_bound(tmp_path):
    prepared, start = _prepared(tmp_path), None
    start = _start(prepared)
    unit = prepared.plan.expected_units[0]
    fence = ps.StageFence(
        "fence-bound", unit.unit_id, unit.process_id, start.lineage_id,
        start.grant_id, start.container_id, start.clock_domain, 10.0,
        start.supervisor_id, start.supervisor_incarnation, start.config_generation,
        start.worker_id, start.worker_generation)
    cache = uw.ParentUnitEvidenceAuthority(max_records=1)
    cache.request_observation_binding(
        start=start, sequence=1, unit=unit, fence=fence,
        recipe_identity_digest=prepared.runtime_pair.anchor.execution_digest)
    cache.next_notice(timeout=0.1)
    with pytest.raises(uw.WorkerBridgeRefused, match="retained-key bound"):
        cache.request_observation_target(
            start=start, sequence=1, unit=unit, fence=fence, pid=123)


def test_fixed_order_and_whole_comparison_deadline_cannot_be_extended(tmp_path):
    prepared, start = _prepared(tmp_path), None
    start = _start(prepared)
    provider = uw.OwnedWorkerStageProvider(
        prepared=prepared, start=start, authority=Authority(),
        membership_probe=lambda _: None, clock=lambda: 1.0)
    with pytest.raises(uw.WorkerBridgeRefused, match="out of order"):
        provider.admit(prepared.plan.digest, prepared.plan.expected_units[1], ps.STAGES)

    class Extending(Authority):
        def admit(self, **kwargs):
            permit = super().admit(**kwargs)
            return uw.UnitPermit(**{**permit.__dict__, "valid_until": 21.0})

    provider = uw.OwnedWorkerStageProvider(
        prepared=prepared, start=start, authority=Extending(),
        membership_probe=lambda _: None, clock=lambda: 1.0)
    with pytest.raises(uw.WorkerBridgeRefused, match="held allocation"):
        provider.admit(prepared.plan.digest, prepared.plan.expected_units[0], ps.STAGES)


def test_continuation_rechecks_live_verifier_and_uses_new_lineage(tmp_path):
    prepared = _prepared(tmp_path, continuation_allowed=True)
    start = _start(prepared)
    first_ref = uw.run_prepared_stage(
        prepared, start, _test_authority=Authority(), _test_membership_probe=lambda _: None,
        _test_measure=_measure, clock=lambda: 1.0,
        wall_clock=iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                         "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z")).__next__)
    store = uw.mc.ArtifactStore(prepared.artifact_root)
    first = uw.PlannedWorkerResult.from_dict(
        store.read(first_ref.result_locator, first_ref.result_sha256)).to_dict()
    store.close()
    body = prepared.body()
    body["previous"] = {"raw_units": first["run"]["raw_units"][:1],
        "previous_lineage_id": "lineage-1", "continuation_proof_ref": "journal:prior",
        "continuation_proof_digest": "e" * 64}
    resumed = uw.PreparedPlannedServingStage.from_dict(
        {**body, "prepared_digest": uw._digest(body)})
    next_start = _start(resumed, lineage="lineage-2")
    authority = Authority()
    ref = uw.run_prepared_stage(
        resumed, next_start, _test_authority=authority,
        _test_membership_probe=lambda _: None, _test_measure=_measure,
        clock=lambda: 1.0,
        wall_clock=iter(("2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z")).__next__)
    store = uw.mc.ArtifactStore(resumed.artifact_root)
    result = uw.PlannedWorkerResult.from_dict(
        store.read(ref.result_locator, ref.result_sha256)).to_dict()
    store.close()
    assert result["lineage_id"] == "lineage-2"
    assert [item[1] for item in authority.units] == ["candidate-0"]
    assert result["completed_unit_ids"] == ["anchor-0", "candidate-0"]


def test_tampered_recipe_dso_is_refused_before_any_authority(tmp_path):
    prepared = _prepared(tmp_path)
    body = prepared.body()
    body["runtime_pair"]["candidate"]["dsos"][0]["sha256"] = "0" * 64
    with pytest.raises(uw.WorkerBridgeRefused, match="typed input"):
        uw.PreparedPlannedServingStage.from_dict(
            {**body, "prepared_digest": uw._digest(body)})


def test_v2_worker_envelopes_are_distinct_and_cannot_wrap_a_v1_plan(tmp_path):
    prepared = _prepared(tmp_path)
    body = prepared.body()
    body["schema"] = uw.PREPARED_SCHEMA_V2
    with pytest.raises(uw.WorkerBridgeRefused, match="schema versions differ"):
        uw.PreparedPlannedServingStage.from_dict(
            {**body, "prepared_digest": uw._digest(body)})

    reference = uw.PlannedWorkerResultReference(
        "nonce-0123456789abcdef", "a" * 64, "worker-1", 7,
        "b" * 64, "result.json", "c" * 64,
        schema=uw.RESULT_REFERENCE_SCHEMA_V2)
    assert uw.PlannedWorkerResultReference.from_dict(reference.to_dict()) == reference


def test_v2_direct_worker_uses_concrete_observer_factory_and_result_envelope(tmp_path):
    prepared, instrument = _v2_prepared(tmp_path)
    start = _start(prepared)
    start_body = start.body() | {"provider_deadline": time.monotonic() + 20.0}
    start = uw.WorkerStart.from_dict(
        {**start_body, "start_digest": uw._digest(start_body)})
    probe_root = tmp_path / "fake-proc-sys"
    probe_root.mkdir()
    probe, proc, _pressure, _vram, _kfd, container = _fixture(probe_root)
    _write_process(proc, 101, start=999, ticks=1)

    class ObservedAuthority(Authority):
        def observation_phase(self, *, sequence, unit, fence, binding, target,
                              phase, boundary_monotonic_s):
            assert phase == "health" and sequence == unit.order_index + 1
            assert binding.fence_id == fence.fence_id and target["pid"] == 101
            assert boundary_monotonic_s > 0
            return {"outcome": "unavailable"}

        def admit(self, *, sequence, plan_digest, unit, prior_completion_digest):
            del plan_digest, prior_completion_digest
            return uw.UnitPermit(
                sequence, unit.unit_id, unit.process_id, f"fence-{sequence}",
                start.provider_deadline, start.grant_id, start.grant_generation,
                start.container_id, True, "test held")

        def observation_binding(self, *, sequence, unit, fence,
                                recipe_identity_digest):
            assert sequence == unit.order_index + 1 and recipe_identity_digest
            return ob.ObservationUnitBinding.from_dict(ob.ObservationUnitBinding(
                f"obs-{unit.unit_id}", unit.unit_id, unit.process_id, fence.fence_id,
                start.clock_domain, "boot-fixture",
                {"worker_id": start.worker_id,
                 "worker_incarnation": start.worker_generation,
                 "grant_id": start.grant_id,
                 "grant_generation": start.grant_generation,
                 "container_identity": container}, start.container_id,
                "claim:test", {"logical_cpus": [0], "gpu_devices": []},
                {"logical_cpus": [0], "numa_nodes": [0], "thp_mode": "madvise"},
                (), (), 10.0, 11.0, _budgets(), instrument).to_dict())

        def observation_target(self, *, sequence, unit, fence, pid):
            del sequence, unit, fence
            return {"pid": pid, "start_ticks": 999, "boot_id": "boot-fixture",
                "worker_binding": {"worker_id": start.worker_id,
                    "worker_incarnation": start.worker_generation,
                    "grant_id": start.grant_id,
                    "grant_generation": start.grant_generation,
                    "container_identity": container},
                "binding_ref": "event:test-owned-descendant"}

    reference = uw.run_prepared_stage(
        prepared, start, _test_authority=ObservedAuthority(),
        _test_membership_probe=lambda _: None, _test_measure=_observed_measure,
        _test_observation_probe=probe, clock=time.monotonic)
    assert reference.schema == uw.RESULT_REFERENCE_SCHEMA_V2
    store = mc.ArtifactStore(prepared.artifact_root)
    result = uw.PlannedWorkerResult.from_dict(
        store.read(reference.result_locator, reference.result_sha256)).to_dict()
    store.close()
    assert result["schema"] == uw.RESULT_SCHEMA_V2
    assert len(result["lifecycle_observation_references"]) == 2
    assert all(row["descendant_binding_ref"] == "event:test-owned-descendant"
               for row in result["lifecycle_observation_references"])
    terminal, fence = _terminal_and_fence(prepared, start, reference)
    captured = {}
    receipts = uw.ingest_deferred_result(
        reference, prepared=prepared, start=start, terminal=terminal, fence=fence,
        capture_transaction=lambda key, value: captured.setdefault(
            key, json.loads(json.dumps(value))))
    assert len(receipts) == len(captured) == 2
    assert all(row["schema"] == mc.CAPTURE_SCHEMA_V2 for row in captured.values())


def test_stale_worker_fence_and_tampered_reference_never_call_parent(tmp_path):
    prepared, calls = _prepared(tmp_path), []
    start = _start(prepared)
    reference = uw.run_prepared_stage(
        prepared, start, _test_authority=Authority(), _test_membership_probe=lambda _: None,
        _test_measure=_measure, clock=lambda: 1.0,
        wall_clock=iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                         "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z")).__next__)
    terminal, stale = _terminal_and_fence(prepared, start, reference, current=False)
    with pytest.raises(uw.WorkerBridgeRefused, match="stale"):
        uw.ingest_deferred_result(
            reference, prepared=prepared, start=start, terminal=terminal, fence=stale,
            capture_transaction=lambda *_: calls.append(1))
    assert calls == []
    changed = reference.to_dict()
    changed["result_sha256"] = "0" * 64
    with pytest.raises(uw.WorkerBridgeRefused, match="digest"):
        uw.PlannedWorkerResultReference.from_dict(changed)

    # Even a second correctly sealed result under the same worker identity is not
    # the exact result-reference envelope accepted by lifecycle.
    store = uw.mc.ArtifactStore(prepared.artifact_root)
    original = uw.PlannedWorkerResult.from_dict(
        store.read(reference.result_locator, reference.result_sha256)).to_dict()
    alternate_body = {key: value for key, value in original.items()
                      if key != "result_digest"}
    alternate_body["completed_unit_ids"] = []
    alternate = uw.PlannedWorkerResult.from_dict(
        {**alternate_body, "result_digest": uw._digest(alternate_body)})
    sealed = store.write("planned-worker-result:substitute", alternate.to_dict())
    store.close()
    substitute = uw.PlannedWorkerResultReference(
        reference.nonce, reference.prepared_digest, reference.worker_id,
        reference.worker_generation, alternate.result_digest, sealed.locator,
        sealed.sha256)
    terminal, current = _terminal_and_fence(prepared, start, reference)
    with pytest.raises(uw.WorkerBridgeRefused, match="stale"):
        uw.ingest_deferred_result(
            substitute, prepared=prepared, start=start, terminal=terminal,
            fence=current, capture_transaction=lambda *_: calls.append(1))
    assert calls == []


def test_bounded_socket_messages_reject_noncanonical_truncation_and_oversize():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        uw.write_bounded_message(left, {"a": 1})
        assert dict(uw.read_bounded_message(right)) == {"a": 1}
        left.sendall((7).to_bytes(4, "big") + b'{"a":1}')
        # This spelling happens to be canonical; whitespace does not.
        assert dict(uw.read_bounded_message(right)) == {"a": 1}
        left.sendall((8).to_bytes(4, "big") + b'{"a": 1}')
        with pytest.raises(uw.WorkerBridgeRefused, match="canonical"):
            uw.read_bounded_message(right)
        left.sendall((uw.MAX_MESSAGE_BYTES + 1).to_bytes(4, "big"))
        with pytest.raises(uw.WorkerBridgeRefused, match="length"):
            uw.read_bounded_message(right)
    finally:
        left.close()
        right.close()


def test_trickled_socket_bytes_cannot_renew_absolute_deadline():
    left, right = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    sender_errors = []

    def trickle():
        try:
            raw = len(b'{"a":1}').to_bytes(4, "big") + b'{"a":1}'
            for byte in raw:
                left.send(bytes((byte,)))
                time.sleep(0.015)
        except (BrokenPipeError, OSError) as exc:
            sender_errors.append(exc)

    thread = threading.Thread(target=trickle)
    thread.start()
    try:
        deadline = time.monotonic() + 0.04
        with pytest.raises(uw.WorkerBridgeRefused, match="deadline"):
            uw.read_bounded_message(right, deadline=deadline)
        assert time.monotonic() < deadline + 0.1
    finally:
        right.close()
        left.close()
        thread.join(timeout=1.0)
        assert not thread.is_alive()


def test_tiny_owned_child_runs_consumer_and_is_reaped(tmp_path):
    prepared = _prepared(tmp_path)
    start_read, start_write = os.pipe2(os.O_CLOEXEC)
    result_read, result_write = os.pipe2(os.O_CLOEXEC)
    error_read, error_write = os.pipe2(os.O_CLOEXEC)
    child_control, parent_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    pid = os.fork()
    if pid == 0:  # pragma: no cover - assertions are made by the parent
        child_status = 0
        try:
            os.close(start_write)
            os.close(result_read)
            os.close(error_read)
            parent_control.close()
            uw.run_from_fds(
                start_fd=start_read, control_fd=child_control.detach(),
                result_fd=result_write,
                _test_membership_probe=lambda _: None,
                _test_measure=_measure)
        except BaseException as exc:
            os.write(error_write, f"{type(exc).__name__}: {exc}".encode()[:1024])
            child_status = 1
        finally:
            os.close(error_write)
            os._exit(child_status)
    os.close(start_read)
    os.close(result_write)
    os.close(error_write)
    child_control.close()
    hello = uw.read_bounded_message(parent_control)
    child_identity = wl.ProcessIdentity(**dict(hello["child_process"]))
    assert child_identity == wl.process_identity(pid)
    start_body = _start(prepared).body()
    start_body.update(nonce=hello["nonce"], child_process=child_identity.to_dict(),
                      provider_deadline=time.monotonic() + 5.0,
                      clock_domain=wl.monotonic_clock_domain())
    start = uw.WorkerStart.from_dict(
        {**start_body, "start_digest": uw._digest(start_body)})
    invocation_body = {"schema": uw.INVOCATION_SCHEMA, "nonce": hello["nonce"],
                       "prepared": prepared.to_dict(), "start": start.to_dict()}
    uw._write_fd_message(
        start_write, {**invocation_body,
                      "invocation_digest": uw._digest(invocation_body)},
        limit=uw.MAX_RESULT_BYTES)
    os.close(start_write)
    authority = Authority()
    evidence = uw.ParentUnitEvidenceAuthority()
    producer_errors = []

    def produce_completion_evidence():
        try:
            for _ in range(2):
                notice = evidence.next_notice(timeout=2.0)
                assert notice["kind"] == "completion"
                completion = authority.complete(
                    sequence=notice["sequence"], fence=notice["fence"],
                    observation=notice["observation"])
                evidence.publish_completion(notice["key"], completion)
        except BaseException as exc:
            producer_errors.append(exc)

    producer = threading.Thread(target=produce_completion_evidence)
    producer.start()
    for _ in range(4):
        try:
            request = uw.read_bounded_message(parent_control)
        except uw.WorkerBridgeRefused as exc:
            detail = os.read(error_read, 1024).decode()
            raise AssertionError(f"owned child failed: {detail}") from exc
        if request["schema"] == uw.UNIT_REQUEST_SCHEMA:
            unit = ep.UnitSpec.from_dict(request["unit"])
            permit = authority.admit(
                sequence=request["sequence"], plan_digest=request["plan_digest"],
                unit=unit, prior_completion_digest=request["prior_completion_digest"])
            body = {"schema": uw.UNIT_PERMIT_SCHEMA, "nonce": start.nonce,
                    "sequence": permit.sequence, "unit_id": permit.unit_id,
                    "process_generation_id": permit.process_generation_id,
                    "fence_id": permit.fence_id,
                    "valid_until": start.provider_deadline,
                    "grant_id": start.grant_id,
                    "grant_generation": start.grant_generation,
                    "container_id": start.container_id, "allowed": True,
                    "reason": "owned test allocation"}
            uw.write_bounded_message(
                parent_control, {**body, "permit_digest": uw._digest(body)})
        else:
            assert request["schema"] == uw.UNIT_COMPLETION_REQUEST_SCHEMA
            fence = ps.StageFence(
                request["fence_id"], prepared.plan.expected_units[
                    request["sequence"] - 1].unit_id,
                prepared.plan.expected_units[request["sequence"] - 1].process_id,
                start.lineage_id, start.grant_id, start.container_id,
                start.clock_domain, start.provider_deadline, start.supervisor_id,
                start.supervisor_incarnation, start.config_generation,
                start.worker_id, start.worker_generation)
            key, completion = evidence.request_completion(
                start=start, sequence=request["sequence"], fence=fence,
                observation=request["observation"])
            deadline = time.monotonic() + 2.0
            while completion is None and time.monotonic() < deadline:
                completion = evidence.poll_completion(key)
                if completion is None:
                    time.sleep(0.001)
            assert completion is not None
            body = {"schema": uw.UNIT_COMPLETION_SCHEMA, "nonce": start.nonce,
                    "sequence": request["sequence"], "fence_id": request["fence_id"],
                    "terminal": completion.terminal,
                    "stage_witnesses": {key: value.to_dict() for key, value
                                        in completion.stage_witnesses.items()},
                    "recorded_screen": completion.recorded_screen,
                    "reason": completion.reason}
            uw.write_bounded_message(
                parent_control, {**body, "completion_digest": uw._digest(body)})
    reference = uw.PlannedWorkerResultReference.from_dict(
        uw._read_fd_message(result_read, limit=uw.MAX_MESSAGE_BYTES))
    os.close(result_read)
    assert os.read(error_read, 1) == b""
    os.close(error_read)
    parent_control.close()
    producer.join(timeout=2.0)
    assert not producer.is_alive() and producer_errors == []
    waited, status = os.waitpid(pid, 0)
    assert waited == pid and os.waitstatus_to_exitcode(status) == 0
    print(f"OWNED_TEST_PID pid={child_identity.pid} start_ticks={child_identity.start_ticks} "
          f"boot_id={child_identity.boot_id} alive={wl.same_process(child_identity)}")
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert reference.worker_generation == start.worker_generation
    entries = {}
    terminal, fence = _terminal_and_fence(prepared, start, reference)
    assert len(uw.ingest_deferred_result(
        reference, prepared=prepared, start=start, terminal=terminal, fence=fence,
        capture_transaction=lambda key, value: entries.setdefault(key, value))) == 2


def test_fixed_isolated_entrypoint_hello_then_malformed_start_is_reaped(tmp_path):
    start_read, start_write = os.pipe2(os.O_CLOEXEC)
    result_read, result_write = os.pipe2(os.O_CLOEXEC)
    child_control, parent_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    for fd in (start_read, result_write, child_control.fileno()):
        os.set_inheritable(fd, True)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    marker = tmp_path / "sitecustomize-ran"
    (hostile / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('bad')\n")
    process = subprocess.Popen(
        [str(Path(sys.executable).resolve()), "-I", "-B", str(Path(uw.__file__).resolve()),
         "--start-fd", str(start_read), "--control-fd", str(child_control.fileno()),
         "--result-fd", str(result_write)],
        cwd=hostile, env={"HOME": str(tmp_path), "PYTHONPATH": str(hostile),
                          "PYTHONUSERBASE": str(hostile), "INHERITED_SECRET": "not-used"},
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        close_fds=True, pass_fds=(start_read, child_control.fileno(), result_write))
    owned_pid = process.pid
    os.close(start_read)
    os.close(result_write)
    child_control.close()
    hello = uw.read_bounded_message(parent_control)
    assert hello["schema"] == uw.HELLO_SCHEMA and hello["child_process"]["pid"] == owned_pid
    os.write(start_write, (8).to_bytes(4, "big") + b'{"x": 1}')
    os.close(start_write)
    parent_control.close()
    stdout, stderr = process.communicate(timeout=3.0)
    assert process.returncode == 125 and stdout == b"" and len(stderr) < 1024
    assert not marker.exists()
    assert os.read(result_read, 1) == b""
    os.close(result_read)
    with pytest.raises(ProcessLookupError):
        os.kill(owned_pid, 0)


def test_parent_large_start_write_is_nonblocking_with_stalled_owned_peer(tmp_path):
    invocation = uw.PlannedWorkerInvocation.open(
        _prepared(tmp_path), uw.ParentUnitEvidenceAuthority())
    try:
        invocation._queue(invocation._start_write, {"payload": "x" * (512 * 1024)},
                          uw.MAX_RESULT_BYTES)
        before = len(invocation._write_buffers[invocation._start_write])
        started = time.monotonic()
        invocation.flush_ready(invocation._start_write)
        assert time.monotonic() - started < 0.25
        assert 0 < len(invocation._write_buffers[invocation._start_write]) < before
    finally:
        invocation.close()


def test_invocation_constructor_closes_acquired_pipe_on_second_pipe_failure(
        tmp_path, monkeypatch):
    original = os.pipe2
    acquired = []

    def fail_second(flags):
        if acquired:
            raise OSError("injected second pipe failure")
        pair = original(flags)
        acquired.extend(pair)
        return pair

    monkeypatch.setattr(uw.os, "pipe2", fail_second)
    authority = uw.ParentUnitEvidenceAuthority()
    with pytest.raises(OSError, match="second pipe"):
        uw.PlannedWorkerInvocation.open(_prepared(tmp_path), authority)
    for fd in acquired:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_invocation_refuses_incompatible_authority_before_descriptor_allocation(
        tmp_path, monkeypatch):
    def unexpected_pipe(_flags):
        raise AssertionError("descriptor allocation preceded authority validation")

    monkeypatch.setattr(uw.os, "pipe2", unexpected_pipe)
    with pytest.raises(uw.WorkerBridgeRefused, match="concrete parent evidence cache"):
        uw.PlannedWorkerInvocation.open(_prepared(tmp_path), object())


def test_parent_evidence_bound_covers_drained_outstanding_and_cached_keys(tmp_path):
    prepared = _prepared(tmp_path, continuation_allowed=True)
    start = _start(prepared)
    unit = prepared.plan.expected_units[0]
    fence = ps.StageFence(
        "fence-1", unit.unit_id, unit.process_id, start.lineage_id,
        start.grant_id, start.container_id, start.clock_domain,
        start.provider_deadline, start.supervisor_id,
        start.supervisor_incarnation, start.config_generation,
        start.worker_id, start.worker_generation)
    evidence = uw.ParentUnitEvidenceAuthority(max_records=1)
    observation = {"schema": "fixture-observation", "value": 1}

    completion_key, completion = evidence.request_completion(
        start=start, sequence=1, fence=fence, observation=observation)
    assert completion is None
    assert evidence.next_notice(timeout=0.1)["key"] == completion_key
    # Draining the notice cannot free the retained request slot, while an exact
    # retry remains legal and does not enqueue a duplicate notice.
    assert evidence.request_completion(
        start=start, sequence=1, fence=fence,
        observation=observation) == (completion_key, None)
    with pytest.raises(uw.WorkerBridgeRefused, match="retained-key bound"):
        evidence.request_completion(
            start=start, sequence=1, fence=fence,
            observation={"schema": "fixture-observation", "value": 2})

    raw = ep.RawUnit.from_dict(raw_dict(prepared.plan, unit.unit_id))
    with pytest.raises(uw.WorkerBridgeRefused, match="retained-key bound"):
        evidence.request_continuation(
            start=start, raw=raw, plan=prepared.plan,
            prompts=prepared.prompts,
            previous_lineage_id="previous-lineage")
    assert evidence._requested == {completion_key: "completion"}
    assert evidence._notices.empty()

    typed = Authority().complete(
        sequence=1, fence=fence, observation=observation)
    evidence.publish_completion(completion_key, typed)
    assert evidence.request_completion(
        start=start, sequence=1, fence=fence,
        observation=observation) == (completion_key, typed)
    with pytest.raises(uw.WorkerBridgeRefused, match="conflicts"):
        evidence.publish_completion(
            completion_key, ps.StageCompletion(
                fence.fence_id, False, typed.stage_witnesses,
                typed.recorded_screen, "conflicting result"))

    preseeded = uw.ParentUnitEvidenceAuthority(
        completions={completion_key: typed}, max_records=1)
    with pytest.raises(uw.WorkerBridgeRefused, match="retained-key bound"):
        preseeded.request_continuation(
            start=start, raw=raw, plan=prepared.plan,
            prompts=prepared.prompts,
            previous_lineage_id="previous-lineage")


def test_owned_bootstrap_reaches_exact_child_then_refuses_mock_containment(tmp_path):
    prepared = _prepared(tmp_path)
    body = prepared.body()
    body.update(max_stage_seconds=1.0, teardown_seconds=0.5)
    prepared = uw.PreparedPlannedServingStage.from_dict(
        {**body, "prepared_digest": uw._digest(body)})
    prepared.artifact_root.mkdir(mode=0o700)
    invocation = uw.PlannedWorkerInvocation.open(
        prepared, uw.ParentUnitEvidenceAuthority())
    request = invocation.stage_request(
        request_id="request-planned", lineage_id="lineage-1",
        stage_id="stage-planned", control_revision=0)
    harness = Harness()
    harness.binding = wl.CampaignBinding(
        "camp-1", "c" * 64, 3, "supervisor-1", 2)
    harness.engine = wl.WorkerLifecycle(
        binding=harness.binding, runtime=harness.runtime,
        event_sink=harness._record_event, provider=harness.provider,
        admission_fence=lambda *_: wl.StageAdmission(True, "admitted"),
        binding_fence=lambda _binding: True,
        runtime_fence=lambda _request, _now: wl.RuntimeDirective("continue", "held"),
        wall_clock=lambda: "2026-09-09T00:00:00Z")
    try:
        with pytest.raises(wl.LifecycleRefused, match="result reference"):
            harness.engine.run_stage(request, planned_invocation=invocation)
        identities = tuple(harness.provider.authorization.container._owned.values())
        assert len(identities) >= 2  # bootstrap plus its exact planned child
        for identity in identities:
            print(f"OWNED_TEST_PID pid={identity.pid} start_ticks={identity.start_ticks} "
                  f"boot_id={identity.boot_id} alive={wl.same_process(identity)}")
            assert not wl.same_process(identity)
    finally:
        harness.close()


def test_accept_append_failure_never_exposes_planned_reference(tmp_path, monkeypatch):
    prepared = _prepared(tmp_path)
    body = prepared.body()
    body.update(max_stage_seconds=1.0, teardown_seconds=0.5)
    prepared = uw.PreparedPlannedServingStage.from_dict(
        {**body, "prepared_digest": uw._digest(body)})
    prepared.artifact_root.mkdir(mode=0o700)
    invocation = uw.PlannedWorkerInvocation.open(
        prepared, uw.ParentUnitEvidenceAuthority())
    request = invocation.stage_request(
        request_id="request-planned", lineage_id="lineage-1",
        stage_id="stage-planned", control_revision=0)
    harness = Harness()
    harness.binding = wl.CampaignBinding(
        "camp-1", "c" * 64, 3, "supervisor-1", 2)

    def event_sink(row):
        harness._record_event(row)
        if row.get("event") == "WORKER_RESULT_ACCEPTED":
            raise OSError("injected durable acceptance failure")

    harness.engine = wl.WorkerLifecycle(
        binding=harness.binding, runtime=harness.runtime, event_sink=event_sink,
        provider=harness.provider,
        admission_fence=lambda *_: wl.StageAdmission(True, "admitted"),
        binding_fence=lambda _binding: True,
        runtime_fence=lambda _request, _now: wl.RuntimeDirective("continue", "held"),
        wall_clock=lambda: "2026-09-09T00:00:00Z")
    reference = uw.PlannedWorkerResultReference(
        "fixture-nonce-0123456789", prepared.prepared_digest, "placeholder-worker",
        1, "d" * 64, "fixture-result.json", "e" * 64)

    def fake_wait(*_args, **_kwargs):
        invocation._reference = reference
        invocation._reference_digest = uw._digest(reference.to_dict())
        return {"schema": wl.OUTCOME_SCHEMA, "nonce": "unused",
                "contract_digest": "f" * 64, "child_pid": 1, "return_code": 0,
                "forwarded_signal": None, "stdout_bytes": 0, "stderr_bytes": 0,
                "stdout_sha256": "0" * 64, "stderr_sha256": "0" * 64,
                "stdout_truncated": False, "stderr_truncated": False}

    monkeypatch.setattr(harness.engine, "_wait_outcome", fake_wait)
    try:
        with pytest.raises(OSError, match="durable acceptance"):
            harness.engine.run_stage(request, planned_invocation=invocation)
        with pytest.raises(uw.WorkerBridgeRefused, match="terminally accepted"):
            invocation.result_reference()
        accepted_event = next(row for row in harness.events
                              if row["event"] == "WORKER_RESULT_ACCEPTED")
        uncommitted = wl.TerminalWorker(
            accepted_event["worker_id"], accepted_event["worker_generation"],
            request.request_id, request.plan_digest, request.lineage_id,
            request.stage_id, accepted_event["grant_id"],
            accepted_event["grant_generation"], accepted_event["container_id"],
            0, invocation.reference_digest, True, None)
        with pytest.raises(wl.LifecycleRefused, match="not owned"):
            harness.engine.trusted_result_fence(uncommitted)
        identities = tuple(harness.provider.authorization.container._owned.values())
        for identity in identities:
            assert not wl.same_process(identity)
    finally:
        harness.close()
