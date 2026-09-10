from dataclasses import replace

import pytest

from . import scheduling, serial_scheduling as ss
from .measurement_capture import ArtifactStore
from .test_scheduling import config, proposal, vector


def manifest():
    return ss.SerialSchedulerManifest.from_dict({
        "schema": ss.MANIFEST_SCHEMA, "scheduler_id": "serial-fixture",
        "config": config(noncoverage_slots=2,
                         capacity=vector(gpus=("mi210_0",), memory=10000)),
        "targets": {
            "cpu": proposal("cpu", backend="cpu", target="cpu-revision",
                            alias="cpu-workload"),
            "gpu": proposal("gpu", backend="gpu", target="gpu-revision",
                            alias="gpu-workload",
                            claims=vector(fraction=0.5, gpus=("mi210_0",), memory=0)),
        },
    })


def bindings():
    return {
        "cpu": {"target_revision": "cpu-revision", "alias_identity": "cpu-workload",
                "backend": "cpu", "eligibility_ref": "trusted:eligible-receipt"},
        "gpu": {"target_revision": "gpu-revision", "alias_identity": "gpu-workload",
                "backend": "gpu", "eligibility_ref": "trusted:eligible-receipt"},
    }


def test_closed_manifest_binds_roster_and_selects_with_owning_pure_scheduler():
    source = manifest()
    ss.validate_target_bindings(source, bindings())
    state = scheduling.initial_state(source.config, source.scheduler_id)
    state, selected, index = ss.select_target(
        source, state, ("cpu", "gpu"), now=1, stage_number=0)
    assert index == 0 and selected.proposal.proposal_id == ss._digest({
        "manifest": source.digest, "selected_id": "cpu", "stage_number": 0})
    assert state.issued_selection_digests == (selected.digest,)
    _next, next_selected, _index = ss.select_target(
        source, scheduling.initial_state(source.config, source.scheduler_id),
        ("cpu", "gpu"), now=2, stage_number=1)
    assert next_selected.proposal.proposal_id != selected.proposal.proposal_id
    assert ss.SerialSchedulerManifest.from_dict(source.to_dict()).digest == source.digest


def test_validation_is_an_explicit_reserved_stage_with_distinct_identity():
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    _state, selected, _index = ss.select_target(
        source, state, ("cpu", "gpu"), now=1, stage_number=0,
        validation_ids=frozenset({"cpu"}))
    assert selected.proposal.stage_class == "validation"
    assert selected.proposal.reservation_kind == "validation"
    assert selected.proposal.proposal_id == ss._digest({
        "manifest": source.digest, "selected_id": "cpu", "stage_number": 0,
        "source_validation": True})

    _state, ordinary, _index = ss.select_target(
        source, scheduling.initial_state(source.config, source.scheduler_id),
        ("cpu", "gpu"), now=1, stage_number=0)
    assert ordinary.proposal.stage_class == "search"
    assert ordinary.proposal.proposal_id != selected.proposal.proposal_id


def test_manifest_refuses_unknown_fields_duplicate_proposals_and_crossed_target():
    row = manifest().to_dict()
    with pytest.raises(ss.SerialSchedulingRefused, match="shape"):
        ss.SerialSchedulerManifest.from_dict(row | {"unknown": True})
    row["targets"]["gpu"]["proposal_id"] = "cpu"
    with pytest.raises(ss.SerialSchedulingRefused, match="unique"):
        ss.SerialSchedulerManifest.from_dict(row)
    source = manifest()
    crossed = bindings()
    crossed["cpu"] = dict(crossed["cpu"], target_revision="gpu-revision")
    with pytest.raises(ss.SerialSchedulingRefused, match="differs"):
        ss.validate_target_bindings(source, crossed)


@pytest.mark.parametrize("native, expected", [
    ("kept", "valid_comparison"),
    ("measured_null", "valid_comparison"),
    ("runtime_observed", "invalid"),
    ("measurement_invalid", "invalid"),
    ("bench_failed", "failed"),
    ("lane_error", "failed"),
    ("planner_transient", "failed"),
])
def test_one_iteration_outcome_does_not_upgrade_unqualified_timing(native, expected):
    assert ss.one_iteration_outcome("complete", {native: 1}) == expected


def test_scheduled_outcome_refuses_batched_or_unknown_result():
    with pytest.raises(ss.SerialSchedulingRefused, match="exactly one"):
        ss.one_iteration_outcome("complete", {"measured_null": 2})
    with pytest.raises(ss.SerialSchedulingRefused, match="unsupported"):
        ss.one_iteration_outcome("complete", {"new_status": 1})
    assert ss.one_iteration_outcome("stopped", {}) == "failed"


def test_manifest_detaches_caller_and_is_immutable():
    source = manifest()
    row = source.to_dict()
    parsed = ss.SerialSchedulerManifest.from_dict(row)
    row["targets"]["cpu"]["backend"] = "gpu"
    assert parsed.proposals["cpu"].backend == "cpu"
    with pytest.raises(TypeError):
        parsed.proposals["new"] = replace(parsed.proposals["cpu"], proposal_id="new")


def _observation(pid, suffix):
    return {"observed_at": 10.0, "started_monotonic_s": 1.0,
            "ended_monotonic_s": 1.1, "owner_pid": pid, "error": None, "status": "held",
            "locks": [{"path": f"/locks/{suffix}", "device": 1,
                       "inode": sum(map(ord, suffix)),
                       "path_unchanged": True,
                       "owners": [{"pid": pid, "kernel_row": "original"}],
                       "same_holder": True}]}


def _component(device, start, end, *, fraction, claim, pid=123):
    start, end = float(start), float(end)
    domain = {"kind": "direct_loop", "clock": "monotonic", "pid": pid,
              "boot_id": "boot", "process_start_ticks": 99, "error": None}
    opened = _observation(pid, claim)
    closed = _observation(pid, claim)
    physical = f"boot:flock:1:{sum(map(ord, claim))}"
    return {"context_id": ss._digest({"domain": domain, "started_at": start,
                                      "locks": opened["locks"]}),
            "domain": domain,
            "ownership_generation": 1, "allocation_generation": 1,
            "started_at": start, "ended_at": end, "device_id": device,
            "physical_claim_ids": [physical], "physical_region_fraction": fraction,
            "gpu_device_ids": [] if device == "cpu" else [device],
            "memory_reservation_bytes": 0, "affinity_cores": ["0", "1"],
            "open": opened, "close": closed, "released": True}


def test_reopen_partitions_original_nested_gpu_interval_without_flattening(tmp_path):
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    _state, selected, _index = ss.select_target(
        source, state, ("gpu",), now=1, stage_number=0)
    target = {"selected_id": "gpu", "original": "fixture"}
    body = {"schema": ss.INTERVAL_SCHEMA, "selection": selected.to_dict(),
            "selection_digest": selected.digest, "target": target,
            "components": [_component("cpu", 1, 9, fraction=0.5, claim="cpu-flock"),
                           _component("mi210_0", 3, 7, fraction=0, claim="gpu-flock")]}
    root = tmp_path / "held-claim-artifacts"
    store = ArtifactStore(root)
    try:
        artifact = store.write("direct-held-intervals", body).to_dict()
    finally:
        store.close()
    reference = {"schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
                 "evidence": artifact}
    receipts = ss.reopen_held_receipts(tmp_path, reference, selection=selected, target=target)
    assert [(item.started_at, item.ended_at, item.gpu_device_ids)
            for item in receipts] == [(1, 3, ()), (3, 7, ("mi210_0",)), (7, 9, ())]
    assert receipts[1].physical_claim_ids == (
        "boot:flock:1:900", "boot:flock:1:904")
    assert all(item.physical_region_fraction == 0.5 for item in receipts)


def test_reopen_refuses_foreign_target_and_lost_original_claim(tmp_path):
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    _state, selected, _index = ss.select_target(
        source, state, ("cpu",), now=1, stage_number=0)
    target = {"selected_id": "cpu"}
    component = _component("cpu", 1, 2, fraction=0.5, claim="cpu-flock")
    component["close"]["status"] = "lost"
    body = {"schema": ss.INTERVAL_SCHEMA, "selection": selected.to_dict(),
            "selection_digest": selected.digest, "target": target,
            "components": [component]}
    store = ArtifactStore(tmp_path / "held-claim-artifacts")
    try:
        artifact = store.write("direct-held-intervals", body).to_dict()
    finally:
        store.close()
    reference = {"schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
                 "evidence": artifact}
    with pytest.raises(ss.SerialSchedulingRefused, match="same-owner"):
        ss.reopen_held_receipts(tmp_path, reference, selection=selected, target=target)
    with pytest.raises(ss.SerialSchedulingRefused, match="identity"):
        clean = dict(body, components=[_component(
            "cpu", 1, 2, fraction=0.5, claim="cpu-flock")])
        (tmp_path / "other").mkdir()
        other = ArtifactStore(tmp_path / "other" / "held-claim-artifacts")
        try:
            other_ref = other.write("direct-held-intervals", clean).to_dict()
        finally:
            other.close()
        ss.reopen_held_receipts(tmp_path / "other", {
            "schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
            "evidence": other_ref}, selection=selected, target={"selected_id": "foreign"})
