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
    ("regression", "valid_comparison"),
    ("abstained", "abstained"),
    ("runtime_observed", "invalid"),
    ("measurement_invalid", "invalid"),
    ("integrity_refused", "invalid"),
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


@pytest.mark.parametrize("native", ["kept", "measured_null", "abstained", "superseded",
                                    "lane_error", "patch_rounds_exhausted"])
def test_one_iteration_stage_outcome_is_exactly_the_historical_contract(native):
    assert ss.stage_outcome("complete", {native: 1}, 1) == \
        ss.one_iteration_outcome("complete", {native: 1})
    assert ss.stage_outcome("stopped", {}, 1) == "failed"
    with pytest.raises(ss.SerialSchedulingRefused, match="exactly one"):
        ss.stage_outcome("complete", {"measured_null": 2}, 1)


@pytest.mark.parametrize("counts,expected", [
    ({"measured_null": 2}, "valid_comparison"),
    ({"kept": 1, "abstained": 1}, "valid_comparison"),
    ({"abstained": 2}, "abstained"),
    # A pooled peer superseded by the other lane's keep is INVALID: worst class wins.
    ({"kept": 1, "superseded": 1}, "invalid"),
    ({"measured_null": 1, "lane_error": 1}, "failed"),
    ({"superseded": 1, "planner_transient": 1}, "failed"),
])
def test_multi_iteration_stage_folds_to_its_worst_iteration_class(counts, expected):
    assert ss.stage_outcome("complete", counts, sum(counts.values())) == expected


def test_multi_iteration_stage_refuses_counts_that_do_not_cover_the_batch():
    assert ss.stage_outcome("stopped", {"kept": 1}, 2) == "failed"
    with pytest.raises(ss.SerialSchedulingRefused, match="finite batch"):
        ss.stage_outcome("complete", {"measured_null": 1}, 2)
    with pytest.raises(ss.SerialSchedulingRefused, match="finite batch"):
        ss.stage_outcome("complete", {}, 2)
    with pytest.raises(ss.SerialSchedulingRefused, match="unsupported"):
        ss.stage_outcome("complete", {"measured_null": 1, "new_status": 1}, 2)
    with pytest.raises(ss.SerialSchedulingRefused, match="iteration count"):
        ss.stage_outcome("complete", {"measured_null": 1}, 0)


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


def _quiet_observation(*, mode="exclusive", relation="ancestor", pid=77, ticks=5, inode=4242):
    return {"path": "/locks/gpu_quiet.lock", "device": 1, "inode": inode,
            "path_unchanged": True, "error": None,
            "owners": [{"pid": pid, "start_ticks": ticks, "mode": mode,
                        "relation": relation, "kernel_row": "original"}]}


def _gpu_only_bundle(tmp_path, quiet, *, fraction=0, name="gpu-only"):
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    _state, selected, _index = ss.select_target(
        source, state, ("gpu",), now=1, stage_number=0)
    target = {"selected_id": "gpu", "original": "fixture"}
    component = _component("mi210_0", 3, 7, fraction=fraction, claim="gpu-flock")
    component["affinity_cores"] = []
    if quiet is not None:
        component["gpu_quiet"] = quiet
    body = {"schema": ss.INTERVAL_SCHEMA, "selection": selected.to_dict(),
            "selection_digest": selected.digest, "target": target, "components": [component]}
    directory = tmp_path / name
    directory.mkdir()
    store = ArtifactStore(directory / "held-claim-artifacts")
    try:
        artifact = store.write("direct-held-intervals", body).to_dict()
    finally:
        store.close()
    reference = {"schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
                 "evidence": artifact}
    return directory, reference, selected, target


def test_reopen_accepts_gpu_only_interval_with_device_claim_and_gpu_quiet(tmp_path):
    directory, reference, selected, target = _gpu_only_bundle(
        tmp_path, {"open": _quiet_observation(), "close": _quiet_observation()})
    (receipt,) = ss.reopen_held_receipts(directory, reference, selection=selected,
                                         target=target)
    assert (receipt.started_at, receipt.ended_at) == (3, 7)
    assert receipt.gpu_device_ids == ("mi210_0",)
    # The device flock plus the gpu-quiet flock are the physical resource receipt.
    assert receipt.physical_claim_ids == ("boot:flock:1:904", "boot:flock:1:4242")
    # Host share: the selected proposal's own estimate (gpu-quiet EXCLUSIVE excludes
    # every CPU-lane measurement), never a smaller invented fraction.
    assert receipt.physical_region_fraction == 0.5 == \
        selected.proposal.estimated_claims.physical_region_fraction
    assert receipt.affinity_cores == ()
    # The scheduler settles it like any other held stage.
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    state, issued, _index = ss.select_target(source, state, ("gpu",), now=1, stage_number=0)
    assert issued.digest == selected.digest
    scheduling.account_stage_components(source.config, state, selected, (receipt,),
                                        outcome="failed")


def _phase_bundle(tmp_path, phases):
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    state, selected, _ = ss.select_target(source, state, ("gpu",), now=1, stage_number=0)
    target = {"selected_id": "gpu"}
    device = _component("mi210_0", 1, 10, fraction=0, claim="gpu-flock")
    body = {"schema": ss.INTERVAL_SCHEMA_V2, "selection": selected.to_dict(),
            "selection_digest": selected.digest, "target": target,
            "components": [device], "phases": phases}
    store = ArtifactStore(tmp_path / "held-claim-artifacts")
    try:
        artifact = store.write("direct-held-intervals", body).to_dict()
    finally:
        store.close()
    reference = {"schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
                 "evidence": artifact}
    return source, state, selected, target, reference


def _phase(kind, start, end):
    cpu = kind == "build"
    row = _component("cpu" if cpu else "gpu_quiet", start, end,
                     fraction=.25 if cpu else 0., claim="cpu-flock" if cpu else "quiet-flock")
    row["gpu_device_ids"] = []
    return {"kind": kind, "component": row}


def test_gpu_phase_partition_charges_only_original_cpu_holds_and_preserves_device_gaps(tmp_path):
    source, state, selected, target, ref = _phase_bundle(
        tmp_path, [_phase("build", 2, 4), _phase("gpu_compute", 6, 8)])
    receipts = ss.reopen_held_receipts(tmp_path, ref, selection=selected, target=target)
    assert [(row.started_at, row.ended_at, row.physical_region_fraction) for row in receipts] \
        == [(1, 2, 0), (2, 4, .25), (4, 6, 0), (6, 8, 0), (8, 10, 0)]
    assert all(row.schema == scheduling.RECEIPT_SCHEMA_V2 for row in receipts)
    view = scheduling.charge_receipts(receipts)
    assert view.physical_region_seconds == .5
    assert view.gpu_device_seconds == {"mi210_0": 9.}
    assert view.held_seconds == 9.
    assert len(receipts[2].physical_claim_ids) == 1  # hosted gap owns only the device
    assert len(receipts[3].physical_claim_ids) == 2  # compute also owns quiet
    settled = scheduling.account_stage_components(source.config, state, selected,
                                                  receipts, outcome="failed")
    assert scheduling.SchedulerState.from_dict(settled.to_dict()) == settled


@pytest.mark.parametrize("defect", ["overlap", "outside", "lost", "quiet_cpu", "gpu_on_cpu"])
def test_gpu_phase_partition_refuses_unowned_or_overlapping_phases(tmp_path, defect):
    phases = [_phase("build", 2, 4), _phase("gpu_compute", 6, 8)]
    if defect == "overlap":
        phases[1] = _phase("gpu_compute", 3, 5)
    elif defect == "outside":
        phases[1] = _phase("gpu_compute", 9, 11)
    elif defect == "lost":
        phases[1]["component"]["close"]["status"] = "lost"
    elif defect == "quiet_cpu":
        phases[1]["component"]["physical_region_fraction"] = .25
    else:
        phases[0]["component"]["gpu_device_ids"] = ["mi210_0"]
    _, _, selected, target, ref = _phase_bundle(tmp_path, phases)
    with pytest.raises(ss.SerialSchedulingRefused):
        ss.reopen_held_receipts(tmp_path, ref, selection=selected, target=target)


def test_v1_device_only_receipt_remains_refused_while_v2_is_explicit(tmp_path):
    _, _, selected, target, ref = _phase_bundle(tmp_path, [])
    (receipt,) = ss.reopen_held_receipts(tmp_path, ref, selection=selected, target=target)
    assert receipt.physical_region_fraction == 0.
    with pytest.raises(scheduling.SchedulingRefused, match="host CPU"):
        replace(receipt, schema=scheduling.RECEIPT_SCHEMA)


@pytest.mark.parametrize("quiet, match", [
    (None, "gpu-quiet EXCLUSIVE"),
    ({"open": _quiet_observation(mode="shared"), "close": _quiet_observation()},
     "gpu-quiet EXCLUSIVE"),
    ({"open": _quiet_observation(relation="other"), "close": _quiet_observation()},
     "gpu-quiet EXCLUSIVE"),
    ({"open": _quiet_observation(), "close": _quiet_observation(pid=78)}, "changed"),
    ({"open": _quiet_observation(), "close": _quiet_observation(ticks=6)}, "changed"),
    ({"open": _quiet_observation(), "close": dict(_quiet_observation(), error="lost")},
     "gpu-quiet EXCLUSIVE"),
])
def test_reopen_refuses_gpu_only_interval_without_continuous_gpu_quiet(tmp_path, quiet, match):
    directory, reference, selected, target = _gpu_only_bundle(tmp_path, quiet)
    with pytest.raises(ss.SerialSchedulingRefused, match=match):
        ss.reopen_held_receipts(directory, reference, selection=selected, target=target)


def test_reopen_refuses_gpu_only_interval_claiming_a_region_or_on_cpu_target(tmp_path):
    quiet = {"open": _quiet_observation(), "close": _quiet_observation()}
    directory, reference, selected, target = _gpu_only_bundle(tmp_path, quiet, fraction=0.5)
    with pytest.raises(ss.SerialSchedulingRefused, match="never held"):
        ss.reopen_held_receipts(directory, reference, selection=selected, target=target)


def test_cpu_targets_still_require_their_cpu_context_and_carry_no_gpu_quiet(tmp_path):
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    _state, selected, _index = ss.select_target(
        source, state, ("cpu",), now=1, stage_number=0)
    target = {"selected_id": "cpu"}
    quiet = {"open": _quiet_observation(), "close": _quiet_observation()}
    cases = {
        "lone-gpu": ([dict(_component("mi210_0", 1, 2, fraction=0, claim="gpu-flock"),
                           gpu_quiet=quiet)], "GPU identity differs"),
        "cpu-quiet": ([dict(_component("cpu", 1, 2, fraction=0.5, claim="cpu-flock"),
                            gpu_quiet=quiet)], "CPU held component carries no gpu-quiet"),
    }
    for name, (components, match) in cases.items():
        body = {"schema": ss.INTERVAL_SCHEMA, "selection": selected.to_dict(),
                "selection_digest": selected.digest, "target": target,
                "components": components}
        (tmp_path / name).mkdir()
        store = ArtifactStore(tmp_path / name / "held-claim-artifacts")
        try:
            artifact = store.write("direct-held-intervals", body).to_dict()
        finally:
            store.close()
        with pytest.raises(ss.SerialSchedulingRefused, match=match):
            ss.reopen_held_receipts(tmp_path / name, {
                "schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
                "evidence": artifact}, selection=selected, target=target)


def test_cpu_gpu_bundle_refuses_a_stray_gpu_quiet_receipt(tmp_path):
    source = manifest()
    state = scheduling.initial_state(source.config, source.scheduler_id)
    _state, selected, _index = ss.select_target(
        source, state, ("gpu",), now=1, stage_number=0)
    target = {"selected_id": "gpu"}
    quiet = {"open": _quiet_observation(), "close": _quiet_observation()}
    body = {"schema": ss.INTERVAL_SCHEMA, "selection": selected.to_dict(),
            "selection_digest": selected.digest, "target": target,
            "components": [_component("cpu", 1, 9, fraction=0.5, claim="cpu-flock"),
                           dict(_component("mi210_0", 3, 7, fraction=0, claim="gpu-flock"),
                                gpu_quiet=quiet)]}
    store = ArtifactStore(tmp_path / "held-claim-artifacts")
    try:
        artifact = store.write("direct-held-intervals", body).to_dict()
    finally:
        store.close()
    with pytest.raises(ss.SerialSchedulingRefused, match="only to a GPU-only"):
        ss.reopen_held_receipts(tmp_path, {
            "schema": ss.REFERENCE_SCHEMA, "selection_digest": selected.digest,
            "evidence": artifact}, selection=selected, target=target)
