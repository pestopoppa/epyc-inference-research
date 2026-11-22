"""Contention/residency witnesses from original in-window observation bytes.

SYNTHETIC fixtures (the shared native parent fixture and hand-edited observation dicts):
they prove derivation and registry plumbing, not host behaviour or any hardware claim.
"""
import copy

import pytest

from . import lifecycle_observation as lo
from . import native_observation_witnesses as ow
from . import native_producer_source as source
from . import native_scientific_witness as scientific
from . import observation_binding as ob
from .test_native_parent_evidence import _case
from .test_native_scientific_witness import _link


def _obs(**over):
    base = {"completeness": "complete", "issues": [], "gap_limit_s": 1.0, "samples": [],
            "held_claim": {"physical_cpus": [0, 1], "gpu_devices": ["gpu0"]}, "intervals": [],
            "residency": None, "phase_boundaries": [], "target_binding": {"pid": 101},
            "ended_monotonic_s": 100.0}
    return {**base, **over}


def _interval(status="observed", overlap=()):
    return {"phase": "measurement", "status": status, "potential_foreign_overlap": list(overlap)}


def _overlap(status, kind=None):
    return {"pid": 7, "start_ticks": 1, "process_total_cpu_tick_delta": 5,
            "potential_physical_claim_overlap": [0], "verification": {"status": status, "kind": kind}}


def _contention(monkeypatch, intervals, **over):
    monkeypatch.setattr(lo, "_intervals", lambda *a: (copy.deepcopy(intervals), []))
    return ow.derive_contention(_obs(intervals=intervals, **over), backend="cpu")


def test_contention_pass_only_with_fully_observed_window(monkeypatch):
    assert _contention(monkeypatch, [_interval()])["status"] == "pass"
    assert _contention(monkeypatch, [])["status"] == "unknown"
    assert _contention(monkeypatch, [_interval("unknown")])["status"] == "unknown"
    assert _contention(monkeypatch, [_interval()], completeness="unknown")["status"] == "unknown"


def test_contention_verified_inference_fails_ordinary_is_diagnostic_unverified_unknown(monkeypatch):
    assert _contention(monkeypatch, [_interval(overlap=[_overlap("verified", "model_inference")])])["status"] == "fail"
    ordinary = _contention(monkeypatch, [_interval(overlap=[_overlap("verified", "ordinary")])])
    assert ordinary["status"] == "pass" and ordinary["facts"]["diagnostic_only_overlap"]
    assert _contention(monkeypatch, [_interval(overlap=[_overlap("unknown")])])["status"] == "unknown"


def test_contention_refuses_recorded_intervals_that_differ_from_replay(monkeypatch):
    monkeypatch.setattr(lo, "_intervals", lambda *a: ([_interval()], []))
    out = ow.derive_contention(_obs(intervals=[_interval(overlap=[_overlap("unknown")])]), backend="cpu")
    assert out["status"] == "unknown" and "differ" in out["reason"]


def test_contention_names_uncaptured_gpu_consumers_on_gpu(monkeypatch):
    monkeypatch.setattr(lo, "_intervals", lambda *a: ([_interval()], []))
    out = ow.derive_contention(_obs(intervals=[_interval()]), backend="gpu")
    assert any("GPU" in item for item in out["facts"]["uncaptured"])
    # Legacy v1 samples: the foreign-GPU witness is unknown, so GPU contention can no longer PASS.
    assert out["status"] == "unknown" and out["facts"]["foreign_gpu_consumers"]["status"] == "unknown"


@pytest.mark.parametrize("replayed,expected", [("observed", "pass"), ("not_observed", "fail"),
                                               ("unknown", "unknown"), ("not_applicable", "unknown")])
def test_residency_maps_owning_protocol_status(monkeypatch, replayed, expected):
    block = {"status": replayed, "measurement_samples": 2, "positive_samples": 1, "reason": "x"}
    monkeypatch.setattr(lo, "_residency", lambda backend, samples: dict(block))
    assert ow.derive_residency(_obs(residency=dict(block)), backend="gpu")["status"] == expected


def test_residency_recorded_block_must_equal_owning_replay(monkeypatch):
    block = {"status": "observed", "measurement_samples": 2, "positive_samples": 1, "reason": "x"}
    monkeypatch.setattr(lo, "_residency", lambda backend, samples: dict(block))
    assert ow.derive_residency(_obs(residency={**block, "status": "not_observed"}), backend="gpu")["status"] == "unknown"


def test_registry_v3_source_is_closed_and_reconstructs_selected_config_only():
    adapters = scientific.ParentScientificWitnessAdapters(
        correctness=scientific.NativeT0WitnessAdapter(max_units=2),
        contention=ow.ObservationWitnessAdapter("contention"),
        residency=ow.ObservationWitnessAdapter("residency"))
    assert adapters.schema == scientific.ADAPTERS_SCHEMA_V3 and adapters.purpose is None
    row = ob._plain(adapters.source_identity())
    assert ob._plain(scientific.validate_scientific_source(row)) == ob._plain(adapters.source_identity())
    assert ob._plain(scientific.installed_scientific_adapters(row).source_identity()) == ob._plain(adapters.source_identity())
    assert scientific.scientific_source_identities(row)
    closure = source.loaded_producer_source_closure(scientific_adapters=adapters)
    assert source.producer_source_closure_complete(closure)
    bad = copy.deepcopy(row)
    bad["purpose"] = adapters.contention.source_identity()
    with pytest.raises(scientific.ScientificWitnessRefused):
        scientific.validate_scientific_source(bad)
    swapped = copy.deepcopy(row)
    swapped["contention"], swapped["residency"] = row["residency"], row["contention"]
    with pytest.raises(scientific.ScientificWitnessRefused):
        scientific.validate_scientific_source(swapped)


@pytest.mark.parametrize("kwargs", [{"purpose": object()}, {"contention": object()},
                                    {"contention": ow.ObservationWitnessAdapter("residency")},
                                    {"residency": ow.ObservationWitnessAdapter("contention")}])
def test_registry_refuses_foreign_or_misplaced_slot_adapters(kwargs):
    with pytest.raises(scientific.ScientificWitnessRefused):
        scientific.ParentScientificWitnessAdapters(**kwargs)


def test_legacy_registries_unchanged():
    legacy = scientific.ParentScientificWitnessAdapters(correctness=scientific.NativeT0WitnessAdapter(max_units=1))
    assert legacy.schema == scientific.ADAPTERS_SCHEMA
    assert legacy.source_identity()["contention"] is None


def test_adapter_replays_original_native_observation_and_receipt_is_bound(tmp_path):
    case = _case(tmp_path)
    try:
        for witness in ("contention", "residency"):
            adapter = ow.ObservationWitnessAdapter(witness)
            ref = adapter.evaluate(case["context"], case["native"], _link(case), (), store=case["store"])
            assert ref.witness == witness
            body = adapter.reopen(ref, context=case["context"], native=case["native"],
                                  lifecycle_link=_link(case), runtime_readbacks=(), store=case["store"])
            assert body["status"] in {"pass", "fail", "unknown"} and body["adapter_id"] == ow.WITNESSES[witness]
            assert body["replay_coverage"] == ow.REPLAY_COVERAGE
            other = ow.ObservationWitnessAdapter("residency" if witness == "contention" else "contention")
            with pytest.raises(scientific.ScientificWitnessRefused):
                other.reopen(ref, context=case["context"], native=case["native"],
                             lifecycle_link=_link(case), runtime_readbacks=(), store=case["store"])
    finally:
        case["store"].close()


def test_producer_and_parent_replay_carry_contention_and_residency_findings(tmp_path, monkeypatch):
    from . import native_parent_evidence as npe
    from . import native_parent_receipt_replay as replay
    adapters = scientific.ParentScientificWitnessAdapters(
        contention=ow.ObservationWitnessAdapter("contention"),
        residency=ow.ObservationWitnessAdapter("residency"))
    seal = ob.seal_loaded_instrument
    monkeypatch.setattr(ob, "seal_loaded_instrument",
                        lambda **kwargs: seal(**{**kwargs, "scientific_adapters": adapters}))
    case = _case(tmp_path)
    try:
        producer = npe.NativeUnitEvidenceProducer(store=case["store"], context=case["context"],
                                                  scientific_adapters=adapters)
        result = producer.evaluate(case["request"])
        body = case["store"].read(result.receipt.locator, result.receipt.sha256)
        for name in ("contention", "residency"):
            assert body["findings"][name]["facts"]["receipt"]["witness"] == name
            assert body["findings"][name]["reason"] == ow.REPLAY_COVERAGE
        assert body["findings"]["purpose"]["status"] == "unknown"
        assert body["findings"]["purpose"]["facts"]["basis"] == "declared_plan_input_not_sealed"
        assert body["findings"]["purpose"]["facts"]["declared"]["category"] == case["context"].plan.category
        registry = replay.IssuedNativeEvidenceRegistry(artifact_root=case["store"].root, max_units=1)
        registry.record(producer=producer, request=case["request"], result=result)
        attempt = {"unit_id": case["context"].unit_id,
            "stage_witnesses": {n: r.to_dict() for n, r in result.completion.stage_witnesses.items()},
            "terminal": result.completion.terminal, "recorded_screen": result.completion.recorded_screen,
            "provider_recorded_screen": result.completion.recorded_screen}
        entry = registry._lookup({"plan": case["context"].plan.to_dict(), "capture_context": {
            "worker_id": case["context"].fence.worker_id,
            "worker_incarnation": case["context"].fence.worker_incarnation}}, case["native"])
        replay.NativeParentReceiptReplayer._replay_unit(entry, case["native"], attempt, case["store"])
    finally:
        case["store"].close()


# --- AKU-07o foreign GPU consumers (FIXTURE KFD trees under tmp_path; no host /sys read) ---

def _kfd_tree(root, rows):
    for pid, files in rows.items():
        (root / str(pid)).mkdir(parents=True)
        for name, value in files.items():
            (root / str(pid) / name).write_text(f"{value}\n")
    return lo.kfd_consumer_reader({"gpu0": "7"}, root=root)


def _gpu_record(tmp_path, rows=None, *, reader=True):
    from .test_lifecycle_observation import _complete, _session
    tmp_path.mkdir(parents=True, exist_ok=True)
    dso = tmp_path / "libfixture.so"
    dso.write_bytes(b"prepared DSO")
    session, probe, _, _, _, kfd, _ = _session(
        tmp_path, backend="gpu", artifact=lo.prepare_artifact_identity(dso, max_bytes=1024))
    (kfd / "101").mkdir()
    for name, value in (("start_ticks", "100"), ("boot_id", "boot-fixture"), ("vram_bytes.gpu0", "4096")):
        (kfd / "101" / name).write_text(value + "\n")
    if reader:
        probe.kfd_consumer_reader = (_kfd_tree(tmp_path / "kfdproc", rows) if rows is not None
                                     else lo.kfd_consumer_reader({"gpu0": "7"}, root=tmp_path / "absent"))
    return lo.validate_observation(_complete(session))


def test_kfd_reader_reads_held_device_and_shares_residency_source(tmp_path):
    from . import residency
    assert lo.KFD_PROC_ROOT == residency.KFD_PROC
    read = _kfd_tree(tmp_path / "k", {101: {"vram_7": 4096}, 202: {"vram_9": 5}, "x": {}})
    budget = {"max_processes": 8, "max_read_bytes": 64}
    assert lo._kfd_block(read(("gpu0",), budget))["processes"] == [{"pid": 101, "vram_bytes": {"gpu0": 4096}},
                                                    {"pid": 202, "vram_bytes": {}}]
    with pytest.raises(lo.ObservationError, match="mapping"):
        read(("gpu1",), budget)


def test_only_owner_pid_is_none_and_samples_are_v2(tmp_path):
    record = _gpu_record(tmp_path, {101: {"vram_7": 4096}, 303: {"vram_9": 1}})
    measured = [row for row in record["samples"] if row["phase"] == "measurement"]
    assert measured and all(row["schema"] == lo.SAMPLE_SCHEMA_V2 for row in measured)
    out = ow.derive_foreign_gpu_consumers(record, backend="gpu")
    assert out["status"] == "none" and out["facts"]["foreign_kfd_without_held_vram"]
    assert out["facts"]["owner_pid"] == 101 and out["facts"]["late_samples_excluded"] == 0


def test_foreign_consumer_present_refuses_contention_pass(tmp_path):
    record = _gpu_record(tmp_path, {101: {"vram_7": 4096}, 202: {"vram_7": 8192}})
    assert ow.derive_foreign_gpu_consumers(record, backend="gpu")["status"] == "present"
    out = ow.derive_contention(record, backend="gpu")
    assert out["status"] == "fail" and out["facts"]["foreign_gpu_consumers"]["facts"]["foreign_consumers"][0]["pid"] == 202


def test_unreadable_or_uncaptured_kfd_source_is_unknown(tmp_path):
    record = _gpu_record(tmp_path / "a")
    census = next(row for row in record["samples"] if row["phase"] == "measurement")["gpu"]["kfd_consumers"]
    assert census["status"] == "unknown" and census["processes"] == []
    assert ow.derive_foreign_gpu_consumers(record, backend="gpu")["status"] == "unknown"
    legacy = _gpu_record(tmp_path / "b", reader=False)
    assert {row["schema"] for row in legacy["samples"]} == {lo.SAMPLE_SCHEMA}
    assert ow.derive_foreign_gpu_consumers(legacy, backend="gpu")["status"] == "unknown"
    assert ow.derive_contention(legacy, backend="gpu")["status"] != "pass"
    owner_hidden = _gpu_record(tmp_path / "c", {})  # empty census never proves absence
    assert ow.derive_foreign_gpu_consumers(owner_hidden, backend="gpu")["status"] == "unknown"


def _sample(start, end, pids, phase="measurement"):
    return {"schema": lo.SAMPLE_SCHEMA_V2, "phase": phase, "read_started_monotonic_s": start,
            "read_ended_monotonic_s": end, "gpu": {"kfd_consumers": {"status": "observed", "reason": None,
                "processes": [{"pid": pid, "vram_bytes": {"gpu0": 1}} for pid in pids]}}}


def test_late_sample_never_counts():
    bounds = [{"phase": "measurement", "monotonic_s": 10.0}, {"phase": "teardown", "monotonic_s": 20.0}]
    late = _sample(19.0, 21.0, [101, 202])
    obs = _obs(phase_boundaries=bounds, samples=[_sample(11, 12, [101]), _sample(13, 14, [101]), late])
    out = ow.derive_foreign_gpu_consumers(obs, backend="gpu")
    assert out["status"] == "none" and out["facts"]["late_samples_excluded"] == 1
    early = _obs(phase_boundaries=bounds, samples=[_sample(9.0, 11.0, [101, 202]), _sample(11, 12, [101])])
    assert ow.derive_foreign_gpu_consumers(early, backend="gpu")["status"] == "unknown"


def test_v1_readers_refuse_v2_samples_and_census_grammar_is_closed(tmp_path):
    record = _gpu_record(tmp_path, {101: {"vram_7": 4096}})
    sample = copy.deepcopy(next(row for row in record["samples"] if row["phase"] == "measurement"))
    lo._validate_sample(sample)
    with pytest.raises(lo.ObservationError):
        lo._validate_sample({**sample, "schema": lo.SAMPLE_SCHEMA})
    for bad in ({"status": "unknown", "processes": [{"pid": 1, "vram_bytes": {}}], "reason": "x"},
                {"status": "observed", "processes": [{"pid": 1, "vram_bytes": []}], "reason": None},
                {"status": "observed", "processes": [], "reason": "x"}):
        with pytest.raises(lo.ObservationError):
            lo._validate_sample({**sample, "gpu": {**sample["gpu"], "kfd_consumers": bad}})


def test_registry_v3_readers_accept_historical_and_current_helper_sets():
    row = ob._plain(ow.ObservationWitnessAdapter("contention").source_identity())
    assert [item["role"] for item in row["helpers"]] == list(ow.HELPER_ROLES)
    assert row["adapter_id"] == "epyc.autokernel.observation_contention_witness.v1"
    ow.validate_observation_source(row, "contention")
    ow.validate_observation_source({**row, "helpers": row["helpers"][:3]}, "contention")
    with pytest.raises((scientific.ScientificWitnessRefused, source.ProducerSourceRefused)):
        ow.validate_observation_source({**row, "helpers": row["helpers"][1:]}, "contention")


def test_purpose_is_declared_input_never_concrete(tmp_path):
    case = _case(tmp_path)
    try:
        out = ow.declared_purpose(case["context"].plan)
        assert out["status"] == "unknown" and out["facts"]["basis"] == "declared_plan_input_not_sealed"
    finally:
        case["store"].close()
