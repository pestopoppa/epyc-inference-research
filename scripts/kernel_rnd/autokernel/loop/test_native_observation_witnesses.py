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
    base = {"completeness": "complete", "issues": [], "held_claim": {"physical_cpus": [0, 1]},
            "gap_limit_s": 1.0, "samples": [], "intervals": [], "residency": None}
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
        assert body["findings"]["purpose"]["reason"] == "no owning evidence adapter implemented"
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
