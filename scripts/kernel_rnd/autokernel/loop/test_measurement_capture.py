"""Prospective native arm capture; all execution/provider behavior is fake."""
from __future__ import annotations

from contextlib import contextmanager
import json
import multiprocessing
import os
import threading

import pytest

from .. import schemas
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import planned_serving as ps
from .test_planned_serving import _measure, _plan, _prompts, _recipes
from .test_planned_serving import _v2_plan


def _process_exclusive(root, ready, acquired):
    store = mc.ArtifactStore(root)
    try:
        ready.set()
        with store.exclusive():
            acquired.set()
    finally:
        store.close()


class CaptureProvider:
    def __init__(self):
        self.lineage = "lineage-1"

    def admit(self, plan_digest, unit, stages):
        return ps.StageFence(
            f"f-{unit.unit_id}", unit.unit_id, unit.process_id, self.lineage,
            "grant-1", "container-1", "monotonic", 100.0,
            "supervisor-1", 2, 3, "worker-1", 4)

    @contextmanager
    def guard(self, fence):
        yield ps.ExecutionGuard(fence.fence_id, fence.unit_id,
                                fence.process_generation_id, fence.lineage_id,
                                fence.grant_id, fence.container_id, True, True)

    def complete(self, fence, observation, *, native_observation=None):
        if native_observation is not None:
            assert set(native_observation) == {"locator", "sha256", "verified"}
        return ps.StageCompletion(
            fence.fence_id, True,
            {name: ep.Witness("pass", f"{name}:{fence.unit_id}")
             for name in ("identity", "teardown", "contention", "placement")},
            "clean", None)


def _context(at, ct, anchor, candidate):
    return mc.CaptureContext.from_dict({
        "campaign_id": "camp-1", "config_digest": "c" * 64,
        "supervisor_id": "supervisor-1", "supervisor_incarnation": 2,
        "config_generation": 3, "worker_id": "worker-1", "worker_incarnation": 4,
        "grant_id": "grant-1", "container_id": "container-1",
        "lineage_id": "lineage-1", "instrument_id": "planned-serving/v1",
        "protocol_id": "P-test", "protocol_status": "ratified",
        "source_identities": {
            "anchor": {"source_revision": "a" * 40,
                       "model_sha256": anchor.model.sha256,
                       "build_sha256": anchor.executable.sha256,
                       "recipe_hash": at.recipe_hash},
            "candidate": {"source_revision": "b" * 40,
                          "model_sha256": candidate.model.sha256,
                          "build_sha256": candidate.executable.sha256,
                          "recipe_hash": ct.recipe_hash}}})


def _run(tmp_path, *, measure=None):
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    plan = ep.ExperimentPlan.from_dict({**plan.to_dict(), "unit": "process"})
    entries = {}

    def transaction(measurement_id, payload):
        plain = json.loads(json.dumps(payload))
        old = entries.get(measurement_id)
        if old is not None and old != plain:
            raise mc.CaptureError("conflicting measurement retry")
        entries[measurement_id] = plain
        return {"record_id": measurement_id, "seq": len(entries)}

    sink = mc.NativeMeasurementSink(
        context=_context(at, ct, anchor, candidate),
        store=mc.ArtifactStore(tmp_path / "artifacts"), capture_transaction=transaction)
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=CaptureProvider(),
        artifact_sink=sink, lineage_id="lineage-1", clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=measure or _measure([]))
    return result, entries, sink


def test_planned_consumer_seals_per_arm_raw_inputs_and_serialized_events(tmp_path):
    result, entries, _ = _run(tmp_path)
    assert result.execution_complete and len(result.capture_receipts) == 2
    assert len(entries) == 2
    for payload in entries.values():
        carrier = payload["carrier"]
        assert carrier["status"] == "measurement"
        assert carrier["measurement"]["value"] == 20.0
        assert carrier["measurement"]["unit"] == "t/s"
        assert carrier["measurement"]["independent_unit"] == "process"
        assert carrier["measurement"]["independent_n"] == 1
        assert carrier["measurement"]["reps_basis"] == \
            "scored independent process launches"
        assert carrier["interval"]["start"].startswith("2026-09-09")
        assert carrier["source_identity"] != payload["artifact"]["sha256"]


def test_changed_same_measurement_is_refused_by_serialized_transaction(tmp_path):
    result, entries, sink = _run(tmp_path)
    measurement_id = next(iter(entries))
    changed = dict(entries[measurement_id])
    changed["carrier"] = dict(changed["carrier"], claim="changed")
    with pytest.raises(mc.CaptureError, match="conflicting"):
        sink.capture_transaction(measurement_id, changed)
    assert len(result.capture_receipts) == 2


def test_deferred_sink_reuses_exact_carrier_builder_without_transaction(tmp_path):
    (tmp_path / "direct").mkdir()
    direct, entries, _ = _run(tmp_path / "direct")
    at, ct, anchor, candidate = _recipes()
    plan = ep.ExperimentPlan.from_dict({**_plan(at, ct, anchor, candidate).to_dict(),
                                       "unit": "process"})
    (tmp_path / "deferred").mkdir()
    sink = mc.DeferredNativeMeasurementSink(
        context=_context(at, ct, anchor, candidate),
        store=mc.ArtifactStore(tmp_path / "deferred" / "artifacts"))
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=_prompts(at), stage_provider=CaptureProvider(),
        artifact_sink=sink, lineage_id="lineage-1", clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=_measure([]))
    assert len(result.capture_receipts) == len(sink.captures) == 2
    assert [mc._plain(item["payload"]) for item in sink.captures] == list(entries.values())
    assert all("journal_entry" not in item for item in result.capture_receipts)
    assert direct.execution_complete and result.execution_complete


def test_v2_deferred_carrier_has_distinct_identity_and_closed_observation_membership(tmp_path):
    at, ct, anchor, candidate = _recipes()
    plan = _v2_plan(at, ct, anchor, candidate)
    context = _context(at, ct, anchor, candidate)
    store = mc.ArtifactStore(tmp_path / "v2-artifacts")
    sink = mc.DeferredNativeMeasurementSink(context=context, store=store)

    class Factory:
        def create(self, *, unit, fence, recipe):
            return unit.unit_id

        def finish_reference(self, *, unit, session):
            assert session == unit.unit_id
            body = {"schema": "epyc.autokernel.lifecycle_observation_reference.v1",
                    "observation_id": f"obs-{unit.unit_id}", "unit_id": unit.unit_id,
                    "process_generation_id": unit.process_id, "fence_id": f"f-{unit.unit_id}",
                    "active_claim_ref": "claim:1", "target_pid": 101,
                    "target_start_ticks": 100, "descendant_binding_ref": "descendant:1",
                    "worker_id": "worker-1",
                    "worker_generation": 1, "grant_id": "grant-1", "grant_generation": 1,
                    "container_id": "container-1", "instrument_identity_sha256": "c" * 64,
                    "observation_content_sha256": "e" * 64, "shutdown_status": "resolved",
                    "successor_permitted": True,
                    "artifact": {"locator": f"{unit.unit_id}.json", "sha256": "f" * 64,
                                 "verified": True}}
            return {**body, "reference_digest": schemas.content_hash(body)}

    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    result = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=_prompts(at), stage_provider=CaptureProvider(),
        artifact_sink=sink, lineage_id="lineage-1", clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=_measure([]),
        observation_session_factory=Factory())
    assert result.schema == ps.RUN_SCHEMA_V2
    assert len(sink.captures) == 2
    for capture in sink.captures:
        carrier = capture["payload"]["carrier"]
        assert carrier["schema"] == mc.CAPTURE_SCHEMA_V2
        assert carrier["producer"] == mc.PRODUCER_ID_V2
        assert carrier["loaded_instrument"] == plan.loaded_instrument
        assert len(carrier["lifecycle_observations"]) == 1
        legacy_id = schemas.content_hash({"producer": mc.PRODUCER_ID,
            "plan_digest": plan.digest, "lineage_id": "lineage-1", "arm": carrier["arm"]})
        assert capture["measurement_id"] != legacy_id
    store.close()


def test_artifact_store_reads_only_exact_pinned_digest(tmp_path):
    store = mc.ArtifactStore(tmp_path / "store")
    body = {"sealed": [1, 2, 3]}
    receipt = store.write("result", body)
    assert mc._plain(store.read(receipt.locator, receipt.sha256)) == body
    with pytest.raises(mc.CaptureError, match="digest"):
        store.read(receipt.locator, "0" * 64)
    with pytest.raises(mc.CaptureError):
        store.read("../escape", receipt.sha256)


def test_failed_or_partial_slot_is_retained_diagnostic_without_scalar(tmp_path):
    result, entries, _ = _run(tmp_path, measure=_measure([], partial=True))
    assert not result.execution_complete
    assert len(entries) == 1
    assert all(payload["carrier"]["status"] == "diagnostic" for payload in entries.values())
    assert all(payload["carrier"]["measurement"] is None for payload in entries.values())
    assert any(payload["carrier"]["raw_artifacts"] for payload in entries.values())


def test_near_but_unequal_slot_sum_is_diagnostic_not_scalar(tmp_path):
    calls = []
    measure = _measure(calls)

    def near_value(*args, **kwargs):
        measure(*args, **kwargs)
        return 20.0 + 5e-13

    _, entries, _ = _run(tmp_path, measure=near_value)
    assert entries and all(payload["carrier"]["status"] == "diagnostic"
                           for payload in entries.values())
    assert all(payload["carrier"]["diagnostic_reason"] ==
               "raw slot sum differs from admitted unit value"
               for payload in entries.values())


def test_context_is_deep_frozen_and_storage_refuses_hardlinks(tmp_path):
    at, ct, anchor, candidate = _recipes()
    context = _context(at, ct, anchor, candidate)
    with pytest.raises(TypeError):
        context.source_identities["anchor"]["source_revision"] = "f" * 40
    store = mc.ArtifactStore(tmp_path / "store")
    receipt = store.write("x", {"a": 1})
    os.link(tmp_path / "store" / receipt.locator, tmp_path / "outside-link")
    with pytest.raises(mc.CaptureError, match="private state|artifact"):
        store.write("x", {"a": 1})


def test_torn_staging_write_publishes_nothing_and_retry_succeeds(tmp_path, monkeypatch):
    store = mc.ArtifactStore(tmp_path / "store")
    real_write = os.write
    calls = 0

    def torn(descriptor, value):
        nonlocal calls
        calls += 1
        if calls == 1:
            real_write(descriptor, value[:1])
            raise OSError("simulated torn staging write")
        return real_write(descriptor, value)

    monkeypatch.setattr(os, "write", torn)
    with pytest.raises(mc.CaptureError, match="publication failed"):
        store.write("torn", {"a": 1})
    assert list((tmp_path / "store").iterdir()) == []
    monkeypatch.setattr(os, "write", real_write)
    assert store.write("torn", {"a": 1}).verified is True


def test_retry_recovers_exact_post_link_pre_unlink_interruption(tmp_path, monkeypatch):
    store = mc.ArtifactStore(tmp_path / "store")
    real_unlink = os.unlink
    interrupted = False

    def fail_once(path, *args, **kwargs):
        nonlocal interrupted
        if not interrupted and str(path).endswith(".stage"):
            interrupted = True
            raise OSError("simulated crash after final link")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_once)
    with pytest.raises(mc.CaptureError, match="publication failed"):
        store.write("linked", {"a": 1})
    entries = list((tmp_path / "store").iterdir())
    assert len(entries) == 2 and len({item.stat().st_ino for item in entries}) == 1
    monkeypatch.setattr(os, "unlink", real_unlink)
    receipt = store.write("linked", {"a": 1})
    assert receipt.verified and list((tmp_path / "store").iterdir()) == [
        tmp_path / "store" / receipt.locator]


def test_noncreating_verify_requires_exact_existing_object(tmp_path):
    store = mc.ArtifactStore(tmp_path / "store")
    with pytest.raises(mc.CaptureError, match="does not exist"):
        store.verify("x", {"a": 1})
    assert list((tmp_path / "store").iterdir()) == []
    written = store.write("x", {"a": 1})
    assert store.verify("x", {"a": 1}) == written
    with pytest.raises(mc.CaptureError, match="does not exist"):
        store.verify("x", {"a": 2})


def test_partial_unpublished_prior_stage_is_quarantined_then_retried(tmp_path):
    store = mc.ArtifactStore(tmp_path / "store")
    name, encoded, _ = store._identity("fixture", {"value": 42})
    stage = store.root / f".{name}.stage"
    stage.write_bytes(encoded[:3])
    stage.chmod(0o600)
    result = store.write("fixture", {"value": 42})
    assert store.verify("fixture", {"value": 42}) == result
    quarantines = list(store.root.glob(f".{name}.stage.quarantine-*"))
    assert len(quarantines) == 1 and quarantines[0].read_bytes() == encoded[:3]


def test_failures_do_not_leak_cleanup_or_constructor_descriptors(tmp_path, monkeypatch):
    store = mc.ArtifactStore(tmp_path / "store")
    before = len(os.listdir("/proc/self/fd"))
    real_write = os.write
    def fail_write(*_):
        raise OSError("write fault")

    monkeypatch.setattr(os, "write", fail_write)
    for index in range(4):
        with pytest.raises(mc.CaptureError):
            store.write("fixture", {"index": index})
    assert len(os.listdir("/proc/self/fd")) == before
    monkeypatch.setattr(os, "write", real_write)
    store.close()
    before = len(os.listdir("/proc/self/fd"))
    def fail_fsync(*_):
        raise OSError("fsync fault")

    monkeypatch.setattr(os, "fsync", fail_fsync)
    with pytest.raises(mc.CaptureError):
        mc.ArtifactStore(tmp_path / "other")
    assert len(os.listdir("/proc/self/fd")) == before


def test_directory_lock_prevents_stealing_an_active_stage(tmp_path):
    first = mc.ArtifactStore(tmp_path / "store")
    second = mc.ArtifactStore(tmp_path / "store")
    entered = threading.Event()
    done = threading.Event()

    def writer():
        entered.set()
        second.write("x", {"a": 1})
        done.set()

    name, encoded, _ = first._identity("x", {"a": 1})
    stage = first.root / f".{name}.stage"
    with first._exclusive():
        stage.write_bytes(encoded[:1])
        stage.chmod(0o600)
        thread = threading.Thread(target=writer)
        thread.start()
        assert entered.wait(1.0)
        assert not done.wait(0.05)
        assert stage.read_bytes() == encoded[:1]
    thread.join(timeout=1.0)
    assert done.is_set() and second.verify("x", {"a": 1}).verified
    first.close()
    second.close()


def test_same_store_serializes_threads_and_nested_context_keeps_outer_lock(tmp_path):
    store = mc.ArtifactStore(tmp_path / "store")
    entered = threading.Event()
    acquired = threading.Event()

    def writer():
        entered.set()
        with store._exclusive():
            acquired.set()

    with store._exclusive():
        with store._exclusive():
            thread = threading.Thread(target=writer)
            thread.start()
            assert entered.wait(1.0)
        assert not acquired.wait(0.05)
    thread.join(timeout=1.0)
    assert acquired.is_set()
    store.close()


def test_public_exclusive_is_reentrant_for_nested_write_and_exact_retry(tmp_path):
    store = mc.ArtifactStore(tmp_path / "store")
    with store.exclusive():
        first = store.write("nested", {"value": 1})
        second = store.write("nested", {"value": 1})
    assert first == second
    assert list(store.root.iterdir()) == [store.root / first.locator]
    store.close()


def test_public_exclusive_refuses_closed_and_replaced_roots(tmp_path):
    store = mc.ArtifactStore(tmp_path / "closed")
    store.close()
    with pytest.raises(mc.CaptureError, match="closed"):
        with store.exclusive():
            pass

    store = mc.ArtifactStore(tmp_path / "replace")
    original = tmp_path / "original-root"
    store.root.rename(original)
    store.root.mkdir(mode=0o700)
    with pytest.raises(mc.SecureRuntimeError, match="identity changed"):
        with store.exclusive():
            pass
    store.close()


def test_public_exclusive_serializes_threads_and_processes_on_same_root(tmp_path):
    root = tmp_path / "store"
    store = mc.ArtifactStore(root)
    thread_entered = threading.Event()
    thread_acquired = threading.Event()

    def thread_waiter():
        thread_entered.set()
        with store.exclusive():
            thread_acquired.set()

    context = multiprocessing.get_context("spawn")
    process_ready = context.Event()
    process_acquired = context.Event()
    with store.exclusive():
        thread = threading.Thread(target=thread_waiter)
        thread.start()
        process = context.Process(
            target=_process_exclusive, args=(root, process_ready, process_acquired))
        process.start()
        assert thread_entered.wait(1.0) and process_ready.wait(1.0)
        assert not thread_acquired.wait(0.05)
        assert not process_acquired.wait(0.05)
    thread.join(timeout=2.0)
    process.join(timeout=2.0)
    assert thread_acquired.is_set() and process_acquired.is_set()
    assert process.exitcode == 0
    store.close()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {"not-json"}])
def test_store_refuses_nonfinite_and_non_json_before_publication(tmp_path, value):
    store = mc.ArtifactStore(tmp_path / "store")
    with pytest.raises(mc.CaptureError, match="canonical JSON"):
        store.write("invalid", {"value": value})
    assert list(store.root.iterdir()) == []
    store.close()


def test_conflicting_stage_never_changes_published_final(tmp_path):
    store = mc.ArtifactStore(tmp_path / "store")
    receipt = store.write("x", {"a": 1})
    final = store.root / receipt.locator
    original = final.read_bytes()
    stage = store.root / f".{receipt.locator}.stage"
    stage.write_bytes(b"conflicting unpublished bytes")
    stage.chmod(0o600)

    with pytest.raises(mc.CaptureError, match="published final conflicts"):
        store.write("x", {"a": 1})

    assert final.read_bytes() == original
    assert stage.read_bytes() == b"conflicting unpublished bytes"


def test_direct_forged_view_and_context_identity_fail_closed(tmp_path):
    result, _, sink = _run(tmp_path)
    summary = {"plan": {}, "plan_digest": "0" * 64}
    with pytest.raises(Exception):
        sink.finalize_run(summary)
    assert result.capture_receipts


def test_continuation_retains_reuse_diagnostic_without_fresh_launch_claim(tmp_path):
    at, ct, anchor, candidate = _recipes()
    plan = _plan(at, ct, anchor, candidate)
    plan = ep.ExperimentPlan.from_dict({**plan.to_dict(), "unit": "process",
                                       "continuation_allowed": True})
    prompts = _prompts(at)
    first_sink = mc.NativeMeasurementSink(
        context=_context(at, ct, anchor, candidate),
        store=mc.ArtifactStore(tmp_path / "first"),
        capture_transaction=lambda measurement_id, payload: {"record_id": measurement_id})
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    first = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=CaptureProvider(),
        artifact_sink=first_sink, lineage_id="lineage-1", clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=_measure([]))
    context = _context(at, ct, anchor, candidate).to_dict()
    context["lineage_id"] = "lineage-2"
    second_entries = {}
    second_sink = mc.NativeMeasurementSink(
        context=mc.CaptureContext.from_dict(context),
        store=mc.ArtifactStore(tmp_path / "second"),
        capture_transaction=lambda measurement_id, payload: second_entries.setdefault(
            measurement_id, json.loads(json.dumps(payload))))
    provider = CaptureProvider()
    provider.lineage = "lineage-2"
    resumed = ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=provider,
        artifact_sink=second_sink, lineage_id="lineage-2", clock=lambda: 1.0,
        previous_raws=first.raw_units, previous_lineage_id="lineage-1",
        continuation_verifier=lambda *_: True, measure=_measure([]))
    assert resumed.execution_complete and len(second_entries) == 2
    assert all(payload["carrier"]["status"] == "diagnostic"
               and payload["carrier"]["measurement"] is None
               and payload["carrier"]["interval"] is None
               for payload in second_entries.values())


def test_unsupported_session_or_dispersion_semantics_are_diagnostic(tmp_path):
    at, ct, anchor, candidate = _recipes()
    plan, prompts = _plan(at, ct, anchor, candidate), _prompts(at)
    context = _context(at, ct, anchor, candidate)
    entries = {}
    sink = mc.NativeMeasurementSink(
        context=context, store=mc.ArtifactStore(tmp_path / "artifacts"),
        capture_transaction=lambda measurement_id, payload: entries.setdefault(
            measurement_id, json.loads(json.dumps(payload))))
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=prompts, stage_provider=CaptureProvider(),
        artifact_sink=sink, lineage_id="lineage-1", clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=_measure([]))
    assert all(item["carrier"]["status"] == "diagnostic" for item in entries.values())
    assert all("unit/estimator/estimand" in item["carrier"]["diagnostic_reason"]
               for item in entries.values())


def test_unknown_protocol_is_preserved_without_relabeling(tmp_path):
    at, ct, anchor, candidate = _recipes()
    plan = _plan(at, ct, anchor, candidate)
    plan = ep.ExperimentPlan.from_dict({**plan.to_dict(), "unit": "process",
                                       "protocol_ref": None, "protocol_status": "unknown"})
    context_body = _context(at, ct, anchor, candidate).to_dict()
    context_body.update(protocol_id=None, protocol_status="unknown")
    entries = {}
    sink = mc.NativeMeasurementSink(
        context=mc.CaptureContext.from_dict(context_body),
        store=mc.ArtifactStore(tmp_path / "artifacts"),
        capture_transaction=lambda measurement_id, payload: entries.setdefault(
            measurement_id, json.loads(json.dumps(payload))))
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=_prompts(at), stage_provider=CaptureProvider(),
        artifact_sink=sink, lineage_id="lineage-1", clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=_measure([]))
    assert entries and all(item["carrier"]["protocol_status"] == "unknown"
                           and item["carrier"]["protocol_id"] is None
                           for item in entries.values())
