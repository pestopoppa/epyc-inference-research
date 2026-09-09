"""Hermetic native producer/controller transaction tests; no real execution."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
import threading
import time

import pytest

from .. import journal as journal_module, schemas
from . import campaign_control as control
from . import experiment_plan as ep
from . import measurement_capture as mc
from . import native_capture_control as native
from . import lifecycle_observation as lo
from . import observation_binding as ob
from . import planned_serving as ps
from .test_campaign_control import _command, _resolved
from .test_measurement_capture import _context
from .test_planned_serving import _measure, _plan, _prompts, _recipes
from .test_lifecycle_observation import Clock, _budgets, _fixture, _resolver, _write_process


class _Provider:
    def __init__(self, context: mc.CaptureContext):
        self.context = context

    def admit(self, plan_digest, unit, stages):
        del plan_digest, stages
        return ps.StageFence(
            f"f-{unit.unit_id}", unit.unit_id, unit.process_id,
            self.context.lineage_id, self.context.grant_id,
            self.context.container_id, "monotonic", 100.0,
            self.context.supervisor_id, self.context.supervisor_incarnation,
            self.context.config_generation, self.context.worker_id,
            self.context.worker_incarnation)

    @contextmanager
    def guard(self, fence):
        yield ps.ExecutionGuard(fence.fence_id, fence.unit_id,
                                fence.process_generation_id, fence.lineage_id,
                                fence.grant_id, fence.container_id, True, True)

    def complete(self, fence, observation):
        del observation
        return ps.StageCompletion(
            fence.fence_id, True,
            {name: ep.Witness("pass", f"{name}:{fence.unit_id}")
             for name in ("identity", "teardown", "contention", "placement")},
            "clean", None)


def _binding(context: mc.CaptureContext) -> native.NativeCaptureBinding:
    return native.NativeCaptureBinding.from_dict({
        "campaign_id": context.campaign_id,
        "config_digest": context.config_digest,
        "config_generation": context.config_generation,
        "supervisor_id": context.supervisor_id,
        "supervisor_incarnation": context.supervisor_incarnation,
    })


def _fence(context: mc.CaptureContext, **changes) -> native.TrustedWorkerResultFence:
    row = {
        "campaign_id": context.campaign_id,
        "config_digest": context.config_digest,
        "config_generation": context.config_generation,
        "supervisor_id": context.supervisor_id,
        "supervisor_incarnation": context.supervisor_incarnation,
        "worker_id": context.worker_id,
        "worker_incarnation": context.worker_incarnation,
        "grant_id": context.grant_id,
        "container_id": context.container_id,
        "lineage_id": context.lineage_id,
        "current": True,
        "result_accepted": True,
    }
    row.update(changes)
    return native.TrustedWorkerResultFence.from_dict(row)


def _validator(context, store, provider=True):
    callback = (lambda measurement_id, actual: _fence(actual)) if provider else None
    return native.NativeCaptureValidator(
        binding=_binding(context), store=store, fence_provider=callback)


def _reseal_carrier(payload, store):
    body = dict(payload["carrier"])
    body.pop("carrier_digest")
    payload["carrier"]["carrier_digest"] = schemas.content_hash(body)
    payload["artifact"] = store.write(
        f"carrier:{payload['measurement_id']}", payload["carrier"]).to_dict()


def _reseal_raw(item, store):
    body = dict(item["document"])
    body.pop("artifact_digest")
    digest = schemas.content_hash(body)
    item["document"]["artifact_digest"] = digest
    item["stored"] = store.write(f"raw:{digest}", item["document"]).to_dict()
    return digest


def _produce(controller, tmp_path, transaction=None, *, partial=False,
             unit="process"):
    at, ct, anchor, candidate = _recipes()
    plan_row = _plan(at, ct, anchor, candidate).to_dict()
    plan_row["campaign_id"] = controller.resolved.campaign_id
    plan_row["unit"] = unit
    plan = ep.ExperimentPlan.from_dict(plan_row)
    prompts = _prompts(at)
    base = _context(at, ct, anchor, candidate).to_dict()
    base.update({
        "campaign_id": controller.resolved.campaign_id,
        "config_digest": controller.config_digest,
        "config_generation": controller.config_generation,
        "supervisor_incarnation": controller.supervisor_incarnation,
    })
    context = mc.CaptureContext.from_dict(base)
    store = mc.ArtifactStore(tmp_path / "artifacts")
    captured = {}
    callback_scope = None
    if transaction is None:
        controller.register_native_capture(_validator(context, store))
        callback_scope = controller.native_capture_callback()
        transaction = callback_scope.__enter__()

    def recording(measurement_id, payload):
        captured[measurement_id] = json.loads(json.dumps(payload))
        return transaction(measurement_id, payload)

    sink = mc.NativeMeasurementSink(
        context=context, store=store, capture_transaction=recording)
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    try:
        result = ps.run_planned_comparison(
            plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
            candidate_recipe=candidate, prompts=prompts,
            stage_provider=_Provider(context), artifact_sink=sink,
            lineage_id=context.lineage_id, clock=lambda: 1.0,
            wall_clock=lambda: next(ticks),
            measure=_measure([], partial=partial))
    finally:
        if callback_scope is not None:
            callback_scope.__exit__(None, None, None)
    return result, captured, context, store


def _prepared(controller, tmp_path, *, partial=False):
    accepted = {}

    def collect(measurement_id, payload):
        accepted[measurement_id] = json.loads(json.dumps(payload))
        return {"record_id": measurement_id}

    result, payloads, context, store = _produce(
        controller, tmp_path, collect, partial=partial)
    assert accepted == payloads
    controller.register_native_capture(_validator(context, store))
    return result, payloads, context, store


def test_producer_store_controller_restart_and_exact_retry(tmp_path):
    service = tmp_path / "service"
    with control.CampaignController(_resolved(), service) as controller:
        result, payloads, _context_value, store = _produce(controller, tmp_path)
        assert result.execution_complete
        assert len(payloads) == 2
        for produced in payloads.values():
            assert produced["carrier"]["status"] == "measurement"
            assert produced["carrier"]["measurement"] == {
                "metric": "aggregate_tok_s", "value": 20.0, "unit": "t/s",
                "independent_unit": "process", "direction": "higher",
                "independent_n": 1,
                "reps_basis": "scored independent process launches",
                "per_launch_values": [20.0],
            }
        measurement_id, payload = next(iter(payloads.items()))
        with controller.native_capture_callback() as callback:
            assert callback(measurement_id, payload).event_id == \
                controller.native_capture(measurement_id).event_id
            controller.apply_command(_command(
                controller.resolved, "drain-after-capture", "drain", 0))
            assert callback(measurement_id, payload).record_id == measurement_id
        store.close()
    with control.CampaignController(_resolved(), service) as replayed:
        assert {measurement_id: replayed.native_capture(measurement_id).payload
                for measurement_id in payloads} == payloads
        measurement_id, payload = next(iter(payloads.items()))
        with replayed.native_capture_callback() as callback:
            assert callback(measurement_id, payload).record_id == measurement_id
            with pytest.raises(native.NativeCaptureRefused, match="drained"):
                callback("f" * 64, payload)


def test_default_and_stale_worker_fences_refuse_new_capture(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        with controller.native_capture_callback() as callback:
            with pytest.raises(native.NativeCaptureRefused, match="not connected"):
                callback(measurement_id, payload)
        stale = native.NativeCaptureValidator(
            binding=_binding(context), store=store,
            fence_provider=lambda mid, actual: _fence(actual, current=False))
        controller.register_native_capture(stale)
        with controller.native_capture_callback() as callback:
            with pytest.raises(native.NativeCaptureRefused, match="stale"):
                callback(measurement_id, payload)
        store.close()


@pytest.mark.parametrize("change", [
    {"worker_incarnation": 5}, {"grant_id": "other"},
    {"container_id": "other"}, {"lineage_id": "other"},
    {"result_accepted": False},
])
def test_worker_result_fence_exact_identity_required(tmp_path, change):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        validator = native.NativeCaptureValidator(
            binding=_binding(context), store=store,
            fence_provider=lambda mid, actual: _fence(actual, **change))
        measurement_id, payload = next(iter(payloads.items()))
        with pytest.raises(native.NativeCaptureRefused):
            validator.validate(measurement_id, payload)
        store.close()


def test_fence_types_and_callback_failures_refuse(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        for provider in (lambda mid, actual: {"current": True},
                         lambda mid, actual: (_ for _ in ()).throw(RuntimeError("x"))):
            validator = native.NativeCaptureValidator(
                binding=_binding(context), store=store, fence_provider=provider)
            with pytest.raises(native.NativeCaptureRefused):
                validator.validate(measurement_id, payload)
        with pytest.raises(native.NativeCaptureRefused):
            native.TrustedWorkerResultFence.from_dict(
                {**_fence(context).to_dict(), "worker_incarnation": True})
        store.close()


def test_v2_large_artifacts_prevalidate_then_exact_current_owner_token(tmp_path):
    at, ct, anchor, candidate = _recipes()
    store = mc.ArtifactStore(tmp_path / "v2-artifacts")
    selected_measure = _measure([])
    instrument = ob.seal_loaded_instrument(
        store=store, measurement_callable=selected_measure,
        fence_clock=time.monotonic, serving_timer=time.time)
    plan_row = _plan(at, ct, anchor, candidate).to_dict()
    plan_row |= {"schema": ep.PLAN_SCHEMA_V2, "unit": "process",
                 "loaded_instrument": instrument.to_dict(),
                 "anchor_identity": ps.arm_identity(
                     at, anchor, loaded_instrument=instrument.to_dict()),
                 "candidate_identity": ps.arm_identity(
                     ct, candidate, loaded_instrument=instrument.to_dict())}
    plan = ep.ExperimentPlan.from_dict(plan_row)
    base_context = _context(at, ct, anchor, candidate).to_dict()
    base_context["instrument_id"] = instrument.identity_sha256
    context = mc.CaptureContext.from_dict(base_context)

    class Factory:
        def __init__(self):
            self.records = {}

        def create(self, *, unit, fence, recipe):
            root = tmp_path / f"observer-{unit.unit_id}"
            root.mkdir()
            probe, proc, _, _, _, container = _fixture(root)
            _write_process(proc, 101, start=100, ticks=1)
            binding = ob.ObservationUnitBinding.from_dict(ob.ObservationUnitBinding(
                f"obs-{unit.unit_id}", unit.unit_id, unit.process_id, fence.fence_id,
                "monotonic", "boot-fixture",
                {"worker_id": context.worker_id,
                 "worker_incarnation": context.worker_incarnation,
                 "grant_id": context.grant_id, "grant_generation": 5,
                 "container_identity": container}, context.container_id,
                "journal:active-claim", {"logical_cpus": [0], "gpu_devices": []},
                {"logical_cpus": [0], "numa_nodes": [0], "thp_mode": "madvise"},
                (), (), 10.0, 11.0, _budgets(), instrument).to_dict())
            observation_context = {"schema": lo.CONTEXT_SCHEMA,
                "observation_id": binding.observation_id, "backend": "cpu",
                "instrument_identity_digest": instrument.identity_sha256,
                "recipe_identity_digest": recipe.execution_digest,
                "clock_domain": binding.clock_domain, "cadence_s": binding.cadence_s,
                "gap_limit_s": binding.gap_limit_s, "boot_id": binding.boot_id,
                "worker_binding": ob._plain(binding.worker_binding),
                "requested_effective_state": ob._plain(binding.requested_effective_state),
                "held_claim": ob._plain(binding.held_claim), "runtime_witness_keys": [],
                "required_gpu_dsos": [], "budgets": ob._plain(binding.budgets)}
            session = lo.ObservationSession(
                observation_context, probe=probe,
                owned_identity_resolver=_resolver(observation_context), monotonic=Clock())
            session.start()
            session.phase("load")
            session.attach_target(101)
            for phase in ("placement", "health", "warmup", "measurement", "teardown"):
                session.phase(phase)
            session.finish()
            self.records[unit.unit_id] = (session, binding)
            return session

        def finish_reference(self, *, unit, session):
            actual, binding = self.records[unit.unit_id]
            assert actual is session
            return ob.seal_observation(
                store=store, binding=binding, record=session.record()).to_dict()

    sink = mc.DeferredNativeMeasurementSink(context=context, store=store)
    ticks = iter(("2026-09-09T00:00:00Z", "2026-09-09T00:00:01Z",
                  "2026-09-09T00:00:02Z", "2026-09-09T00:00:03Z"))
    ps.run_planned_comparison(
        plan, anchor_template=at, candidate_template=ct, anchor_recipe=anchor,
        candidate_recipe=candidate, prompts=_prompts(at), stage_provider=_Provider(context),
        artifact_sink=sink, lineage_id=context.lineage_id, clock=lambda: 1.0,
        wall_clock=lambda: next(ticks), measure=selected_measure,
        observation_session_factory=Factory())
    measurement_id, payload = sink.captures[0]["measurement_id"], sink.captures[0]["payload"]
    validator = native.NativeCaptureValidator(binding=_binding(context), store=store)
    with pytest.raises(native.NativeCaptureRefused, match="requires outside-lock"):
        validator.validate(measurement_id, payload)
    prevalidated = validator.prevalidate(measurement_id, payload)
    violations = journal_module._validate_native_payload(
        journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED,
        prevalidated.validated.payload())
    assert violations == [], violations
    assert len(prevalidated.observation_links) == 1
    assert prevalidated.observation_links[0].observation_status == "unknown"
    assert prevalidated.observation_links[0].purpose_status == "unknown"
    token = native.CurrentOwnerToken(measurement_id, prevalidated.payload_digest, 5,
                                     _fence(context))
    assert validator.validate_prevalidated(prevalidated, token) == prevalidated.validated
    with pytest.raises(native.NativeCaptureRefused, match="another payload"):
        validator.validate_prevalidated(prevalidated,
            native.CurrentOwnerToken(measurement_id, prevalidated.payload_digest, 6,
                                     _fence(context)))
    store.close()


@pytest.mark.parametrize(("path", "value"), [
    (("arm",), []),
    (("environment_verdicts", 0, "contention", "verdict"), []),
])
def test_malformed_nested_carrier_values_return_typed_refusal(tmp_path, path, value):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        changed = copy.deepcopy(payload)
        target = changed["carrier"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        _reseal_carrier(changed, store)
        with pytest.raises(native.NativeCaptureRefused):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_diagnostic_is_retained_without_scientific_progress(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid}, partial=True)
        validator = _validator(context, store)
        measurement_id, payload = next(iter(payloads.items()))
        accepted = validator.validate(measurement_id, payload)
        assert accepted.status == "diagnostic"
        assert accepted.scientific_progress is False
        controller.register_native_capture(validator)
        with controller.native_capture_callback() as callback:
            callback(measurement_id, payload)
        assert controller.snapshot()["last_scientific_result_at"] is None
        store.close()


def test_unsupported_session_unit_is_retained_as_diagnostic(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, _context_value, store = _produce(
            controller, tmp_path, unit="session")
        assert payloads
        assert all(payload["carrier"]["status"] == "diagnostic"
                   for payload in payloads.values())
        assert all(payload["carrier"]["measurement"] is None
                   for payload in payloads.values())
        store.close()


def test_mutated_carrier_source_prompt_interval_and_artifacts_refuse(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        validator = _validator(context, store)
        measurement_id, original = next(iter(payloads.items()))
        changes = []
        for mutate in (
                lambda row: row["carrier"]["source_identity"].update(
                    {"build_sha256": "f" * 64}),
                lambda row: row["carrier"]["prompt_manifest"].update(
                    {"version": "changed"}),
                lambda row: row["carrier"]["interval"].update(
                    {"end": "2026-09-09T00:10:00Z"}),
                lambda row: row["carrier"]["raw_artifacts"][0]["stored"].update(
                    {"sha256": "f" * 64})):
            row = copy.deepcopy(original)
            mutate(row)
            body = dict(row["carrier"])
            body.pop("carrier_digest")
            row["carrier"]["carrier_digest"] = schemas.content_hash(body)
            changes.append(row)
        for row in changes:
            with pytest.raises(native.NativeCaptureRefused):
                validator.validate(measurement_id, row)
        store.close()


def test_measurement_scalar_is_rederived_from_selected_launch_values(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        changed = copy.deepcopy(payload)
        changed["carrier"]["measurement"]["value"] = 29.0
        changed["carrier"]["measurement"]["per_launch_values"] = [29.0]
        _reseal_carrier(changed, store)
        with pytest.raises(native.NativeCaptureRefused, match="derived"):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_full_comparison_identities_equal_frozen_plan_not_hash_subset(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        changed = copy.deepcopy(payload)
        changed["carrier"]["comparison_identities"]["anchor"]["backend"] = "gpu"
        _reseal_carrier(changed, store)
        with pytest.raises(native.NativeCaptureRefused, match="frozen plan"):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_environment_summary_is_rederived_from_completed_witnesses(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        changed = copy.deepcopy(payload)
        changed["carrier"]["environment_verdicts"][0]["contention"]["verdict"] = \
            "contaminated"
        _reseal_carrier(changed, store)
        with pytest.raises(native.NativeCaptureRefused, match="environment verdicts"):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_native_request_hash_is_bound_to_frozen_prompt_after_full_reseal(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        changed = copy.deepcopy(payload)
        artifacts = changed["carrier"]["raw_artifacts"]
        native_item = next(item for item in artifacts
                           if item["document"]["kind"] == "native_observation")
        attempt_item = next(item for item in artifacts
                            if item["document"]["kind"] == "completed_attempt")
        native_item["document"]["selected_observation"]["requests"][0][
            "request_sha256"] = "f" * 64
        native_item["document"]["observations"][-1]["requests"][0][
            "request_sha256"] = "f" * 64
        native_digest = _reseal_raw(native_item, store)
        attempt_item["document"]["native_observation_digest"] = native_digest
        attempt_digest = _reseal_raw(attempt_item, store)
        view = changed["carrier"]["admissible_view"]
        row = next(row for row in view["selected_rows"]
                   if row["arm"] == changed["carrier"]["arm"])
        row["artifact_digest"] = attempt_digest
        view_body = dict(view)
        view_body.pop("view_digest")
        view["view_digest"] = schemas.content_hash(view_body)
        _reseal_carrier(changed, store)
        with pytest.raises(native.NativeCaptureRefused, match="frozen prompt"):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_unknown_instrument_refuses_measurement_use(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        changed = copy.deepcopy(payload)
        changed["carrier"]["capture_context"]["instrument_id"] = "unknown/v9"
        changed["carrier"]["instrument_id"] = "unknown/v9"
        _reseal_carrier(changed, store)
        with pytest.raises(native.NativeCaptureRefused, match="unsupported native"):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_missing_artifact_verification_does_not_create_it(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        measurement_id, payload = next(iter(payloads.items()))
        missing = store.root / payload["carrier"]["raw_artifacts"][0]["stored"]["locator"]
        missing.unlink()
        with pytest.raises(native.NativeCaptureRefused, match="byte verification"):
            _validator(context, store).validate(measurement_id, payload)
        assert not missing.exists()
        store.close()


def test_swapped_raw_artifact_receipt_refuses(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        (measurement_id, payload), (_, other) = list(payloads.items())
        changed = copy.deepcopy(payload)
        changed["carrier"]["raw_artifacts"][0]["stored"] = copy.deepcopy(
            other["carrier"]["raw_artifacts"][0]["stored"])
        body = dict(changed["carrier"])
        body.pop("carrier_digest")
        changed["carrier"]["carrier_digest"] = schemas.content_hash(body)
        with pytest.raises(native.NativeCaptureRefused, match="locator/hash"):
            _validator(context, store).validate(measurement_id, changed)
        store.close()


def test_capability_is_same_thread_scoped_and_closed_lifetime_fenced(tmp_path):
    controller = control.CampaignController(_resolved(), tmp_path / "service")
    controller.__enter__()
    _, payloads, context, store = _produce(
        controller, tmp_path, lambda mid, row: {"record_id": mid})
    controller.register_native_capture(_validator(context, store))
    measurement_id, payload = next(iter(payloads.items()))
    errors = []
    with controller.native_capture_callback() as callback:
        thread = threading.Thread(
            target=lambda: errors.append(pytest.raises(control.ControlRefused,
                callback, measurement_id, payload).value))
        thread.start()
        thread.join()
    with pytest.raises(control.ControlRefused, match="not current"):
        callback(measurement_id, payload)
    controller.close()
    with pytest.raises(control.ControlRefused, match="not current"):
        callback(measurement_id, payload)
    assert errors
    store.close()


def test_append_uncertainty_and_projection_failure_require_restart(tmp_path, monkeypatch):
    service = tmp_path / "service"
    controller = control.CampaignController(_resolved(), service)
    controller.__enter__()
    _, payloads, context, store = _produce(
        controller, tmp_path, lambda mid, row: {"record_id": mid})
    controller.register_native_capture(_validator(context, store))
    measurement_id, payload = next(iter(payloads.items()))
    real_append = controller._journal.append

    def after_append(*args, **kwargs):
        real_append(*args, **kwargs)
        raise OSError("uncertain after fsync")

    monkeypatch.setattr(controller._journal, "append", after_append)
    with controller.native_capture_callback() as callback:
        with pytest.raises(OSError, match="uncertain"):
            callback(measurement_id, payload)
        with pytest.raises(control.ControlRefused, match="poisoned"):
            callback(measurement_id, payload)
    controller.close()
    with control.CampaignController(_resolved(), service) as replayed:
        assert replayed.native_capture(measurement_id).payload == payload
    store.close()


def test_failure_before_append_poisons_without_creating_native_event(tmp_path, monkeypatch):
    service = tmp_path / "service"
    controller = control.CampaignController(_resolved(), service)
    controller.__enter__()
    _, payloads, context, store = _produce(
        controller, tmp_path, lambda mid, row: {"record_id": mid})
    controller.register_native_capture(_validator(context, store))
    measurement_id, payload = next(iter(payloads.items()))

    def refuse_append(*args, **kwargs):
        del args, kwargs
        raise OSError("before append")

    monkeypatch.setattr(controller._journal, "append", refuse_append)
    with controller.native_capture_callback() as callback:
        with pytest.raises(OSError, match="before append"):
            callback(measurement_id, payload)
        with pytest.raises(control.ControlRefused, match="poisoned"):
            callback(measurement_id, payload)
    controller.close()
    with control.CampaignController(_resolved(), service) as replayed:
        assert replayed.native_capture(measurement_id) is None
    store.close()


def test_durable_append_then_index_failure_is_poisoned_and_replayed(tmp_path):
    class BrokenIndex(dict):
        def __setitem__(self, key, value):
            del key, value
            raise RuntimeError("projection update failed")

    service = tmp_path / "service"
    controller = control.CampaignController(_resolved(), service)
    controller.__enter__()
    _, payloads, context, store = _produce(
        controller, tmp_path, lambda mid, row: {"record_id": mid})
    controller.register_native_capture(_validator(context, store))
    measurement_id, payload = next(iter(payloads.items()))
    controller._native_records = BrokenIndex(controller._native_records)
    with controller.native_capture_callback() as callback:
        with pytest.raises(RuntimeError, match="projection update"):
            callback(measurement_id, payload)
        with pytest.raises(control.ControlRefused, match="poisoned"):
            callback(measurement_id, payload)
    controller.close()
    with control.CampaignController(_resolved(), service) as replayed:
        assert replayed.native_capture(measurement_id).payload == payload
    store.close()


def test_concurrent_duplicate_serializes_and_conflicting_retry_refuses(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        controller.register_native_capture(_validator(context, store))
        measurement_id, payload = next(iter(payloads.items()))
        barrier = threading.Barrier(2)
        results = []

        def submit():
            with controller.native_capture_callback() as callback:
                barrier.wait()
                results.append(callback(measurement_id, payload).event_id)

        threads = [threading.Thread(target=submit) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 2 and len(set(results)) == 1
        changed = copy.deepcopy(payload)
        changed["carrier"]["claim"] = "conflicting same-id claim"
        body = dict(changed["carrier"])
        body.pop("carrier_digest")
        changed["carrier"]["carrier_digest"] = schemas.content_hash(body)
        with controller.native_capture_callback() as callback:
            with pytest.raises(native.NativeCaptureRefused, match="different"):
                callback(measurement_id, changed)
        store.close()


def test_old_incarnation_validator_refuses(tmp_path):
    service = tmp_path / "service"
    old_validator = None
    store = None
    with control.CampaignController(_resolved(), service) as first:
        _, payloads, context, store = _produce(
            first, tmp_path, lambda mid, row: {"record_id": mid})
        old_validator = _validator(context, store)
    with control.CampaignController(_resolved(), service) as current:
        with pytest.raises(control.ControlRefused, match="not current"):
            current.register_native_capture(old_validator)
        # No fresh validator: historical absence is not reconstructed authority.
        with current.native_capture_callback() as callback:
            with pytest.raises(native.NativeCaptureRefused, match="not connected"):
                callback(next(iter(payloads)), next(iter(payloads.values())))
    assert store is not None
    store.close()


def test_drained_controller_refuses_new_capture(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, context, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        controller.register_native_capture(_validator(context, store))
        controller.apply_command(_command(
            controller.resolved, "drain-native", "drain", 0))
        measurement_id, payload = next(iter(payloads.items()))
        with controller.native_capture_callback() as callback:
            with pytest.raises(native.NativeCaptureRefused, match="drained"):
                callback(measurement_id, payload)
        store.close()


def test_journal_native_validator_is_closed_and_total_for_malformed_json():
    kind = journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED
    for payload in ({}, {"schema": journal_module.PLANNED_SERVING_ARM_CAPTURE_SCHEMA,
                         "measurement_id": [], "carrier": [], "artifact": []},
                    {"schema": journal_module.PLANNED_SERVING_ARM_CAPTURE_SCHEMA,
                     "measurement_id": "f" * 64, "carrier": {"carrier_digest": []},
                     "artifact": {"locator": {}, "sha256": [], "verified": 1}}):
        assert journal_module._validate_native_payload(kind, payload)
    future = {"schema": "epyc.autokernel.unified_arm_capture.v2",
              "measurement_id": "f" * 64, "carrier": {}, "artifact": {}}
    assert journal_module._validate_native_payload(kind, future)


def test_journal_native_validator_returns_violations_for_nested_arbitrary_json(tmp_path):
    with control.CampaignController(_resolved(), tmp_path / "service") as controller:
        _, payloads, _context_value, store = _produce(
            controller, tmp_path, lambda mid, row: {"record_id": mid})
        original = next(iter(payloads.values()))
        for field, value in (("capture_context", []), ("plan", []),
                             ("prompt_manifest", []),
                             ("comparison_identities", []),
                             ("raw_artifacts", {}),
                             ("environment_verdicts", {})):
            changed = copy.deepcopy(original)
            changed["carrier"][field] = value
            body = dict(changed["carrier"])
            body.pop("carrier_digest")
            changed["carrier"]["carrier_digest"] = schemas.content_hash(body)
            assert journal_module._validate_native_payload(
                journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED, changed)
        store.close()


def test_replay_rejects_duplicate_native_measurement_id(tmp_path):
    service = tmp_path / "service"
    with control.CampaignController(_resolved(), service) as controller:
        _, payloads, _context_value, store = _produce(controller, tmp_path)
    measurement_id, payload = next(iter(payloads.items()))
    duplicate_writer = journal_module.Journal(
        str(service / "journal"), campaign_id=_resolved().campaign_id)
    duplicate_writer.append(
        journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED, payload,
        record_id=measurement_id)
    with pytest.raises(journal_module.JournalCorruption, match="repeats"):
        control.CampaignController(_resolved(), service).__enter__()
    store.close()
