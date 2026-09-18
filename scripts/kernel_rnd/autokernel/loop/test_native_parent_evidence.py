"""Factual producer acceptance: fake proc/sys, no grants, kernels, or inference.

This test also runs standalone beside the new module while all existing package
modules remain read-only in the native owner's checkout.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import time

import pytest

from autokernel import loop as _loop_package

_acceptance_loop = str(Path(__file__).parent)
if _acceptance_loop not in _loop_package.__path__:
    _loop_package.__path__.insert(0, _acceptance_loop)

from autokernel.loop import native_parent_evidence as npe
from autokernel.loop import experiment_plan as ep
from autokernel.loop import lifecycle_observation as lo
from autokernel.loop import measurement_capture as mc
from autokernel.loop import observation_binding as ob
from autokernel.loop import planned_serving as ps
from autokernel.loop import serving
from autokernel.loop import worker_lifecycle as wl
from autokernel.loop.test_experiment_plan import plan_dict
from autokernel.loop.test_lifecycle_observation import _fixture, _write_process, _budgets
from autokernel.loop.test_planned_serving import _prompts
from autokernel.loop.test_resolved_recipe import _resolve, _policy


def _case(tmp_path, *, capture=True, thp="0", required=(), status_field="THP_enabled",
          capture_label="health"):
    probe, proc, _pressure, _vram, _kfd, container = _fixture(tmp_path)
    _write_process(proc, 101, start=100, ticks=1)
    status_path = proc / "101" / "status"
    if thp is not None:
        status_path.write_text(status_path.read_text() + f"{status_field}:\t{thp}\n")
    store = mc.ArtifactStore(tmp_path / "artifacts")
    instrument = ob.seal_loaded_instrument(store=store, measurement_callable=serving._measure_once,
                                          fence_clock=time.monotonic, serving_timer=time.time)
    template = serving.Recipe(name="fixture-cpu", model="/fixture-model", np=2,
        device="none", ngl=0, cpu_list="0", env={"THP_CONTROL": "off"},
        env_readback=({"env": "THP_CONTROL", "field": status_field, "expect": "0"},))
    recipe = _resolve(template, backend="cpu", policy=_policy("THP_CONTROL", witness="recipe_readback"))
    prompts = _prompts(template)
    plan_body = plan_dict(n=1, instrument="serving", unit="process")
    identity = ps.arm_identity(template, recipe, loaded_instrument=instrument.to_dict())
    plan_body.update(schema=ep.PLAN_SCHEMA_V2, loaded_instrument=instrument.to_dict(),
        metric="aggregate_tok_s", anchor_identity=identity, candidate_identity=identity,
        required_witnesses=["identity", "placement", "runtime_readback", *required])
    plan = ep.ExperimentPlan.from_dict(plan_body)
    unit = plan.expected_units[0]
    binding = ob.ObservationUnitBinding.from_dict(ob.ObservationUnitBinding(
        "fixture-obs", unit.unit_id, unit.process_id, "fence-1", "fixture-monotonic", "boot-fixture",
        {"worker_id": "worker-1", "worker_incarnation": 2, "grant_id": "grant-1",
         "grant_generation": 3, "container_identity": container}, "container-1", "fixture-claim",
        {"logical_cpus": [0], "gpu_devices": []},
        {"logical_cpus": [0], "numa_nodes": [0], "thp_mode": "madvise"},
        (status_field,), (), 10.0, 11.0, _budgets(), instrument).to_dict())
    fence = ps.StageFence("fence-1", unit.unit_id, unit.process_id, "lineage-1", "grant-1",
        "container-1", binding.clock_domain, time.monotonic() + 60,
        "supervisor-1", 1, 1, "worker-1", 2)
    event = {"schema": wl.EVENT_SCHEMA, "event": "OWNED_DESCENDANT_CAPTURED",
        "campaign_id": plan.campaign_id, "config_digest": "a" * 64, "config_generation": 1,
        "supervisor_id": "supervisor-1", "supervisor_incarnation": 1,
        "worker_id": "worker-1", "worker_generation": 2, "request_id": "request-1",
        "plan_digest": plan.digest, "lineage_id": "lineage-1", "stage_id": "stage-1",
        "grant_id": "grant-1", "grant_generation": 3, "container_id": "container-1",
        "control_revision": 1, "occurred_at": ps._utc_now(), "data": {
            "role": "planned-serving-server", "unit_id": unit.unit_id,
            "process_generation_id": unit.process_id, "fence_id": "fence-1",
            "process": {"pid": 101, "start_ticks": 100, "boot_id": "boot-fixture"},
            "container_identity": container, "binding_digest": binding.to_dict()["binding_digest"]}}
    claim = {"schema": wl.ACTIVE_OBSERVATION_CLAIM_SCHEMA, "grant_id": "grant-1",
        "grant_generation": 3, "container_id": "container-1",
        "held_claim": ob._plain(binding.held_claim), "active_claim_ref": binding.active_claim_ref}
    claim["claim_digest"] = wl._digest(claim)
    session_context = {"schema": lo.CONTEXT_SCHEMA, "observation_id": binding.observation_id,
        "backend": "cpu", "instrument_identity_digest": instrument.identity_sha256,
        "recipe_identity_digest": recipe.execution_digest, "clock_domain": binding.clock_domain,
        "cadence_s": binding.cadence_s, "gap_limit_s": binding.gap_limit_s,
        "boot_id": binding.boot_id, "worker_binding": ob._plain(binding.worker_binding),
        "requested_effective_state": ob._plain(binding.requested_effective_state),
        "held_claim": ob._plain(binding.held_claim), "runtime_witness_keys": [status_field],
        "required_gpu_dsos": [], "budgets": ob._plain(binding.budgets)}

    def resolver(pid):
        assert pid == 101
        event["occurred_at"] = ps._utc_now()
        return {**event["data"]["process"], "worker_binding": ob._plain(binding.worker_binding),
                "binding_ref": wl._digest(event)}

    session = lo.ObservationSession(session_context, probe=probe, owned_identity_resolver=resolver)
    session.start()
    try:
        session.phase("load")
        session.attach_target(101)
        context = npe.ParentUnitContext(plan, unit.unit_id, recipe, prompts, fence, binding,
                                        event, claim, "parent-nonce", template=template)
        producer = npe.NativeUnitEvidenceProducer(store=store, context=context, runtime_probe=probe)
        for phase in ("placement", "health", "warmup", "measurement"):
            session.phase(phase)
            if phase == "health" and capture:
                producer.capture_runtime_readback(phase=capture_label)
        session.checkpoint("measurement_end")
        session.phase("teardown")
    finally:
        record = session.finish()
    reference = ob.seal_observation(store=store, binding=binding, record=record)
    requests = [{"phase": "measurement", "slot_index": index,
                 "prompt_id": item.prompt_id, "request_sha256": item.request_digest,
                 "predicted_n": template.n_predict, "predicted_per_second": 10.0,
                 "terminal": True, "error": None} for index, item in enumerate(prompts.prompts)]
    selected = {"schema": "epyc.autokernel.serving_observation.v1", "process_pid": 101,
        "requests": requests, "residency": {"status": "not_applicable"},
        "teardown": "terminated", "failure": None}
    native = {"schema": ps.ARTIFACT_SCHEMA_V2, "kind": "native_observation",
        "plan_digest": plan.digest, "unit_id": unit.unit_id, "arm": unit.arm,
        "process_generation_id": unit.process_id, "lineage_id": fence.lineage_id,
        "fence_id": fence.fence_id, "grant_id": fence.grant_id, "container_id": fence.container_id,
        "worker_identity": {key: getattr(fence, key) for key in ("supervisor_id",
            "supervisor_incarnation", "config_generation", "worker_id", "worker_incarnation")},
        "observed_started_at": record["started_at"], "observed_ended_at": record["ended_at"],
        "prompt_manifest_digest": prompts.digest,
        "comparison_identities": {"anchor": identity, "candidate": identity},
        "observations": [selected], "selected_observation": selected, "value": 20.0,
        "error": None, "lifecycle_observation": reference.to_dict()}
    case = {"store": store, "context": context, "producer": producer, "observation": record,
            "native": native, "proc": proc, "event": event, "claim": claim}
    _seal(case)
    return case


def _seal(case, *, reseal_observation=False):
    if reseal_observation:
        record = case["observation"]
        record["intervals"], _ = lo._intervals(record["samples"],
            set(record["held_claim"]["physical_cpus"]), record["gap_limit_s"])
        record["content_sha256"] = wl._digest({key: value for key, value in record.items()
                                              if key != "content_sha256"})
        reference = ob.seal_observation(store=case["store"], binding=case["context"].binding, record=record)
        case["native"]["lifecycle_observation"] = reference.to_dict()
    native = case["native"]
    native["artifact_digest"] = wl._digest({key: value for key, value in native.items()
                                           if key != "artifact_digest"})
    ref = case["store"].write(f"raw:{native['artifact_digest']}", native)
    case["request"] = npe.completion_request(case["context"], ref)


@pytest.fixture
def case(tmp_path):
    value = _case(tmp_path)
    try:
        yield value
    finally:
        value["store"].close()


def _result(case):
    result = case["producer"].evaluate(case["request"])
    receipt = ob._plain(case["store"].read(result.receipt.locator, result.receipt.sha256))
    return result, receipt


def test_honest_factual_path_retains_sources_and_no_scientific_pass(case):
    result, receipt = _result(case)
    assert result.completion.terminal is True
    assert result.completion.recorded_screen == "flagged_but_retained"
    for name in ("identity", "request_completeness", "placement", "runtime_readback"):
        witness = result.completion.stage_witnesses[name]
        assert witness.status == "pass", receipt["findings"][name]
        assert witness.ref == f"parent-unit-evidence:{result.receipt_digest}#{name}"
    for name in ("correctness", "contention", "purpose", "residency"):
        assert result.completion.stage_witnesses[name] == ep.Witness("unknown", None)
    assert receipt["parent_descendant_event"] == case["event"]
    assert receipt["recipe_source_pins"]["execution_digest"] == case["context"].recipe.execution_digest
    assert receipt["producer_source_pin"]["implementation_status"] == "pinned"
    assert receipt["phase_coverage"]["complete"] is True
    raw_ref = receipt["runtime_readbacks"][0]
    raw = case["store"].read(raw_ref["locator"], raw_ref["sha256"])
    assert "THP_enabled:\t0" in raw["status_text"]
    assert raw["source"]["proc_root"] == str(case["proc"])


def test_duplicate_request_returns_exact_one_receipt(case):
    first, _ = _result(case)
    assert case["producer"].evaluate(copy.deepcopy(case["request"])) is first
    case["native"]["value"] = 30.0
    _seal(case)
    with pytest.raises(npe.NativeEvidenceRefused, match="retry"):
        _result(case)
    with pytest.raises(npe.NativeEvidenceRefused, match="terminal"):
        case["producer"].capture_runtime_readback(phase="measurement")


@pytest.mark.parametrize("field,value", [("nonce", "another"), ("sequence", True),
    ("sequence", 2), ("fence_id", "other"), ("schema", "invented")])
def test_closed_completion_joins_parent_fence(case, field, value):
    case["request"][field] = value
    case["request"]["request_digest"] = wl._digest({key: item for key, item in case["request"].items()
                                                   if key != "request_digest"})
    with pytest.raises(npe.NativeEvidenceRefused):
        _result(case)


def test_child_verified_bit_never_substitutes_for_reopening(case):
    case["request"]["native_observation"]["verified"] = False
    case["request"]["request_digest"] = wl._digest({key: value for key, value in case["request"].items()
                                                   if key != "request_digest"})
    assert _result(case)[0].completion.stage_witnesses["identity"].status == "pass"


def test_wrong_sealed_bytes_refused_even_when_verified(case):
    case["request"]["native_observation"]["sha256"] = "0" * 64
    case["request"]["request_digest"] = wl._digest({key: value for key, value in case["request"].items()
                                                   if key != "request_digest"})
    with pytest.raises(mc.CaptureError, match="digest"):
        _result(case)


@pytest.mark.parametrize("field,value", [("unit_id", "wrong"), ("plan_digest", "b" * 64),
    ("process_generation_id", "wrong"), ("grant_id", "wrong"), ("container_id", "wrong")])
def test_one_native_identity_mutation_refused(case, field, value):
    case["native"][field] = value
    _seal(case)
    with pytest.raises(npe.NativeEvidenceRefused, match="native"):
        _result(case)


def test_wrong_parent_descendant_reference_refused(case):
    case["observation"]["target_binding"]["binding_ref"] = "child-invented-reference"
    _seal(case, reseal_observation=True)
    with pytest.raises(npe.NativeEvidenceRefused, match="descendant event reference"):
        _result(case)


def test_wrong_sample_pid_start_refused(case):
    sample = next(row for row in case["observation"]["samples"] if row["target"] is not None)
    sample["target"]["start_ticks"] += 1
    _seal(case, reseal_observation=True)
    with pytest.raises(npe.NativeEvidenceRefused, match="sample target identity"):
        _result(case)


def test_wrong_recipe_observation_refused(case):
    case["observation"]["recipe_identity_digest"] = "b" * 64
    _seal(case, reseal_observation=True)
    with pytest.raises(npe.NativeEvidenceRefused, match="recipe_identity_digest"):
        _result(case)


def test_affinity_mismatch_is_factual_failure_despite_child_match_label(case):
    sample = next(row for row in case["observation"]["samples"] if row["phase"] == "measurement")
    assert sample["effective_readback"]["logical_cpus_match"] is True
    sample["target"]["cpus_allowed"] = [1]
    sample["target"]["physical_affinity_footprint"] = [1]
    _seal(case, reseal_observation=True)
    result, _ = _result(case)
    assert result.completion.stage_witnesses["placement"].status == "fail"
    assert result.completion.stage_witnesses["contention"].status == "unknown"


def test_missing_measurement_phase_is_unknown_even_if_record_claims_complete(case):
    case["observation"]["samples"] = [sample for sample in case["observation"]["samples"]
                                       if sample["phase"] != "measurement"]
    assert case["observation"]["completeness"] == "complete"
    _seal(case, reseal_observation=True)
    result, receipt = _result(case)
    assert result.completion.stage_witnesses["placement"].status == "unknown"
    assert receipt["phase_coverage"]["observed_samples_by_phase"]["measurement"] == 0


@pytest.mark.parametrize("phase", ["load", "placement", "health", "warmup", "measurement"])
def test_missing_live_phase_target_is_unknown_not_placement_pass(case, phase):
    for sample in case["observation"]["samples"]:
        if sample["phase"] == phase:
            sample["target"] = None
            sample["effective_readback"] = None
    _seal(case, reseal_observation=True)
    result, receipt = _result(case)
    assert result.completion.stage_witnesses["placement"].status == "unknown"
    assert receipt["findings"]["placement"]["facts"]["target_samples_by_required_phase"][phase] == 0


def test_target_reads_outside_declared_phase_do_not_supply_placement_coverage(case):
    start = next(row["monotonic_s"] for row in case["observation"]["phase_boundaries"]
                 if row["phase"] == "measurement")
    for sample in case["observation"]["samples"]:
        if sample["phase"] == "warmup":
            sample["read_started_monotonic_s"] = start
            sample["read_ended_monotonic_s"] = start + 0.000001
    _seal(case, reseal_observation=True)
    result, receipt = _result(case)
    assert result.completion.stage_witnesses["placement"].status == "unknown"
    assert receipt["findings"]["placement"]["facts"]["target_samples_by_required_phase"]["warmup"] == 0


@pytest.mark.parametrize("field,value", [("prompt_id", "absent"), ("request_sha256", "f" * 64),
                                       ("terminal", False), ("predicted_n", 0)])
def test_request_fact_mutations_do_not_become_complete(case, field, value):
    case["native"]["selected_observation"]["requests"][0][field] = value
    _seal(case)
    result, _ = _result(case)
    assert result.completion.terminal is False
    assert result.completion.stage_witnesses["request_completeness"].status == "fail"


@pytest.mark.parametrize("delta", [-1, 1])
def test_positive_wrong_token_count_is_not_exact_frozen_completion(case, delta):
    row = case["native"]["selected_observation"]["requests"][0]
    row["predicted_n"] += delta
    assert row["predicted_n"] > 0
    _seal(case)
    result, receipt = _result(case)
    assert result.completion.terminal is False
    assert result.completion.stage_witnesses["request_completeness"].status == "fail"
    expected = case["context"].prompts.prompts[0].n_predict
    assert receipt["findings"]["request_completeness"]["facts"]["expected_predicted_n"][0] == expected


@pytest.mark.parametrize("thp,expected", [("1", "fail"), (None, "unknown")])
def test_runtime_readback_does_not_pass_mismatch_or_missing_field(tmp_path, thp, expected):
    case = _case(tmp_path, thp=thp)
    try:
        assert _result(case)[0].completion.stage_witnesses["runtime_readback"].status == expected
    finally:
        case["store"].close()


def test_no_parent_readback_does_not_trust_child_fired_label(tmp_path):
    case = _case(tmp_path, capture=False)
    try:
        for sample in case["observation"]["samples"]:
            for runtime in sample["runtime_witnesses"]:
                runtime.update(status="fired_under_target", evidence_ref="child:claims-pass", reason=None)
        _seal(case, reseal_observation=True)
        assert _result(case)[0].completion.stage_witnesses["runtime_readback"].status == "unknown"
    finally:
        case["store"].close()


def test_runtime_sample_after_unit_cannot_retrofit_evidence(tmp_path):
    case = _case(tmp_path, capture=False)
    try:
        case["producer"].capture_runtime_readback(phase="measurement")
        result, receipt = _result(case)
        assert result.completion.stage_witnesses["runtime_readback"].status == "unknown"
        assert receipt["findings"]["runtime_readback"]["facts"]["samples"][0]["within_lifecycle"] is False
    finally:
        case["store"].close()


def test_runtime_sample_must_join_declared_phase_not_only_lifecycle(tmp_path):
    case = _case(tmp_path, capture_label="measurement")
    try:
        result, receipt = _result(case)
        assert result.completion.stage_witnesses["runtime_readback"].status == "unknown"
        facts = receipt["findings"]["runtime_readback"]["facts"]["samples"][0]
        assert facts["within_lifecycle"] is True
        assert facts["within_declared_phase"] is False
    finally:
        case["store"].close()


def test_readback_from_moved_container_is_unknown(tmp_path):
    case = _case(tmp_path, capture=False)
    try:
        (case["proc"] / "101" / "cgroup").write_text("0::/other-container\n")
        captured = case["producer"].capture_runtime_readback(phase="measurement")
        raw = case["store"].read(captured.locator, captured.sha256)
        assert raw["status_text"] is None and raw["error"] is not None
        assert "container" in raw["error"] or "cgroup" in raw["error"]
        assert _result(case)[0].completion.stage_witnesses["runtime_readback"].status == "unknown"
    finally:
        case["store"].close()


def test_unsupported_witnesses_and_status_fields_stay_unknown(tmp_path):
    case = _case(tmp_path, required=("native-capture-v1", "invented-gpu-correctness"),
                 status_field="InventedStatus")
    try:
        result, _ = _result(case)
        for name in ("native-capture-v1", "invented-gpu-correctness", "runtime_readback"):
            assert result.completion.stage_witnesses[name] == ep.Witness("unknown", None)
    finally:
        case["store"].close()


def test_parent_context_detaches_inputs_and_rejects_claim_mutation(case):
    original = case["context"].identity
    case["event"]["data"]["process"]["pid"] = 999
    assert case["context"].identity == original
    bad_claim = copy.deepcopy(case["claim"])
    bad_claim["held_claim"]["logical_cpus"] = [1]
    bad_claim["claim_digest"] = wl._digest({key: value for key, value in bad_claim.items()
                                           if key != "claim_digest"})
    with pytest.raises(npe.NativeEvidenceRefused, match="held resources"):
        replace(case["context"], active_claim=bad_claim)


def test_permissive_verifier_is_not_an_injectable_probe(case):
    with pytest.raises(npe.NativeEvidenceRefused, match="bounded filesystem reader"):
        npe.NativeUnitEvidenceProducer(store=case["store"], context=case["context"],
                                       runtime_probe=lambda *_: {"status": "pass"})
