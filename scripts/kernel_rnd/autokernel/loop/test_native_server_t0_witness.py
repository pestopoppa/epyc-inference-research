"""Synthetic files/lifecycle plus real owning reducers; no model or hardware run."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import pytest

from ..evaluator import api, correctness
from ..execution import t0_provider as t0
from ..execution.test_t0_provider import (
    FakeClaim, candidate_build, evaluation_request, execution_plan, t0_policy)
from . import experiment_plan as ep
from . import lifecycle_observation as lo
from . import native_parent_evidence as npe
from . import native_parent_receipt_replay as replay
from . import native_scientific_witness as scientific
from . import native_server_response as raw
from . import native_server_t0_witness as server
from . import observation_binding as ob
from . import planned_serving as ps
from . import worker_lifecycle as wl
from . import test_native_parent_evidence as native_fixture
from .test_planned_serving import _prompts
from .test_resolved_recipe import _resolve, _policy


def _owning_run(_self, argv, *, env, cwd, timeout_s):
    binary = str(Path(cwd) / "build" / "bin" / "llama-server")
    library = str(Path(binary).with_name("libggml-base.so"))
    bindir = str(Path(binary).parent)
    if "test-backend-ops" in " ".join(argv):
        receipt = "AK_REF_V1 metric=test_backend_ops_error/v1 observed=0 tolerance=1e-07 comparisons=2 oracle=ggml_cpu_reference/v1"
        output = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
            f"  MUL_MAT(type_a=f32,type_b=f32,m=16,n=1,k=256): {receipt} OK\n"
            f"  MUL_MAT_ID(type_a=f32,type_b=f32,n_mats=4,n_used=2): {receipt} OK\n"
            "  2/2 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")
    else:
        output = (f"binary : {binary}\nexpect : libraries under {bindir}\n"
            f"  OK   libggml-base.so -> {library}\n"
            f"PASS: all linked ggml libraries resolve inside {bindir}\n")
    return t0.CompletedProcess(tuple(argv), tuple(sorted(env.items())), cwd, 0,
                              output, "", 0.01, False, False)


@pytest.fixture
def pair(tmp_path, monkeypatch, request):
    monkeypatch.setattr(t0.SubprocessRunner, "run", _owning_run)
    issuer = scientific.NativeT0WitnessAdapter(max_units=4)
    adapter = server.NativeServerT0WitnessAdapter(max_units=4, owning_issuer=issuer)
    registry = scientific.ParentScientificWitnessAdapters(correctness=adapter)
    marker_mode = getattr(request, "param", None)
    if marker_mode in ("required_marker", "missing_required_marker", "missing_legacy_marker"):
        loaded = ob.loaded_planned_serving_identity
        def declared_markers(**kwargs):
            row = ob._plain(loaded(**kwargs))
            if marker_mode == "missing_legacy_marker":
                row["used_constants"].pop("parent_window_markers", None)
            else:
                row["used_constants"]["parent_window_markers"] = [
                    "health", "warmup", "measurement", "measurement_end", "teardown"]
            row.pop("sha256")
            row["sha256"] = wl._digest(row)
            return lo.validate_instrument_identity(row)
        monkeypatch.setattr(ob, "loaded_planned_serving_identity", declared_markers)
    seal = ob.seal_loaded_instrument
    monkeypatch.setattr(ob, "seal_loaded_instrument",
        lambda **kwargs: seal(**(kwargs | {"scientific_adapters": registry})))
    base = native_fixture._case(tmp_path)
    store = base["store"]
    build = tmp_path / "candidate" / "build"
    bindir = build / "bin"
    bindir.mkdir(parents=True)
    binary, library = bindir / "llama-server", bindir / "libggml-base.so"
    binary.write_bytes(b"synthetic non-executable server")
    library.write_bytes(b"synthetic DSO")
    modelroot = tmp_path / "six-shard-model"
    modelroot.mkdir()
    members = []
    for index in range(6):
        path = modelroot / f"model-{index}.gguf"
        path.write_bytes(f"tiny shard {index}".encode())
        members.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    material = {"model_path": str(modelroot), "files": members}
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"schema": "epyc.autokernel.model_identity.v1", **material}))
    model_identity = scientific.tensor_capture.CaptureModelIdentity(str(modelroot), manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        hashlib.sha256(scientific.tensor_capture._canonical(material).encode()).hexdigest())
    entry = modelroot / members[0]["path"]
    adapter.prepare_model_identity(store=store, identity=model_identity, entry_path=entry,
                                  preparation_claim=FakeClaim(claim_id="scheduled-fixture-preparation"))
    template = replace(base["context"].template, model=str(entry), top_k=1, temperature=0.0)
    def artifact(role, path):
        return {"schema": "epyc.autokernel.artifact_digest.v1", "role": role,
                "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    from . import resolved_recipe as rr
    artifacts = {"model": artifact("model", entry), "drafter": None,
                 "executable": artifact("executable", binary), "dsos": [artifact("dso", library)]}
    for item in (artifacts["model"], artifacts["executable"], *artifacts["dsos"]):
        item["schema"] = rr.ARTIFACT_SCHEMA
    recipe = _resolve(template, backend="cpu", build=build, artifacts=artifacts,
                      policy=_policy("THP_CONTROL", witness="recipe_readback"))
    prompts = _prompts(template)
    identity = ps.arm_identity(template, recipe, loaded_instrument=base["context"].binding.instrument.to_dict())
    plan = ep.ExperimentPlan.from_dict(base["context"].plan.to_dict() |
        {"campaign_id": "ak-server-fixture", "anchor_identity": identity, "candidate_identity": identity})
    if getattr(request, "param", None) == "candidate_first":
        document = plan.to_dict()
        for unit in document["expected_units"]:
            unit["order_index"] = len(document["expected_units"]) - 1 - unit["order_index"]
        plan = ep.ExperimentPlan.from_dict(document)
    owning_plan = execution_plan(candidate=candidate_build(worktree=str(build.parent),
        build_dir=str(build), binary=str(binary), library_path=str(bindir),
        test_backend_ops=str(bindir / "test-backend-ops")), base_env=recipe.launch_env)
    linkage = t0.ExecutedT0EvidenceProvider.linkage_digest(t0.parse_linkage_report(
        f"binary : {binary}\nexpect : libraries under {bindir}\n"
        f"  OK   libggml-base.so -> {library}\n"
        f"PASS: all linked ggml libraries resolve inside {bindir}\n"))
    anchor_identity = api.AnchorIdentity(owning_plan.candidate.source_commit,
                                       artifacts["executable"]["sha256"], linkage)
    cases = []
    for unit in sorted(plan.expected_units, key=lambda unit: unit.order_index):
        binding = replace(base["context"].binding, unit_id=unit.unit_id,
            process_generation_id=unit.process_id, observation_id=f"server-observation:{unit.unit_id}",
            fence_id=f"fence:{unit.unit_id}")
        binding = ob.ObservationUnitBinding.from_dict(binding.to_dict())
        fence = replace(base["context"].fence, unit_id=unit.unit_id,
                        process_generation_id=unit.process_id, fence_id=binding.fence_id)
        event = ob._plain(base["context"].descendant_event)
        event.update(plan_digest=plan.digest, campaign_id=plan.campaign_id)
        event["data"].update(unit_id=unit.unit_id, process_generation_id=unit.process_id,
                            fence_id=fence.fence_id, binding_digest=binding.to_dict()["binding_digest"])
        session_context = {key: ob._plain(base["observation"][key]) for key in (
            "backend", "clock_domain", "cadence_s", "gap_limit_s", "boot_id", "worker_binding",
            "requested_effective_state", "budgets")}
        session_context.update(schema=lo.CONTEXT_SCHEMA, observation_id=binding.observation_id,
            instrument_identity_digest=binding.instrument.identity_sha256,
            recipe_identity_digest=recipe.execution_digest, held_claim=ob._plain(binding.held_claim),
            runtime_witness_keys=list(binding.runtime_witness_keys), required_gpu_dsos=[])
        def resolver(pid):
            event["occurred_at"] = ps._utc_now()
            return {**event["data"]["process"], "worker_binding": ob._plain(binding.worker_binding),
                    "binding_ref": wl._digest(event)}
        session = lo.ObservationSession(session_context, probe=base["producer"]._probe,
                                       owned_identity_resolver=resolver)
        session.start()
        try:
            session.phase("load")
            session.attach_target(101)
            context = npe.ParentUnitContext(plan, unit.unit_id, recipe, prompts, fence, binding,
                                           event, base["claim"], "server-parent", template=template)
            session.phase("placement")
            session.phase("health")
            requests = prompts.requests(unit.expected_prompt_ids, template)
            capture = raw.ServerResponseCapture(store=store, plan=plan, unit=unit, fence=fence,
                recipe=recipe, prompts=prompts, frozen_requests=requests)
            started = time.monotonic()
            rows = []
            for phase in raw.PHASES:
                session.phase(phase)
                if phase == "measurement" and marker_mode == "response_after_marker":
                    session.checkpoint("measurement_end")
                for index, (name, request) in enumerate(requests):
                    begin = time.monotonic()
                    response = json.dumps({"content": f"answer:{name}", "stop": True,
                        "tokens": [], "timings": {"predicted_n": template.n_predict,
                                                "predicted_per_second": 10.0}}).encode()
                    rows.append(raw.RawServerResponse(phase, index, name, request,
                                response, begin, time.monotonic(), None))
            if marker_mode not in ("missing_required_marker", "missing_legacy_marker", "response_after_marker"):
                session.checkpoint("measurement_end")
            retained = capture.seal(rows, process_pid=101, request_started_monotonic_s=started,
                                   request_ended_monotonic_s=time.monotonic())
            session.phase("teardown")
        finally:
            record = session.finish()
        lifecycle = ob.seal_observation(store=store, binding=binding, record=record)
        selected = {"schema": "epyc.autokernel.serving_observation.v1", "process_pid": 101,
            "requests": [{"phase": "measurement", "slot_index": index, "prompt_id": name,
                "request_sha256": hashlib.sha256(body).hexdigest(), "predicted_n": template.n_predict,
                "predicted_per_second": 10.0, "terminal": True, "error": None}
                for index, (name, body) in enumerate(requests)],
            "residency": {"status": "not_applicable"}, "teardown": "terminated", "failure": None,
            "server_responses": retained}
        native = ob._plain(base["native"])
        native.update(plan_digest=plan.digest, unit_id=unit.unit_id, arm=unit.arm,
            process_generation_id=unit.process_id, fence_id=fence.fence_id,
            prompt_manifest_digest=prompts.digest, comparison_identities={"anchor": identity, "candidate": identity},
            observed_started_at=record["started_at"], observed_ended_at=record["ended_at"],
            observations=[selected], selected_observation=selected, lifecycle_observation=lifecycle.to_dict())
        case = {"store": store, "context": context, "native": native, "observation": record}
        native_fixture._seal(case)
        link = ob.validate_reopened_observation(lifecycle, store=store, expected={
            "unit_id": unit.unit_id, "process_generation_id": unit.process_id,
            "fence_id": fence.fence_id, "active_claim_ref": binding.active_claim_ref,
            "container_id": binding.container_id, "capture_context": ob._plain(binding.worker_binding)},
            instrument=binding.instrument)
        request = replace(evaluation_request(anchor=anchor_identity,
            source_sha256=owning_plan.candidate.source_sha256,
            binary_sha256=recipe.executable.sha256, linkage_sha256=linkage),
            campaign_id=plan.campaign_id)
        original = issuer.collect_issued(context=context, store=store, plan=owning_plan,
            request=request, policy=t0_policy(), claim=FakeClaim(claim_id=binding.active_claim_ref))
        requests_by_slot = tuple(replace(request, event_id=f"{request.event_id}:{unit.unit_id}:{index}")
                                 for index in range(len(unit.expected_prompt_ids)))
        adapter.register_owning_inputs(context=context, store=store, original_context=context,
            original_artifact=original, request_by_slot=requests_by_slot,
            anchor_unit_id=None if unit.arm == "anchor" else plan.expected_units[0].unit_id)
        case.update(link=link, original=original)
        cases.append(case)
    try:
        yield {"adapter": adapter, "issuer": issuer, "registry": registry, "cases": cases,
               "store": store, "modelroot": modelroot}
    finally:
        store.close()


def _evaluate(pair, index):
    case = pair["cases"][index]
    return pair["adapter"].evaluate(case["context"], case["native"], case["link"], (),
                                    store=pair["store"])


def _issue_parents(pair, *, count=None):
    registry = replay.IssuedNativeEvidenceRegistry(artifact_root=pair["store"].root,
                                                   max_units=4)
    results = []
    for case in pair["cases"][:count]:
        producer = npe.NativeUnitEvidenceProducer(store=pair["store"], context=case["context"],
                                                  scientific_adapters=pair["registry"])
        result = producer.evaluate(case["request"])
        registry.record(producer=producer, request=case["request"], result=result)
        results.append(result)
    return registry, results


def test_original_server_anchor_content_coherence_full_slot_reports(pair):
    _evaluate(pair, 0)
    reference = _evaluate(pair, 1)
    body = pair["store"].read(reference.artifact.locator, reference.artifact.sha256)
    assert body["status"] == "unknown"
    assert body["replay_coverage"] == server.REPLAY_COVERAGE
    slots = body["report"]["slots"]
    assert len(slots) == len(pair["cases"][1]["context"].unit.expected_prompt_ids) == 2
    for index, slot in enumerate(slots):
        assert slot["slot_index"] == index
        assert len(slot["report"]["gates"]) == 17
        gates = {item["gate_id"]: item for item in slot["report"]["gates"]}
        assert gates[correctness.GID_COHERENCE]["outcome"] == "PASS"
        assert gates[correctness.GID_DETERMINISM]["outcome"] != "PASS"
        assert gates[correctness.GID_NO_FALLBACK]["outcome"] != "PASS"
        assert slot["static_bundle"] is None
        assert slot["generation"]["fields"]["seed"] is None
    assert body["original_anchor_native"] is not None
    assert body["scope_defects"] == ()


@pytest.mark.parametrize("pair", [None, "candidate_first"], indirect=True)
def test_post_arm_pair_uses_original_parent_issuance_without_rewriting_units(pair):
    registry, results = _issue_parents(pair)
    store, adapter = pair["store"], pair["adapter"]
    plan = pair["cases"][0]["context"].plan
    originals = [store.read(item.receipt.locator, item.receipt.sha256) for item in results]
    candidate_index = next(index for index, case in enumerate(pair["cases"])
                           if case["context"].unit.arm == "candidate")
    old_candidate = originals[candidate_index]
    old_slot = old_candidate["findings"]["correctness"]["facts"]["report"]["slots"][0]
    old_gate = next(row for row in old_slot["report"]["gates"]
                    if row["gate_id"] == correctness.GID_COHERENCE)
    assert (old_gate["outcome"] == "PASS") == (candidate_index != 0)
    reference = adapter.finalize_pair(plan=plan, registry=registry, store=store)
    body = adapter.reopen_pair(reference, plan=plan, registry=registry, store=store)
    assert len(body["ordered_units"]) == len(plan.expected_units) == 2
    candidate = next(item for item in body["ordered_units"] if item["arm"] == "candidate")
    assert candidate["final_evidence"]["status"] == "unknown"
    for slot in candidate["final_evidence"]["report"]["slots"]:
        assert len(slot["report"]["gates"]) == 17
        coherence = next(gate for gate in slot["report"]["gates"]
                         if gate["gate_id"] == correctness.GID_COHERENCE)
        assert coherence["outcome"] == "PASS"
        assert slot["evidence"]["fields"]["coherence"]["fields"]["token_agreement_ratio"] is None
        assert slot["report"]["unevaluated"]
    assert adapter.finalize_pair(plan=plan, registry=registry, store=store) == reference
    assert [store.read(item.receipt.locator, item.receipt.sha256) for item in results] == originals
    assert adapter.reopen_pair(reference, plan=plan, registry=registry, store=store) == body


@pytest.mark.parametrize("count", [0, 1])
def test_pair_refuses_empty_or_prefix_registry_without_waiting(pair, count):
    registry, _ = _issue_parents(pair, count=count)
    with pytest.raises(replay.ParentReceiptRefused, match="complete original plan membership"):
        pair["adapter"].finalize_pair(plan=pair["cases"][0]["context"].plan,
                                     registry=registry, store=pair["store"])


def test_lost_pair_issuance_is_not_restored_from_matching_artifact(pair):
    registry, _ = _issue_parents(pair)
    adapter, store, plan = pair["adapter"], pair["store"], pair["cases"][0]["context"].plan
    reference = adapter.finalize_pair(plan=plan, registry=registry, store=store)
    adapter._pairs.clear()
    with pytest.raises(scientific.ScientificWitnessRefused, match="original pair issuance is lost"):
        adapter.reopen_pair(reference, plan=plan, registry=registry, store=store)


def test_new_pair_never_recovers_lost_original_owning_issuance(pair):
    registry, _ = _issue_parents(pair)
    pair["issuer"]._issued.clear()
    with pytest.raises(scientific.ScientificWitnessRefused):
        pair["adapter"].finalize_pair(plan=pair["cases"][0]["context"].plan,
                                     registry=registry, store=pair["store"])


def test_registry_source_v2_is_closed_complete_and_reconstructs_no_issuance(pair):
    from . import native_producer_source as source
    selected = pair["registry"].source_identity()
    assert selected["schema"] == scientific.ADAPTERS_SCHEMA_V2
    assert source.producer_source_closure_complete(source.loaded_producer_source_closure(
        scientific_adapters=pair["registry"]))
    fresh = scientific.installed_scientific_adapters(selected)
    assert not fresh.correctness._inputs and not fresh.correctness.owning_issuer._issued
    case = pair["cases"][0]
    with pytest.raises(scientific.ScientificWitnessRefused, match="inputs unavailable"):
        fresh.correctness.evaluate(case["context"], case["native"], case["link"], (), store=pair["store"])


@pytest.mark.parametrize("pair", ["required_marker", "missing_legacy_marker"], indirect=True)
def test_declared_window_marker_and_legacy_absence_preserve_honest_evaluation(pair):
    reference = _evaluate(pair, 0)
    assert pair["store"].read(reference.artifact.locator, reference.artifact.sha256)["status"] == "unknown"


@pytest.mark.parametrize("pair,reason", [
    ("missing_required_marker", "required original measurement_end is missing"),
    ("response_after_marker", "server response exceeds original measurement_end")], indirect=["pair"])
def test_required_marker_absence_and_response_crossing_refuse(pair, reason):
    with pytest.raises(scientific.ScientificWitnessRefused, match=reason):
        _evaluate(pair, 0)
