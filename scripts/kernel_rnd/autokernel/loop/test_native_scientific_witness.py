"""No models, grants or hardware: original owning collectors read synthetic captures.

The selected live runner's process boundary is replaced in tests, not the owning
collector, parser, reducer, gate list, or production adapter. Synthetic missing
surfaces stay unknown; a complete 17-gate report is not an all-passing report.
"""
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import time

import pytest

from ..evaluator import correctness
from ..execution import t0_provider as t0
from ..execution.test_t0_provider import (
    FakeClaim, candidate_build, evaluation_request, execution_plan, t0_policy)
from . import native_parent_evidence as npe
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_parent_receipt_replay as replay
from . import native_producer_source as source
from . import native_scientific_witness as scientific
from . import observation_binding as ob
from . import serving
from .test_native_parent_evidence import _case


@pytest.fixture
def original(tmp_path, monkeypatch):
    adapter = scientific.NativeT0WitnessAdapter(max_units=1)
    adapters = scientific.ParentScientificWitnessAdapters(correctness=adapter)
    seal = ob.seal_loaded_instrument

    def configured_seal(**kwargs):
        return seal(**{**kwargs, "scientific_adapters": adapters})

    # Build the native plan with the installed adapter config, before its issue.
    monkeypatch.setattr(ob, "seal_loaded_instrument", configured_seal)
    build = tmp_path / "candidate" / "build"
    bindir = build / "bin"
    bindir.mkdir(parents=True)
    binary, library = bindir / "llama-cli", bindir / "libggml-base.so"
    binary.write_bytes(b"synthetic non-executable fixture binary")
    library.write_bytes(b"synthetic library bytes")
    plan = execution_plan(candidate=candidate_build(worktree=str(build.parent),
        build_dir=str(build), binary=str(binary), library_path=str(bindir),
        test_backend_ops=str(bindir / "test-backend-ops")))
    calls = []

    def run(_self, argv, *, env, cwd, timeout_s):
        calls.append((tuple(argv), dict(env), cwd, timeout_s))
        if "test-backend-ops" in " ".join(argv):
            receipt = "AK_REF_V1 metric=test_backend_ops_error/v1 observed=0 tolerance=1e-07 comparisons=2 oracle=ggml_cpu_reference/v1"
            text = ("Testing 1 devices\n\nBackend 1/1: CPU\n"
                f"  MUL_MAT(type_a=f32,type_b=f32,m=16,n=1,k=256): {receipt} OK\n"
                f"  MUL_MAT_ID(type_a=f32,type_b=f32,n_mats=4,n_used=2): {receipt} OK\n"
                "  2/2 tests passed\n  Backend CPU: OK\n1/1 backends passed\nOK\n")
        else:
            text = (f"binary : {binary}\nexpect : libraries under {bindir}\n\n"
                f"  OK   libggml-base.so -> {library}\n\n"
                f"PASS: all linked ggml libraries resolve inside {bindir}\n")
        return t0.CompletedProcess(tuple(argv), tuple(sorted(env.items())), cwd, 0,
            text, "", 0.01, False, False)

    monkeypatch.setattr(t0.SubprocessRunner, "run", run)
    case = _case(tmp_path)
    claim = FakeClaim(claim_id=case["context"].binding.active_claim_ref)
    request = evaluation_request(binary_sha256=t0.sha256_file(str(binary)),
        linkage_sha256=t0.ExecutedT0EvidenceProvider.linkage_digest(t0.parse_linkage_report(
            f"binary : {binary}\nexpect : libraries under {bindir}\n"
            f"  OK   libggml-base.so -> {library}\n"
            f"PASS: all linked ggml libraries resolve inside {bindir}\n")))
    artifact = adapter.collect_issued(context=case["context"], store=case["store"],
        plan=plan, request=request, policy=t0_policy(), claim=claim)
    case.update(adapter=adapter, artifact=artifact, plan=plan, t0_request=request,
                policy=t0_policy(), calls=calls, claim=claim)
    yield case
    case["store"].close()


def _link(case):
    context = case["context"]
    reference = ob.LifecycleObservationReference.from_dict(case["native"]["lifecycle_observation"])
    return ob.validate_reopened_observation(reference, store=case["store"], expected={
        "unit_id": context.unit_id, "process_generation_id": context.unit.process_id,
        "fence_id": context.fence.fence_id, "active_claim_ref": context.binding.active_claim_ref,
        "container_id": context.binding.container_id,
        "capture_context": ob._plain(context.binding.worker_binding)}, instrument=context.binding.instrument)


def test_original_complete_owning_report_replays_after_claim_release(original):
    case = original
    count = len(case["calls"])
    assert count == 2
    case["claim"]._held = False
    issued = case["adapter"]._replay(case["context"], case["store"])
    assert len(issued.report["gates"]) == len(correctness.T0_GATE_IDS) == 17
    assert issued.report["unevaluated"]
    assert len(case["calls"]) == count
    assert len(issued.raw) == count
    assert issued.original_body["replay_coverage"] == scientific.REPLAY_COVERAGE
    assert case["adapter"]._replay(case["context"], case["store"]) is issued


def test_no_cross_tool_correctness_transfer_and_missing_surfaces_stay_unknown(original):
    case = original
    ref = case["adapter"].evaluate(case["context"], case["native"], _link(case), (), store=case["store"])
    body = case["adapter"].reopen(ref, context=case["context"], native=case["native"],
        lifecycle_link=_link(case), runtime_readbacks=(), store=case["store"])
    assert body["status"] == "unknown"
    assert "T0 tool artifact is not the native serving artifact" in body["scope_defects"]
    assert len(body["report"]["gates"]) == 17


@pytest.mark.parametrize("field", ["stdout", "argv", "env", "cwd", "duration_s", "exit_code", "sequence", "timeout_s"])
def test_one_original_raw_fact_mutation_refuses(original, field):
    case = original
    issued = case["adapter"]._replay(case["context"], case["store"])
    row = issued.raw[0]
    path = case["store"].root / row["artifact"]["locator"]
    raw = ob._plain(row["body"])
    if field == "sequence":
        raw[field] += 1
    elif field == "timeout_s":
        raw["invocation"][field] += 1
    else:
        raw["capture"][field] = "mutated"
    # Keep the original filename/reference: the bounded reader must catch bytes.
    import json
    path.write_text(json.dumps(raw))
    with pytest.raises(Exception, match="digest|differs"):
        case["adapter"]._replay(case["context"], case["store"])


def test_hash_valid_artifacts_cannot_reconstruct_lost_issuance(original):
    with pytest.raises(scientific.ScientificWitnessRefused, match="unavailable"):
        scientific.NativeT0WitnessAdapter()._replay(original["context"], original["store"])


def test_original_nested_inputs_are_deeply_immutable(original):
    issued = original["adapter"]._replay(original["context"], original["store"])
    with pytest.raises(FrozenInstanceError):
        issued.request.candidate_id = "akc-other"
    with pytest.raises(TypeError):
        issued.raw[0]["body"]["capture"]["env"][0][1] = "mutated"
    with pytest.raises(TypeError):
        issued.report["gates"][0]["outcome"] = "PASS"
    with pytest.raises(AttributeError):
        original["adapter"].max_units = 3


def test_duplicate_same_issue_does_not_collect_again_and_drift_refuses(original):
    case = original
    kwargs = {"context": case["context"], "store": case["store"], "plan": case["plan"],
        "request": case["t0_request"], "policy": case["policy"], "claim": case["claim"]}
    count = len(case["calls"])
    case["claim"]._held = False
    assert case["adapter"].collect_issued(**kwargs) == case["artifact"]
    assert len(case["calls"]) == count
    with pytest.raises(scientific.ScientificWitnessRefused, match="differs"):
        case["adapter"].collect_issued(**{**kwargs, "policy": replace(case["policy"], policy_ref="policy:changed")})


def test_registered_adapter_reaches_native_completion_and_parent_replay(original):
    case = original
    adapters = scientific.ParentScientificWitnessAdapters(correctness=case["adapter"])
    producer = npe.NativeUnitEvidenceProducer(store=case["store"], context=case["context"], scientific_adapters=adapters)
    result = producer.evaluate(case["request"])
    body = case["store"].read(result.receipt.locator, result.receipt.sha256)
    assert body["findings"]["correctness"]["facts"]["receipt"]["schema"] == scientific.WITNESS_SCHEMA
    assert result.completion.stage_witnesses["correctness"].status == "unknown"
    assert result.completion.stage_witnesses["purpose"].status == "unknown"
    registry = replay.IssuedNativeEvidenceRegistry(artifact_root=case["store"].root, max_units=1)
    registry.record(producer=producer, request=case["request"], result=result)
    attempt = {"unit_id": case["context"].unit_id,
        "stage_witnesses": {name: row.to_dict() for name, row in result.completion.stage_witnesses.items()},
        "terminal": result.completion.terminal, "recorded_screen": result.completion.recorded_screen,
        "provider_recorded_screen": result.completion.recorded_screen}
    entry = registry._lookup({"plan": case["context"].plan.to_dict(), "capture_context": {
        "worker_id": case["context"].fence.worker_id, "worker_incarnation": case["context"].fence.worker_incarnation}}, case["native"])
    replay.NativeParentReceiptReplayer._replay_unit(entry, case["native"], attempt, case["store"])


def test_unknown_adapters_and_callbacks_are_not_registered():
    for kwargs in ({"correctness": lambda *args: "pass"}, {"purpose": object()},
                   {"residency": object()}, {"contention": object()}):
        with pytest.raises(scientific.ScientificWitnessRefused):
            scientific.ParentScientificWitnessAdapters(**kwargs)


def test_v1_preserved_v2_source_closed_and_actual_selected_configuration_bound(tmp_path):
    old = source.loaded_producer_source_closure()
    assert old["schema"] == source.PRODUCER_SOURCE_SCHEMA
    assert "scientific_adapters" not in old
    adapters = scientific.ParentScientificWitnessAdapters(correctness=scientific.NativeT0WitnessAdapter(max_units=3))
    new = source.loaded_producer_source_closure(scientific_adapters=adapters)
    assert new["schema"] == source.PRODUCER_SOURCE_SCHEMA_V2
    assert source.producer_source_closure_complete(new)
    assert scientific.installed_scientific_adapters(new["scientific_adapters"]).source_identity() == adapters.source_identity()
    row = ob._plain(new)
    row["scientific_adapters"]["correctness"]["configuration"]["max_units"] += 1
    assert source.validate_producer_source_closure(row) != new
    row["scientific_adapters"]["invented"] = None
    with pytest.raises((ob.ObservationBindingError, scientific.ScientificWitnessRefused)):
        source.validate_producer_source_closure(row)
    # Prospective plan identity covers the selected config before plan issuance.
    identity = ob.loaded_planned_serving_identity(measurement_callable=serving._measure_once,
        fence_clock=time.monotonic, serving_timer=time.time, scientific_adapters=adapters)
    assert identity["used_constants"]["producer_source_closure"] == new


@pytest.mark.parametrize("bad", [object(), {"bad": object()}, float("nan"), {1: "key"}])
def test_arbitrary_objects_are_not_immutable_evidence(bad):
    with pytest.raises(scientific.ScientificWitnessRefused):
        scientific._immutable(bad)


@pytest.fixture
def prepared_model(tmp_path):
    root = tmp_path / "six-shard-model"
    root.mkdir()
    files = []
    for index in range(6):
        path = root / f"model-{index}.gguf"
        path.write_bytes(f"tiny synthetic shard {index}".encode())
        files.append({"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    material = {"model_path": str(root), "files": files}
    manifest = tmp_path / "model.json"
    manifest.write_text(json.dumps({"schema": "epyc.autokernel.model_identity.v1", **material}))
    owner = scientific.tensor_capture
    identity = owner.CaptureModelIdentity(str(root), manifest,
        hashlib.sha256(manifest.read_bytes()).hexdigest(),
        hashlib.sha256(owner._canonical(material).encode()).hexdigest())
    adapter = scientific.NativeT0WitnessAdapter(max_units=2)
    store = mc.ArtifactStore(tmp_path / "artifacts")
    entry = root / files[0]["path"]
    kwargs = {"store": store, "identity": identity, "entry_path": entry,
              "preparation_claim": FakeClaim(claim_id="scheduled-model-preparation")}
    ref = adapter.prepare_model_identity(**kwargs)
    context = SimpleNamespace(recipe=SimpleNamespace(model=SimpleNamespace(
        path=str(entry), sha256=files[0]["sha256"])))
    yield adapter, store, kwargs, ref, context, root
    store.close()


def test_six_shard_original_inventory_and_entry_digest_are_distinct(prepared_model):
    adapter, store, kwargs, ref, context, _root = prepared_model
    body = store.read(ref.locator, ref.sha256)
    assert len(body["manifest"]["files"]) == 6
    assert hashlib.sha256(body["manifest_raw_utf8"].encode("utf-8")).hexdigest() == kwargs["identity"].model_manifest_sha256
    assert body["entry_sha256"] == context.recipe.model.sha256
    assert body["inventory_sha256"] == kwargs["identity"].model_sha256
    assert body["entry_sha256"] != body["inventory_sha256"]
    assert len(body["verified_file_stats"]) == 7
    fact = adapter._model_fact(context, store)
    assert fact["error"] is None
    assert fact["original"] == ref.to_dict()


def test_reuse_and_unit_readback_do_not_rehash_model(prepared_model, monkeypatch):
    adapter, store, kwargs, ref, context, root = prepared_model
    original_open = Path.open

    def no_model_open(path, *args, **kw):
        assert not path.is_relative_to(root), "model contents reread during reuse/unit"
        return original_open(path, *args, **kw)

    monkeypatch.setattr(Path, "open", no_model_open)
    assert adapter.prepare_model_identity(**kwargs) == ref
    kwargs["preparation_claim"]._held = False
    assert adapter._model_fact(context, store)["error"] is None


def test_nonentry_shard_change_invalidates_original_inventory(prepared_model):
    adapter, store, kwargs, _ref, context, root = prepared_model
    (root / "model-5.gguf").write_bytes(b"changed nonentry shard")
    assert adapter._model_fact(context, store)["error"] == "model continuity changed"
    with pytest.raises(scientific.ScientificWitnessRefused, match="continuity"):
        adapter.prepare_model_identity(**kwargs)


def test_model_receipt_bytes_and_lost_issuance_are_not_reconstructed(prepared_model):
    adapter, store, _kwargs, ref, context, _root = prepared_model
    assert scientific.NativeT0WitnessAdapter()._model_fact(context, store)["original"] is None
    path = store.root / ref.locator
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(mc.CaptureError, match="digest"):
        adapter._model_fact(context, store)


def test_full_model_verification_requires_scheduled_claim(prepared_model):
    _adapter, _store, kwargs, _ref, _context, _root = prepared_model
    kwargs["preparation_claim"]._held = False
    with pytest.raises(t0.ClaimNotHeld):
        scientific.NativeT0WitnessAdapter().prepare_model_identity(**kwargs)


def test_collection_reservation_does_not_hold_registry_lock_or_repeat_work(original, monkeypatch):
    case = original
    adapter = scientific.NativeT0WitnessAdapter(max_units=1)
    entered, release = threading.Event(), threading.Event()
    original_run = t0.SubprocessRunner.run
    errors = []

    def blocked_run(self, argv, **kwargs):
        entered.set()
        if not release.wait(3):
            raise AssertionError("test release was not signalled")
        return original_run(self, argv, **kwargs)

    monkeypatch.setattr(t0.SubprocessRunner, "run", blocked_run)
    kwargs = {"context": case["context"], "store": case["store"], "plan": case["plan"],
        "request": case["t0_request"], "policy": case["policy"], "claim": case["claim"]}

    def collect():
        try:
            adapter.collect_issued(**kwargs)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=collect)
    thread.start()
    try:
        assert entered.wait(2)
        assert adapter._lock.acquire(timeout=0.1)
        adapter._lock.release()
        with pytest.raises(scientific.ScientificWitnessRefused, match="in flight"):
            adapter.collect_issued(**kwargs)
        with pytest.raises(scientific.ScientificWitnessRefused, match="conflict"):
            adapter.collect_issued(**{**kwargs, "policy": replace(case["policy"], policy_ref="changed")})
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert not errors


def test_bounded_bytes_code_constants_are_typed_not_configuration_support():
    def function():
        return b""
    code = function.__code__
    base = lo._code_projection(code)[1]
    observed = []
    for item in (b"", b"\x00\xff", "00ff", (0, 255)):
        status, body = lo._code_projection(code.replace(co_consts=(None, item)))
        assert status == "pinned"
        observed.append(body)
    assert all(observed[i] != observed[j] for i in range(4) for j in range(i))
    assert base["constants"][1]["type"] == "builtins.bytes"
    class BytesSubclass(bytes):
        pass
    for value in (bytearray(b"x"), BytesSubclass(b"x"), b"x" * 4097):
        assert lo._code_projection(code.replace(co_consts=(None, value)))[0] == "unproven"
    assert lo._stable_json_value(b"x")[0] == "unproven"


def test_bytes_identity_is_reproducible_across_hashseeds():
    import os
    import subprocess
    import sys
    package_root = str(Path(lo.__file__).resolve().parents[2])
    program = ("from autokernel.loop import lifecycle_observation as lo; "
               "f=lambda:b'\\x00\\xff'; print(lo.callable_identity(f)['implementation_sha256'])")
    outputs = [subprocess.check_output([sys.executable, "-c", program], text=True,
        env={**os.environ, "PYTHONPATH": package_root, "PYTHONHASHSEED": seed})
        for seed in ("1", "91")]
    assert outputs[0] == outputs[1]
