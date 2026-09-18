"""Final owner integration; tiny synthetic files/HTTP, no model/device execution."""
from dataclasses import replace
import hashlib
from pathlib import Path
import shlex
import socket

import pytest

from .. import journal as journal_module
from ..evaluator import api, correctness
from ..execution import t0_provider as t0
from ..execution.test_t0_provider import candidate_build, evaluation_request, execution_plan, t0_policy
from . import experiment_plan as ep
from . import campaign
from . import campaign_control
from . import test_unified_driver as driver_fixtures
from . import lifecycle_observation as lo
from . import measurement_capture as mc
from . import native_capture_control as nc
from . import native_final_trial as final
from . import native_model_preparation as model_prep
from . import native_parent_receipt_replay as replay
from . import native_parent_service as service
from . import native_scientific_witness as scientific
from . import native_server_t0_witness as server
from . import resolved_recipe as rr
from . import unified_worker as uw
from . import worker_lifecycle as wl
from .test_driver_execution import _run_real_controller_child_v2_capture_and_restart
from .test_native_model_preparation import _inventory
from .test_native_server_response import _CONTAINED_HTTP_FIXTURE
from .test_native_server_t0_witness import _owning_run
from .test_unified_planner import canonical_recipe
from .test_experiment_plan import plan_dict


def _prefix_reference(reference, *, prepared, start, terminal, fence, store):
    original, captures = uw.reopen_deferred_result(reference, prepared=prepared, start=start,
                                                   terminal=terminal, fence=fence)
    body = original.to_dict()
    carrier = mc._plain(captures[0][1]["carrier"])
    sink = mc.DeferredNativeMeasurementSink(
        context=mc.CaptureContext.from_dict(carrier["capture_context"]), store=store)
    for item in carrier["raw_artifacts"]:
        sink(item["document"])
    raws = body["run"]["raw_units"][:1]
    observations = body["lifecycle_observation_references"][:1]
    view = ep.admissible_units(prepared.plan, (ep.RawUnit.from_dict(row) for row in raws))
    summary = {"plan": prepared.plan.to_dict(), "plan_digest": prepared.plan.digest,
        "prompt_manifest": prepared.prompts.to_dict(), "prompt_manifest_digest": prepared.prompts.digest,
        "lineage_id": start.lineage_id, "comparison_identities": carrier["comparison_identities"],
        "raw_units": raws, "admissible_view": view.to_dict(), "execution_complete": False,
        "paused_reason": "fixture stopped at fixed prefix", "lifecycle_observation_references": observations}
    receipts = sink.finalize_run(summary)
    body["completed_unit_ids"] = body["completed_unit_ids"][:1]
    body["lifecycle_observation_references"] = observations
    body["captures"] = [mc._plain(item) for item in sink.captures]
    body["run"].update(raw_units=raws, lifecycle_observation_references=observations,
        admissible_view=view.to_dict(), execution_complete=False,
        paused_reason="fixture stopped at fixed prefix", capture_receipts=[mc._plain(item) for item in receipts])
    body.pop("result_digest")
    partial = uw.PlannedWorkerResult.from_dict(body | {"result_digest": uw._digest(body)})
    artifact = store.write(f"planned-worker-result:{prepared.prepared_digest}:{start.nonce}", partial.to_dict())
    partial_ref = uw.PlannedWorkerResultReference(start.nonce, prepared.prepared_digest,
        start.worker_id, start.worker_generation, partial.result_digest, artifact.locator,
        artifact.sha256, uw.RESULT_REFERENCE_SCHEMA_V2)
    partial_terminal = replace(terminal, result_digest=uw._digest(partial_ref.to_dict()))
    # The existing real reader deliberately accepts a well-formed fixed prefix.
    uw.reopen_deferred_result(partial_ref, prepared=prepared, start=start,
                              terminal=partial_terminal, fence=fence)
    return partial_ref, partial_terminal


def _check_final_journal_grammar(payload):
    """Grammar mutations of an actually issued final carrier, not fake appends."""
    kind = journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED
    assert journal_module._validate_native_payload(kind, payload) == []
    mutations = (
        (("schema",), "future"),
        (("unknown",), True),
        (("measurement_id",), "0" * 64),
        (("carrier", "unknown"), True),
        (("carrier", "producer"), mc.PRODUCER_ID_V2),
        (("carrier", "arm_locator"), "planned-serving:foreign"),
        (("carrier", "original_arm_capture"), None),
        (("carrier", "original_arm_capture", "unknown"), True),
        (("carrier", "original_arm_capture", "measurement_id"), "0" * 64),
        (("carrier", "original_arm_capture", "carrier_digest"), "invalid"),
        (("carrier", "original_arm_capture", "artifact", "verified"), False),
        (("carrier", "original_arm_capture", "artifact", "sha256"), "invalid"),
        (("carrier", "original_arm_capture", "artifact", "locator"), ""),
        (("carrier", "original_arm_capture", "artifact", "unknown"), True),
        (("carrier", "parent_final_trial"), None),
        (("carrier", "parent_final_trial", "schema"), "future"),
        (("carrier", "parent_final_trial", "unknown"), True),
        (("carrier", "parent_final_trial", "digest"), "0" * 64),
        (("carrier", "parent_final_trial", "artifact", "verified"), False),
        (("carrier", "parent_final_trial", "artifact", "sha256"), "invalid"),
        (("carrier", "parent_final_trial", "artifact", "locator"), ""),
        (("carrier", "parent_final_trial", "artifact", "unknown"), True),
    )
    for path, value in mutations:
        changed = mc._plain(payload)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        changed["carrier"].pop("carrier_digest")
        changed["carrier"]["carrier_digest"] = wl._digest(changed["carrier"])
        assert journal_module._validate_native_payload(kind, changed), path
    missing = mc._plain(payload)
    del missing["carrier"]["parent_final_trial"]
    missing["carrier"].pop("carrier_digest")
    missing["carrier"]["carrier_digest"] = wl._digest(missing["carrier"])
    assert journal_module._validate_native_payload(kind, missing)


@pytest.mark.parametrize("screen", ["flagged_but_retained", "rejected"])
@pytest.mark.parametrize("original_correctness", ["unknown", "fail"])
def test_final_projection_uses_owning_screen_and_never_overwrites_original_fail(screen, original_correctness):
    # Reducer-only synthetic witness inputs, not an issuance/production positive.
    document = plan_dict(n=1)
    document["required_witnesses"] = ["correctness"]
    plan = ep.ExperimentPlan.from_dict(document)
    rows, pair = [], []
    for unit in sorted(plan.expected_units, key=lambda item: item.order_index):
        rows.append(ep.RawUnit(ep.UNIT_SCHEMA, plan.digest, unit.unit_id, unit.arm,
            unit.process_id, unit.expected_prompt_ids, True, 10.0,
            {"correctness": ep.Witness(original_correctness,
                None if original_correctness == "unknown" else "original:failed")},
            screen, "original retained screen reason", "a" * 64, unit.order_index).to_dict())
        pair.append({"unit_id": unit.unit_id, "final_evidence": {"status": "pass"}})
    final_rows, view = final.project_final_rows(plan, rows,
        {"plan_digest": plan.digest, "ordered_units": pair})
    assert view.complete == (screen != "rejected" and original_correctness != "fail")
    assert all(row.recorded_screen == screen and row.reason == "original retained screen reason"
               for row in final_rows)
    if original_correctness == "fail":
        assert all(row.witnesses["correctness"] == ep.Witness("fail", "original:failed") for row in final_rows)


def test_actual_child_http_original_issuer_final_pair_capture_and_restart(tmp_path, monkeypatch):
    identity, entry, entry_sha = _inventory(tmp_path)
    build = tmp_path / "candidate" / "build"
    bindir = build / "bin"
    bindir.mkdir(parents=True)
    binary, library = bindir / "llama-server", bindir / "libggml-base.so"
    binary.write_bytes(b"synthetic declared server executable; HTTP fixture owns the actual process")
    library.write_bytes(b"synthetic mapped-library input, not a real ELF or linkage pass")
    ops, linkage_tool = bindir / "test-backend-ops", tmp_path / "fixture-linkage.sh"
    for tool, arguments in ((ops, (str(ops),)), (linkage_tool, ("fixture-linkage",))):
        output = _owning_run(None, arguments, env={}, cwd=str(build.parent), timeout_s=1.0).stdout
        tool.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(output) + "\n")
        tool.chmod(0o755)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    assert port != 8000
    original_recipe = canonical_recipe(model_path=str(entry), model_sha256=entry_sha)
    # Existing export fixture carries context/np/threads only, so its reimport
    # retains the published sampler defaults. This composed case is diagnostic;
    # the separate original-issuer pair fixture proves greedy byte coherence.
    template = original_recipe.template
    full = template.server_argv(build, port)
    def artifact(role, path):
        return {"schema": rr.ARTIFACT_SCHEMA, "role": role, "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    recipe = rr.resolve_canonical_launch(template, build_dir=build,
        topology_prefix=full[:3], command_argv=full[3:],
        launch_environment=template.server_env(build, base={}), artifact_identities={
            "model": artifact("model", entry), "drafter": None,
            "executable": artifact("executable", binary), "dsos": [artifact("dso", library)]},
        backend="cpu", environment_policy=original_recipe.environment_policy,
        port=port, runtime_binary_dir=str(bindir), runtime_ld_paths=[str(bindir)],
        provenance=dict(original_recipe.provenance))
    issuer = scientific.NativeT0WitnessAdapter(max_units=4)
    adapter = server.NativeServerT0WitnessAdapter(max_units=4, owning_issuer=issuer)
    adapters = scientific.ParentScientificWitnessAdapters(correctness=adapter)
    replayer, registries, scopes, final_records = replay.NativeParentReceiptReplayer(), [], [], []
    injected = pytest.MonkeyPatch()
    text_writer, original_validator_init = Path.write_text, nc.NativeCaptureValidator.__init__
    original_ingest = uw.ingest_deferred_result
    original_enroll = driver_fixtures.campaign_for_recipe
    original_experiment = driver_fixtures.experiment
    original_controller_enter = campaign_control.CampaignController.__enter__
    selected_cases, validators = [], []
    restart_duplicates = []

    def enroll_physical_fixture(recipe, **kwargs):
        row = original_enroll(recipe, **(kwargs | {"campaign_id": "ak-final-trial-fixture"})).to_dict()
        for target in row["targets"]:
            target["execution"]["build"].update(path=recipe.executable.path,
                                                sha256=recipe.executable.sha256)
            execution = campaign.ResolvedExecution.from_dict(target["execution"])
            baseline = campaign.ArtifactIdentity.from_dict(target["baseline"],
                                                           kind="build", ref=target["baseline_ref"])
            target["workload_signature"] = campaign._workload_signature(
                execution, baseline, target["baseline_ref"])
        return campaign.ResolvedCampaign.from_dict(row)

    def experiment_fixture(*args, **kwargs):
        return original_experiment(*args, **kwargs) | {"campaign_id": "ak-final-trial-fixture"}

    def fixture_writer(path, text, *args, **kwargs):
        if path == tmp_path / "fixture_measure.py":
            text = _CONTAINED_HTTP_FIXTURE
        return text_writer(path, text, *args, **kwargs)

    def validator_init(self, *args, **kwargs):
        original_validator_init(self, *args, **(kwargs | {"parent_receipt_replayer": replayer}))
        validators.append(self)

    def controller_enter(self):
        value = original_controller_enter(self)
        if selected_cases:
            # A restart restores durable duplicate identity, not a live grant or
            # original registry authority. New/different bytes still refuse.
            assert self._native_validator is None
            prior_count = len(self._journal.read_all())
            with self.native_capture_callback() as capture:
                for measurement_id, payload in selected_cases[0][3]:
                    existing = self.native_capture(measurement_id)
                    assert existing is not None
                    duplicate = capture(measurement_id, mc._plain(payload))
                    assert duplicate.record_id == existing.record_id
                    restart_duplicates.append(duplicate.record_id)
                    mutated = mc._plain(payload)
                    mutated["carrier"]["parent_final_trial"]["digest"] = "0" * 64
                    with pytest.raises(nc.NativeCaptureRefused, match="different capture bytes"):
                        capture(measurement_id, mutated)
            assert len(self._journal.read_all()) == prior_count
        return value

    class OriginalIssuingService(service.NativeParentEvidenceService):
        def _producer_for(self, notice):
            producer = super()._producer_for(notice)
            context = producer.context
            key = (context.plan.digest, context.unit_id)
            if key not in adapter._inputs:
                selected = context.recipe
                tools = t0.ToolPaths(bash="/bin/bash", verify_ggml_linkage_sh=str(linkage_tool),
                                     cmake="/usr/bin/cmake")
                plan = execution_plan(candidate=candidate_build(worktree=str(build.parent),
                    build_dir=str(build), binary=str(binary), library_path=str(bindir),
                    test_backend_ops=str(ops)), tools=tools, base_env=selected.launch_env)
                linkage = t0.ExecutedT0EvidenceProvider.linkage_digest(t0.parse_linkage_report(
                    _owning_run(None, ("fixture-linkage",), env={}, cwd=str(build.parent), timeout_s=1.0).stdout))
                anchor_identity = api.AnchorIdentity(plan.candidate.source_commit,
                    selected.executable.sha256, linkage)
                request = replace(evaluation_request(anchor=anchor_identity,
                    source_sha256=plan.candidate.source_sha256,
                    binary_sha256=selected.executable.sha256, linkage_sha256=linkage),
                    campaign_id=context.plan.campaign_id)
                claim = model_prep.ActiveObservationPreparationClaim(lifecycle=self.lifecycle,
                    start=notice["start"], unit=context.unit, fence=context.fence,
                    initial_claim=context.active_claim)
                original = issuer.collect_issued(context=context, store=self._native_store,
                    plan=plan, request=request, policy=t0_policy(), claim=claim)
                anchor_id = next(unit.unit_id for unit in context.plan.expected_units if unit.arm == "anchor")
                adapter.register_owning_inputs(context=context, store=self._native_store,
                    original_context=context, original_artifact=original,
                    request_by_slot=tuple(replace(request, event_id=f"{request.event_id}:{index}")
                                          for index in range(len(context.unit.expected_prompt_ids))),
                    anchor_unit_id=None if context.unit.arm == "anchor" else anchor_id)
            return producer

    def producer_factory(authority, prepared, lifecycle, configuration):
        target = prepared.dispatch["proposal"]["target_revision_digest"]
        preparations = {}
        for selected in (prepared.runtime_pair.anchor, prepared.runtime_pair.candidate):
            body = {"schema": model_prep.SPEC_SCHEMA, "target_revision_digest": target,
                "recipe_execution_digest": selected.execution_digest, "entry_path": selected.model.path,
                "entry_sha256": selected.model.sha256, "inventory_identity": identity}
            preparations[selected.execution_digest] = {**body, "preparation_digest": wl._digest(body)}
        registry = replay.IssuedNativeEvidenceRegistry(artifact_root=prepared.artifact_root,
                                                       max_units=len(prepared.plan.expected_units))
        scope = replayer.using(registry)
        scope.__enter__()
        registries.append(registry)
        scopes.append(scope)
        root = tmp_path / "fixture-probe"
        return OriginalIssuingService(authority, prepared, lifecycle, configuration,
            registry=registry, runtime_probe=lo.FilesystemProbe(proc_root=root / "proc",
                sysfs_cpu_root=root / "cpu", boot_id_path=root / "boot", cgroup_root=root / "cgroup"),
            scientific_adapters=adapters, model_preparations=preparations)

    def ingest(reference, **kwargs):
        originals = original_ingest(reference, **kwargs)
        store = mc.ArtifactStore(kwargs["prepared"].artifact_root)
        try:
            options = {name: kwargs[name] for name in ("prepared", "start", "terminal", "fence")}
            options |= {"result": reference, "registry": registries[0], "store": store}
            partial, partial_terminal = _prefix_reference(reference, **{
                name: options[name] for name in ("prepared", "start", "terminal", "fence", "store")})
            with pytest.raises(scientific.ScientificWitnessRefused, match="incomplete original unit/lifecycle prefix"):
                adapter.final_trial_owner.finalize(**(options | {"result": partial, "terminal": partial_terminal}))
            with pytest.raises(uw.WorkerBridgeRefused, match="stale|current worker fence"):
                adapter.final_trial_owner.finalize(**(options | {"fence": replace(options["fence"], current=False)}))
            issued = adapter.final_trial_owner.finalize(**options)
            validated = adapter.final_trial_owner.reopen(issued, registry=registries[0], store=store)
            assert not validated.final_view.complete
            assert validated.control_reference is validated.calibration_reference is None
            assert set(validated.t0_reports_by_unit) == {unit.unit_id for unit in kwargs["prepared"].plan.expected_units}
            for _unit, reports in validated.t0_reports_by_unit.items():
                assert all(isinstance(report, correctness.T0Report) and report.unevaluated for report in reports)
            captures = adapter.final_trial_owner.captures(issued, registry=registries[0], store=store)
            for measurement_id, payload in captures:
                assert payload["carrier"]["status"] == "diagnostic"
                _check_final_journal_grammar(mc._plain(payload))
                captured = kwargs["capture_transaction"](measurement_id, mc._plain(payload))
                duplicate = kwargs["capture_transaction"](measurement_id, mc._plain(payload))
                assert duplicate.record_id == captured.record_id
                final_records.append(captured.record_id)
                mutated = mc._plain(payload)
                mutated["carrier"]["parent_final_trial"]["digest"] = "0" * 64
                with pytest.raises(scientific.ScientificWitnessRefused):
                    adapter.final_trial_owner.prevalidate(
                        validators[0], measurement_id, mutated)
            # Well-shaped forged original artifact references pass the grammar
            # but cannot pass the owning replay. Grammar is not an issuer.
            measurement_id, payload = captures[0]
            mutated = mc._plain(payload)
            mutated["carrier"]["original_arm_capture"]["artifact"]["sha256"] = "0" * 64
            mutated["carrier"].pop("carrier_digest")
            mutated["carrier"]["carrier_digest"] = wl._digest(mutated["carrier"])
            assert journal_module._validate_native_payload(
                journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED, mutated) == []
            with pytest.raises(scientific.ScientificWitnessRefused, match="independently rebuilt final carrier"):
                adapter.final_trial_owner.prevalidate(validators[0], measurement_id, mutated)
            assert adapter.final_trial_owner.finalize(**options) == issued
            selected_cases.append((options, issued, validated, captures))
        finally:
            store.close()
        return originals  # Existing helper still verifies the two original child records.

    injected.setattr(Path, "write_text", fixture_writer)
    injected.setattr(nc.NativeCaptureValidator, "__init__", validator_init)
    injected.setattr(uw, "ingest_deferred_result", ingest)
    injected.setattr(driver_fixtures, "campaign_for_recipe", enroll_physical_fixture)
    injected.setattr(driver_fixtures, "experiment", experiment_fixture)
    injected.setattr(campaign_control.CampaignController, "__enter__", controller_enter)
    try:
        _run_real_controller_child_v2_capture_and_restart(tmp_path, monkeypatch,
            producer_type=producer_factory, recipe=recipe, scientific_adapters=adapters)
        assert len(selected_cases) == 1 and len(final_records) == 2
        assert set(restart_duplicates) == set(final_records)
        rows = journal_module.Journal(str(tmp_path / "controller" / "journal")).read_all()
        native = [row for row in rows if row.kind == journal_module.KIND_PLANNED_SERVING_ARM_CAPTURED]
        assert len(native) == 4
        assert {row.record_id for row in native if row.payload["schema"] == final.CAPTURE_SCHEMA} == set(final_records)
        assert all(row.payload["carrier"]["status"] == "diagnostic" for row in native)
    finally:
        for scope in reversed(scopes):
            scope.__exit__(None, None, None)
        injected.undo()
