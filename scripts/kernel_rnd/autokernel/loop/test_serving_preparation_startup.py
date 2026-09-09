"""Prospective startup expansion; fixtures label their non-authoritative policy."""
from dataclasses import replace

import pytest

from . import serving_preparation as prep, serving_preparation_startup as startup
from . import unified_driver
from .test_experiment_plan import plan_dict
from .test_serving_preparation import request
from .test_unified_driver import runtime_driver


def configuration(tmp_path):
    original = request(tmp_path, blocks=1, max_attempts=2)
    driver, engine, resolved, _target, digest = runtime_driver(git_source=True)
    anchor = driver.runtime_anchors.recipes[digest]
    declared = replace(original.declaration,
        campaign_id=resolved.campaign_id, target_revision=digest,
        statistics=replace(original.declaration.statistics,
            commitment=replace(original.declaration.statistics.commitment,
                               campaign_id=resolved.campaign_id)),
        aa_pair=prep.PreparationArmPair("aa", anchor, anchor, None),
        resources=engine.config.capacity,
        frame={**prep._plain(original.declaration.frame), "quant": driver.profiles[digest].quant,
               "model_sha256": anchor.model.sha256})
    policy = {key: plan_dict()[key] for key in startup.PROTOCOL_FIELDS}
    cfg = startup.PreparationStartupConfiguration.from_dict({"schema": startup.SCHEMA,
        "entries": [{"declaration": declared.to_dict(), "protocol": policy,
                     "submitted_at": 1.0, "estimated_duration_seconds": 30}]})
    execution = unified_driver.ExecutionInput(digest, original.prompts,
        declared.max_stage_seconds, declared.teardown_seconds,
        original.plan.loaded_instrument["identity_sha256"])
    return cfg, {"resolved": resolved, "anchors": driver.runtime_anchors,
        "execution_inputs": {digest: execution}, "profiles": driver.profiles,
        "scheduler_config": engine.config, "loaded_instrument": original.plan.loaded_instrument,
        "expected_epoch": declared.frame["epoch"]}


def test_compact_configuration_derives_original_complete_pool_and_retry_ids(tmp_path, monkeypatch):
    cfg, args = configuration(tmp_path)
    from . import measurement_capture, native_model_preparation
    monkeypatch.setattr(measurement_capture.ArtifactStore, "__init__",
                        lambda *_args, **_kwargs: pytest.fail("startup opened artifact store"))
    monkeypatch.setattr(native_model_preparation, "validate_model_identity",
                        lambda *_args, **_kwargs: pytest.fail("startup hashed model"), raising=False)
    values = startup.materialize(cfg, **args)
    assert len(values) == 4
    assert len({unit.process_id for item in values for unit in item.plan.expected_units}) == 8
    assert len({item.chunk_identity for item in values}) == 2
    assert len({item.declaration.digest for item in values}) == 1
    assert all(item.plan.changed_factors == () and item.plan.intended_use == "explore" for item in values)
    assert all(item.stage_proposal.production_frontier for item in values)
    assert all(item.stage_proposal.eligibility_ref == item.declaration.digest for item in values)
    assert [item.to_dict() for item in startup.materialize(cfg, **args)] == [item.to_dict() for item in values]
    for chunk in {item.chunk_identity for item in values}:
        initial, retry = sorted((item for item in values if item.chunk_identity == chunk),
                                key=lambda item: item.attempt_ordinal)
        assert initial.logical_membership == retry.logical_membership
        assert [unit.arm for unit in initial.plan.expected_units] == list(reversed(
            [unit.arm for unit in retry.plan.expected_units]))


def test_expansion_budget_refuses_before_any_native_pair_allocation(tmp_path, monkeypatch):
    cfg, args = configuration(tmp_path)
    args["scheduler_config"] = replace(args["scheduler_config"], campaign_attempt_cap=3)
    monkeypatch.setattr(startup.ps, "arm_identity",
                        lambda *_args, **_kwargs: pytest.fail("allocated native identity"))
    with pytest.raises(prep.PreparationRefused, match="expansion"):
        startup.materialize(cfg, **args)


@pytest.mark.parametrize("field", ["campaign_id", "target_revision", "epoch", "prompt_manifest_digest",
                                  "metric", "metric_direction"])
def test_foreign_original_input_cannot_be_rebound_during_startup(tmp_path, field):
    cfg, args = configuration(tmp_path)
    row = cfg.to_dict()
    declaration = row["entries"][0]["declaration"]
    if field in {"metric", "metric_direction"}:
        declaration["frame"][field] = "wrong_metric" if field == "metric" else "lower_better"
    elif field == "epoch":
        declaration["frame"][field] = "foreign-epoch"
    else:
        declaration[field] = "f" * 64
        if field == "campaign_id":
            declaration["statistics"]["commitment"]["campaign_id"] = "f" * 64
    changed = startup.PreparationStartupConfiguration.from_dict(row)
    with pytest.raises(prep.PreparationRefused, match="enrolled"):
        startup.materialize(changed, **args)


def test_startup_config_rejects_fields_and_detaches_nested_original_inputs(tmp_path):
    cfg, _args = configuration(tmp_path)
    row = cfg.to_dict()
    restored = startup.PreparationStartupConfiguration.from_dict(row)
    row["entries"][0]["protocol"]["required_witnesses"].append("invented")
    assert restored.to_dict() == cfg.to_dict()
    row["grant"] = True
    with pytest.raises(prep.PreparationRefused, match="fields"):
        startup.PreparationStartupConfiguration.from_dict(row)


def factory_request(tmp_path, monkeypatch, *, neutral=False, synthetic_measure=True):
    """Real exporter machinery; fixture production bytes/provider/observations."""
    import hashlib
    import json
    import os
    import subprocess
    from pathlib import Path
    from . import campaign, campaign_cli, production_enrollment, scheduling, unified_planner
    from . import native_model_preparation as nmp, native_scientific_witness as scientific
    from . import standalone_inputs as inputs, startup_factory, worker_lifecycle as wl
    from . import serving, lifecycle_observation as observation
    from .test_feed_runtime import config as feed_config
    from .test_production_enrollment import _seal_recipe_artifacts
    from .test_startup_factory import pin, request_for
    from .test_standalone_native_inputs import _budgets
    from .test_unified_planner import profile
    from .test_unified_worker import _prompt
    from .test_serving_preparation import declaration
    from .test_serving_preparation_execution import install_observation_fixture

    request = request_for(tmp_path, include_candidate=False)
    pid_log = tmp_path / "owned-calibration-pids.txt"
    if synthetic_measure:
        measurement, pid_log = install_observation_fixture(tmp_path, monkeypatch, installed_measure=True)
        monkeypatch.setattr(serving, "_measure_once", measurement)
    exported = json.loads(Path(request["production_export"]["path"]).read_text())
    raw_target = exported["targets"][0]
    model_path = str(tmp_path / "model-shards" / "entry.gguf")
    raw_target["command_argv"][raw_target["command_argv"].index("-m") + 1] = model_path
    for artifact in raw_target["artifacts"]:
        if artifact["use"] == "model":
            artifact["path"] = model_path
    raw_target["command_argv"][raw_target["command_argv"].index("-np") + 1] = "1"
    raw_target["workload"]["np"] = "1"
    source = Path(unified_driver.__file__).resolve()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=source.parent, text=True).strip()
    exported["context"]["sources"] = [{"name": "launcher", "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "revision": revision}]
    raw_target["source_revisions"] = {"launcher": revision}
    cpus = sorted(os.sched_getaffinity(0))[:4]
    prefix = ["numactl", "--interleave=all", "--", "taskset", "-c", ",".join(map(str, cpus))]
    raw_target["topology"]["argv_prefix"] = prefix
    raw_target["argv"] = prefix + raw_target["command_argv"]
    raw_target["artifacts"] = [item for item in raw_target["artifacts"] if item["use"] != "recipe"]
    for artifact in raw_target["artifacts"]:
        path = Path(artifact["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("synthetic preparation fixture " + artifact["use"]).encode())
        artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    exported = _seal_recipe_artifacts(exported, tmp_path)
    request["production_export"] = pin(Path(request["production_export"]["path"]), exported)
    config_path = tmp_path / "campaign-config.json"
    config = json.loads(config_path.read_text())
    config["resources"]["cpu_logical"] = cpus
    pin(config_path, config)
    output = Path(request["resolved_export"]["path"])
    assert campaign_cli.main(["--production-enrollment", request["production_export"]["path"],
        "--production-campaign-config", str(config_path), "--out", str(output)]) == 0
    request["resolved_export"] = {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    resolved = campaign.ResolvedCampaign.from_dict(json.loads(output.read_text())["resolved_campaign"])
    target = resolved.targets[0]
    digest = unified_planner._target_digest(target)
    anchor = unified_planner.prepare_runtime_anchors(resolved, {digest: {
        "schema": unified_planner.ANCHOR_SCHEMA, "target_revision_digest": digest,
        "target_id": target.target_ids[0], "production_export": production_enrollment.load_export(exported),
        "environment_policy": request["environment_policy"]}}).recipes[digest]
    raw_profile = profile(target)
    raw_profile["opportunities"] = []
    prompts = _prompt(anchor.template)
    material = {"model_path": str(Path(anchor.model.path).parent),
                "files": [{"path": Path(anchor.model.path).name, "sha256": anchor.model.sha256}]}
    model_manifest = tmp_path / "original-model-inventory.json"
    model_pin = pin(model_manifest, {"schema": "epyc.autokernel.model_identity.v1", **material})
    model_identity = {"model_id": material["model_path"], "model_manifest": str(model_manifest),
        "model_manifest_sha256": model_pin["sha256"],
        "model_sha256": hashlib.sha256(scientific.tensor_capture._canonical(material).encode()).hexdigest()}
    model_body = {"schema": nmp.SPEC_SCHEMA, "target_revision_digest": digest,
        "recipe_execution_digest": anchor.execution_digest,
        "entry_path": anchor.model.path, "entry_sha256": anchor.model.sha256,
        "inventory_identity": model_identity}
    model_spec = {**model_body, "preparation_digest": wl._digest(model_body)}
    feed = replace(feed_config(tmp_path), source_root=request["store_path"] + "/journal")
    scheduler = scheduling.SchedulerConfig.from_dict(request["scheduler_config"])
    original = declaration(max_attempts=2)
    neutral_pair = None
    if neutral:
        from .resolved_recipe import resolve_canonical_launch
        directory = tmp_path / "neutral-build"
        executable = directory / "bin" / "llama-server"
        executable.parent.mkdir(parents=True)
        executable.write_bytes(Path(anchor.executable.path).read_bytes())
        candidate = resolve_canonical_launch(
            anchor.template, build_dir=str(directory),
            command_argv=(str(executable), *anchor.command_argv[1:]),
            topology_prefix=anchor.topology_prefix, launch_environment=dict(anchor.launch_env),
            artifact_identities={"model": anchor.model.to_dict(), "drafter": None,
                "executable": {**anchor.executable.to_dict(), "path": str(executable)},
                "dsos": [item.to_dict() for item in anchor.dsos]},
            backend=anchor.backend, environment_policy=anchor.environment_policy, port=anchor.port,
            runtime_binary_dir=str(executable.parent), runtime_ld_paths=anchor.runtime_ld_paths,
            provenance={**dict(anchor.provenance), "fixture": "original byte-identical neutral copy"})
        neutral_pair = prep.PreparationArmPair("neutral", anchor, candidate,
            str(executable) + "#sha256=" + anchor.executable.sha256)
    declared = replace(original, campaign_id=resolved.campaign_id, target_revision=digest,
        statistics=replace(original.statistics, commitment=replace(original.statistics.commitment,
                           campaign_id=resolved.campaign_id)),
        aa_pair=prep.PreparationArmPair("aa", anchor, anchor, None),
        neutral_pair=neutral_pair,
        prompt_manifest_digest=prep._digest(prompts.to_dict()), resources=scheduler.capacity,
        max_stage_seconds=5, teardown_seconds=2,
        frame={**prep._plain(original.frame), "epoch": feed.expected_epoch,
               "model_sha256": anchor.model.sha256, "quant": raw_profile["quant"]})
    request.update(schema=startup_factory.PREPARATION_REQUEST_SCHEMA,
        native_artifact_sink_ref=str(Path(request["store_path"]) / "unified-native-artifacts"),
        target_defaults={"cpu": {"profile": pin(tmp_path / "profile.json", raw_profile),
            "profile_request": None, "runtime_dimensions": [], "execution": {
                "prompt_manifest": pin(tmp_path / "prompts.json", prompts.to_dict()),
                "max_stage_seconds": 5, "teardown_seconds": 2}}},
        experiment_plans={}, evidence_feed=feed.to_dict(),
        native_evidence={"scientific_adapters": {"schema": inputs.SCIENTIFIC_SELECTION_SCHEMA,
            "correctness": {"adapter_id": scientific.ADAPTER_ID, "max_units": 8},
            "purpose": None, "contention": None, "residency": None},
            "model_preparations": {digest: {anchor.execution_digest: model_spec}},
            "search_window_configuration": None,
            "observation_configuration": {"requested_effective_states": {anchor.execution_digest: {
                "logical_cpus": cpus, "numa_nodes": [0, 1], "thp_mode": "madvise"}},
                "required_gpu_dsos": {}, "cadence_s": 0.01, "gap_limit_s": 0.05,
                "budgets": {**_budgets(), "max_samples": observation.required_sample_capacity(
                    max_duration_s=declared.max_stage_seconds + declared.teardown_seconds,
                    cadence_s=0.01, nonperiodic_samples=len(observation.PHASES) + 2)}}},
        serving_preparation=pin(tmp_path / "original-preparation.json", {
            "schema": startup.SCHEMA, "entries": [{"declaration": declared.to_dict(),
                "protocol": {key: plan_dict()[key] for key in startup.PROTOCOL_FIELDS},
                "submitted_at": 1, "estimated_duration_seconds": 3}]}))
    request.pop("evidence_index")
    request["providers"].pop("evidence_verifier")
    return request, pid_log


@pytest.mark.parametrize("neutral,interrupt_before_settlement", [
    (False, False), (True, False),
    pytest.param(False, True, marks=pytest.mark.xfail(strict=True,
        reason="OP-AKU-HELD: original provider held receipt is not durable before settlement")),
])
def test_actual_factory_materialize_runtime_tick_and_restart_preserve_preissued_pairs(
        tmp_path, monkeypatch, neutral, interrupt_before_settlement):
    import json
    import time
    import threading
    from pathlib import Path
    from . import standalone_inputs as inputs, standalone_runtime, startup_factory
    from . import driver_execution, worker_lifecycle as wl
    from .test_feed_runtime import binding
    from .test_standalone_inputs import FullHeldProvider
    from .test_driver_execution import _resume

    request, pid_log = factory_request(tmp_path, monkeypatch, neutral=neutral)
    output = tmp_path / "bundle"
    with monkeypatch.context() as dry:
        from . import measurement_capture as mc
        dry.setattr(mc.ArtifactStore, "__init__", lambda *_a, **_k: pytest.fail("factory opened store"))
        receipt = startup_factory.build_startup(request, output_dir=output)
        manifest = inputs.StartupManifest.from_dict(json.loads((output / "startup.json").read_text()))
        materialized = inputs.materialize(manifest)
    count = 4 if neutral else 2
    assert receipt["preflight"]["serving_preparation"]["request_count"] == count * 2
    assert not Path(request["store_path"]).exists()
    original_requests = tuple(item.digest for item in materialized.inputs.calibration_requests)
    declared = materialized.inputs.calibration_requests[0].declaration
    if neutral:
        snapshots = materialized.inputs.retention_runtime_recipe_snapshots[declared.target_revision]
        assert len(snapshots) == 2
        assert len({recipe.execution_digest for recipe in snapshots.values()}) == 1
        roots = {artifact.path for artifact in materialized.inputs.retention_catalog_seed.artifacts}
        assert {declared.aa_pair.anchor.executable.path,
                declared.neutral_pair.candidate.executable.path} <= roots
    feed = manifest.evidence_feed
    for path in (Path(feed.corpus_root), Path(feed.store_root), Path(feed.ledger_path).parent,
                 tmp_path / "containers"):
        path.mkdir(parents=True, exist_ok=True)
    provider = FullHeldProvider(tmp_path / "containers")
    original_close = provider.close_held_receipt

    def close_calibration(**kwargs):
        trusted = original_close(**kwargs)
        return replace(trusted, receipt=replace(trusted.receipt, stage_class="calibration",
                                                memory_reservation_bytes=0))

    monkeypatch.setattr(provider, "close_held_receipt", close_calibration)
    # Actual active-claim shape from the fixture resource owner, not a witness.
    from .test_driver_execution import HeldProvider
    monkeypatch.setattr(provider, "describe_active_observation_claim",
                        HeldProvider.describe_active_observation_claim.__get__(provider))
    registry = inputs.ProviderRegistry({
        request["providers"]["lifecycle"]: inputs.ProviderBinding(lifecycle_provider=provider),
        request["providers"]["readiness"]: inputs.ProviderBinding(readiness_check=lambda: (True, None))},
        evidence_feeds={feed.binding_id: binding()})
    build = inputs.runtime_factory(materialized, registry)

    class Args:
        store = request["store_path"]
        config_generation = 1
        snapshot_version = 3

    controller = runtime = None
    producer_errors = []
    previous_profile = threading.getprofile()

    def capture_errors(frame, event, _arg):
        if event == "return" and frame.f_code is driver_execution.UnknownParentEvidenceProducer._run.__code__:
            producer_errors.extend(str(error) for error in frame.f_locals["self"]._errors)

    threading.setprofile(capture_errors)
    original_chunks = original_solve = None
    try:
        for incarnation in range(2):
            if incarnation:
                # A real startup reloads the immutable manifest; never reuse
                # the previous incarnation's mutable scheduler engine.
                materialized = inputs.materialize(inputs.StartupManifest.from_dict(
                    json.loads((output / "startup.json").read_text())))
                build = inputs.runtime_factory(materialized, registry)
            controller, runtime = build(materialized.resolved, Args())
            assert runtime.driver.calibration_requests
            assert tuple(item.digest for item in runtime.driver.calibration_requests) == original_requests
            recovered = runtime.recover()
            assert recovered.status == ("settled" if incarnation and interrupt_before_settlement
                                        else "recovered"), recovered.to_dict()
            _resume(controller, materialized.resolved.campaign_id)
            before = pid_log.read_text() if pid_log.exists() else ""
            settled = []
            deadline = time.monotonic() + count * 7
            while time.monotonic() < deadline and len(settled) < (count if incarnation == 0 else 0):
                if interrupt_before_settlement and len(settled) == count - 1:
                    def interrupted(_request):
                        raise OSError("fixture interruption after raw completion before settlement")

                    with monkeypatch.context() as interruption:
                        interruption.setattr(controller, "unified_driver_settle", interrupted)
                        with pytest.raises(standalone_runtime.StandaloneRuntimeUncertain):
                            runtime.tick()
                    assert len(pid_log.read_text().splitlines()) == count * 2
                    assert controller.unified_driver_pending_intent() is not None
                    assert runtime.driver.preparation_owner.solve_collected(declared) is None
                    break
                result = runtime.tick()
                if result.status == "settled":
                    assert result.execution_receipt["settlement_request"]["outcome"] == "calibration", producer_errors
                    settled.append(result)
                elif result.status == "waiting":
                    time.sleep(min(result.retry_after_seconds, 0.01))
                else:
                    pytest.fail(str(result.to_dict()))
            if incarnation == 0 and interrupt_before_settlement:
                runtime.close()
                controller.close()
                runtime = controller = None
                continue
            if incarnation == 0:
                assert len(settled) == count
                for result in settled:
                    parsed = driver_execution.DriverExecutionReceipt.from_dict(
                        result.to_dict()["execution_receipt"])
                    assert parsed.schema == driver_execution.EXECUTION_RECEIPT_SCHEMA_V2
                    assert parsed.settlement_request["outcome"] == "calibration"
                assert len(pid_log.read_text().splitlines()) == count * 2
            else:
                assert runtime.driver.preparation_owner.pending_requests() == ()
                assert runtime.tick().status == "waiting"
                assert pid_log.read_text() == before
            history = controller.unified_driver_preparation_history(runtime.driver.calibration_requests)
            by_digest = {item.digest: item for item in runtime.driver.calibration_requests}
            chunks = []
            for record in history["records"]:
                refs = [value for value in record["settlement"]["terminal_refs"]
                        if value.startswith("calibration-collected:")]
                assert len(refs) == 1
                reference = prep.CollectedCalibrationReference.from_dict(
                    json.loads(refs[0].removeprefix("calibration-collected:")))
                original_request = by_digest[record["request_digest"]]
                chunk = runtime.driver.preparation_owner.reopen_chunk(reference, request=original_request)
                assert chunk["qualification"] == "unavailable" and not chunk["ranking_authorized"]
                chunks.append(prep._plain(chunk))
            chunks.sort(key=lambda item: item["request_digest"])
            solved = runtime.driver.preparation_owner.solve_collected(declared)
            if neutral:
                assert solved is not None
                body = runtime.driver.preparation_owner.reopen_solve(solved, declaration=declared)
                assert body["qualification"] == "unavailable" and not body["ranking_authorized"]
                assert body["numeric_solve"] is not None or body["numeric_reasons"]
                assert "frame_phase_cell_scope_unverified" in body["qualification_debt"]
            else:
                assert solved is None
            if incarnation == 0:
                original_chunks, original_solve = chunks, solved
            else:
                assert chunks == original_chunks
                assert solved == original_solve
            runtime.close()
            controller.close()
            runtime = controller = None
    finally:
        threading.setprofile(previous_profile)
        if runtime is not None:
            runtime.close()
        if controller is not None:
            controller.close()
        if pid_log.exists():
            for row in pid_log.read_text().splitlines():
                pid, ticks = map(int, row.split())
                assert not wl.same_process(wl.ProcessIdentity(
                    pid, ticks, Path("/proc/sys/kernel/random/boot_id").read_text().strip()))


def test_emitted_v4_dry_run_command_is_runnable_and_old_versions_refuse_new_configuration(tmp_path, monkeypatch):
    import json
    import os
    from pathlib import Path
    import subprocess
    from . import standalone_inputs as inputs, startup_factory

    request, pid_log = factory_request(tmp_path, monkeypatch, synthetic_measure=False)
    output = tmp_path / "dry-bundle"
    receipt = startup_factory.build_startup(request, output_dir=output)
    process = subprocess.run(receipt["command"]["argv"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": receipt["command"]["PYTHONPATH"]}, timeout=15)
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    assert report["execution_authorized"] is False
    assert report["serving_preparation"]["status"] == "planned_uncollected"
    assert report["serving_preparation"]["request_count"] == 4
    assert report["serving_preparation"]["ranking_authorized"] is False
    assert not Path(request["store_path"]).exists() and not pid_log.exists()
    document = json.loads((output / "startup.json").read_text())
    for schema in (inputs.MANIFEST_SCHEMA, inputs.FEED_MANIFEST_SCHEMA, inputs.NATIVE_MANIFEST_SCHEMA):
        old = dict(document, schema=schema)
        if schema != inputs.NATIVE_MANIFEST_SCHEMA:
            old.pop("native_evidence")
        if schema == inputs.MANIFEST_SCHEMA:
            old.pop("evidence_feed")
            old.update(evidence_index={}, evidence_verifier_id="explicit-unavailable")
        old["manifest_digest"] = inputs._digest({k: v for k, v in old.items() if k != "manifest_digest"})
        with pytest.raises(inputs.StandaloneInputsRefused, match="fields/schema"):
            inputs.StartupManifest.from_dict(old)
    for schema in (startup_factory.REQUEST_SCHEMA, startup_factory.FEED_REQUEST_SCHEMA,
                   startup_factory.NATIVE_REQUEST_SCHEMA):
        old = dict(request, schema=schema)
        if schema != startup_factory.NATIVE_REQUEST_SCHEMA:
            old.pop("native_evidence")
        if schema == startup_factory.REQUEST_SCHEMA:
            old.pop("evidence_feed")
            old["evidence_index"] = {}
        with pytest.raises(startup_factory.StartupFactoryRefused, match="fields"):
            startup_factory.build_startup(old, output_dir=tmp_path / "unused")
    identity = inputs.materialize(inputs.StartupManifest.from_dict(document)).inputs.loaded_instrument_identity
    pinned = identity["used_constants"]["serving_preparation_source"]
    assert any(item["module"].endswith("serving_preparation_startup") for item in pinned["callables"])
    assert all(item["implementation_status"] == item["configuration_status"] == "pinned"
               for item in pinned["callables"])


def test_execution_equivalent_model_alias_cannot_share_wrong_preparation_path(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from . import startup_factory, standalone_inputs
    from .resolved_recipe import resolve_canonical_launch
    from .test_startup_factory import pin

    request, _pids = factory_request(tmp_path, monkeypatch, neutral=True, synthetic_measure=False)
    raw = json.loads(Path(request["serving_preparation"]["path"]).read_text())
    declared = prep.ServingPreparationDeclaration.from_dict(raw["entries"][0]["declaration"])
    original = declared.neutral_pair.candidate
    model_path = str(tmp_path / "other-inventory" / "same-bytes.gguf")
    argv = list(original.command_argv)
    argv[argv.index("-m") + 1] = model_path
    changed = resolve_canonical_launch(replace(original.template, model=model_path),
        build_dir=original.build_dir, command_argv=argv, topology_prefix=original.topology_prefix,
        launch_environment=dict(original.launch_env), artifact_identities={
            "model": {**original.model.to_dict(), "path": model_path}, "drafter": None,
            "executable": original.executable.to_dict(), "dsos": [item.to_dict() for item in original.dsos]},
        backend=original.backend, environment_policy=original.environment_policy, port=original.port,
        runtime_binary_dir=original.runtime_binary_dir, runtime_ld_paths=original.runtime_ld_paths,
        provenance=dict(original.provenance))
    assert changed.execution_digest == original.execution_digest
    assert changed.snapshot_digest != original.snapshot_digest
    declared = replace(declared, neutral_pair=replace(declared.neutral_pair, candidate=changed))
    raw["entries"][0]["declaration"] = declared.to_dict()
    request["serving_preparation"] = pin(Path(request["serving_preparation"]["path"]), raw)
    with pytest.raises(standalone_inputs.StandaloneInputsRefused, match="incompatible native recipes"):
        startup_factory.build_startup(request, output_dir=tmp_path / "rejected-bundle")
