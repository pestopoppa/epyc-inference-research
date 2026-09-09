"""Build closed standalone startup inputs offline from pinned enrollment artifacts.

This CLI creates only its explicitly requested, new output bundle. It never enters
a campaign store, resolves provider callbacks, acquires claims, or launches work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence

from .. import schemas
from . import campaign, campaign_cli, experiment_plan, planned_serving, production_enrollment
from . import scheduling, scoped_evidence, unified_driver, unified_planner

REQUEST_SCHEMA = "epyc.autokernel.startup_factory_request.v1"
FEED_REQUEST_SCHEMA = "epyc.autokernel.startup_factory_request.v2"
RECEIPT_SCHEMA = "epyc.autokernel.startup_factory_receipt.v1"
MAX_INPUT_BYTES = 4 * 1024 * 1024
TARGET_FIELDS = {"profile", "profile_request", "execution", "runtime_dimensions"}


class StartupFactoryRefused(ValueError):
    """A required choice, artifact pin, or typed binding is absent or inconsistent."""


def _closed(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise StartupFactoryRefused(f"{label}: missing or unknown fields")
    return dict(value)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode()


def _read(path: Path, label: str) -> bytes:
    if not path.is_absolute():
        raise StartupFactoryRefused(f"{label}: path must be absolute")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_uid != os.getuid() or not 0 < before.st_size <= MAX_INPUT_BYTES):
            raise StartupFactoryRefused(f"{label}: unsafe file identity or size")
        chunks = []
        remaining = MAX_INPUT_BYTES + 1
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    def identity(info: os.stat_result) -> tuple[int, ...]:
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
                info.st_uid, info.st_nlink, info.st_mode)
    if len(raw) > MAX_INPUT_BYTES or identity(before) != identity(after) \
            or identity(after) != identity(named):
        raise StartupFactoryRefused(f"{label}: changed during bounded read")
    return raw


class _Pins:
    def __init__(self) -> None:
        self.records: dict[str, str] = {}

    def read(self, value: Any, label: str) -> Any:
        pin = _closed(value, {"path", "sha256"}, label)
        path = Path(pin["path"])
        raw = _read(path, label)
        observed = hashlib.sha256(raw).hexdigest()
        if observed != pin["sha256"]:
            raise StartupFactoryRefused(f"{label}: SHA-256 mismatch")
        previous = self.records.setdefault(str(path), observed)
        if previous != observed:
            raise StartupFactoryRefused(f"{label}: conflicting artifact pin")
        return json.loads(raw)

    def recheck(self) -> None:
        for path, expected in self.records.items():
            if hashlib.sha256(_read(Path(path), "input closure")).hexdigest() != expected:
                raise StartupFactoryRefused("input closure changed before bundle publication")


def _profile_request(target: campaign.TargetRevision, digest: str, spec: Any,
                     config: scheduling.SchedulerConfig) -> dict[str, Any]:
    row = _closed(spec, {"adapter_id", "adapter_digest", "estimated_duration_seconds",
                         "estimated_claims", "submitted_at"}, "profile request configuration")
    claims = scheduling.ResourceVector.from_dict(row["estimated_claims"])
    if (claims.physical_region_fraction > config.capacity.physical_region_fraction
            or not set(claims.gpu_devices) <= set(config.capacity.gpu_devices)
            or claims.memory_reservation_bytes > config.capacity.memory_reservation_bytes
            or row["estimated_duration_seconds"] > config.max_stage_seconds
            or (target.execution.backend == "cpu" and claims.gpu_devices)
            or (target.execution.backend == "gpu" and not claims.gpu_devices)):
        raise StartupFactoryRefused("profile request exceeds configured capacity/budget or backend")
    production = "production" in target.enrolled_as
    stage = scheduling.StageProposal(
        proposal_id="profile:" + digest, submitted_at=row["submitted_at"],
        backend=target.execution.backend, target_revision=digest, alias_identity=digest,
        frontier_id="production:" + digest if production else None,
        production_frontier=production,
        seed_id="seed:" + digest if "seed" in target.enrolled_as else None,
        stage_class="prerequisite", estimated_duration_seconds=row["estimated_duration_seconds"],
        estimated_claims=claims, eligible=True,
        eligibility_ref="profile_configuration:" + row["adapter_digest"],
        reservation_kind=None, full_region=claims.physical_region_fraction == 1.0,
        compatibility_authority_refs=(), safe_chunking_declared=False)
    return unified_driver.ProfilePreparationRequest.from_dict({
        "schema": unified_driver.PROFILE_REQUEST_SCHEMA, "target_revision_digest": digest,
        "stage_proposal": stage.to_dict(), "profile_contract": {
            "schema": unified_driver.PROFILE_CONTRACT_SCHEMA,
            "adapter_id": row["adapter_id"], "adapter_digest": row["adapter_digest"]},
    }).to_dict()


def _write_new(directory: Path, name: str, value: Any) -> str:
    raw = _json_bytes(value)
    descriptor = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(raw)
        while view:
            view = view[os.write(descriptor, view):]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def build_startup(request: Mapping[str, Any], *, output_dir: Path) -> dict[str, Any]:
    """Write a new startup bundle from explicit choices and verified small inputs."""
    from . import standalone_inputs

    fields = {"schema", "resolved_export", "production_export", "candidate_target_ids",
              "store_path", "config_generation", "scheduler_config", "scheduler_state",
              "environment_policy", "target_defaults", "targets", "experiment_plans",
              "evidence_index", "actor_identities", "providers", "native_artifact_sink_ref",
              "dry_run_runner"}
    feed_mode = request.get("schema") == FEED_REQUEST_SCHEMA
    if feed_mode:
        fields = (fields - {"evidence_index"}) | {"evidence_feed"}
    request = _closed(request, fields, "factory request")
    if request["schema"] not in {REQUEST_SCHEMA, FEED_REQUEST_SCHEMA}:
        raise StartupFactoryRefused("unsupported factory request schema")
    output_dir = Path(output_dir)
    store_path = Path(request["store_path"])
    if not output_dir.is_absolute() or output_dir.exists() or not output_dir.parent.is_dir():
        raise StartupFactoryRefused("output directory must be a new absolute path with an existing parent")
    resolved_output, resolved_store = output_dir.resolve(), store_path.resolve()
    if (not store_path.is_absolute() or resolved_store == resolved_output
            or resolved_output in resolved_store.parents or resolved_store in resolved_output.parents):
        raise StartupFactoryRefused("campaign store must be absolute and outside the output bundle")
    pins = _Pins()
    envelope = pins.read(request["resolved_export"], "campaign CLI v2 output")
    envelope = _closed(envelope, {"schema", "mode", "admission_ready", "disposition",
        "resolved_campaign", "target_dispositions", "summary", "verification",
        "production_enrollment"}, "campaign CLI v2 output")
    if (envelope["schema"] != campaign_cli.PRODUCTION_DRY_RESOLUTION_SCHEMA
            or envelope["mode"] != "offline_dry_resolution"
            or envelope["admission_ready"] is not False):
        raise StartupFactoryRefused("expected non-authoritative campaign CLI v2 output")
    raw_export = _closed(pins.read(request["production_export"], "sealed production export"),
        {"schema", "context", "targets", "disposition", "export_sha256"}, "sealed production export")
    if (raw_export["schema"] != production_enrollment.EXPORT_SCHEMA
            or raw_export["export_sha256"] != production_enrollment._digest(
                {key: value for key, value in raw_export.items() if key != "export_sha256"})
            or not isinstance(raw_export["targets"], list)):
        raise StartupFactoryRefused("sealed production export integrity/schema differs")
    # Bound sidecar reads before the existing production loader reopens them.
    # Their canonical launch meaning remains exclusively the loader's contract.
    for target in raw_export["targets"]:
        if not isinstance(target, Mapping) or not isinstance(target.get("artifacts", []), list):
            raise StartupFactoryRefused("production target artifacts must be an array")
        for artifact in target.get("artifacts", []):
            if not isinstance(artifact, Mapping):
                raise StartupFactoryRefused("production artifact must be an object")
            if artifact.get("use") == "recipe":
                pins.read({"path": artifact["path"], "sha256": artifact["sha256"]},
                          "production recipe sidecar")
    export = production_enrollment.load_export(raw_export)
    if envelope["production_enrollment"] != production_enrollment.production_enrollment_diagnostics(export):
        raise StartupFactoryRefused("campaign CLI output differs from the sealed production export")
    resolved = campaign.ResolvedCampaign.from_dict(envelope["resolved_campaign"])
    registry = production_enrollment.registry_snapshot_from_export(export)
    expected_sources = {row["name"]: "production-source:" + row["name"] + ":" + row["revision"]
                        for row in export["context"]["sources"]}
    if dict(resolved.source_refs) != expected_sources or any(
            identity is None or identity.to_dict() != registry["source"].get(expected_sources[name])
            for name, identity in resolved.source_snapshot):
        raise StartupFactoryRefused("resolved source pins differ from the production export")
    exported_ids = {row["target_id"] for row in export["targets"]}
    aliases = {alias: target for target in resolved.targets for alias in target.target_ids}
    if len(aliases) != sum(len(target.target_ids) for target in resolved.targets):
        raise StartupFactoryRefused("resolved target IDs are not unique")
    projected = {alias for row in envelope["production_enrollment"]["targets"]
                 for alias in row["enrolled_target_ids"]}
    if not projected <= set(aliases):
        raise StartupFactoryRefused("resolved campaign omits exported targets")
    for row in export["targets"]:
        if row["target_id"] not in projected:
            continue
        role = "seed" if row.get("optional_seed", False) else "production"
        target = aliases[row["target_id"]]
        if (role not in target.enrolled_as
                or not set(row.get("obligations", ())) <= set(target.required_obligations)):
            raise StartupFactoryRefused("resolved target enrollment/obligations differ from export")
        for use, identity in (("model", target.execution.model),
                              ("executable", target.execution.build),
                              ("recipe", target.execution.recipe),
                              ("drafter", target.execution.drafter)):
            expected = [item for item in row.get("artifacts", ()) if item["use"] == use]
            if identity is not None and (len(expected) != 1 or
                    (identity.path, identity.sha256) != (expected[0]["path"], expected[0]["sha256"])):
                raise StartupFactoryRefused("resolved production artifact pin differs from export")
    candidate_ids = request["candidate_target_ids"]
    if (not isinstance(candidate_ids, list)
            or any(not isinstance(alias, str) for alias in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)
            or any(alias not in aliases or "seed" not in aliases[alias].enrolled_as
                   for alias in candidate_ids)):
        raise StartupFactoryRefused("explicit candidate target IDs must identify enrolled seeds")
    if not isinstance(request["targets"], Mapping) or set(request["targets"]) - set(aliases):
        raise StartupFactoryRefused("target configuration names an unknown enrolled target")
    defaults = request["target_defaults"]
    if not isinstance(defaults, Mapping) or set(defaults) - {"cpu", "gpu"}:
        raise StartupFactoryRefused("target defaults must be explicitly keyed by backend")
    config = scheduling.SchedulerConfig.from_dict(request["scheduler_config"])
    if (config.config_id != resolved.campaign_id
            or config.max_stage_seconds > resolved.resources.stage_timeout_s
            or not set(config.capacity.gpu_devices) <= set(resolved.resources.gpu_ids)):
        raise StartupFactoryRefused("scheduler differs from the resolved campaign resource envelope")
    if request["scheduler_state"] is None:
        if store_path.exists():
            raise StartupFactoryRefused("existing campaign store requires an explicit scheduler state pin")
        state = scheduling.initial_state(config, resolved.campaign_id)
    else:
        state = scheduling.SchedulerState.from_dict(pins.read(request["scheduler_state"], "scheduler state"))
    if state.scheduler_id != resolved.campaign_id:
        raise StartupFactoryRefused("scheduler state belongs to another campaign")
    scheduling.SchedulerEngine(config, state)
    evidence = None
    feed = None
    if feed_mode:
        from .feed_runtime import FeedConfig, validate_paths
        feed = FeedConfig.from_dict(request["evidence_feed"])
        validate_paths(feed, store_path)
    else:
        evidence = scoped_evidence.EvidenceIndex.from_dict(
            pins.read(request["evidence_index"], "evidence index"))
    policy = unified_planner.EnvironmentPolicy.from_dict(request["environment_policy"])
    anchors, profiles, profile_requests, dimensions, executions = {}, {}, {}, {}, {}
    target_map = {}
    for target in resolved.targets:
        digest = unified_planner._target_digest(target)
        target_map.update({alias: digest for alias in target.target_ids})
        overrides = [request["targets"][alias] for alias in target.target_ids if alias in request["targets"]]
        if len(overrides) > 1:
            raise StartupFactoryRefused("deduplicated target has multiple alias configurations")
        supplied = overrides[0] if overrides else defaults.get(target.execution.backend)
        settings = _closed(supplied, TARGET_FIELDS, "target configuration")
        if settings["profile"] is not None and settings["profile_request"] is not None:
            raise StartupFactoryRefused("target must select an existing profile or preparation, not both")
        if target.status != "ready":
            continue
        recipe_identity = target.execution.recipe
        pins.read({"path": recipe_identity.path, "sha256": recipe_identity.sha256},
                  "enrolled recipe sidecar")
        production_ids = [alias for alias in target.target_ids if alias in exported_ids]
        if production_ids:
            anchors[digest] = unified_planner.RuntimeAnchor.from_dict({
                "schema": unified_planner.ANCHOR_SCHEMA, "target_revision_digest": digest,
                "target_id": production_ids[0], "production_export": export,
                "environment_policy": policy.to_dict()}).to_dict()
        else:
            if "production" in target.enrolled_as:
                raise StartupFactoryRefused("production target is absent from sealed export")
            recipe = target.execution.recipe
            anchors[digest] = unified_planner.LocalRuntimeAnchor.from_dict({
                "schema": unified_planner.LOCAL_ANCHOR_SCHEMA, "target_revision_digest": digest,
                "recipe_sidecar_path": recipe.path, "recipe_sidecar_sha256": recipe.sha256}).to_dict()
        dimensions[digest] = [unified_planner.RuntimeDimension.from_dict(item).to_dict()
                              for item in settings["runtime_dimensions"]]
        if settings["profile"] is not None:
            profile = unified_planner.TargetProfile.from_dict(pins.read(settings["profile"], "target profile"))
            if profile.target_revision_digest != digest:
                raise StartupFactoryRefused("profile differs from enrolled target")
            profiles[digest] = profile.to_dict()
        elif settings["profile_request"] is not None:
            profile_requests[digest] = _profile_request(target, digest, settings["profile_request"], config)
        if settings["execution"] is not None:
            execution = _closed(settings["execution"], {"prompt_manifest", "max_stage_seconds",
                                "teardown_seconds", "instrument_id"}, "execution configuration")
            prompts = planned_serving.FrozenPromptManifest.from_dict(pins.read(
                execution["prompt_manifest"], "prompt manifest"))
            parsed = unified_driver.ExecutionInput(
                digest, prompts, execution["max_stage_seconds"], execution["teardown_seconds"],
                execution["instrument_id"])
            if parsed.max_stage_seconds > config.max_stage_seconds:
                raise StartupFactoryRefused("execution input exceeds configured stage budget")
            executions[digest] = parsed.to_dict()
    prepared_anchors = unified_planner.prepare_runtime_anchors(resolved, anchors)
    if not isinstance(request["experiment_plans"], Mapping):
        raise StartupFactoryRefused("experiment plans must be an explicit mapping")
    plans = {key: experiment_plan.ExperimentPlan.from_dict(pins.read(pin, "experiment plan")).to_dict()
             for key, pin in request["experiment_plans"].items()}
    provider_fields = {"lifecycle", "readiness"}
    if not feed_mode:
        provider_fields.add("evidence_verifier")
    providers = _closed(request["providers"], provider_fields, "providers")
    driver_config = unified_driver.DriverConfig.from_dict({
        "schema": unified_driver.CONFIG_SCHEMA,
        "resolved_campaign_path": str(output_dir / "resolved-campaign.json"),
        "store_path": str(store_path), "config_generation": request["config_generation"],
        "scheduler_config": config.to_dict(), "scheduler_state": state.to_dict(),
        "runtime_anchors": anchors, "runtime_dimensions": dimensions, "profiles": profiles,
        "profile_requests": profile_requests, "experiment_plans": plans, "execution_inputs": executions,
        "native_artifact_sink_ref": request["native_artifact_sink_ref"],
    })
    config_body = {field: unified_driver._thaw(getattr(driver_config, field))
                   for field in driver_config.__dataclass_fields__}
    body = {"schema": standalone_inputs.MANIFEST_SCHEMA, "driver_config": config_body,
            "actor_identities": request["actor_identities"],
            "lifecycle_provider_id": providers["lifecycle"], "readiness_provider_id": providers["readiness"],
            }
    if feed is not None:
        body.update(schema=standalone_inputs.FEED_MANIFEST_SCHEMA,
                    evidence_feed=feed.to_dict())
    else:
        assert evidence is not None
        body.update(evidence_index=evidence.to_dict(),
                    evidence_verifier_id=providers["evidence_verifier"])
    manifest = standalone_inputs.StartupManifest.from_dict(
        body | {"manifest_digest": standalone_inputs._digest(body)})
    runner = _closed(request["dry_run_runner"], {"python", "pythonpath"}, "dry-run runner")
    interpreter, package_root = Path(runner["python"]), Path(runner["pythonpath"])
    if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise StartupFactoryRefused("dry-run interpreter must be an existing absolute executable")
    if not package_root.is_absolute():
        raise StartupFactoryRefused("dry-run package root must be absolute")
    runner_sources = {}
    for name in ("unified_driver", "standalone_inputs", "standalone_runtime"):
        path = package_root / "autokernel" / "loop" / (name + ".py")
        runner_sources[str(path)] = hashlib.sha256(_read(path, "dry-run source")).hexdigest()
    loaded_sources = {}
    for module in (sys.modules[__name__], standalone_inputs, unified_driver,
                   unified_planner, campaign_cli, production_enrollment):
        path = Path(module.__file__).resolve()
        loaded_sources[str(path)] = hashlib.sha256(_read(path, "loaded source")).hexdigest()
    pins.recheck()
    output_dir.mkdir(mode=0o700)
    files = {"resolved-campaign.json": _write_new(output_dir, "resolved-campaign.json", envelope)}
    materialized = standalone_inputs.materialize(manifest)
    report = materialized.preflight()
    files["startup.json"] = _write_new(output_dir, "startup.json", manifest.to_dict())
    receipt = {"schema": RECEIPT_SCHEMA, "manifest_digest": manifest.manifest_digest,
               "request_digest": schemas.content_hash(request),
               "input_pins": pins.records, "output_files": files, "target_revision_map": target_map,
               "loaded_sources": loaded_sources, "dry_run_sources": runner_sources,
               "artifact_verification": envelope["verification"],
               "candidate_target_ids": candidate_ids, "preflight": report,
               "anchor_prerequisites": {key: list(value) for key, value in prepared_anchors.runtime_prerequisites.items()},
               "target_dispositions": envelope["target_dispositions"], "execution_authorized": False,
               "command": {"argv": [str(interpreter), "-B", "-m", "autokernel.loop.unified_driver",
                                     "--config", str(output_dir / "startup.json"), "--dry-run"],
                           "PYTHONPATH": str(package_root)}}
    _write_new(output_dir, "factory-receipt.json", receipt)
    descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        request = json.loads(_read(args.request, "factory request"))
        result = build_startup(request, output_dir=args.out_dir)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError) as exc:
        print(f"startup factory refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
