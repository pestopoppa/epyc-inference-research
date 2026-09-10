from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest

from .production_enrollment import (
    CAMPAIGN_CONFIG_SCHEMA,
    EXPORT_SCHEMA,
    ProductionEnrollmentError,
    load_export,
    manifest_from_export,
    production_enrollment_diagnostics,
    registry_snapshot_from_export,
    resolve_exported_recipes,
)
from . import production_enrollment as pe
from . import resolved_recipe as rr
from .resolved_recipe import CanonicalResolvedRecipe, ENVIRONMENT_POLICY_SCHEMA
from . import serving


def _export(tmp_path: Path, *, compatible: bool = False, backend: str = "cpu") -> dict:
    build = tmp_path / "build"
    model = tmp_path / "model.gguf"
    executable = build / "bin" / "llama-server"
    dso = build / "bin" / "libggml.so"
    device = "none" if backend == "cpu" else "ROCm0"
    command = [str(executable), "-m", str(model), "--host", "127.0.0.1",
               "--port", "8070", "-np", "2", "-c", "4096", "-t", "4",
               "-ub", "2048", "--flash-attn", "on", "--jinja", "-ctk", "q8_0",
               "-ctv", "q8_0", "--no-mmap", "--spec-type", "draft-mtp",
               "--spec-draft-n-max", "4", "--reasoning", "off",
               "--slot-save-path", str(tmp_path / "slots"), "--device", device,
               "--device-draft", device]
    if backend == "gpu":
        command[command.index("--device"):command.index("--device")] = ["-ngl", "all"]
    prefix = ["numactl", "--interleave=all", "--", "taskset", "-c", "0-3"]
    environment = {"LD_LIBRARY_PATH": str(build / "bin"), "OMP_PROC_BIND": "spread"}
    target = {
        "target_id": "frontdoor@8070", "status": "ready", "backend": backend,
        "primary_role": "frontdoor", "aliases": ["worker_summarize"],
        "obligations": ["frontdoor", "worker_summarize"], "optional_seed": False,
        "workload": {"np": "2", "context": "4096", "threads": "4"},
        "speculation": "draft-mtp",
        "port": 8070, "argv": prefix + command, "command_argv": command,
        "environment": environment,
        "environment_unsets": [], "numa_instance": 0,
        "topology": {"argv_prefix": prefix},
        "runtime_requirements": {"binary_dir": None, "ld_library_path": []},
        "source_revisions": {"launcher": "rev"},
        "source_revision_kinds": {"launcher": "caller_declared"},
        "artifacts": [
            {"use": "model", "path": str(model), "sha256": "1" * 64},
            {"use": "executable", "path": str(executable), "sha256": "2" * 64},
            {"use": "dso", "path": str(dso), "sha256": "3" * 64}],
    }
    del compatible  # retained for call-site compatibility with the initial fixture
    body = {
        "schema": EXPORT_SCHEMA,
        "context": {"instance_mode": "full",
                    "sources": [{"name": "launcher", "path": str(tmp_path / "source.py"),
                                 "sha256": "4" * 64, "revision": "rev"}]},
        "targets": [target],
        "disposition": {"ready": 1, "waiting_artifact": 0, "unsupported": 0},
    }
    body["export_sha256"] = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return body


def _seal_recipe_artifacts(export: dict, tmp_path: Path) -> dict:
    for target in export["targets"]:
        if target.get("backend") not in {"cpu", "gpu"} or not target.get("command_argv"):
            continue
        raw = (json.dumps(pe._recipe_body(target), sort_keys=True,
                          separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        path = tmp_path / f"{digest}.recipe.json"
        path.write_bytes(raw)
        target.setdefault("artifacts", []).append(
            {"use": "recipe", "path": str(path), "sha256": digest})
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return export


def _policy() -> dict:
    return {"schema": ENVIRONMENT_POLICY_SCHEMA, "version": "test-v1",
            "measurement_keys": [], "allowed_inherit_keys": ["OMP_PROC_BIND"],
            "witnesses": {}}


def _campaign_config(backend: str = "cpu") -> dict:
    return {"schema": CAMPAIGN_CONFIG_SCHEMA, "campaign_id": "production-test",
            "request_id": "request-1",
            "resources": {"schema": "epyc.autokernel.resource_request.v1",
                          "cpu_logical": [0, 1, 2, 3],
                          "gpu_ids": ["ROCm0"] if backend == "gpu" else [],
                          "stage_timeout_s": 60, "build_timeout_s": 60,
                          "build_jobs": 1, "max_builds": 1},
            "objective_ref": "objective/aggregate-throughput-v1",
            "actors": {"planner": "planner"}, "fallbacks": {"planner": []},
            "metric": "aggregate_tok_s", "metric_direction": "higher"}


@pytest.mark.parametrize("extra", [[], ["--draft-p-min", "0.0", "--threads-draft", "16"],
                                   ["--log-colors", "off"]])
def test_production_optional_flags_and_implicit_ubatch(tmp_path: Path, extra):
    export = _export(tmp_path)
    target = export["targets"][0]
    command = target["command_argv"]
    index = command.index("-ub")
    del command[index:index + 2]
    command.extend(extra)
    target["argv"] = target["topology"]["argv_prefix"] + command
    export = _seal_recipe_artifacts(export, tmp_path)
    row = resolve_exported_recipes(export, environment_policy=_policy())["targets"][0]
    assert row["status"] == "resolved"
    resolved = CanonicalResolvedRecipe.from_dict(row["resolved_recipe"])
    assert resolved.template.ubatch == 512
    assert list(resolved.command_argv) == command
    assert all(flag in resolved.template.extra_flags for flag in extra[::2])


@pytest.mark.parametrize("extra", [["--draft-p-min", "nan"],
    ["--draft-p-min", "0", "--spec-draft-p-min", "0"],
    ["--threads-draft", "bad"], ["--log-colors", "unknown"]])
def test_production_optional_flags_reject_invalid_values(tmp_path: Path, extra):
    command = _export(tmp_path)["targets"][0]["command_argv"] + extra
    with pytest.raises(rr.ResolutionError):
        rr.canonical_recipe_projection(name="invalid", command_argv=command,
                                       topology_prefix=["taskset", "-c", "0-3"])


def test_split_model_entry_is_not_overwritten_by_last_shard(tmp_path: Path):
    export = _export(tmp_path)
    target = export["targets"][0]
    original = target["artifacts"][0]
    first = str(tmp_path / "model-00001-of-00003.gguf")
    original["path"] = first
    target["command_argv"][2] = first
    target["argv"] = target["topology"]["argv_prefix"] + target["command_argv"]
    target["artifacts"].extend({"use": "model",
        "path": str(tmp_path / f"model-{i:05d}-of-00003.gguf"), "sha256": str(i) * 64}
        for i in (2, 3))
    export = _seal_recipe_artifacts(export, tmp_path)
    registry = registry_snapshot_from_export(export)
    assert registry["model"]["production:frontdoor@8070:model"]["path"] == first
    resolved = resolve_exported_recipes(export, environment_policy=_policy())
    assert resolved["targets"][0]["status"] == "resolved"
    assert len([a for a in load_export(export)["targets"][0]["artifacts"]
                if a["use"] == "model"]) == 3


def test_integrity_and_registry_projection(tmp_path: Path):
    export = _export(tmp_path)
    registry = registry_snapshot_from_export(export)
    assert "production-source:launcher:rev" in registry["source"]
    assert registry["recipe"]["production:frontdoor@8070:recipe"]["status"] == \
        "unsupported_capability"
    export["targets"][0]["argv"].append("--moved")
    with pytest.raises(ProductionEnrollmentError, match="integrity"):
        load_export(export)


@pytest.mark.parametrize(
    ("producer_status", "expected_status"),
    [("unsupported", "unsupported_capability"),
     ("waiting_artifact", "missing_artifact")],
)
def test_sealed_recipe_does_not_erase_producer_prerequisite_disposition(
        tmp_path: Path, producer_status: str, expected_status: str):
    export = _seal_recipe_artifacts(_export(tmp_path), tmp_path)
    export["targets"][0]["status"] = producer_status
    export["targets"][0]["reasons"] = [f"producer_{producer_status}"]
    export["disposition"] = {
        "ready": 0,
        "waiting_artifact": int(producer_status == "waiting_artifact"),
        "unsupported": int(producer_status == "unsupported"),
    }
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    from . import campaign
    manifest = manifest_from_export(export, campaign_config=_campaign_config())
    resolved = campaign.resolve_manifest(
        manifest, registry_snapshot=registry_snapshot_from_export(export))

    assert resolved.targets[0].status == expected_status
    assert resolved.targets[0].status != "ready"


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_canonical_cpu_and_gpu_roundtrip_validate_and_reach_measure_once(
        tmp_path: Path, backend: str):
    result = resolve_exported_recipes(_export(tmp_path, compatible=True, backend=backend),
                                      environment_policy=_policy())
    row = result["targets"][0]
    assert row["status"] == "resolved"
    assert row["resolved_recipe"]["argv"] == row["source_argv"]
    assert row["resolved_recipe"]["launch_env"] == row["source_environment"]
    loaded = CanonicalResolvedRecipe.from_dict(row["resolved_recipe"])
    loaded.validate_launch(loaded.template, loaded.build_dir, loaded.port)
    assert result["admission_ready"] is False

    class FakeProcess:
        pid = 123
        returncode = None
        def poll(self): return None
        def terminate(self): return None
        def wait(self, timeout): return 0
        def kill(self): raise AssertionError("unexpected kill")

    class FakeSampler:
        proof = {"samples": 2, "vram_reads": 2, "resident": True,
                 "peak_vram_bytes": 2**30, "median_vram_bytes": 2**30,
                 "peak_kfd_processes": 1, "sclk_min_mhz": 1000,
                 "sclk_max_mhz": 1000, "clock_stable": True}
        def __enter__(self): return self
        def __exit__(self, *_): return False

    class Response:
        def __init__(self, body): self._body = body
        def read(self): return self._body

    def urlopen(request, timeout):
        if isinstance(request, str):
            return Response(b"ok")
        return Response(json.dumps({"stop": True, "timings": {
            "predicted_n": 256, "predicted_per_second": 10.0}}).encode())

    requests = (("p0", b'{"prompt":"a"}'), ("p1", b'{"prompt":"b"}'))
    with mock.patch.object(serving.subprocess, "Popen", return_value=FakeProcess()) as popen, \
         mock.patch.object(serving.residency, "Sampler", return_value=FakeSampler()), \
         mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen), \
         mock.patch.object(serving, "verify_env_readback"):
        assert serving._measure_once(
            loaded.template, Path(loaded.build_dir), loaded.port,
            frozen_requests=requests, resolved_recipe=loaded) == 20.0
    assert popen.call_args.args[0] == list(loaded.argv)
    assert popen.call_args.kwargs["env"] == dict(loaded.launch_env)


def test_missing_canonical_command_is_typed_per_target(tmp_path: Path):
    export = _export(tmp_path)
    del export["targets"][0]["command_argv"]
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    row = resolve_exported_recipes(export, environment_policy=_policy())["targets"][0]
    assert row["status"] == "unsupported"
    assert row["reason"].startswith("not_exactly_resolvable")


def test_manifest_factory_derives_alias_obligations_without_target_transcription(tmp_path: Path):
    manifest = manifest_from_export(_export(tmp_path), campaign_config=_campaign_config())
    assert len(manifest.production) == 1
    target = manifest.production[0]
    assert target.roles == ("worker_summarize", "frontdoor")
    assert target.required_obligations == ("frontdoor", "worker_summarize")
    assert target.model_ref == "production:frontdoor@8070:model"


def test_malformed_nonfinite_and_unknown_schema_fail_closed(tmp_path: Path):
    with pytest.raises(ProductionEnrollmentError, match="object"):
        load_export([])
    malformed = _export(tmp_path)
    malformed["disposition"]["ready"] = float("nan")
    with pytest.raises(ProductionEnrollmentError, match="non-finite"):
        load_export(malformed)
    with pytest.raises(rr.ResolutionError, match="unsupported schema"):
        rr.resolved_recipe_from_dict({"schema": "legacy"})


@pytest.mark.parametrize("mutation,reason", [
    (lambda row: row["command_argv"].extend(["-np", "9"]), "not_exactly_resolvable"),
    (lambda row: row["environment"].update({"AUTH_TOKEN": "secret"}),
     "not_exactly_resolvable"),
])
def test_duplicate_execution_flag_and_secret_environment_are_per_target_refusals(
        tmp_path: Path, mutation, reason):
    export = _export(tmp_path)
    mutation(export["targets"][0])
    export["targets"][0]["argv"] = (export["targets"][0]["topology"]["argv_prefix"]
                                      + export["targets"][0]["command_argv"])
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    row = resolve_exported_recipes(export, environment_policy=_policy())["targets"][0]
    assert row["status"] == "unsupported"
    assert row["reason"].startswith(reason)


def test_direct_canonical_object_mutation_cannot_reuse_integrity_hashes(tmp_path: Path):
    row = resolve_exported_recipes(_export(tmp_path), environment_policy=_policy())["targets"][0]
    resolved = CanonicalResolvedRecipe.from_dict(row["resolved_recipe"])
    forged = replace(resolved, topology_prefix=("taskset", "-c", "9"),
                     argv=("taskset", "-c", "9") + resolved.command_argv)
    with pytest.raises(rr.ResolutionError, match="integrity"):
        forged.validate_launch(forged.template, forged.build_dir, forged.port)


@pytest.mark.parametrize("alteration", ["kv", "thread_batch", "draft_device"])
def test_rehashed_canonical_command_must_match_complete_semantic_projection(
        tmp_path: Path, alteration: str):
    row = resolve_exported_recipes(_export(tmp_path),
                                   environment_policy=_policy())["targets"][0]
    frozen = CanonicalResolvedRecipe.from_dict(row["resolved_recipe"])
    command = list(frozen.command_argv)
    if alteration == "kv":
        command.append("--kv-unified")
    elif alteration == "thread_batch":
        command.extend(["-tb", "999"])
    else:
        command[command.index("--device-draft") + 1] = "ROCm0"
    changed = replace(frozen, command_argv=tuple(command),
                      argv=frozen.topology_prefix + tuple(command))
    changed = replace(changed, snapshot_digest=rr._digest(changed._snapshot_dict()),
                      execution_digest=rr._digest(changed._normalized_execution_dict()))
    with pytest.raises(rr.ResolutionError):
        CanonicalResolvedRecipe.from_dict(changed.to_dict())


@pytest.mark.parametrize("alteration", ["unknown_numa", "flash_alias"])
def test_unknown_numa_policy_and_flash_alias_conflict_are_scoped_refusals(
        tmp_path: Path, alteration: str):
    export = _export(tmp_path)
    target = export["targets"][0]
    if alteration == "unknown_numa":
        target["topology"]["argv_prefix"][1] = "--unknown-policy=all"
    else:
        target["command_argv"].extend(["-fa", "off"])
    target["argv"] = target["topology"]["argv_prefix"] + target["command_argv"]
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    row = resolve_exported_recipes(export, environment_policy=_policy())["targets"][0]
    assert row["status"] == "unsupported"


def test_diagnostics_retain_non_campaign_targets_and_projection_failure(tmp_path: Path):
    export = _export(tmp_path)
    export["targets"].extend([
        {"target_id": "speech:whisper", "status": "unsupported",
         "reasons": ["speech_instrument_unsupported"], "argv": ["whisper"],
         "environment": {}, "aliases": [], "obligations": ["whisper"],
         "backend": "cpu"},
        {"target_id": "unknown-seed", "status": "unsupported",
         "reasons": ["role_not_in_selected_fleet"], "argv": [],
         "environment": {}, "aliases": [], "obligations": ["unknown-seed"],
         "backend": "unknown"},
    ])
    export["disposition"] = {"ready": 1, "waiting_artifact": 0, "unsupported": 2}
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    diagnostics = production_enrollment_diagnostics(export)
    by_id = {row["target_id"]: row for row in diagnostics["targets"]}
    assert by_id["frontdoor@8070"]["enrolled_target_ids"] == ["frontdoor@8070"]
    assert by_id["speech:whisper"]["status"] == "unsupported"
    assert by_id["speech:whisper"]["enrolled_target_ids"] == []
    assert "campaign_projection:workload_missing" in by_id["speech:whisper"]["reasons"]
    assert by_id["unknown-seed"]["enrolled_target_ids"] == []


def test_distinct_thread_recipes_have_distinct_actual_artifacts_and_do_not_alias(
        tmp_path: Path):
    export = _export(tmp_path)
    other = json.loads(json.dumps(export["targets"][0]))
    other["target_id"] = "frontdoor-quarter@8071"
    other["primary_role"] = "frontdoor-quarter"
    other["aliases"] = []
    other["obligations"] = ["frontdoor-quarter"]
    other["command_argv"][other["command_argv"].index("-t") + 1] = "8"
    other["workload"]["threads"] = "8"
    other["argv"] = other["topology"]["argv_prefix"] + other["command_argv"]
    export["targets"].append(other)
    export["disposition"]["ready"] = 2
    _seal_recipe_artifacts(export, tmp_path)
    path = tmp_path / "export.json"
    path.write_text(json.dumps(export), encoding="utf-8")
    manifest = manifest_from_export(path, campaign_config=_campaign_config())
    from . import campaign
    resolved = campaign.resolve_manifest(
        manifest, registry_snapshot=registry_snapshot_from_export(path))
    assert len(resolved.targets) == 2
    assert len({target.execution.recipe.sha256 for target in resolved.targets}) == 2


def test_recipe_sidecar_tamper_or_row_drift_is_refused(tmp_path: Path):
    export = _seal_recipe_artifacts(_export(tmp_path), tmp_path)
    recipe = next(item for item in export["targets"][0]["artifacts"]
                  if item["use"] == "recipe")
    path = Path(recipe["path"])
    original = path.read_bytes()
    path.write_bytes(b"tampered")
    with pytest.raises(ProductionEnrollmentError, match="SHA-256"):
        load_export(export)
    path.write_bytes(original)
    export["targets"][0]["workload"]["threads"] = "99"
    export["export_sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in export.items() if key != "export_sha256"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    with pytest.raises(ProductionEnrollmentError, match="differs from exported launch"):
        load_export(export)


def test_local_seed_cannot_reuse_opposite_backend_production_recipe(tmp_path: Path):
    config = _campaign_config()
    target = {
        "schema": "epyc.autokernel.target_spec.v1", "request_id": "request-1",
        "target_id": "future", "backend": "gpu", "model_ref": "future-model",
        "build_ref": "production:frontdoor@8070:executable",
        "recipe_ref": "production:frontdoor@8070:recipe", "context": 4096,
        "concurrency": 1, "speculation": "none", "env": {},
        "metric": "aggregate_tok_s", "metric_direction": "higher",
        "roles": ["future"], "required_obligations": [],
    }
    config["resources"]["gpu_ids"] = ["ROCm0"]
    config["local_seeds"] = {"targets": [target], "artifacts": {}}
    with pytest.raises(ProductionEnrollmentError, match="backend differs"):
        manifest_from_export(_export(tmp_path), campaign_config=config)
