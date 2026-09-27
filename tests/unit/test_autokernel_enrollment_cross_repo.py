"""Optional end-to-end contract with the research consumer.

Run with both repository roots on PYTHONPATH. A standalone orchestrator checkout skips
this test instead of importing a copied research schema.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from unittest import mock

import pytest
import yaml

production = pytest.importorskip("autokernel.loop.production_enrollment")
resolved_recipe = pytest.importorskip("autokernel.loop.resolved_recipe")
serving = pytest.importorskip("autokernel.loop.serving")
campaign_cli = pytest.importorskip("autokernel.loop.campaign_cli")
campaign = pytest.importorskip("autokernel.loop.campaign")

from scripts.server import orchestrator_stack as launcher
from scripts.server.autokernel_enrollment import (
    ArtifactPin, capture_current_context, export_production_enrollment, seal_export_bundle)

ROOT = Path(__file__).resolve().parents[2]


def _fresh_prior(tmp_path: Path, monkeypatch) -> None:
    prior = yaml.safe_load((ROOT / "orchestration/derived/stack_priors.yaml").read_text())
    sources = {
        "registry": ROOT / "orchestration/model_registry.yaml",
        "descriptors": ROOT / "orchestration/model_descriptors.yaml",
        "launch_manifest": ROOT / "orchestration/launch_manifest.yaml",
        "stack_topology": ROOT / "orchestration/stack_topology.yaml",
        "stack_runtime": ROOT / "scripts/server/stack_runtime.py",
        "stack_paths": ROOT / "scripts/server/stack_paths.py",
    }
    for name, path in sources.items():
        prior["source_artifacts"][name]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = tmp_path / "stack-priors.yaml"
    path.write_text(yaml.safe_dump(prior, sort_keys=False))
    monkeypatch.setattr(launcher, "STACK_PRIORS_PATH", path)


def test_actual_cpu_gpu_export_resolves_reaches_campaign_cli_and_fake_measurement(
        tmp_path, monkeypatch, capsys):
    _fresh_prior(tmp_path, monkeypatch)
    context = capture_current_context(
        master_registry=ROOT / "orchestration/model_registry.yaml",
        revision="caller-declared-test", instance_mode="full")
    first = export_production_enrollment(context)
    pins = []
    # The GPU leg is the MI210 27B on :8083. Its primary role is architect_critic
    # since the 2026-09-27 ARCHITECT SWAP (target_id architect_general@8083 before).
    for role in ("frontdoor", "architect_critic"):
        row = next(item for item in first["targets"] if item["primary_role"] == role)
        command = row["command_argv"]
        pins.extend((
            ArtifactPin("executable", command[0], hashlib.sha256(f"{role}:exe".encode()).hexdigest()),
            ArtifactPin("model", command[2], hashlib.sha256(f"{role}:model".encode()).hexdigest()),
        ))
        directories = ({str(Path(command[0]).parent)}
                       | set(row["runtime_requirements"]["ld_library_path"] or []))
        for index, directory in enumerate(sorted(directories)):
            pins.append(ArtifactPin(
                "dso", str(Path(directory) / f"lib-{role}-{index}.so"),
                hashlib.sha256(f"{role}:dso:{index}".encode()).hexdigest()))
    export = export_production_enrollment(replace(context, artifacts=tuple(pins)))
    selected = [row for row in export["targets"]
                if row["primary_role"] in {"frontdoor", "architect_critic"}]
    allowed_env = sorted({key for row in selected for key in row["environment"]
                          if key != "LD_LIBRARY_PATH"})
    policy = {"schema": resolved_recipe.ENVIRONMENT_POLICY_SCHEMA,
              "version": "cross-repo-test", "measurement_keys": [],
              "allowed_inherit_keys": allowed_env, "witnesses": {}}
    export_path = tmp_path / "production-enrollment.json"
    export = seal_export_bundle(export, export_path)
    result = production.resolve_exported_recipes(export_path, environment_policy=policy)
    config = {
        "schema": production.CAMPAIGN_CONFIG_SCHEMA,
        "campaign_id": "cross-repo", "request_id": "cross-repo-1",
        "resources": {"schema": campaign.RESOURCE_SCHEMA, "cpu_logical": [0],
                      "gpu_ids": ["ROCm0"], "stage_timeout_s": 60,
                      "build_timeout_s": 60, "build_jobs": 1, "max_builds": 1},
        "objective_ref": "objective/aggregate-throughput-v1",
        "actors": {"planner": "fixture"}, "fallbacks": {"planner": []},
        "metric": "aggregate_tok_s", "metric_direction": "higher",
    }
    config_path = tmp_path / "campaign-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert campaign_cli.main([
        "--production-campaign-config", str(config_path),
        "--production-enrollment", str(export_path)]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["schema"] == campaign_cli.PRODUCTION_DRY_RESOLUTION_SCHEMA
    assert dry["admission_ready"] is False
    diagnostics = {row["target_id"]: row
                   for row in dry["production_enrollment"]["targets"]}
    assert diagnostics["speech:whisper"]["status"] == "unsupported"
    assert diagnostics["speech:tts"]["status"] == "unsupported"
    assert {"frontdoor@8070", "architect_critic@8083"}.issubset(
        {target_id for target in dry["resolved_campaign"]["targets"]
         for target_id in target["target_ids"]})

    class Process:
        pid = 99
        returncode = None
        def poll(self): return None
        def terminate(self): return None
        def wait(self, timeout): return 0
        def kill(self): raise AssertionError("unexpected kill")

    class Sampler:
        proof = {"samples": 2, "vram_reads": 2, "resident": True,
                 "peak_vram_bytes": 2**30, "median_vram_bytes": 2**30,
                 "peak_kfd_processes": 1, "sclk_min_mhz": 1000,
                 "sclk_max_mhz": 1000, "clock_stable": True}
        def __enter__(self): return self
        def __exit__(self, *_): return False

    class Response:
        def __init__(self, body): self.body = body
        def read(self): return self.body

    def urlopen(request, timeout):
        if isinstance(request, str):
            return Response(b"ok")
        return Response(json.dumps({"stop": True, "timings": {
            "predicted_n": 256, "predicted_per_second": 10.0}}).encode())

    for target_id in ("frontdoor@8070", "architect_critic@8083"):
        item = next(row for row in result["targets"] if row["target_id"] == target_id)
        assert item["status"] == "resolved"
        frozen = resolved_recipe.resolved_recipe_from_dict(item["resolved_recipe"])
        assert dict(frozen.provenance)["export_sha256"] == export["export_sha256"]
        campaign_target = next(target for target in dry["resolved_campaign"]["targets"]
                               if target_id in target["target_ids"])
        sealed_recipe = next(artifact for artifact in export["targets"]
                             if artifact["target_id"] == target_id)
        sealed_recipe = next(artifact for artifact in sealed_recipe["artifacts"]
                             if artifact["use"] == "recipe")
        assert campaign_target["execution"]["recipe"]["sha256"] == sealed_recipe["sha256"]
        frozen.validate_launch(frozen.template, frozen.build_dir, frozen.port)
        requests = tuple((f"p{i}", json.dumps({"prompt": str(i)}).encode())
                         for i in range(frozen.template.np))
        with mock.patch.object(serving.subprocess, "Popen", return_value=Process()) as popen, \
             mock.patch.object(serving.residency, "Sampler", return_value=Sampler()), \
             mock.patch.object(serving.urllib.request, "urlopen", side_effect=urlopen), \
             mock.patch.object(serving, "verify_env_readback"):
            assert serving._measure_once(
                frozen.template, Path(frozen.build_dir), frozen.port,
                frozen_requests=requests, resolved_recipe=frozen) == 10.0 * len(requests)
        assert popen.call_args.args[0] == list(frozen.argv)
