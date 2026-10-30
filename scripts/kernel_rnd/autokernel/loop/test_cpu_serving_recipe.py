"""The real serving launcher consumes resolved CPU/GPU recipes without live compute."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from . import resolved_recipe as rr, serving
from .test_glm_frozen_requests import _canonical_launch
from .test_resolved_recipe import BUILD, _policy, _rehash, _resolve
from .test_serving_residency import _proof, _sampler_class


def _launch(recipe, resolved, proof, evidence=None):
    seen = {}

    class Process:
        pid = 4321
        returncode = 0

        def poll(self):
            return None

        def terminate(self):
            seen["terminated"] = True

        def wait(self, _timeout=None):
            return 0

    def popen(argv, **kwargs):
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        return Process()

    class Response:
        def read(self):
            return json.dumps({"timings": {
                "predicted_n": recipe.n_predict,
                "predicted_per_second": 25.0}}).encode()

    # Synthetic launch fixture; actual socket ownership has separate real-child controls.
    with mock.patch.object(serving, "require_free_port"), \
            mock.patch.object(serving, "ServerPortOwner"), \
            mock.patch.object(serving.subprocess, "Popen", side_effect=popen), \
            mock.patch.object(serving.urllib.request, "urlopen", return_value=Response()), \
            mock.patch.object(serving.residency, "Sampler", _sampler_class(proof)), \
            mock.patch.object(serving, "verify_env_readback") as verify:
        value = serving._measure_once(recipe, BUILD, 18311, evidence=evidence,
                                      resolved_recipe=resolved)
    return value, seen, verify


def test_cpu_zero_vram_is_not_a_gpu_veto_and_evidence_stays_unproven():
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0, np=2)
    resolved = _resolve(recipe, backend="cpu")
    evidence = []
    value, seen, _ = _launch(recipe, resolved, _proof(peak=0, median=0, kfd=0), evidence)
    assert value == 50.0
    assert seen["argv"] == list(resolved.argv)
    assert evidence[0]["backend"] == "cpu"
    assert evidence[0]["gpu_residency"] == serving.RESIDENCY_NOT_APPLICABLE
    assert evidence[0]["cpu_placement"] == "unproven"
    assert evidence[0]["contention"] == "unproven"


def test_numa_is_opt_in_identity_and_exact_cpu_launch_prefix():
    legacy = serving.Recipe(name="cpu", model="/m", device="none", ngl=0,
                            cpu_list="0-95", threads=48)
    old_row = legacy.to_dict()
    old_hash = legacy.recipe_hash
    assert "numa" not in old_row
    assert serving.Recipe.from_dict(old_row).to_dict() == old_row
    assert serving.Recipe.from_dict(old_row).recipe_hash == old_hash
    gpu = serving.Recipe(name="gpu", model="/m")
    assert "numa" not in gpu.to_dict()
    assert gpu.recipe_hash == (
        "7f2f98b728201ab769d86f99906582260137e8cc38117c4dc5ee5dcfc0a672b5")
    pinned = replace(legacy, numa="--interleave=0-3")
    assert pinned.to_dict()["numa"] == "--interleave=0-3"
    assert "numa=--interleave=0-3" in pinned.describe()
    assert serving.Recipe.from_dict(pinned.to_dict()).to_dict() == pinned.to_dict()
    assert pinned.recipe_hash != old_hash
    assert pinned.server_argv(BUILD, 18311)[:6] == [
        "numactl", "--interleave=0-3", "--", "taskset", "-c", "0-95"]
    resolved = _resolve(pinned, backend="cpu")
    assert rr.ResolvedRecipe.from_dict(resolved.to_dict()) == resolved
    value, seen, _ = _launch(pinned, resolved, _proof(peak=0, median=0, kfd=0))
    assert value == 25.0 * pinned.np
    assert seen["argv"][:6] == pinned.server_argv(BUILD, 18311)[:6]


@pytest.mark.parametrize("bad", ["interleave=all", "--interleave=", "--membind=all",
                                     "--membind=3;true", "--interleave=3-",
                                     "--interleave=3-1", "--membind=2-0",
                                     "--interleave=0,3-1", "--preferred=0", ""])
def test_numa_rejects_unvalidated_policy(bad):
    with pytest.raises(serving.RecipeError, match="numa"):
        serving.Recipe(name="cpu", model="/m", device="none", ngl=0,
                       cpu_list="0-95", numa=bad)


def test_numa_requires_cpu_recipe_and_affinity():
    for options in ({"device": "ROCm0", "ngl": 99, "cpu_list": "0-95"},
                    {"device": "none", "ngl": 1, "cpu_list": "0-95"},
                    {"device": "none", "ngl": 0, "cpu_list": None}):
        with pytest.raises(serving.RecipeError, match="CPU recipe"):
            serving.Recipe(name="bad", model="/m", numa="--interleave=all", **options)


@pytest.mark.parametrize("bad", ("--interleave=3-1", "--membind=2-0"))
def test_canonical_topology_rejects_descending_numa_ranges(bad):
    _, old = _canonical_launch(18311)
    for prefix in (("numactl", bad, "--", "taskset", "-c", "0-95"),
                   ("taskset", "-c", "0-95", "numactl", bad)):
        with pytest.raises(rr.ResolutionError, match="unsupported canonical numactl policy"):
            rr.canonical_recipe_projection(
                name="bad", command_argv=old.command_argv,
                topology_prefix=prefix, include_numa=True)


def test_legacy_canonical_cpu_prefix_does_not_upgrade_recipe_identity():
    template, old = _canonical_launch(18311)
    assert old.topology_prefix[:2] == ("numactl", "--interleave=all")
    assert "numa" not in template.to_dict()
    assert template.recipe_hash == (
        "42414714a17895ccfedff608d59d3e4df1bbb22a2706f12ad54a059c121b91eb")
    assert old.snapshot_digest == (
        "7cc3b513d68d681cb1c05495a81c6e5e8378bdc36d51c6a9ec78e6782e28891a")
    assert rr.CanonicalResolvedRecipe.from_dict(old.to_dict()).to_dict() == old.to_dict()
    replay = rr.resolve_canonical_launch(
        template, build_dir=old.build_dir, command_argv=old.command_argv,
        topology_prefix=old.topology_prefix, launch_environment=dict(old.launch_env),
        artifact_identities={"model": old.model.to_dict(),
                             "drafter": old.drafter.to_dict() if old.drafter else None,
                             "executable": old.executable.to_dict(),
                             "dsos": [item.to_dict() for item in old.dsos]},
        backend="cpu", environment_policy=old.environment_policy, port=old.port,
        runtime_binary_dir=old.runtime_binary_dir,
        runtime_ld_paths=old.runtime_ld_paths, provenance=dict(old.provenance))
    assert replay.template_hash == old.template_hash == template.recipe_hash
    assert replay.to_dict() == old.to_dict()


def test_explicit_canonical_numa_policy_must_match_resolved_prefix():
    legacy, old = _canonical_launch(18311)
    pinned = replace(legacy, numa="--interleave=all")
    assert rr.canonical_recipe_projection(
        name=legacy.name, command_argv=old.command_argv,
        topology_prefix=old.topology_prefix,
        n_predict=legacy.n_predict, temperature=legacy.temperature,
        top_p=legacy.top_p, top_k=legacy.top_k, metric=legacy.metric,
        include_numa=True).to_dict() == pinned.to_dict()
    kwargs = dict(build_dir=old.build_dir, command_argv=old.command_argv,
                  launch_environment=dict(old.launch_env),
                  artifact_identities={"model": old.model.to_dict(),
                                       "drafter": None,
                                       "executable": old.executable.to_dict(),
                                       "dsos": [item.to_dict() for item in old.dsos]},
                  backend="cpu", environment_policy=old.environment_policy, port=old.port,
                  runtime_binary_dir=old.runtime_binary_dir,
                  runtime_ld_paths=old.runtime_ld_paths, provenance=dict(old.provenance))
    resolved = rr.resolve_canonical_launch(
        pinned, topology_prefix=old.topology_prefix, **kwargs)
    assert resolved.template_hash == pinned.recipe_hash != old.template_hash
    assert rr.CanonicalResolvedRecipe.from_dict(resolved.to_dict()).to_dict() == resolved.to_dict()
    with pytest.raises(rr.ResolutionError, match="NUMA policy|semantic projection"):
        rr.resolve_canonical_launch(
            pinned, topology_prefix=("numactl", "--membind=0", "--",
                                     "taskset", "-c", "0-95"), **kwargs)
    with pytest.raises(rr.ResolutionError, match="NUMA policy|semantic projection"):
        rr.resolve_canonical_launch(
            pinned, topology_prefix=("taskset", "-c", "0-95"), **kwargs)


def test_gpu_zero_vram_still_refuses_and_forged_not_applicable_cannot_bypass():
    recipe = serving.Recipe(name="gpu", model="/m", np=2)
    resolved = _resolve(recipe)
    with pytest.raises(serving.ServingNotResident):
        _launch(recipe, resolved, _proof(peak=0, median=0, kfd=0), [])
    forged = {"status": serving.RESIDENCY_NOT_APPLICABLE, "sampled": True,
              "covers_request_phase": True, "samples": 2}
    with pytest.raises(serving.ServingNotResident):
        serving._refuse_if_not_resident(recipe, forged, backend="gpu")


def test_frozen_argv_environment_and_effective_readback_reach_popen():
    check = ({"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
              "expect": {"1": "0", serving.UNSET: "1"}},)
    recipe = serving.Recipe(name="gpu", model="/m", env_readback=check)
    parent = {"GGML_NOHUGEPAGE_PROCESS": "1"}
    policy = _policy("GGML_NOHUGEPAGE_PROCESS",
                     inherit=("GGML_NOHUGEPAGE_PROCESS",), witness="recipe_readback")
    resolved = _resolve(recipe, policy=policy, inherited=parent)
    parent["GGML_NOHUGEPAGE_PROCESS"] = "mutated"
    _, seen, verify = _launch(recipe, resolved, _proof())
    assert seen["argv"] == list(resolved.argv)
    assert seen["env"] == dict(resolved.launch_env)
    verify.assert_called_once_with(
        recipe, 4321, expectations=(("THP_enabled", "0"),))


def test_rehashed_env_forgery_still_refuses_before_popen_against_template():
    recipe = serving.Recipe(name="gpu", model="/m", env={"KNOB": "1"})
    resolved = _resolve(recipe, policy=_policy("KNOB"))
    launch = dict(resolved.launch_env)
    launch["KNOB"] = "2"
    forged = _rehash(rr.ResolvedRecipe(**{
        **resolved.__dict__, "launch_env": tuple(sorted(launch.items())),
        "relevant_environment": (("KNOB", "2"),)}))
    with mock.patch.object(serving.subprocess, "Popen") as popen:
        with pytest.raises(rr.ResolutionError, match="set values differ"):
            serving._measure_once(recipe, BUILD, 18311, resolved_recipe=forged)
    popen.assert_not_called()


def test_loader_really_removes_inherited_hsa_override_without_changing_recipe_hash():
    recipe = serving.Recipe(name="gpu", model="/m")
    legacy_hash = recipe.recipe_hash
    env = recipe.server_env(
        BUILD, base={"HSA_OVERRIDE_GFX_VERSION": "forbidden", "PATH": "/usr/bin"})
    assert "HSA_OVERRIDE_GFX_VERSION" not in env
    assert env["PATH"] == "/usr/bin"
    assert recipe.recipe_hash == legacy_hash


def test_unsupported_capability_and_forged_supported_record_refuse_before_popen():
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=9)
    resolved = _resolve(recipe, backend="cpu")
    with mock.patch.object(serving.subprocess, "Popen") as popen, \
            mock.patch.object(serving.residency, "Sampler") as sampler:
        with pytest.raises(rr.UnsupportedRecipeCapability):
            serving._measure_once(recipe, BUILD, 18311, resolved_recipe=resolved)
    popen.assert_not_called()
    sampler.assert_not_called()

    gpu = serving.Recipe(name="gpu", model="/m")
    valid = _resolve(gpu)
    replace_capability = rr.CapabilityReport(
        supported=True, backend="cpu", speculation=valid.capability.speculation,
        reasons=(), witnesses=valid.capability.witnesses,
        gpu_residency="not_applicable", cpu_placement="unproven", contention="unproven")
    forged = rr.ResolvedRecipe(**{**valid.__dict__, "backend": "cpu",
                                  "capability": replace_capability})
    with mock.patch.object(serving.subprocess, "Popen") as popen:
        with pytest.raises(rr.ResolutionError):
            serving._measure_once(gpu, BUILD, 18311, resolved_recipe=forged)
    popen.assert_not_called()


def test_write_floor_delegates_to_hardened_writer_and_propagates_fault():
    recipe = serving.Recipe(name="gpu", model="/m")
    row = {"recipe_hash": recipe.recipe_hash, "floor_pct": 1.0, "n": 5}
    with mock.patch.object(serving.status, "write_json", side_effect=OSError("fsync fault")) as write:
        with pytest.raises(OSError, match="fsync fault"):
            serving.write_floor(Path("/store"), recipe, row,
                                unit=serving.CALIBRATION_UNIT)
    assert write.call_args.kwargs["prefix"] == ".sv-floor-"
