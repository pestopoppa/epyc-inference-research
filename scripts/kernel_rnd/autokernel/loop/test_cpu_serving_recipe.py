"""The real serving launcher consumes resolved CPU/GPU recipes without live compute."""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

from . import resolved_recipe as rr, serving
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

    with mock.patch.object(serving.subprocess, "Popen", side_effect=popen), \
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
    row = {"recipe_hash": recipe.recipe_hash, "floor_pct": 1.0}
    with mock.patch.object(serving.status, "write_json", side_effect=OSError("fsync fault")) as write:
        with pytest.raises(OSError, match="fsync fault"):
            serving.write_floor(Path("/store"), recipe, row)
    assert write.call_args.kwargs["prefix"] == ".sv-floor-"
