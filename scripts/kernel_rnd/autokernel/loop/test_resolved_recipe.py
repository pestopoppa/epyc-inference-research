"""Pure tests for immutable resolved serving recipes."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from . import resolved_recipe as rr
from . import serving


BUILD = Path("/build")


def _artifact(role: str, path: str, digit: str) -> dict:
    return {"schema": rr.ARTIFACT_SCHEMA, "role": role, "path": path,
            "sha256": digit * 64}


def _artifacts(recipe: serving.Recipe, *, build: Path = BUILD) -> dict:
    drafter = recipe.spec_decode.get("drafter")
    return {
        "model": _artifact("model", recipe.model, "a"),
        "drafter": (_artifact("drafter", drafter, "b") if drafter else None),
        "executable": _artifact(
            "executable", str(build / "bin" / "llama-server"), "c"),
        "dsos": [_artifact("dso", str(build / "bin" / "libggml.so"), "d")],
    }


def _policy(*keys: str, inherit=(), witness="runtime_set") -> dict:
    return {"schema": rr.ENVIRONMENT_POLICY_SCHEMA, "version": "policy-v1",
            "measurement_keys": list(keys), "allowed_inherit_keys": list(inherit),
            "witnesses": {key: witness for key in keys}}


def _resolve(recipe: serving.Recipe, *, backend="gpu", policy=None, inherited=None,
             artifacts=None, build=BUILD, port=18311):
    return rr.resolve_recipe(
        recipe, build_dir=build,
        artifact_identities=artifacts or _artifacts(recipe, build=build), backend=backend,
        environment_policy=policy or _policy(),
        inherited_environment=inherited or {}, port=port)


def _rehash(value: rr.ResolvedRecipe) -> rr.ResolvedRecipe:
    provisional = rr.ResolvedRecipe(**{**value.__dict__, "snapshot_digest": "0" * 64,
                                      "execution_digest": "0" * 64})
    return rr.ResolvedRecipe(**{
        **provisional.__dict__,
        "snapshot_digest": rr._digest(provisional._snapshot_dict()),
        "execution_digest": rr._digest(provisional._normalized_execution_dict())})


def test_resolved_recipe_is_immutable_round_trips_and_allows_repeated_argv_values():
    recipe = serving.Recipe(name="gpu", model="/models/main.gguf")
    resolved = _resolve(recipe)
    assert resolved.argv.count("8") >= 2
    assert rr.ResolvedRecipe.from_dict(resolved.to_dict()) == resolved
    with pytest.raises(FrozenInstanceError):
        resolved.port = 7
    with pytest.raises(TypeError):
        resolved.launch_env[0][1] = "changed"


def test_resolution_does_not_change_legacy_hash_or_shipped_gpu_commands():
    root = Path(__file__).resolve().parents[4]
    for path in sorted((root / "artifacts/serving-recipes").glob("*.json")):
        recipe = serving.Recipe.load(path)
        before_hash = recipe.recipe_hash
        before_argv = recipe.server_argv(BUILD, 18311)
        resolved = _resolve(recipe)
        assert resolved.template_hash == before_hash == recipe.recipe_hash
        assert list(resolved.argv) == before_argv
        assert resolved.capability.supported


def test_normalized_identity_ignores_only_relocation_label_and_listen_port():
    first_recipe = serving.Recipe(name="first-label", model="/models/one.gguf")
    first = _resolve(first_recipe, build=Path("/build-one"), port=18001)
    moved_recipe = replace(first_recipe, name="other-label", model="/relocated/model.gguf")
    moved = _resolve(moved_recipe, build=Path("/relocated/build"), port=19002)
    assert first.execution_digest == moved.execution_digest
    assert first.snapshot_digest != moved.snapshot_digest


def test_normalized_identity_changes_for_execution_and_artifact_content():
    recipe = serving.Recipe(name="gpu", model="/m", env={"KNOB": "1"})
    base = _resolve(recipe, policy=_policy("KNOB"))
    assert _resolve(replace(recipe, threads=9), policy=_policy("KNOB")).execution_digest \
        != base.execution_digest
    assert _resolve(replace(recipe, spec_decode={"type": "draft-mtp"}),
                    policy=_policy("KNOB")).execution_digest != base.execution_digest
    assert _resolve(replace(recipe, env={"KNOB": "2"}),
                    policy=_policy("KNOB")).execution_digest != base.execution_digest
    artifacts = _artifacts(recipe)
    artifacts["model"]["sha256"] = "e" * 64
    assert _resolve(recipe, policy=_policy("KNOB"),
                    artifacts=artifacts).execution_digest != base.execution_digest
    artifacts = _artifacts(recipe)
    artifacts["dsos"][0]["sha256"] = "f" * 64
    assert _resolve(recipe, policy=_policy("KNOB"),
                    artifacts=artifacts).execution_digest != base.execution_digest
    artifacts = _artifacts(recipe)
    artifacts["dsos"][0]["path"] = "/build/bin/libdifferent-load-name.so"
    assert _resolve(recipe, policy=_policy("KNOB"),
                    artifacts=artifacts).execution_digest != base.execution_digest


def test_parent_environment_is_copied_allowlisted_and_credentials_are_absent():
    recipe = serving.Recipe(name="gpu", model="/m", env={"EMPTY": ""})
    parent = {"PATH": "/usr/bin", "AUTHORIZATION_TOKEN": "do-not-serialize"}
    resolved = _resolve(recipe, policy=_policy("EMPTY", inherit=("PATH",)),
                        inherited=parent)
    parent["PATH"] = "/mutated"
    serialized = json.dumps(resolved.to_dict())
    assert dict(resolved.launch_env)["PATH"] == "/usr/bin"
    assert dict(resolved.relevant_environment)["EMPTY"] == ""
    assert "do-not-serialize" not in serialized
    assert "AUTHORIZATION_TOKEN" not in serialized


def test_inherited_treatment_and_explicit_unset_control_freeze_opposite_readbacks():
    check = ({"field": "THP_enabled", "env": "GGML_NOHUGEPAGE_PROCESS",
              "expect": {"1": "0", serving.UNSET: "1"}},)
    template = serving.Recipe(name="base", model="/m", env_readback=check)
    policy = _policy("GGML_NOHUGEPAGE_PROCESS",
                     inherit=("GGML_NOHUGEPAGE_PROCESS",), witness="recipe_readback")
    parent = {"GGML_NOHUGEPAGE_PROCESS": "1"}
    treatment = _resolve(template, policy=policy, inherited=parent)
    control_recipe = template.with_env(name="control", GGML_NOHUGEPAGE_PROCESS=None)
    control = _resolve(control_recipe, policy=policy, inherited=parent)
    parent["GGML_NOHUGEPAGE_PROCESS"] = "changed-after-resolution"
    assert treatment.readback_expectations == (("THP_enabled", "0"),)
    assert control.readback_expectations == (("THP_enabled", "1"),)
    assert dict(treatment.launch_env)["GGML_NOHUGEPAGE_PROCESS"] == "1"
    assert "GGML_NOHUGEPAGE_PROCESS" not in dict(control.launch_env)


def test_runtime_set_and_master_off_are_declared_unknown_not_inferred_from_env():
    recipe = serving.Recipe(name="gpu", model="/m", env={"RUNTIME_SET": "1",
                                                           "MASTER_OFF": "0"})
    policy = _policy("RUNTIME_SET", "MASTER_OFF")
    policy["witnesses"]["MASTER_OFF"] = "master_off"
    resolved = _resolve(recipe, policy=policy)
    assert {(item.kind, item.status, item.reason) for item in resolved.capability.witnesses} == {
        ("runtime_set", "unknown", "witness_sampler_not_implemented"),
        ("master_off", "unknown", "witness_sampler_not_implemented")}


@pytest.mark.parametrize("field,value", [
    ("temperature", float("nan")), ("top_p", float("inf")), ("np", 0), ("ngl", -1),
])
def test_nonfinite_or_malformed_numeric_fields_refuse(field, value):
    recipe = replace(serving.Recipe(name="gpu", model="/m"), **{field: value})
    with pytest.raises(rr.ResolutionError, match=field):
        _resolve(recipe)


@pytest.mark.parametrize("flag", ["-ngl", "-ngl=999", "--device=none", "-m"])
def test_extra_flags_cannot_override_structured_execution_fields(flag):
    recipe = serving.Recipe(name="cpu", model="/m", device="none", ngl=0,
                            extra_flags=(flag, "999"))
    with pytest.raises(rr.ResolutionError, match="structured launch fields"):
        _resolve(recipe, backend="cpu")


def test_unknown_speculation_type_and_mismatched_artifact_paths_refuse():
    recipe = serving.Recipe(name="gpu", model="/m", spec_decode={"type": "invented"})
    with pytest.raises(rr.ResolutionError, match="unknown recipe speculation"):
        _resolve(recipe)
    recipe = serving.Recipe(name="gpu", model="/m")
    artifacts = _artifacts(recipe)
    artifacts["model"]["path"] = "/other"
    with pytest.raises(rr.ResolutionError, match="model artifact path"):
        _resolve(recipe, artifacts=artifacts)


def test_none_self_and_external_speculation_are_distinct():
    none = serving.Recipe(name="none", model="/m")
    self_draft = replace(none, spec_decode={"type": "draft-mtp"})
    external = replace(none, spec_decode={"type": "draft-dflash", "drafter": "/d",
                                         "ngld": 99})
    assert _resolve(none).capability.speculation == "none"
    assert _resolve(self_draft).capability.speculation == "self_draft"
    assert _resolve(external).capability.speculation == "external_draft"


def test_cpu_requires_explicit_none_device_zero_ngl_and_mixed_draft_is_structured():
    cpu = serving.Recipe(name="cpu", model="/m", device="none", ngl=0)
    assert _resolve(cpu, backend="cpu").capability.supported
    wrong = replace(cpu, ngl=1)
    report = _resolve(wrong, backend="cpu").capability
    assert not report.supported
    assert "cpu_requires_ngl_zero" in {reason.code for reason in report.reasons}
    mixed = replace(cpu, spec_decode={"type": "draft-dflash", "drafter": "/d", "ngld": 99})
    report = _resolve(mixed, backend="cpu").capability
    assert "mixed_cpu_gpu_draft_unsupported" in {reason.code for reason in report.reasons}

    forged_capability = replace(report, supported=True, reasons=())
    forged = _rehash(rr.ResolvedRecipe(**{
        **_resolve(mixed, backend="cpu").__dict__, "capability": forged_capability}))
    with pytest.raises(rr.ResolutionError, match="mixed draft placement"):
        rr.ResolvedRecipe.from_dict(forged.to_dict())


def test_multi_device_and_rpc_are_explicitly_unsupported():
    multi = serving.Recipe(name="gpu", model="/m", device="ROCm0,ROCm1")
    rpc = serving.Recipe(name="gpu", model="/m", extra_flags=("--rpc-server", "x"))
    assert "multi_device_unsupported" in {
        reason.code for reason in _resolve(multi).capability.reasons}
    assert "rpc_unsupported" in {reason.code for reason in _resolve(rpc).capability.reasons}


@pytest.mark.parametrize(("draft_flag", "reason"), [
    (("--device-draft", "ROCm1"), "multi_device_unsupported"),
    (("--device-draft", "banana"), "unsupported_draft_device"),
    (("--device-draft=ROCm0,ROCm1",), "multi_device_unsupported"),
])
def test_external_gpu_draft_device_must_be_valid_and_same_device(draft_flag, reason):
    recipe = serving.Recipe(
        name="gpu", model="/m", spec_decode={"type": "draft-dflash", "drafter": "/d",
                                               "ngld": 99},
        extra_flags=draft_flag)
    report = _resolve(recipe).capability
    assert not report.supported
    assert reason in {item.code for item in report.reasons}


def test_forged_supported_cpu_on_gpu_argv_and_env_inconsistency_refuse_roundtrip():
    resolved = _resolve(serving.Recipe(name="gpu", model="/m"))
    capability = rr.CapabilityReport(
        True, "cpu", resolved.capability.speculation, (), resolved.capability.witnesses,
        "not_applicable", "unproven", "unproven")
    row = _rehash(rr.ResolvedRecipe(**{**resolved.__dict__, "backend": "cpu",
                                       "capability": capability})).to_dict()
    with pytest.raises(rr.ResolutionError, match="supported CPU capability"):
        rr.ResolvedRecipe.from_dict(row)

    forged = _rehash(rr.ResolvedRecipe(**{
        **resolved.__dict__, "launch_env": resolved.launch_env + (("UNAPPROVED", "x"),)}))
    row = forged.to_dict()
    with pytest.raises(rr.ResolutionError, match="exceeds its allowlist"):
        rr.ResolvedRecipe.from_dict(row)


def test_duplicate_dso_paths_and_mutated_typed_policy_refuse():
    recipe = serving.Recipe(name="gpu", model="/m")
    artifacts = _artifacts(recipe)
    artifacts["dsos"].append(dict(artifacts["dsos"][0]))
    with pytest.raises(rr.ResolutionError, match="unique loader"):
        _resolve(recipe, artifacts=artifacts)
    artifacts = _artifacts(recipe)
    duplicate_name = dict(artifacts["dsos"][0])
    duplicate_name["path"] = "/other/location/libggml.so"
    artifacts["dsos"].append(duplicate_name)
    with pytest.raises(rr.ResolutionError, match="unique loader names"):
        _resolve(recipe, artifacts=artifacts)
    malformed = rr.EnvironmentPolicy("v", ("A",), (), ())
    with pytest.raises(rr.ResolutionError, match="witnesses"):
        _resolve(recipe, policy=malformed)
